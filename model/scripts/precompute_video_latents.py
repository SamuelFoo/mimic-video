"""Precompute VAE latents for the mimic-video training and validation datasets.

For each sample i in MimicDataset, runs the frozen Cosmos VAE encoder on the
concatenated obs+action raw video (`[3, obs+action, 480, 640]`, currently
`[3, 21, 480, 640]`) and saves the resulting `[state_ch, state_t, H/8, W/8]`
latent (currently `[16, 6, 60, 80]`) into a single contiguous tensor file.
Training then loads the file with mmap and skips the encoder.

Prerequisite: pixel-space augmentations must be disabled in the transform
config — see cosmos_predict2/configs/dataloading/dataset/transform/. Otherwise
the cached latent reflects one random augmentation roll forever.

Caches live at:
    ${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/{train,val}_<stats_id>.pt

Usage (same flags as scripts/train.py — sources the same experiment config):
    cd mimic-video/model
    python -m scripts.precompute_video_latents \\
        --config=cosmos_predict2/configs/config.py \\
        -- experiment=<exp_name>
"""

import argparse
import importlib
import os
import pathlib

import torch
import torch.utils.data
import tqdm
from loguru import logger as logging

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.defaults.data_action import get_dataset
from imaginaire.lazy_config import instantiate
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override


class _TokenizerOnlyPipe:
    """Minimal shim that implements just `Video2WorldPipeline.encode`.

    The full World2ActionModel instantiates the 1.96B-param video DiT + the
    500M-param action decoder on GPU even though precompute only needs the
    ~100M VAE tokenizer. Skipping them frees ~5 GB of VRAM, which lets us
    push BATCH_SIZE higher and saves ~5s of init.
    """

    def __init__(self, tokenizer, sigma_data: float, state_ch: int, state_t: int) -> None:
        self.tokenizer = tokenizer
        self.sigma_data = sigma_data
        self.state_ch = state_ch
        self.state_t = state_t

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if (T - 1) % 4 != 0 or (C, H, W) != (3, 480, 640):
            raise ValueError(f"Unexpected raw video shape {state.shape}")
        encoded = self.tokenizer.encode(state) * self.sigma_data
        if encoded.shape[2] == self.state_t:
            return encoded
        if encoded.shape[2] > self.state_t:
            raise ValueError(
                f"Encoded latent has too many frames: encoded.shape={encoded.shape}, "
                f"expected state_t={self.state_t}"
            )
        res = torch.zeros(
            (B, self.state_ch, self.state_t, encoded.shape[3], encoded.shape[4]),
            device=state.device,
            dtype=encoded.dtype,
        )
        res[:, :, : encoded.shape[2], :, :] = encoded
        return res

BATCH_SIZE = int(os.environ.get("LATENT_PRECOMPUTE_BATCH_SIZE", 10))
NUM_WORKERS = int(os.environ.get("LATENT_PRECOMPUTE_NUM_WORKERS", 12))
PREFETCH_FACTOR = int(os.environ.get("LATENT_PRECOMPUTE_PREFETCH", 2))


def _encode_split(
    pipe,
    data_config,
    *,
    is_train: bool,
    cache_dir: pathlib.Path,
    latent_shape: tuple[int, int, int, int],
) -> None:
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_rank0 = distributed.is_rank0()

    split_name = "train" if is_train else "val"
    # ChunkReader raises ValueError("ChunkReader on no episodes doesn't make
    # sense.") before we ever get a dataset to call len() on, so the empty-split
    # case has to be caught here rather than via `n == 0` below.
    try:
        dataset = get_dataset(data_config, is_train=is_train)
    except ValueError as e:
        if "no episodes" in str(e):
            if is_rank0:
                logging.warning(f"Empty {split_name} dataset, skipping.")
            return
        raise
    n = len(dataset)
    if n == 0:
        if is_rank0:
            logging.warning(f"Empty {split_name} dataset, skipping.")
        return

    cache_path = cache_dir / f"{split_name}_{dataset.stats_id}.pt"
    if cache_path.exists():
        if is_rank0:
            logging.info(f"{cache_path} already exists; reusing ({n} samples).")
        return

    # DistributedSampler scatters the n dataset indices across ranks. When
    # n % world_size != 0 it pads the tail by repeating early indices — those
    # samples get encoded twice and the merge writes identical data to the
    # same slot, so it's wasted compute, not a correctness issue.
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )
    local_indices = list(sampler)

    # DataLoader overlaps the CPU side (zarr reads + PIL resize in
    # CosmosProcessImage) with the GPU encoder forward.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    local_latents = torch.empty((len(local_indices), *latent_shape), dtype=torch.bfloat16)

    # Progress in samples, not batches — bumps by `bs` per loader iteration so
    # the ETA reflects actual samples-per-second regardless of batch size.
    write_pos = 0
    pbar = tqdm.tqdm(
        total=len(local_indices),
        unit="sample",
        desc=f"Encoding {split_name} [rank {rank}]",
        position=rank,
    )
    for batch in loader:
        obs = batch["obs/workspace_rgb"].cuda(non_blocking=True)
        action = batch["action/workspace_rgb"].cuda(non_blocking=True)
        raw = torch.concat((obs, action), dim=2)  # [B, 3, obs+action, 480, 640]

        with torch.no_grad():
            encoded = pipe.encode(raw)  # [B, state_ch, state_t, H/8, W/8]

        bs = encoded.shape[0]
        local_latents[write_pos : write_pos + bs] = encoded.detach().cpu().to(torch.bfloat16)
        write_pos += bs
        pbar.update(bs)
    pbar.close()

    assert write_pos == len(local_indices), (
        f"Wrote {write_pos} latents but sampler has {len(local_indices)}"
    )

    # Each rank writes a shard tagged with its source indices; rank 0 then
    # scatters them into one merged tensor. Same path for world_size == 1.
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / f".{split_name}_{dataset.stats_id}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank{rank:04d}.pt"
    torch.save(
        {
            "indices": torch.tensor(local_indices, dtype=torch.int64),
            "latents": local_latents,
        },
        shard_path,
    )

    distributed.barrier()

    if not is_rank0:
        return

    merged = torch.empty((n, *latent_shape), dtype=torch.bfloat16)
    for r in range(world_size):
        shard = torch.load(shard_dir / f"rank{r:04d}.pt", map_location="cpu")
        merged[shard["indices"]] = shard["latents"]

    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save(merged, tmp_path)
    tmp_path.rename(cache_path)
    size_gb = cache_path.stat().st_size / 1e9
    logging.info(f"Saved {n} {split_name} latents to {cache_path} ({size_gb:.2f} GB)")

    for r in range(world_size):
        (shard_dir / f"rank{r:04d}.pt").unlink()
    shard_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute VAE latents.")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Hydra-style overrides after `--`.")
    args = parser.parse_args()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)

    distributed.init()
    config.validate()
    config.freeze()

    dataset_dir = os.environ.get("MIMIC_VIDEO_DATASET_DIR")
    if not dataset_dir:
        raise RuntimeError(
            "MIMIC_VIDEO_DATASET_DIR is not set. Source it the same way the training "
            "script does (it should match the staged dataset dir)."
        )
    cache_dir = pathlib.Path(dataset_dir) / ".latent_cache"

    # Skip the 2.46B-param model — we only need the VAE tokenizer to encode.
    # Build it directly from the same config the model would have used,
    # bypassing the DiT and action decoder entirely.
    logging.info("Loading VAE tokenizer (skipping DiT and action decoder)...")
    video_pipe_config = get_cosmos_predict2_video2world_pipeline(
        model_size="2B", resolution="480", fps=10,
    )
    # TokenizerInterface is not an nn.Module; the underlying VAE loads on CUDA
    # by default and is set to eval/no-grad in its own constructor.
    tokenizer = instantiate(video_pipe_config.tokenizer)
    pipe = _TokenizerOnlyPipe(
        tokenizer,
        sigma_data=video_pipe_config.sigma_data,
        state_ch=video_pipe_config.state_ch,
        state_t=video_pipe_config.state_t,
    )

    sf = tokenizer.spatial_compression_factor
    latent_shape = (video_pipe_config.state_ch, video_pipe_config.state_t, 480 // sf, 640 // sf)

    # data_config is a LazyCall that returns the resolved Hydra DictConfig; we
    # need the concrete one to pass to get_dataset(is_train=...).
    data_config = instantiate(config.data_config)

    _encode_split(pipe, data_config, is_train=True, cache_dir=cache_dir, latent_shape=latent_shape)
    _encode_split(pipe, data_config, is_train=False, cache_dir=cache_dir, latent_shape=latent_shape)

    logging.info("Done.")


if __name__ == "__main__":
    main()
