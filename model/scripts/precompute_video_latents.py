"""Precompute VAE latents for the mimic-video training and validation datasets.

For each sample i in MimicDataset, runs the frozen Cosmos VAE encoder on the
concatenated obs+action raw video (`[3, 61, 480, 640]`) and saves the resulting
`[16, 16, 60, 80]` latent into a single contiguous tensor file. Training then
loads the file with mmap and skips the encoder.

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

    def __init__(self, tokenizer, sigma_data: float) -> None:
        self.tokenizer = tokenizer
        self.sigma_data = sigma_data

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T not in {1, 5, 61} or (C, H, W) != (3, 480, 640):
            raise ValueError(f"Unexpected raw video shape {state.shape}")
        encoded = self.tokenizer.encode(state) * self.sigma_data
        if encoded.shape[2] == 16:
            return encoded
        res = torch.zeros((B, 16, 16, 60, 80), device=state.device, dtype=state.dtype)
        res[:, :, : encoded.shape[2], :, :] = encoded
        return res

BATCH_SIZE = int(os.environ.get("LATENT_PRECOMPUTE_BATCH_SIZE", 10))
NUM_WORKERS = int(os.environ.get("LATENT_PRECOMPUTE_NUM_WORKERS", 12))
PREFETCH_FACTOR = int(os.environ.get("LATENT_PRECOMPUTE_PREFETCH", 2))
LATENT_SHAPE = (16, 16, 60, 80)  # (T_latent, C, H, W) per sample


def _encode_split(pipe, data_config, *, is_train: bool, cache_dir: pathlib.Path) -> None:
    dataset = get_dataset(data_config, is_train=is_train)
    n = len(dataset)
    split_name = "train" if is_train else "val"
    if n == 0:
        logging.warning(f"Empty {split_name} dataset, skipping.")
        return

    cache_path = cache_dir / f"{split_name}_{dataset.stats_id}.pt"
    if cache_path.exists():
        logging.info(f"{cache_path} already exists; reusing ({n} samples).")
        return

    latents = torch.empty((n, *LATENT_SHAPE), dtype=torch.bfloat16)

    # DataLoader overlaps the CPU side (zarr reads + PIL resize in
    # CosmosProcessImage) with the GPU encoder forward. shuffle=False keeps
    # cache index == dataset index.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    # Progress in samples, not batches — bumps by `bs` per loader iteration so
    # the ETA reflects actual samples-per-second regardless of batch size.
    write_pos = 0
    pbar = tqdm.tqdm(total=n, unit="sample", desc=f"Encoding {split_name}")
    for batch in loader:
        obs = batch["obs/workspace_rgb"].cuda(non_blocking=True)
        action = batch["action/workspace_rgb"].cuda(non_blocking=True)
        raw = torch.concat((obs, action), dim=2)  # [B, 3, 61, 480, 640]

        with torch.no_grad():
            encoded = pipe.encode(raw)  # [B, 16, 16, 60, 80]

        bs = encoded.shape[0]
        latents[write_pos : write_pos + bs] = encoded.detach().cpu().to(torch.bfloat16)
        write_pos += bs
        pbar.update(bs)
    pbar.close()

    assert write_pos == n, f"Wrote {write_pos} latents but dataset has {n}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".pt.tmp")
    torch.save(latents, tmp_path)
    tmp_path.rename(cache_path)
    size_gb = cache_path.stat().st_size / 1e9
    logging.info(f"Saved {n} {split_name} latents to {cache_path} ({size_gb:.2f} GB)")


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
    pipe = _TokenizerOnlyPipe(tokenizer, sigma_data=video_pipe_config.sigma_data)

    # data_config is a LazyCall that returns the resolved Hydra DictConfig; we
    # need the concrete one to pass to get_dataset(is_train=...).
    data_config = instantiate(config.data_config)

    _encode_split(pipe, data_config, is_train=True, cache_dir=cache_dir)
    _encode_split(pipe, data_config, is_train=False, cache_dir=cache_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
