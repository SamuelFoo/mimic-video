"""Convert a LeRobot dataset to mimic-video's per-episode zarr layout.

Output layout (one .zarr directory per episode):
    workspace_rgb               (T, H, W, 3) uint8
    workspace_rgb_timestamps    (T,) uint64 nanoseconds
    joint_pos_lowdim            (T, D_state) float32
    joint_pos_lowdim_timestamps (T,) uint64
    joint_action_lowdim         (T, D_action) float32
    joint_action_lowdim_timestamps (T,) uint64
    language_instruction        (1,) bytes
    language_instruction_timestamps (1,) uint64

After running this, run ``precompute_t5.py --dataset-path <out_dir>`` to add
the `language_embedding` / `language_embedding_timestamps` keys.

Notes:
  * This script depends on the ``lerobot`` package (the LeRobotDataset class).
    The preprocessing step does not need the cosmos uv env -- run it from
    whichever env has lerobot installed (e.g. the existing ``lerobot`` conda
    env that train_act.sbatch uses).
  * LeRobot v3 datasets store image frames inside encoded videos. Reading a
    single frame at a time is slow; we therefore iterate the dataset
    sequentially per-episode, which lets the underlying decoder reuse decode
    state.
"""

import argparse
import logging
import pathlib
from typing import Iterable

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

from lerobot.datasets.lerobot_dataset import LeRobotDataset

NS_PER_SEC = 1_000_000_000
COMP = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)


def _to_numpy_uint8_image(img) -> np.ndarray:
    """LeRobot returns image tensors as float32 in [0, 1] with shape (C, H, W)."""
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _to_numpy_vec(x) -> np.ndarray:
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return arr.astype(np.float32).reshape(-1)


def _episode_frame_indices(ds: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    """Return [start, end) frame indices for the given episode."""
    # LeRobotDataset exposes per-episode frame ranges via episode_data_index
    # or the meta object; cover both APIs to be robust across versions.
    edi = getattr(ds, "episode_data_index", None)
    if edi is not None:
        return int(edi["from"][ep_idx]), int(edi["to"][ep_idx])
    meta = getattr(ds, "meta", None)
    if meta is not None and hasattr(meta, "episodes"):
        ep = meta.episodes[ep_idx]
        # lerobot v3 uses dataset_from_index/dataset_to_index; older versions used from/to.
        if "dataset_from_index" in ep:
            return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        return int(ep["from"]), int(ep["to"])
    raise RuntimeError("Could not locate per-episode frame index on LeRobotDataset.")


def _resolve_task_string(ds: LeRobotDataset, ep_idx: int, default: str) -> str:
    meta = getattr(ds, "meta", None)
    if meta is None:
        return default
    # v3 stores tasks per frame; pull the most common task for this episode.
    try:
        episodes = meta.episodes
        ep = episodes[ep_idx]
        tasks = ep.get("tasks") or ep.get("task")
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
        if isinstance(tasks, str) and tasks:
            return tasks
    except Exception:
        pass
    return default


def _write_array(root: zarr.Group, name: str, data: np.ndarray, t_chunk: int) -> None:
    chunks = (min(t_chunk, data.shape[0]), *data.shape[1:]) if data.ndim > 1 else (min(t_chunk, data.shape[0]),)
    root.create_dataset(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        compressor=COMP,
        overwrite=True,
    )[...] = data


def process_episode(
    ds: LeRobotDataset,
    ep_idx: int,
    image_key: str,
    state_key: str,
    action_key: str,
    out_path: pathlib.Path,
    overwrite: bool,
    default_lang: str,
    language_instruction: str | None,
    fps: float | None,
) -> str:
    if out_path.exists() and not overwrite:
        return f"skip (exists): {out_path}"

    start, end = _episode_frame_indices(ds, ep_idx)
    n = end - start
    if n <= 0:
        return f"skip (empty): episode {ep_idx}"

    # Use dataset fps if not explicitly provided.
    fps_eff = fps if fps is not None else float(getattr(ds.meta, "fps", 30.0))
    dt_ns = NS_PER_SEC / fps_eff
    timestamps = (np.arange(n, dtype=np.uint64) * np.uint64(dt_ns)).astype(np.uint64)

    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for i in range(start, end):
        sample = ds[i]
        images.append(_to_numpy_uint8_image(sample[image_key]))
        states.append(_to_numpy_vec(sample[state_key]))
        actions.append(_to_numpy_vec(sample[action_key]))

    images_np = np.stack(images, axis=0)  # (T, H, W, 3)
    states_np = np.stack(states, axis=0).astype(np.float32)
    actions_np = np.stack(actions, axis=0).astype(np.float32)

    if not (images_np.shape[0] == states_np.shape[0] == actions_np.shape[0] == n):
        return f"skip (length mismatch): episode {ep_idx}"

    lang = language_instruction or _resolve_task_string(ds, ep_idx, default=default_lang)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_img = min(65, n)
    t_ld = min(1024, n)

    root = zarr.open(str(out_path), mode="w")

    _write_array(root, "workspace_rgb", images_np, t_img)
    _write_array(root, "workspace_rgb_timestamps", timestamps, t_ld)

    _write_array(root, "joint_pos_lowdim", states_np, t_ld)
    _write_array(root, "joint_pos_lowdim_timestamps", timestamps, t_ld)

    _write_array(root, "joint_action_lowdim", actions_np, t_ld)
    _write_array(root, "joint_action_lowdim_timestamps", timestamps, t_ld)

    root.create_dataset(
        "language_instruction",
        shape=(1,),
        dtype=bytes,
        chunks=(1,),
        compressor=COMP,
        overwrite=True,
    )[...] = np.array([lang.encode()])
    root.create_dataset(
        "language_instruction_timestamps",
        shape=(1,),
        dtype="uint64",
        chunks=(1,),
        compressor=COMP,
        overwrite=True,
    )[...] = np.array([0], dtype=np.uint64)

    return f"ok: {out_path} (T={n})"


def iter_episode_indices(ds: LeRobotDataset, only: Iterable[int] | None) -> Iterable[int]:
    total = ds.meta.total_episodes if hasattr(ds.meta, "total_episodes") else len(ds.episode_data_index["from"])
    if only is not None:
        only = list(only)
        for i in only:
            if i < 0 or i >= total:
                raise ValueError(f"episode index {i} out of range [0, {total})")
        return only
    return range(total)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="LeRobot repo_id, e.g. user/record-test-merged")
    p.add_argument("--root", type=pathlib.Path, default=None, help="Local LeRobot dataset root (skips HF download)")
    p.add_argument("--output-dir", type=pathlib.Path, required=True, help="Where to write per-episode zarrs")
    p.add_argument("--image-key", default="observation.images.front", help="LeRobot image feature name")
    p.add_argument("--state-key", default="observation.state", help="LeRobot proprio feature name")
    p.add_argument("--action-key", default="action", help="LeRobot action feature name")
    p.add_argument("--default-lang", default="perform the demonstrated task.", help="Fallback language string")
    p.add_argument("--language-instruction", default=None, help="Override language string for all episodes")
    p.add_argument("--fps", type=float, default=None, help="Override dataset fps")
    p.add_argument("--episodes", type=int, nargs="+", default=None, help="Only process these episode indices")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ds_kwargs = {"repo_id": args.repo_id}
    if args.root is not None:
        ds_kwargs["root"] = args.root
    ds = LeRobotDataset(**ds_kwargs)

    indices = list(iter_episode_indices(ds, args.episodes))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in tqdm.tqdm(indices, desc=f"Converting {args.repo_id}"):
        out_path = args.output_dir / f"episode_{ep_idx:06d}.zarr"
        msg = process_episode(
            ds,
            ep_idx=ep_idx,
            image_key=args.image_key,
            state_key=args.state_key,
            action_key=args.action_key,
            out_path=out_path,
            overwrite=args.overwrite,
            default_lang=args.default_lang,
            language_instruction=args.language_instruction,
            fps=args.fps,
        )
        logging.info(msg)


if __name__ == "__main__":
    main()
