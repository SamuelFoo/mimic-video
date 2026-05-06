import abc
import re
import typing
from collections.abc import Iterable, Iterator
from functools import partial
from operator import itemgetter, methodcaller

import einops
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from cosmos_predict2.data.action import convert_pose_repr
from cosmos_predict2.data.action.types import SCIPY_ROTATION_CONVERSIONS, LieRepr, ObsMeta, ObsType


class DataTransform(abc.ABC):
    def __init__(
        self, targets: list[str], metas: dict[str, ObsMeta], additional_data: dict[str, str] | None = None
    ) -> None:
        self._targets = targets
        self._metas = metas
        self._additional_data = additional_data or {}

    @abc.abstractmethod
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        raise NotImplementedError

    @property
    def targets(self) -> list[str]:
        return self._targets

    @property
    def additional_data(self) -> dict[str, str]:
        return self._additional_data

    @property
    def new_components(self) -> dict[str, dict]:
        return {}

    @property
    @abc.abstractmethod
    def remove_original(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ignore_for_normalization(self) -> bool:
        raise NotImplementedError

    @property
    def train_only(self) -> bool:
        return False


class DeconstructPoseMat(DataTransform):
    def __init__(self, out_key_prefix: str, **kwargs):
        super().__init__(**kwargs)

        self._out_key_prefix = out_key_prefix

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        """Assumes input is a single pose matrix."""
        mat = targets[0][1]

        rot = mat[:, :3, :3]
        yield f"{self._out_key_prefix}_rot_lowdim", rot

        trans = mat[:, :3, 3]
        yield f"{self._out_key_prefix}_pos_lowdim", trans

    @property
    def new_components(self) -> dict[str, dict]:
        pose_mat_meta = next(
            (item for key, item in self._metas.items() if re.search(self._targets[0], key) is not None)
        )
        return {
            f"{self._out_key_prefix}_rot_lowdim": {"obs_type": ObsType.ROTATION_MATRIX, "repr": pose_mat_meta["repr"]},
            f"{self._out_key_prefix}_pos_lowdim": {"obs_type": ObsType.CARTESIAN_POS, "repr": pose_mat_meta["repr"]},
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class Concat(DataTransform):
    def __init__(self, out_key: str, **kwargs):
        super().__init__(**kwargs)

        self._out_key = out_key

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        """Assumes all actions are flattened and have the same horizon."""
        _, targets = zip(*targets, strict=False)
        yield self._out_key, np.concatenate(targets, axis=-1)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True


class Flatten(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, value in targets:
            T, *_ = value.shape
            yield key, value.reshape(T, -1).astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class ConvertLowdimRepr(DataTransform):
    def __init__(self, relative_base_value_idx: int | None = None, *, is_action: bool | None = None, **kwargs):
        super().__init__(**kwargs)

        self._relative_base_value_idx = relative_base_value_idx
        self._is_action = is_action

    def __call__(
        self, targets: list[tuple[str, np.ndarray]], relative_base_value: np.ndarray | None = None
    ) -> Iterator[tuple[str, np.ndarray]]:
        kwargs = (
            {"relative_base_value": relative_base_value[self._relative_base_value_idx]}
            if relative_base_value is not None
            else {"is_action": self._is_action}
        )
        for key, value in targets:
            res = convert_pose_repr.convert_to_repr(
                value,
                self._metas[key]["obs_type"],
                typing.cast(LieRepr, self._metas[key]["repr"]),
                self._metas[key]["target_repr"],
                **kwargs,  # ty:ignore[invalid-argument-type]
            )
            yield key, res

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "repr": meta["target_repr"]}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class RotationMatrixTo6D(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, rot_mat in targets:
            yield key, rot_mat[:, :2].reshape(-1, 6)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class ToRotationMatrix(DataTransform):
    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, rot in targets:
            rot = SCIPY_ROTATION_CONVERSIONS[self._metas[key]["obs_type"]][0](rot)
            yield key, rot.as_matrix()

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "obs_type": ObsType.ROTATION_MATRIX}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return False


class CosmosProcessImage(DataTransform):
    def __init__(
        self,
        resize_sizes: dict[str, tuple[int, int]] | tuple[int, int],
        obs_types: dict[str, ObsType] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._resize_sizes = resize_sizes
        self._obs_types = obs_types or {}

    def _get_resize_sizes(self, key: str) -> tuple[int, int]:
        return self._resize_sizes[key] if isinstance(self._resize_sizes, dict) else self._resize_sizes

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        for key, imgs in targets:
            imgs = np.stack(
                [transforms.Resize(self._get_resize_sizes(key))(PIL.Image.fromarray(img, "RGB")) for img in imgs],
                axis=0,
            )
            imgs = einops.rearrange(imgs, "t h w c -> c t h w")
            yield key, 2.0 * (imgs.astype(np.float32) / 255.0 - 0.5)

    @property
    def new_components(self) -> dict[str, dict]:
        return {
            key: {**meta, "obs_type": ObsType.RGB}
            for key, meta in self._metas.items()
            if any(re.search(pattern, key) is not None for pattern in self._targets)
        }

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True


class RandomPhotometric(DataTransform):
    """Per-call brightness, contrast, and per-channel scale jitter.

    Operates on float images already produced by `CosmosProcessImage` (range
    [-1, 1], shape `c t h w`). Samples one set of params per call and applies
    them to every matching key, so multiple cameras / obs+action streams of
    the same frame share lighting consistently.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        channel: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._brightness = brightness
        self._contrast = contrast
        self._channel = channel

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        b = np.random.uniform(-self._brightness, self._brightness)
        c = np.random.uniform(1.0 - self._contrast, 1.0 + self._contrast)
        ch = np.random.uniform(1.0 - self._channel, 1.0 + self._channel, size=(3, 1, 1, 1)).astype(np.float32)
        for key, imgs in targets:
            mean = imgs.mean(axis=(2, 3), keepdims=True)
            out = (imgs - mean) * c + mean + b
            out = out * ch
            yield key, np.clip(out, -1.0, 1.0).astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class RandomShiftCrop(DataTransform):
    """Edge-pad then crop a window of original size at a random offset.

    Standard DrQ-v2 style translational augmentation. One offset is sampled
    per call and applied to every matching key (and to all timesteps within
    each key), so the augmentation is temporally consistent and consistent
    across obs/action streams of the same frames.
    """

    def __init__(self, max_shift_pixels: int = 4, **kwargs):
        super().__init__(**kwargs)
        self._max_shift = int(max_shift_pixels)

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        if self._max_shift <= 0:
            for key, imgs in targets:
                yield key, imgs
            return

        dy = int(np.random.randint(0, 2 * self._max_shift + 1))
        dx = int(np.random.randint(0, 2 * self._max_shift + 1))
        s = self._max_shift
        for key, imgs in targets:
            _, _, h, w = imgs.shape
            padded = np.pad(imgs, ((0, 0), (0, 0), (s, s), (s, s)), mode="edge")
            yield key, padded[:, :, dy : dy + h, dx : dx + w].astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class RandomErasing(DataTransform):
    """Zero-fill a random rectangle of the image (per-call same patch across keys/time).

    Useful as a mild occlusion regularizer. Disabled by default in our
    transform yaml; opt in if your task does not depend on always-visible
    fine details (gripper, small target objects).
    """

    def __init__(self, prob: float = 0.25, max_area_frac: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self._prob = prob
        self._max_area_frac = max_area_frac

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        if np.random.rand() >= self._prob:
            for key, imgs in targets:
                yield key, imgs
            return

        # Use the first target's spatial size to pick a patch.
        _, first = targets[0]
        _, _, h, w = first.shape
        area = h * w * np.random.uniform(0.0, self._max_area_frac)
        ph = int(np.clip(np.sqrt(area), 1, h))
        pw = int(np.clip(area / max(ph, 1), 1, w))
        y0 = int(np.random.randint(0, max(h - ph, 1)))
        x0 = int(np.random.randint(0, max(w - pw, 1)))
        for key, imgs in targets:
            out = imgs.copy()
            out[:, :, y0 : y0 + ph, x0 : x0 + pw] = 0.0
            yield key, out

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class GaussianNoise(DataTransform):
    """Per-pixel additive Gaussian noise on float [-1, 1] images.

    One std is sampled per call; same std is used across all keys, but each
    pixel/frame draws independent noise. Cheap sensor-noise simulation.
    """

    def __init__(self, std_range: tuple[float, float] = (0.0, 0.04), **kwargs):
        super().__init__(**kwargs)
        self._std_range = tuple(std_range)

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        std = float(np.random.uniform(*self._std_range))
        for key, imgs in targets:
            if std <= 0.0:
                yield key, imgs
                continue
            noise = np.random.normal(0.0, std, size=imgs.shape).astype(np.float32)
            yield key, np.clip(imgs + noise, -1.0, 1.0).astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class GaussianBlur(DataTransform):
    """Random Gaussian blur via torchvision.

    Same sigma applied to every key/frame in the call (consistent defocus
    across multi-cam streams of the same physical frame).
    """

    def __init__(self, prob: float = 0.3, sigma_range: tuple[float, float] = (0.1, 1.5), **kwargs):
        super().__init__(**kwargs)
        self._prob = float(prob)
        self._sigma_range = tuple(sigma_range)

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        if np.random.rand() >= self._prob:
            for key, imgs in targets:
                yield key, imgs
            return

        sigma = float(np.random.uniform(*self._sigma_range))
        ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
        for key, imgs in targets:
            t = torch.from_numpy(imgs)  # (C, T, H, W)
            t = t.permute(1, 0, 2, 3)  # (T, C, H, W) -- treat T as batch
            t = TF.gaussian_blur(t, kernel_size=ksize, sigma=sigma)
            t = t.permute(1, 0, 2, 3).contiguous()
            yield key, t.numpy().astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class MotionBlur(DataTransform):
    """Linear motion blur with random angle and length.

    Builds a 1-D line kernel rotated to a random angle and convolves all
    targets with it. Same kernel applied across keys for consistency.
    """

    def __init__(self, prob: float = 0.2, kernel_size_range: tuple[int, int] = (3, 9), **kwargs):
        super().__init__(**kwargs)
        self._prob = float(prob)
        lo, hi = kernel_size_range
        # ensure odd kernel sizes
        self._sizes = [k for k in range(int(lo), int(hi) + 1) if k % 2 == 1]
        if not self._sizes:
            raise ValueError("MotionBlur kernel_size_range must contain at least one odd integer")

    def _build_kernel(self) -> torch.Tensor:
        ksize = int(np.random.choice(self._sizes))
        angle = float(np.random.uniform(0.0, np.pi))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        c = ksize // 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(ksize):
            off = i - c
            x = int(round(c + off * cos_a))
            y = int(round(c + off * sin_a))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        kernel /= max(kernel.sum(), 1.0)
        return torch.from_numpy(kernel).view(1, 1, ksize, ksize)

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        if np.random.rand() >= self._prob:
            for key, imgs in targets:
                yield key, imgs
            return

        kernel = self._build_kernel()
        ksize = kernel.shape[-1]
        pad = ksize // 2
        for key, imgs in targets:
            c, t, h, w = imgs.shape
            x = torch.from_numpy(imgs).reshape(c * t, 1, h, w)
            x = torch.nn.functional.conv2d(x, kernel, padding=pad)
            x = x.reshape(c, t, h, w).contiguous()
            yield key, x.numpy().astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


class HueShift(DataTransform):
    """Random hue rotation in HSV space.

    `max_shift` is in units of full-circle hue (0.5 = 180 degrees). Keep
    small (~0.05) for tasks that depend on color cues.
    """

    def __init__(self, max_shift: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self._max_shift = float(max_shift)

    def __call__(self, targets: list[tuple[str, np.ndarray]]) -> Iterator[tuple[str, np.ndarray]]:
        delta = float(np.random.uniform(-self._max_shift, self._max_shift))
        for key, imgs in targets:
            t = torch.from_numpy(imgs)  # (C, T, H, W) in [-1, 1]
            t01 = ((t + 1.0) * 0.5).clamp(0.0, 1.0)
            t01 = t01.permute(1, 0, 2, 3)  # (T, C, H, W)
            t01 = TF.adjust_hue(t01, delta)
            t01 = t01.permute(1, 0, 2, 3).contiguous()
            out = t01 * 2.0 - 1.0
            yield key, out.numpy().astype(np.float32)

    @property
    def remove_original(self) -> bool:
        return True

    @property
    def ignore_for_normalization(self) -> bool:
        return True

    @property
    def train_only(self) -> bool:
        return True


def make_data_transforms(
    data_transform_specs: list[dict], data_components: dict[str, ObsMeta]
) -> Iterator[DataTransform]:
    new_components = {}
    for data_transform_spec in data_transform_specs:
        data_transform_cls = globals()[data_transform_spec.pop("name")]
        data_transform = data_transform_cls(**data_transform_spec, metas=data_components | new_components)
        new_components |= data_transform.new_components
        yield data_transform


def apply_data_transforms(
    data: dict,
    ops: Iterable[DataTransform],
    *,
    should_ignore_transforms_for_norm: bool,
    is_train: bool = True,
) -> dict:
    for op in ops:
        if should_ignore_transforms_for_norm and op.ignore_for_normalization:
            continue
        if (not is_train) and op.train_only:
            continue

        get_fn = partial(methodcaller, "pop") if op.remove_original else itemgetter
        targets = [
            (key, get_fn(key)(data))
            for pattern in op.targets
            for key in sorted(data)
            if re.search(pattern, key) is not None
        ]

        if targets:
            additional_data = {key: data[val] for key, val in op.additional_data.items()}
            data.update(op(targets, **additional_data))

    return data
