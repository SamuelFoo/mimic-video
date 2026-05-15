import pathlib
from collections.abc import Sequence
from typing import TypeVar

from cosmos_predict2.data.action.types import NormalizationType

T = TypeVar("T", bound=float)


def get_paths(
    data_dir: pathlib.Path,
    *,
    verbose: bool = False,
) -> list[pathlib.Path]:
    # Cache removed: glob is ~30 ms on a 100-episode local-disk dataset
    return sorted(data_dir.glob("**/*.zarr"))


def dict_apply(x: dict, func) -> dict:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def extract_normalization_types(policy_io: dict[str, dict]) -> dict[str, NormalizationType]:
    return {
        f"{category}/{key}": val["normalization_type"]
        for category, values in policy_io.items()
        for key, val in values.items()
    }


def linear_search_with_initial_guess_left(seq: Sequence[T], target: T, initial_guess: int) -> int:
    if seq[0] >= target:
        return 0
    if seq[-1] <= target:
        return len(seq) - 1

    res = max(0, min(len(seq) - 1, initial_guess))
    while True:
        if seq[res] <= target:
            if seq[res + 1] > target:
                return res
            res += 1
        else:
            res -= 1


def linear_search_with_initial_guess_right(seq: Sequence[T], target: T, initial_guess: int) -> int:
    if seq[0] >= target:
        return 0
    if seq[-1] <= target:
        return len(seq) - 1

    res = max(0, min(len(seq) - 1, initial_guess))
    while True:
        if seq[res] >= target:
            if seq[res - 1] < target:
                return res
            res -= 1
        else:
            res += 1
