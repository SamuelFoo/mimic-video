# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import huggingface_hub
from huggingface_hub import snapshot_download

"""Download mimic-video models from Hugging Face."""

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


_CHECKPOINTS = [
    "pretrained_cosmos_bridge",
    "finetuned_cosmos_bridge",
    "libero_goal_half",
    "libero_goal_tenth",
    "libero_goal_one",
    "libero_object_full",
    "libero_object_half",
    "libero_object_tenth",
    "libero_object_one",
    "libero_spatial_full",
    "libero_spatial_tenth",
    "libero_spatial_one",
]


def get_checkpoints(policy_name: str) -> tuple[str, ...]:
    if policy_name == "pretrained_cosmos_bridge":
        return (
            "action_decoder/w2a_bridge_v2w_pretrained_cosmos*",
            "video_backbone/v2w_pretrained_cosmos.pt",
            "dataset_statistics/bridge.json",
        )

    if policy_name == "finetuned_cosmos_bridge":
        return (
            "action_decoder/w2a_bridge_v2w_bridge_lora*",
            "video_backbone/v2w_bridge_lora*",
            "dataset_statistics/bridge.json",
        )

    return (
        f"action_decoder/w2a_{policy_name}*",
        f"video_backbone/v2w_{'_'.join(policy_name.split('_')[:2])}*",
        f"dataset_statistics/{policy_name}.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=_CHECKPOINTS,
        choices=_CHECKPOINTS,
        help="Which models to download.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to save the downloaded checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script and print the download commands without actually downloading the files.",
    )
    args = parser.parse_args()
    return args


def main(args):
    args.checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    allow_patterns = [
        "text_encoder/*",
        "video_backbone/tokenizer/*",
        *{ckpt for model in args.models for ckpt in get_checkpoints(model)},
    ]
    revision = huggingface_hub.HfApi().repo_info(repo_id="jonpai/mimic-video").sha

    print(f"Downloading jonpai/mimic-video to {args.checkpoint_dir} ...")
    print(f"Revision: {revision}")

    if not args.dry_run:
        try:
            snapshot_download(
                repo_id="jonpai/mimic-video",
                local_dir=args.checkpoint_dir,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        except Exception as e:
            print(f"\033[91mError downloading jonpai/mimic-video: {e}\033[0m")
    print("-" * 20)

    print("Checkpoint downloading done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
