from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_ROOT = PROJECT_ROOT / "utils"
import sys

for path in (PROJECT_ROOT, UTILS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.dataset_handling import process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a dataset with multiple video backbones.")
    parser.add_argument("--dataset", default="mpii_cooking_2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["res50", "b32", "l14", "pe-l14", "pe-g14"],
        help="Backbones to embed sequentially.",
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--pe-video-batch-size", type=int, default=None)
    parser.add_argument("--pe-target-t", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-intermediate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-embedder-pickle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-portable-state-pickle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for model in args.models:
        pe_video_batch_size = args.pe_video_batch_size
        if pe_video_batch_size is None:
            if model == "pe-g14":
                pe_video_batch_size = 1
            elif model == "pe-l14":
                pe_video_batch_size = 1
        print(
            f"[embed] dataset={args.dataset} model={model} window={args.window_size} "
            f"num_gpus={args.num_gpus} pe_video_batch_size={pe_video_batch_size}",
            flush=True,
        )
        output_path = process_dataset(
            args.dataset,
            model,
            window_size=args.window_size,
            random=args.random_windows,
            batch_size=args.batch_size,
            pe_video_batch_size=pe_video_batch_size,
            pe_target_T=args.pe_target_t,
            enable_tf32=args.enable_tf32,
            seed=args.seed,
            num_gpus=args.num_gpus,
            use_parallel=True,
            cleanup_intermediate=args.cleanup_intermediate,
            save_embedder_pickle=args.save_embedder_pickle,
            save_portable_state_pickle=args.save_portable_state_pickle,
        )
        print(f"[embed] completed {model}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
