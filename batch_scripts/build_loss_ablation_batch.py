#!/usr/bin/env python3
"""Build the matched PE-L14 loss-ablation training matrix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs/main_models_trace_pe_l14.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/generated/loss_ablation_pe_l14_128_3seed.json"
PROTOCOLS = {"barista_s1", "breakfast_s1", "mpii_attr"}
ARMS: tuple[tuple[str, dict[str, object]], ...] = (
    ("full", {}),
    ("no_concept_forecast", {"concept_forecast_loss_weight": 0.0}),
    (
        "no_transition_tolerance",
        {"forecast_transition_tolerance_radius": 0, "forecast_transition_tolerance_weight": 0.0},
    ),
    (
        "no_sil_penalty",
        {"activity_sil_false_positive_penalty": 0.0, "forecast_sil_false_positive_penalty": 0.0},
    ),
    ("no_classifier_l1", {"classifier_l1_weight": 0.0}),
)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.base_config.read_text(encoding="utf-8"))
    defaults = copy.deepcopy(source["defaults"])
    seeds = [42] if args.smoke else parse_seeds(args.seeds)
    if args.smoke:
        defaults.update(
            {
                "num_epochs": 1,
                "patience": 1,
                "no_wandb": True,
                "wandb_mode": "offline",
                "max_sequences_per_split": 4,
                "batch_size": 4,
            }
        )

    selected_runs = [
        source_run
        for source_run in source["runs"]
        if any(str(source_run["name"]).startswith(f"{protocol}_trace_") for protocol in PROTOCOLS)
    ]
    selected_names = {
        str(source_run["name"]).split("_trace_", maxsplit=1)[0]
        for source_run in selected_runs
    }
    if selected_names != PROTOCOLS:
        raise ValueError(
            f"Loss ablation protocol mismatch: expected={sorted(PROTOCOLS)}, "
            f"actual={sorted(selected_names)}"
        )

    runs: list[dict[str, object]] = []
    for source_run in selected_runs:
        for arm, overrides in ARMS:
            run = copy.deepcopy(source_run)
            run["name"] = str(run["name"]).replace("_trace_", f"_loss_{arm}_")
            run.update(copy.deepcopy(overrides))
            runs.append(run)

    payload = {
        "batch_name": "iclr2027_loss_ablation_pe_l14_128_3seed" + ("_smoke" if args.smoke else ""),
        "batch_output_dir": source.get("batch_output_dir", "runs/train_models"),
        "gpus": source.get("gpus", [0, 1, 2, 3]),
        "workers_per_gpu": source.get("workers_per_gpu", 2),
        "max_parallel": source.get("max_parallel", 8),
        "seeds": seeds,
        "defaults": defaults,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {len(ARMS)} loss arms x "
        f"{len(selected_runs)} protocols x {len(seeds)} seeds = {len(runs) * len(seeds)} runs"
    )


if __name__ == "__main__":
    main()
