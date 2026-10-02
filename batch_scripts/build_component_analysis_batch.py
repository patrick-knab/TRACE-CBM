#!/usr/bin/env python3
"""Build the single-split factorial component analysis for TRACE.

The full jointly trained checkpoint is reused from the PE-L14 main batch. This
builder trains the other three cells of the calibration x transition design on
BARISTA s1, Breakfast official_s1, and MPII Cooking 2 Attr.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs/main_models_trace_pe_l14.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/generated/component_analysis_train.json"
PROTOCOLS = {"barista_s1", "breakfast_s1", "mpii_attr"}
NO_TRANSITIONS: dict[str, object] = {
    "motif_z_attention_layers": 0,
    "st_graph_layers": 0,
    "st_task_graph_layers": 0,
    "st_spatial_top_k": 0,
    "st_temporal_top_k": 0,
    "st_enable_same_concept_temporal": False,
    "st_enable_cross_temporal": False,
    "st_cross_temporal_top_k": 0,
    "st_past_context_mode": "none",
    # A no-graph arm is a persistence baseline, not an empty graph stack:
    # every forecast horizon reads the current concept state directly.
    "st_forecast_rollout_mode": "persistence",
    "st_activity_feedback_mode": "none",
    "st_activity_feedback_top_k": 0,
    "st_activity_feedback_history_steps": 0,
    "transition_prior_enabled": False,
    "transition_prior_weight": 0.0,
}
ARMS: tuple[tuple[str, dict[str, object]], ...] = (
    ("no_transitions", NO_TRANSITIONS),
    (
        "no_calibration",
        {
            "use_concept_calibrator": False,
        },
    ),
    (
        "neither",
        {
            **NO_TRANSITIONS,
            "use_concept_calibrator": False,
        },
    ),
)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.base_config.read_text(encoding="utf-8"))
    defaults = copy.deepcopy(source["defaults"])
    base_hparams = copy.deepcopy(defaults["model_hparams"])
    seeds = [42] if args.smoke else parse_int_list(args.seeds)
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
            f"Single-split protocol mismatch: expected={sorted(PROTOCOLS)}, "
            f"actual={sorted(selected_names)}"
        )

    runs: list[dict[str, object]] = []
    for source_run in selected_runs:
        if source_run.get("base_method", defaults.get("base_method")) not in {
            "trace",
            "graph_cbm",
            "concept_forecast_cbm",  # Existing source artifacts.
        }:
            raise ValueError("Component analysis expects only TRACE templates")
        for arm, hparam_overrides in ARMS:
            run = copy.deepcopy(source_run)
            run["name"] = str(run["name"]).replace("_trace_", f"_component_{arm}_")
            hparams = copy.deepcopy(base_hparams)
            hparams.update(copy.deepcopy(run.get("model_hparams", {})))
            hparams.update(hparam_overrides)
            run["model_hparams"] = hparams
            if arm in {"no_calibration", "neither"}:
                run["learn_concept_threshold"] = False
            runs.append(run)

    expected = len(selected_runs) * len(ARMS) * len(seeds)
    payload = {
        "batch_name": "iclr2027_component_analysis" + ("_smoke" if args.smoke else ""),
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
        f"Wrote {args.output}: {len(ARMS)} factorial arms x "
        f"{len(selected_runs)} protocols x {len(seeds)} seeds = {expected} runs"
    )


if __name__ == "__main__":
    main()
