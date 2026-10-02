"""Zero-shot recursive H1--H50 stability evaluation for H3-trained TRACE checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROTOCOLS,
    deterministic_sample,
    discover_checkpoints,
    entropy,
    mean,
    sample_std,
    write_csv,
    write_json,
)
from utils.graph_concept_ui import load_workspace
from utils.intervention_notebook import (
    _forecast_logits_by_horizon,
    _forward,
    _instance_tensors,
    select_instance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extend_rollout_horizon(workspace, max_horizon: int) -> None:
    model = workspace.model
    if isinstance(model, dict):
        raise TypeError("Long-horizon extrapolation currently requires a TRACE module checkpoint.")
    if not hasattr(model, "forecast_horizons") or not hasattr(model, "activity_head"):
        raise TypeError("Checkpoint does not expose the shared-head recursive forecast interface.")
    model.forecast_horizons = tuple(range(1, int(max_horizon) + 1))


def calibrated_target(workspace, dataset_split: dict[str, np.ndarray], video_index: int, timestep: int) -> np.ndarray:
    key = "concepts_std" if "concepts_std" in dataset_split else "concepts"
    raw = torch.as_tensor(
        dataset_split[key][video_index, timestep],
        dtype=torch.float32,
        device=workspace.device,
    ).unsqueeze(0)
    calibrator = getattr(workspace.model, "calibrator", None)
    with torch.no_grad():
        target = calibrator(raw) if callable(calibrator) else raw
    return target[0].detach().cpu().numpy()


def candidate_instances(workspace, stride: int, maximum: int, seed: int) -> list[tuple[int, int]]:
    split = workspace.standardized_splits["test"]
    candidates: list[tuple[int, int]] = []
    for video_index, raw_length in enumerate(split["lengths"]):
        length = int(raw_length)
        start = max(int(workspace.history_length) - 1, 0)
        candidates.extend((video_index, timestep) for timestep in range(start, max(start, length - 1), int(stride)))
    return [tuple(item) for item in deterministic_sample(candidates, int(maximum), int(seed))]


def evaluate_record(record, args: argparse.Namespace) -> Path:
    workspace = load_workspace(record.checkpoint, device=args.device)
    extend_rollout_horizon(workspace, args.max_horizon)
    split = workspace.standardized_splits["test"]
    rows: list[dict[str, object]] = []
    instances = candidate_instances(workspace, args.stride, args.max_examples, record.seed)
    for example_index, (video_index, timestep) in enumerate(instances):
        instance = select_instance(workspace, split="test", video_index=video_index, timestep=timestep)
        concepts, key_padding_mask = _instance_tensors(workspace, instance)
        with torch.no_grad():
            outputs = _forward(workspace, concepts, key_padding_mask)
        logits_by_horizon = _forecast_logits_by_horizon(workspace, outputs)
        predicted_concepts = outputs.get("predicted_concepts_by_step", {})
        current_label = int(instance["current_label"])
        for horizon in range(1, int(args.max_horizon) + 1):
            true_label = instance["future_labels"].get(horizon)
            if true_label is None or horizon not in logits_by_horizon:
                continue
            logits = logits_by_horizon[horizon][0, -1]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            top = np.argsort(-probabilities)
            prediction = int(top[0])
            row: dict[str, object] = {
                "dataset": record.dataset_key,
                "seed": record.seed,
                "run_name": record.run_name,
                "video_id": instance["video_id"],
                "video_index": video_index,
                "timestep": timestep,
                "example_index": example_index,
                "horizon": horizon,
                "true_label": int(true_label),
                "prediction": prediction,
                "top1_correct": int(prediction == int(true_label)),
                "top3_correct": int(int(true_label) in top[: min(3, len(top))]),
                "true_probability": float(probabilities[int(true_label)]),
                "entropy": entropy(probabilities),
                "persistence_prediction": current_label,
                "persistence_correct": int(current_label == int(true_label)),
            }
            predicted = predicted_concepts.get(horizon)
            target_timestep = timestep + horizon
            if torch.is_tensor(predicted) and target_timestep < int(split["lengths"][video_index]):
                predicted_np = predicted[0, -1].detach().cpu().numpy()
                target_np = calibrated_target(workspace, split, video_index, target_timestep)
                row.update(
                    {
                        "concept_mae": float(np.mean(np.abs(predicted_np - target_np))),
                        "concept_rmse": float(np.sqrt(np.mean((predicted_np - target_np) ** 2))),
                        "concept_variance": float(np.var(predicted_np)),
                        "concept_saturation_fraction": float(np.mean((predicted_np <= 0.01) | (predicted_np >= 0.99))),
                    }
                )
            rows.append(row)

    destination = Path(args.output_root) / f"long_horizon_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{record.dataset_key}_seed{record.seed}"
    write_csv(destination / f"{stem}_rows.csv", rows)
    summary_rows: list[dict[str, object]] = []
    for horizon in range(1, int(args.max_horizon) + 1):
        selected = [row for row in rows if int(row["horizon"]) == horizon]
        if not selected:
            continue
        summary_rows.append(
            {
                "dataset": record.dataset_key,
                "seed": record.seed,
                "horizon": horizon,
                "n": len(selected),
                "top1": mean(float(row["top1_correct"]) for row in selected),
                "top3": mean(float(row["top3_correct"]) for row in selected),
                "persistence_top1": mean(float(row["persistence_correct"]) for row in selected),
                "true_probability": mean(float(row["true_probability"]) for row in selected),
                "entropy": mean(float(row["entropy"]) for row in selected),
                "concept_mae": mean(float(row["concept_mae"]) for row in selected if "concept_mae" in row),
                "concept_variance": mean(float(row["concept_variance"]) for row in selected if "concept_variance" in row),
                "concept_saturation_fraction": mean(
                    float(row["concept_saturation_fraction"])
                    for row in selected
                    if "concept_saturation_fraction" in row
                ),
            }
        )
    write_csv(destination / f"{stem}_summary.csv", summary_rows)
    write_json(
        destination / f"{stem}_metadata.json",
        {
            "checkpoint": str(record.checkpoint),
            "dataset": record.dataset_key,
            "seed": record.seed,
            "max_horizon": int(args.max_horizon),
            "stride": int(args.stride),
            "num_instances": len(instances),
            "interpretation": "zero-shot recursive extrapolation of an H3-trained checkpoint",
        },
    )
    return destination


def aggregate(output_dir: Path) -> None:
    import csv

    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*_summary.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["horizon"])), []).append(row)
    aggregate_rows = []
    for (dataset, horizon), selected in sorted(grouped.items()):
        row: dict[str, object] = {"dataset": dataset, "horizon": horizon, "seeds": len(selected)}
        for metric in ("top1", "top3", "persistence_top1", "true_probability", "entropy", "concept_mae", "concept_variance", "concept_saturation_fraction"):
            values = [float(item[metric]) for item in selected if item.get(metric) not in (None, "", "nan")]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)
        aggregate_rows.append(row)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)


def main() -> None:
    args = parse_args()
    records = discover_checkpoints(args.source_batch, args.protocols)
    if args.dry_run:
        print(json.dumps({"records": [str(row.checkpoint) for row in records], "max_horizon": args.max_horizon}, indent=2))
        return
    if args.record_index is None:
        for record in records:
            destination = evaluate_record(record, args)
        aggregate(destination)
        return
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError(f"record-index must be in [0, {len(records) - 1}]")
    evaluate_record(records[args.record_index], args)


if __name__ == "__main__":
    main()
