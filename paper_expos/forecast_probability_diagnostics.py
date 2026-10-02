#!/usr/bin/env python3
"""Report matched H1--H3 forecast accuracy and probability diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    CheckpointRecord,
    discover_checkpoints,
    mean,
    require_batch_complete,
    sample_std,
    write_csv,
    write_json,
)
from utils.graph_concept_ui import load_workspace
from utils.model import (
    _example_batch_to_torch,
    _forecast_logits_by_step,
    _forward_outputs,
    _iter_example_batches,
    _prepared_splits_with_forecast,
    _sliding_window_examples,
    _standardized_splits,
)


DEFAULT_PROTOCOLS = Path(__file__).resolve().parent / "configs" / "main_protocols_single_split_128_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def destination(args: argparse.Namespace) -> Path:
    name = f"forecast_probability_diagnostics_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    path = Path(args.output_root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _empty_accumulator(ece_bins: int) -> dict[str, object]:
    return {
        "n": 0,
        "top1_hits": 0,
        "top3_hits": 0,
        "nll_sum": 0.0,
        "brier_sum": 0.0,
        "bin_count": np.zeros(ece_bins, dtype=np.int64),
        "bin_confidence": np.zeros(ece_bins, dtype=np.float64),
        "bin_correct": np.zeros(ece_bins, dtype=np.int64),
    }


def _update_accumulator(
    accumulator: dict[str, object],
    logits: torch.Tensor,
    targets: torch.Tensor,
    ece_bins: int,
) -> None:
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    top_k = min(3, int(probabilities.shape[-1]))
    top3 = torch.topk(probabilities, k=top_k, dim=-1).indices.eq(targets.unsqueeze(-1)).any(dim=-1)
    accumulator["n"] = int(accumulator["n"]) + int(targets.numel())
    accumulator["top1_hits"] = int(accumulator["top1_hits"]) + int(predictions.eq(targets).sum().item())
    accumulator["top3_hits"] = int(accumulator["top3_hits"]) + int(top3.sum().item())
    accumulator["nll_sum"] = float(accumulator["nll_sum"]) + float(F.cross_entropy(logits, targets, reduction="sum").item())
    squared_norm = probabilities.square().sum(dim=-1)
    target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    accumulator["brier_sum"] = float(accumulator["brier_sum"]) + float(
        (squared_norm + 1.0 - 2.0 * target_probability).sum().item()
    )
    confidence = probabilities.max(dim=-1).values.detach().cpu().numpy()
    correctness = predictions.eq(targets).detach().cpu().numpy().astype(np.int64)
    indices = np.minimum((confidence * int(ece_bins)).astype(np.int64), int(ece_bins) - 1)
    counts = accumulator["bin_count"]
    confidence_sums = accumulator["bin_confidence"]
    correct_sums = accumulator["bin_correct"]
    assert isinstance(counts, np.ndarray)
    assert isinstance(confidence_sums, np.ndarray)
    assert isinstance(correct_sums, np.ndarray)
    np.add.at(counts, indices, 1)
    np.add.at(confidence_sums, indices, confidence)
    np.add.at(correct_sums, indices, correctness)


def _summary_row(record: CheckpointRecord, horizon: int, accumulator: dict[str, object], ece_bins: int) -> dict[str, object]:
    count = int(accumulator["n"])
    if count <= 0:
        raise RuntimeError(f"No valid H{horizon} examples for {record.run_name}")
    bin_count = accumulator["bin_count"]
    bin_confidence = accumulator["bin_confidence"]
    bin_correct = accumulator["bin_correct"]
    assert isinstance(bin_count, np.ndarray)
    assert isinstance(bin_confidence, np.ndarray)
    assert isinstance(bin_correct, np.ndarray)
    nonempty = bin_count > 0
    mean_confidence = np.zeros(ece_bins, dtype=np.float64)
    accuracy = np.zeros(ece_bins, dtype=np.float64)
    mean_confidence[nonempty] = bin_confidence[nonempty] / bin_count[nonempty]
    accuracy[nonempty] = bin_correct[nonempty] / bin_count[nonempty]
    ece = float(np.sum((bin_count / count) * np.abs(accuracy - mean_confidence)))
    return {
        "dataset": record.dataset_key,
        "seed": record.seed,
        "run_name": record.run_name,
        "checkpoint": str(record.checkpoint),
        "horizon": int(horizon),
        "n": count,
        "top1": int(accumulator["top1_hits"]) / count,
        "top3": int(accumulator["top3_hits"]) / count,
        "nll": float(accumulator["nll_sum"]) / count,
        "brier": float(accumulator["brier_sum"]) / count,
        "ece": ece,
        "ece_bins": int(ece_bins),
    }


def _ece_rows(record: CheckpointRecord, horizon: int, accumulator: dict[str, object], ece_bins: int) -> list[dict[str, object]]:
    bin_count = accumulator["bin_count"]
    bin_confidence = accumulator["bin_confidence"]
    bin_correct = accumulator["bin_correct"]
    assert isinstance(bin_count, np.ndarray)
    assert isinstance(bin_confidence, np.ndarray)
    assert isinstance(bin_correct, np.ndarray)
    rows = []
    for index in range(ece_bins):
        count = int(bin_count[index])
        confidence = float(bin_confidence[index] / count) if count else float("nan")
        accuracy = float(bin_correct[index] / count) if count else float("nan")
        rows.append(
            {
                "dataset": record.dataset_key,
                "seed": record.seed,
                "horizon": int(horizon),
                "bin": index,
                "lower": index / ece_bins,
                "upper": (index + 1) / ece_bins,
                "n": count,
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "gap": abs(accuracy - confidence) if count else float("nan"),
            }
        )
    return rows


@torch.no_grad()
def evaluate_record(record: CheckpointRecord, args: argparse.Namespace) -> Path:
    workspace = load_workspace(record.checkpoint, device=args.device)
    if isinstance(workspace.model, dict):
        raise TypeError("This diagnostic requires a module checkpoint with recursive forecast logits.")
    horizon = int(workspace.horizon)
    if horizon < 3:
        raise ValueError(f"Expected an H3-trained checkpoint, found horizon={horizon} for {record.run_name}")
    model = workspace.model.eval()
    prepared_splits = _prepared_splits_with_forecast(workspace.preprocessed_data, horizon)
    standardized_splits = _standardized_splits(prepared_splits)
    examples = _sliding_window_examples(
        standardized_splits["test"],
        horizon=horizon,
        history_length=int(workspace.history_length),
        memory_prefix_length=int(getattr(model, "st_memory_prefix_length", 0)),
    )
    accumulators = {step: _empty_accumulator(args.ece_bins) for step in (1, 2, 3)}
    example_count = int(examples["concepts"].shape[0])
    for indices in _iter_example_batches(np.arange(example_count), args.batch_size):
        batch = _example_batch_to_torch(examples, indices, workspace.device)
        outputs = _forward_outputs(
            model,
            workspace.base_method,
            batch["concepts"],
            batch["key_padding_mask"],
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )
        logits_by_step = _forecast_logits_by_step(outputs, workspace.base_method, horizon)
        for step in (1, 2, 3):
            if step not in logits_by_step:
                raise RuntimeError(f"Checkpoint did not return H{step} logits: {record.run_name}")
            targets = batch["teacher_forcing_labels"][:, step] if step < horizon else batch["forecast_labels"]
            _update_accumulator(accumulators[step], logits_by_step[step][:, -1, :], targets, args.ece_bins)

    destination_path = destination(args)
    stem = f"{record.dataset_key}_seed{record.seed}"
    summary_rows = [_summary_row(record, step, accumulators[step], args.ece_bins) for step in (1, 2, 3)]
    write_csv(destination_path / f"{stem}_summary.csv", summary_rows)
    write_csv(
        destination_path / f"{stem}_ece_bins.csv",
        [row for step in (1, 2, 3) for row in _ece_rows(record, step, accumulators[step], args.ece_bins)],
    )
    write_json(
        destination_path / f"{stem}_metadata.json",
        {
            "checkpoint": str(record.checkpoint),
            "dataset": record.dataset_key,
            "seed": record.seed,
            "horizon": horizon,
            "num_examples": example_count,
            "ece_bins": int(args.ece_bins),
            "brier_definition": "mean over examples of sum_k (p_k - one_hot_k)^2",
            "ece_definition": "fixed-width confidence ECE using the maximum predicted class probability",
            "evaluation": "all valid H3 sliding-window test examples; H1--H3 use the same windows and autoregressive logits as the training evaluator",
        },
    )
    del workspace, model, examples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return destination_path


def aggregate(output_dir: Path, expected_tasks: int = 9) -> None:
    summary_paths = sorted(output_dir.glob("*_seed*_summary.csv"))
    if len(summary_paths) != int(expected_tasks):
        raise RuntimeError(
            f"Refusing to aggregate incomplete diagnostic output: expected {expected_tasks} seed summaries, found {len(summary_paths)}"
        )
    rows: list[dict[str, str]] = []
    for path in summary_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["horizon"]))].append(row)
    aggregate_rows: list[dict[str, object]] = []
    for (dataset, horizon), selected in sorted(grouped.items()):
        if len(selected) != 3:
            raise RuntimeError(f"Expected three seeds for {(dataset, horizon)}, found {len(selected)}")
        aggregate_row: dict[str, object] = {
            "dataset": dataset,
            "horizon": horizon,
            "seeds": len(selected),
            "n_per_seed": ";".join(str(row["n"]) for row in selected),
        }
        for metric in ("top1", "top3", "nll", "brier", "ece"):
            values = [float(row[metric]) for row in selected]
            aggregate_row[f"{metric}_mean"] = mean(values)
            aggregate_row[f"{metric}_std"] = sample_std(values)
        aggregate_rows.append(aggregate_row)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    write_json(
        output_dir / "aggregate_metadata.json",
        {
            "seed_count": 3,
            "summary": "mean and sample standard deviation over seeds",
            "probability_metrics": "NLL, multiclass Brier score, and 15-bin maximum-confidence ECE",
        },
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.ece_bins < 2:
        raise ValueError("ece-bins must be at least two")
    require_batch_complete(args.source_batch)
    records = discover_checkpoints(args.source_batch, args.protocols)
    if args.dry_run:
        print(json.dumps({"records": [str(record.checkpoint) for record in records]}, indent=2))
        return
    if args.record_index is None:
        for record in records:
            evaluate_record(record, args)
        aggregate(destination(args), expected_tasks=len(records))
        return
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError(f"record-index must be in [0, {len(records) - 1}]")
    evaluate_record(records[args.record_index], args)


if __name__ == "__main__":
    main()
