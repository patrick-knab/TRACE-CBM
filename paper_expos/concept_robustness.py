"""Concept dropout/noise robustness for the final TRACE checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from paper_expos.common import DEFAULT_OUTPUT_ROOT, discover_checkpoints, mean, sample_std, write_csv, write_json
from utils.graph_concept_ui import load_workspace
from utils.intervention_notebook import _workspace_examples
from utils.model import _evaluate_sequence, _forecast_horizon_metric


@dataclass(frozen=True)
class Record:
    checkpoint: Path
    protocol: str
    dataset: str
    seed: int
    run_name: str


def corruption_seed(dataset: str, seed: int, kind: str, level: float) -> int:
    text = f"{dataset}|{seed}|{kind}|{level:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def array_checksum(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def corrupt_examples(
    examples: Mapping[str, np.ndarray], dataset: str, seed: int, kind: str, level: float
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    concepts = np.array(examples["concepts"], dtype=np.float32, copy=True)
    valid = ~np.asarray(examples["key_padding_mask"], dtype=bool)
    valid_3d = valid[..., None]
    rng_seed = corruption_seed(dataset, seed, kind, level)
    rng = np.random.default_rng(rng_seed)
    if kind == "gaussian":
        perturbation = rng.normal(0.0, level, size=concepts.shape).astype(np.float32)
        perturbation *= valid_3d
        concepts += perturbation
        changed = perturbation != 0.0
        checksum_source = perturbation
    elif kind == "dropout":
        dropped = (rng.random(concepts.shape) < level) & valid_3d
        concepts[dropped] = 0.0
        changed = dropped
        checksum_source = dropped
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    concepts *= valid_3d
    corrupted = dict(examples)
    corrupted["concepts"] = concepts
    metadata = {
        "kind": kind,
        "level": float(level),
        "seed": int(rng_seed),
        "mask_or_noise_checksum": array_checksum(np.asarray(checksum_source)),
        "corrupted_concepts_checksum": array_checksum(concepts),
        "changed_fraction_of_valid_entries": float(
            changed.sum() / max(valid_3d.sum() * concepts.shape[-1], 1)
        ),
    }
    return corrupted, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--gaussian-levels", default="0.1,0.25,0.5")
    parser.add_argument("--dropout-levels", default="0.1,0.25,0.5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover(batch: Path) -> list[Record]:
    return [
        Record(row.checkpoint, row.dataset_key, row.dataset, row.seed, row.run_name)
        for row in discover_checkpoints(batch)
    ]


def parse_levels(value: str) -> tuple[float, ...]:
    levels = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("Corruption levels must be positive.")
    return levels


def metrics_row(record: Record, condition: str, level: float, metrics: dict[str, object]) -> dict[str, object]:
    activity = metrics.get("activity", {})
    horizon = 3
    return {
        "protocol": record.protocol,
        "dataset": record.dataset,
        "seed": record.seed,
        "condition": condition,
        "level": level,
        "activity_accuracy": activity.get("accuracy"),
        "activity_top3": activity.get("top3_accuracy"),
        "activity_macro_f1": activity.get("macro_f1"),
        "h3_accuracy": _forecast_horizon_metric(metrics, horizon, "accuracy"),
        "h3_top3": _forecast_horizon_metric(metrics, horizon, "top3_accuracy"),
        "h3_macro_f1": _forecast_horizon_metric(metrics, horizon, "macro_f1"),
    }


def evaluate(record: Record, args: argparse.Namespace) -> Path:
    workspace = load_workspace(record.checkpoint, device=args.device)
    examples = _workspace_examples(workspace, "test")
    sil_index = workspace.activity_names.index("SIL") if "SIL" in workspace.activity_names else None
    clean = _evaluate_sequence(
        workspace.model,
        examples,
        workspace.base_method,
        workspace.horizon,
        args.batch_size,
        len(workspace.activity_names),
        workspace.device,
        sil_index=sil_index,
    )
    rows = [metrics_row(record, "clean", 0.0, clean)]
    for kind, levels in (("gaussian", parse_levels(args.gaussian_levels)), ("dropout", parse_levels(args.dropout_levels))):
        for level in levels:
            corrupted, metadata = corrupt_examples(examples, record.dataset, record.seed, kind, level)
            metrics = _evaluate_sequence(
                workspace.model,
                corrupted,
                workspace.base_method,
                workspace.horizon,
                args.batch_size,
                len(workspace.activity_names),
                workspace.device,
                sil_index=sil_index,
            )
            row = metrics_row(record, kind, level, metrics)
            row.update(
                {
                    "corruption_seed": metadata["seed"],
                    "changed_fraction": metadata["changed_fraction_of_valid_entries"],
                    "corruption_checksum": metadata["mask_or_noise_checksum"],
                }
            )
            rows.append(row)
    destination = Path(args.output_root) / f"concept_robustness_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / f"{record.protocol}_seed{record.seed}.csv", rows)
    write_json(
        destination / f"{record.protocol}_seed{record.seed}.json",
        {"checkpoint": str(record.checkpoint), "rows": rows},
    )
    return destination


def aggregate(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.csv")):
        if path.name == "aggregate_summary.csv":
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    grouped = {}
    for row in rows:
        key = (row["protocol"], row["dataset"], row["condition"], float(row["level"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        result = {"protocol": key[0], "dataset": key[1], "condition": key[2], "level": key[3], "seeds": len(selected)}
        for metric in ("activity_accuracy", "activity_top3", "activity_macro_f1", "h3_accuracy", "h3_top3", "h3_macro_f1"):
            values = [float(row[metric]) for row in selected if row.get(metric) not in (None, "", "nan")]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        output.append(result)
    write_csv(output_dir / "aggregate_summary.csv", output)


def main() -> None:
    args = parse_args()
    records = discover(args.source_batch)
    if args.dry_run:
        print(json.dumps([record.__dict__ | {"checkpoint": str(record.checkpoint)} for record in records], indent=2))
        return
    if args.record_index is None:
        destination = None
        for record in records:
            destination = evaluate(record, args)
        if destination:
            aggregate(destination)
        return
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError(f"record-index must be in [0, {len(records) - 1}]")
    evaluate(records[args.record_index], args)


if __name__ == "__main__":
    main()
