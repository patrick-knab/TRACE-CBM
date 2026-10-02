"""Aggregate the matched PE-L14 loss-ablation matrix after training."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    CheckpointRecord,
    discover_checkpoints,
    load_protocols,
    mean,
    read_json,
    require_batch_complete,
    sample_std,
    write_csv,
    write_json,
)


ARM_ORDER = (
    "full",
    "no_concept_forecast",
    "no_transition_tolerance",
    "no_sil_penalty",
    "no_classifier_l1",
)
ARM_LABELS = {
    "full": "Full loss",
    "no_concept_forecast": "No concept-forecast loss",
    "no_transition_tolerance": "Exact forecast CE",
    "no_sil_penalty": "No SIL false-positive penalty",
    "no_classifier_l1": "No classifier L1 penalty",
}
METRICS = {
    "activity": ("accuracy", "top3_accuracy"),
    "h1": ("accuracy", "top3_accuracy"),
    "h2": ("accuracy", "top3_accuracy"),
    "h3": ("accuracy", "top3_accuracy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--protocols",
        type=Path,
        default=Path("paper_expos/configs/main_protocols_single_split_128_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def arm_from_name(name: str) -> str | None:
    for arm in ARM_ORDER:
        if f"_loss_{arm}_seed" in name:
            return arm
    return None


def collect_records(batch: Path, protocols: Path) -> dict[tuple[str, int, str], CheckpointRecord]:
    payload = load_protocols(protocols)
    expected = {
        (str(row["key"]), int(seed), arm)
        for row in payload["datasets"]
        for seed in payload["seeds"]
        for arm in ARM_ORDER
    }
    records: dict[tuple[str, int, str], CheckpointRecord] = {}
    for record in discover_checkpoints(
        batch,
        protocols,
        require_complete_matrix=False,
        allow_duplicate_runs=True,
    ):
        arm = arm_from_name(record.run_name)
        if arm is None:
            continue
        key = (record.dataset_key, record.seed, arm)
        if key in records:
            raise RuntimeError(f"Duplicate loss-ablation checkpoint for {key}")
        records[key] = record
    if set(records) != expected:
        raise RuntimeError(
            "Incomplete loss-ablation matrix: "
            f"missing={sorted(expected - set(records))}, extra={sorted(set(records) - expected)}"
        )
    return records


def performance_row(record: CheckpointRecord, arm: str) -> dict[str, object]:
    test = read_json(record.checkpoint.parent / "metrics.json")["test"]
    endpoints = {"activity": test["activity"]}
    endpoints.update({f"h{horizon}": test["forecast_by_horizon"][str(horizon)] for horizon in (1, 2, 3)})
    row: dict[str, object] = {
        "protocol": record.dataset_key,
        "seed": record.seed,
        "arm": arm,
        "arm_order": ARM_ORDER.index(arm),
        "variant": ARM_LABELS[arm],
        "run_name": record.run_name,
        "checkpoint": str(record.checkpoint),
    }
    for endpoint, metric_names in METRICS.items():
        for metric in metric_names:
            row[f"{endpoint}_{metric}"] = float(endpoints[endpoint][metric])
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["arm"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (protocol, arm), selected in sorted(
        grouped.items(), key=lambda item: (item[0][0], ARM_ORDER.index(item[0][1]))
    ):
        summary: dict[str, object] = {
            "protocol": protocol,
            "arm": arm,
            "arm_order": ARM_ORDER.index(arm),
            "variant": ARM_LABELS[arm],
            "seeds": len(selected),
        }
        for endpoint, metric_names in METRICS.items():
            for metric in metric_names:
                values = [float(row[f"{endpoint}_{metric}"]) for row in selected]
                summary[f"{endpoint}_{metric}_mean"] = mean(values)
                summary[f"{endpoint}_{metric}_std"] = sample_std(values)
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    require_batch_complete(args.batch)
    records = collect_records(args.batch, args.protocols)
    rows = [
        performance_row(record, arm)
        for (_, _, arm), record in sorted(
            records.items(), key=lambda item: (item[0][0], item[0][1], ARM_ORDER.index(item[0][2]))
        )
    ]
    destination = args.output_root / f"loss_ablation_{os.environ.get('SLURM_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "performance.csv", rows)
    write_csv(destination / "summary.csv", summarize(rows))
    write_json(
        destination / "manifest.json",
        {
            "batch": str(args.batch.resolve()),
            "protocols": str(args.protocols.resolve()),
            "arms": list(ARM_ORDER),
            "loss_ablation_note": "Each non-full arm removes exactly one active loss component from the standard objective.",
            "records": [
                {
                    "protocol": protocol,
                    "seed": seed,
                    "arm": arm,
                    "checkpoint": str(record.checkpoint),
                }
                for (protocol, seed, arm), record in sorted(records.items())
            ],
        },
    )
    print(f"Wrote {destination} with {len(rows)} per-seed rows and {len(summarize(rows))} summaries")


if __name__ == "__main__":
    main()
