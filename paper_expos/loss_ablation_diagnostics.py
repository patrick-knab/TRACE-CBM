"""Aggregate targeted behavioral diagnostics for the matched loss ablation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    CheckpointRecord,
    discover_checkpoints,
    load_protocols,
    mean,
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
    "no_transition_tolerance": "No transition tolerance",
    "no_sil_penalty": "No SIL penalty",
    "no_classifier_l1": r"No classifier $L_1$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument(
        "--protocols",
        type=Path,
        default=Path("paper_expos/configs/main_protocols_single_split_128_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def arm_from_name(name: str) -> str:
    for arm in ARM_ORDER:
        if f"_loss_{arm}_seed" in name:
            return arm
    raise ValueError(f"Could not identify loss arm from {name}")


def decoder_support(checkpoint: Path) -> tuple[float, float]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload["model"]
    weights = model.activity_head.weight.detach().float().abs()
    row_l1 = weights.sum(dim=1).clamp_min(1e-12)
    effective_support = (row_l1.square() / weights.square().sum(dim=1).clamp_min(1e-12)).mean()
    top_k = min(5, int(weights.shape[1]))
    top5_mass = (weights.topk(top_k, dim=1).values.sum(dim=1) / row_l1).mean()
    return float(effective_support.item()), float(top5_mass.item())


def intervention_rows(interventions: Path) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for path in sorted(interventions.glob("*_summary.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("intervention_type") != "concept"
                    or row.get("policy") != "directional_oracle"
                    or row.get("treatment") != "oracle_pe_target"
                    or int(row.get("budget", -1)) != 5
                ):
                    continue
                run_name = str(row["run_name"])
                if run_name in selected:
                    raise RuntimeError(f"Duplicate intervention summary for {run_name}")
                selected[run_name] = {
                    "oracle_w2c": float(row["wrong_to_correct_by_budget_rate"]),
                    "oracle_c2w": float(row["correct_to_wrong_by_budget_rate"]),
                    "oracle_delta_p": float(row["true_probability_delta"]),
                }
    return selected


def per_seed_rows(
    records: list[CheckpointRecord],
    interventions: Path,
) -> list[dict[str, object]]:
    intervention_by_run = intervention_rows(interventions)
    rows = []
    for record in records:
        arm = arm_from_name(record.run_name)
        metrics = json.loads((record.checkpoint.parent / "metrics.json").read_text())
        sil_fp = float(metrics["test"]["forecast_by_horizon"]["3"]["sil_false_positive_rate"])
        n_eff, top5 = decoder_support(record.checkpoint)
        if record.run_name not in intervention_by_run:
            raise RuntimeError(f"Missing directional-oracle summary for {record.run_name}")
        row = {
            "protocol": record.dataset_key,
            "seed": record.seed,
            "arm": arm,
            "arm_order": ARM_ORDER.index(arm),
            "variant": ARM_LABELS[arm],
            "run_name": record.run_name,
            "checkpoint": str(record.checkpoint),
            "sil_fp": 100.0 * sil_fp,
            "decoder_n_eff": n_eff,
            "decoder_top5_mass": 100.0 * top5,
        }
        row.update(intervention_by_run[record.run_name])
        for key in ("oracle_w2c", "oracle_c2w", "oracle_delta_p"):
            row[key] = 100.0 * float(row[key])
        rows.append(row)
    return rows


def summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["arm"]))].append(row)
    metrics = (
        "sil_fp",
        "decoder_n_eff",
        "decoder_top5_mass",
        "oracle_w2c",
        "oracle_c2w",
        "oracle_delta_p",
    )
    output = []
    for (protocol, arm), selected in sorted(
        grouped.items(), key=lambda item: (item[0][0], ARM_ORDER.index(item[0][1]))
    ):
        row: dict[str, object] = {
            "protocol": protocol,
            "arm": arm,
            "arm_order": ARM_ORDER.index(arm),
            "variant": ARM_LABELS[arm],
            "seeds": len(selected),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    protocols = load_protocols(args.protocols)
    records = discover_checkpoints(
        args.source_batch,
        args.protocols,
        require_complete_matrix=True,
        allow_duplicate_runs=True,
    )
    expected = len(protocols["datasets"]) * len(protocols["seeds"]) * len(ARM_ORDER)
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} loss checkpoints, found {len(records)}")
    rows = per_seed_rows(records, args.interventions)
    destination = args.output_root / "loss_ablation_behavior_local"
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "per_seed.csv", rows)
    write_csv(destination / "summary.csv", summaries(rows))
    write_json(
        destination / "manifest.json",
        {
            "source_batch": str(args.source_batch.resolve()),
            "interventions": str(args.interventions.resolve()),
            "protocols": str(args.protocols.resolve()),
            "oracle_node_policy": "directional_oracle",
            "oracle_node_treatment": "oracle_pe_target",
            "oracle_node_budget": 5,
            "decoder_support": "mean over activity-head rows of ||w||_1^2 / ||w||_2^2",
        },
    )
    print(f"Wrote {destination} with {len(rows)} per-seed rows and {len(summaries(rows))} summaries")


if __name__ == "__main__":
    main()
