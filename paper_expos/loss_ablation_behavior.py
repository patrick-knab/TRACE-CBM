"""Matched post-hoc SIL, decoder-selectivity, and node-intervention diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from paper_expos.intervention_budget import (
    calibrated_future_target,
    concept_intervention,
    probabilities_for_horizon,
    score_chunk,
    select_cases,
)
from utils.graph_concept_ui import forward_outputs, load_workspace
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance


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
    parser.add_argument(
        "--protocols",
        type=Path,
        default=Path("paper_expos/configs/main_protocols_single_split_128_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--budgets", default="1,3,5")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--errors", type=int, default=30)
    parser.add_argument("--correct", type=int, default=30)
    parser.add_argument("--forward-chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-dir", type=Path)
    return parser.parse_args()


def arm_from_name(name: str) -> str:
    for arm in ARM_ORDER:
        if f"_loss_{arm}_seed" in name:
            return arm
    raise ValueError(f"Could not identify loss arm from {name}")


def decoder_support(workspace) -> tuple[float, float]:
    weights = workspace.model.activity_head.weight.detach().float().abs()
    row_l1 = weights.sum(dim=1).clamp_min(1e-12)
    n_eff = (row_l1.square() / weights.square().sum(dim=1).clamp_min(1e-12)).mean()
    top5 = (weights.topk(min(5, weights.shape[1]), dim=1).values.sum(dim=1) / row_l1).mean()
    return float(n_eff.item()), float(top5.item())


def matched_cases(reference_workspace, target_workspace, seed: int, stride: int, errors: int, correct: int):
    reference = select_cases(reference_workspace, stride, errors, correct, seed)
    matched = []
    horizon = max(int(value) for value in target_workspace.forecast_horizons)
    for case in reference:
        ref_instance = case["instance"]
        instance = select_instance(
            target_workspace,
            "test",
            int(ref_instance["video_index"]),
            int(ref_instance["timestep"]),
        )
        baseline = forward_outputs(target_workspace, instance)
        probabilities = probabilities_for_horizon(target_workspace, baseline, horizon)
        true_label = int(instance["future_labels"][horizon])
        matched.append(
            {
                "instance": instance,
                "baseline": baseline,
                "true_label": true_label,
                "prediction": int(probabilities.argmax()),
                "correct": int(probabilities.argmax()) == true_label,
                "reference_correct": bool(case["correct"]),
            }
        )
    return matched


def annotate_cumulative(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["case_index"]),
                str(row["intervention_type"]),
                str(row["policy"]),
                str(row["treatment"]),
            )
        ].append(row)
    for selected in groups.values():
        corrected = False
        harmed = False
        for row in sorted(selected, key=lambda item: int(item["budget"])):
            baseline_correct = bool(row["baseline_correct"])
            if not baseline_correct and bool(row["after_correct"]):
                corrected = True
            if baseline_correct and not bool(row["after_correct"]):
                harmed = True
            row["wrong_to_correct_by_budget"] = int(corrected)
            row["correct_to_wrong_by_budget"] = int(harmed)


def directional_oracle_specs(workspace, case, budgets, case_seed: int):
    baseline = case["baseline"]
    instance = case["instance"]
    horizon = max(int(value) for value in workspace.forecast_horizons)
    predicted = baseline["predicted_concepts_by_step"][1][0, -1].detach().cpu().numpy()
    target = calibrated_future_target(
        workspace,
        int(instance["video_index"]),
        int(instance["timestep"]) + 1,
    )
    logits = _forecast_logits_by_horizon(workspace, baseline)[horizon][0, -1].detach()
    runner_logits = logits.clone()
    runner_logits[int(case["true_label"])] = -torch.inf
    runner = int(torch.argmax(runner_logits).item())
    head = workspace.model.activity_head.weight.detach().cpu().numpy()
    score = (target - predicted) * (head[int(case["true_label"])] - head[runner])
    order = [int(index) for index in np.argsort(-score) if score[index] > 0.0]
    order.extend(int(index) for index in np.argsort(-score) if int(index) not in order)
    specs = []
    for budget in budgets:
        specs.append(
            {
                "intervention_type": "concept",
                "policy": "directional_oracle",
                "treatment": "oracle_pe_target",
                "budget": int(budget),
                "selected": order[: int(budget)],
                "payload": concept_intervention(
                    baseline,
                    target,
                    order,
                    int(budget),
                    "oracle_pe_target",
                ),
            }
        )
    return specs


def evaluate_record(
    record: CheckpointRecord,
    full_record: CheckpointRecord,
    args: argparse.Namespace,
    destination: Path,
) -> None:
    workspace = load_workspace(record.checkpoint, device=args.device)
    reference_workspace = (
        workspace
        if record.checkpoint == full_record.checkpoint
        else load_workspace(full_record.checkpoint, device=args.device)
    )
    cases = matched_cases(
        reference_workspace,
        workspace,
        record.seed,
        args.stride,
        args.errors,
        args.correct,
    )
    all_rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        specs = directional_oracle_specs(
            workspace,
            case,
            tuple(int(value) for value in args.budgets.split(",") if value.strip()),
            record.seed * 100_000 + case_index,
        )
        for start in range(0, len(specs), args.forward_chunk):
            scored = score_chunk(workspace, case, specs[start : start + args.forward_chunk])
            for row in scored:
                row.update({"case_index": case_index})
                all_rows.append(row)
    annotate_cumulative(all_rows)
    selected = [
        row
        for row in all_rows
        if row["intervention_type"] == "concept"
        and row["policy"] == "directional_oracle"
        and row["treatment"] == "oracle_pe_target"
        and int(row["budget"]) == 5
    ]
    if len(selected) != len(cases):
        raise RuntimeError(f"Expected one budget-5 oracle row per case, found {len(selected)}")
    wrong = [row for row in selected if not bool(row["baseline_correct"])]
    correct = [row for row in selected if bool(row["baseline_correct"])]
    metrics = json.loads((record.checkpoint.parent / "metrics.json").read_text())
    n_eff, top5 = decoder_support(workspace)
    output = {
        "protocol": record.dataset_key,
        "seed": record.seed,
        "arm": arm_from_name(record.run_name),
        "arm_order": ARM_ORDER.index(arm_from_name(record.run_name)),
        "variant": ARM_LABELS[arm_from_name(record.run_name)],
        "run_name": record.run_name,
        "checkpoint": str(record.checkpoint),
        "matched_cases": len(cases),
        "matched_reference_wrong": sum(not bool(case["reference_correct"]) for case in cases),
        "model_wrong": len(wrong),
        "model_correct": len(correct),
        "sil_fp": 100.0 * float(metrics["test"]["forecast_by_horizon"]["3"]["sil_false_positive_rate"]),
        "decoder_n_eff": n_eff,
        "decoder_top5_mass": 100.0 * top5,
        "oracle_w2c": 100.0 * mean(float(row["wrong_to_correct_by_budget"]) for row in wrong),
        "oracle_c2w": 100.0 * mean(float(row["correct_to_wrong_by_budget"]) for row in correct),
        "oracle_delta_p": 100.0 * mean(float(row["true_probability_delta"]) for row in selected),
    }
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in record.run_name)
    write_json(destination / f"{safe_name}.json", output)
    write_csv(destination / f"{safe_name}.csv", [output])


def aggregate(directory: Path, output_root: Path) -> None:
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "manifest.json":
            continue
        rows.append(json.loads(path.read_text()))
    if len(rows) != 45:
        raise RuntimeError(f"Expected 45 completed diagnostic rows, found {len(rows)}")
    metrics = (
        "sil_fp",
        "decoder_n_eff",
        "decoder_top5_mass",
        "oracle_w2c",
        "oracle_c2w",
        "oracle_delta_p",
    )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["arm"]))].append(row)
    summaries = []
    for (protocol, arm), selected in sorted(
        grouped.items(), key=lambda item: (item[0][0], ARM_ORDER.index(item[0][1]))
    ):
        summary = {
            "protocol": protocol,
            "arm": arm,
            "arm_order": ARM_ORDER.index(arm),
            "variant": ARM_LABELS[arm],
            "seeds": len(selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
        summaries.append(summary)
    write_csv(output_root / "per_seed.csv", rows)
    write_csv(output_root / "summary.csv", summaries)
    write_json(
        output_root / "manifest.json",
        {
            "diagnostic_directory": str(directory.resolve()),
            "case_matching": "same 30 wrong and 30 correct video windows selected from the full-loss checkpoint for each dataset and seed",
            "oracle_node_policy": "directional_oracle",
            "oracle_node_treatment": "oracle_pe_target",
            "oracle_node_budget": 5,
            "decoder_support": "mean over activity-head rows of ||w||_1^2 / ||w||_2^2",
        },
    )


def main() -> None:
    args = parse_args()
    if args.aggregate_dir is not None:
        output_root = args.output_root / "loss_ablation_behavior_matched"
        output_root.mkdir(parents=True, exist_ok=False)
        aggregate(args.aggregate_dir, output_root)
        print(f"Wrote {output_root}")
        return
    records = discover_checkpoints(
        args.source_batch,
        args.protocols,
        require_complete_matrix=True,
        allow_duplicate_runs=True,
    )
    if len(records) != 45:
        raise RuntimeError(f"Expected 45 loss checkpoints, found {len(records)}")
    by_key = {(record.dataset_key, record.seed, arm_from_name(record.run_name)): record for record in records}
    record_order = sorted(records, key=lambda record: (record.dataset_key, record.seed, arm_from_name(record.run_name)))
    if args.record_index is None:
        raise ValueError("--record-index is required for per-checkpoint evaluation")
    if args.record_index < 0 or args.record_index >= len(record_order):
        raise IndexError(f"record-index must be in [0, {len(record_order) - 1}]")
    record = record_order[args.record_index]
    full_record = by_key[(record.dataset_key, record.seed, "full")]
    destination = args.output_root / f"loss_ablation_behavior_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=True)
    evaluate_record(record, full_record, args, destination)
    print(f"Wrote {record.run_name} to {destination}")


if __name__ == "__main__":
    main()
