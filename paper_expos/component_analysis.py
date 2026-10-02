#!/usr/bin/env python3
"""Evaluate factorial TRACE components on predictions and interventions."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROTOCOLS,
    CheckpointRecord,
    discover_checkpoints,
    load_protocols,
    mean,
    read_json,
    sample_std,
    write_csv,
    write_json,
)
from paper_expos.intervention_budget import (
    activity_intervention,
    calibrated_future_target,
    probabilities_for_horizon,
    score_chunk,
    select_cases,
)
from utils.graph_concept_ui import forward_outputs, load_workspace
from utils.intervention_notebook import _forecast_logits_by_horizon


ARM_ORDER = ("full", "no_transitions", "no_calibration", "neither")
ARM_LABELS = {
    "full": "Full model",
    "no_transitions": "No transitions",
    "no_calibration": "No concept calibration",
    "neither": "No calibration or transitions",
}
ENDPOINT_METRICS = {
    "activity": ("accuracy", "macro_f1", "top3_accuracy"),
    "h1": ("accuracy", "macro_f1", "top3_accuracy", "sil_false_positive_rate"),
    "h2": ("accuracy", "macro_f1", "top3_accuracy", "sil_false_positive_rate"),
    "h3": ("accuracy", "macro_f1", "top3_accuracy", "sil_false_positive_rate"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-batch", type=Path, required=True)
    parser.add_argument("--component-batch", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--concept-budgets", default="1,3,5")
    parser.add_argument("--class-budgets", default="1,3")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--errors", type=int, default=30)
    parser.add_argument("--correct", type=int, default=30)
    parser.add_argument("--forward-chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_unique_ints(value: str, *, maximum: int | None = None) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or parsed != tuple(sorted(set(parsed))) or min(parsed) < 1:
        raise ValueError("Expected unique ascending positive integers")
    if maximum is not None and max(parsed) > maximum:
        raise ValueError(f"Values must not exceed {maximum}")
    return parsed


def trace_records(batch: Path, protocols: Path) -> list[CheckpointRecord]:
    records = discover_checkpoints(
        batch,
        protocols,
        require_complete_matrix=False,
        allow_duplicate_runs=True,
    )
    return [
        record
        for record in records
        if str(read_json(record.args_path).get("base_method")) in {"trace", "graph_cbm", "concept_forecast_cbm"}
    ]


def component_arm(run_name: str) -> str | None:
    if "component_no_transitions" in run_name:
        return "no_transitions"
    if "component_no_calibration" in run_name:
        return "no_calibration"
    if "component_neither" in run_name:
        return "neither"
    return None


def task_matrix(args: argparse.Namespace) -> list[tuple[str, int, dict[str, CheckpointRecord]]]:
    protocols = load_protocols(args.protocols)
    seeds = parse_unique_ints(args.seeds)
    expected_keys = {
        (str(protocol["key"]), seed)
        for protocol in protocols["datasets"]
        for seed in seeds
    }
    by_arm: dict[str, dict[tuple[str, int], CheckpointRecord]] = {
        arm: {} for arm in ARM_ORDER
    }

    for record in trace_records(args.main_batch, args.protocols):
        key = (record.dataset_key, record.seed)
        if key not in expected_keys:
            continue
        if key in by_arm["full"]:
            raise RuntimeError(f"Duplicate full checkpoint for {key}")
        by_arm["full"][key] = record

    for record in trace_records(args.component_batch, args.protocols):
        arm = component_arm(record.run_name)
        key = (record.dataset_key, record.seed)
        if arm is None or key not in expected_keys:
            continue
        if key in by_arm[arm]:
            raise RuntimeError(f"Duplicate {arm} checkpoint for {key}")
        by_arm[arm][key] = record

    for arm in ARM_ORDER:
        actual = set(by_arm[arm])
        if actual != expected_keys:
            raise RuntimeError(
                f"Incomplete {arm} matrix: missing={sorted(expected_keys-actual)}, "
                f"extra={sorted(actual-expected_keys)}"
            )

    protocol_order = {str(row["key"]): index for index, row in enumerate(protocols["datasets"])}
    return [
        (key[0], key[1], {arm: by_arm[arm][key] for arm in ARM_ORDER})
        for key in sorted(expected_keys, key=lambda value: (protocol_order[value[0]], value[1]))
    ]


def performance_row(arm: str, record: CheckpointRecord) -> dict[str, object]:
    metrics_path = record.checkpoint.parent / "metrics.json"
    metrics = read_json(metrics_path)
    test = metrics["test"]
    row: dict[str, object] = {
        "arm": arm,
        "arm_order": ARM_ORDER.index(arm),
        "variant": ARM_LABELS[arm],
        "protocol": record.dataset_key,
        "seed": record.seed,
        "run_name": record.run_name,
        "checkpoint": str(record.checkpoint),
    }
    endpoints = {"activity": test["activity"]}
    endpoints.update(
        {f"h{horizon}": test["forecast_by_horizon"][str(horizon)] for horizon in (1, 2, 3)}
    )
    for endpoint, values in endpoints.items():
        for metric in ENDPOINT_METRICS[endpoint]:
            row[f"{endpoint}_{metric}"] = float(values[metric])
    return row


def concept_specs(
    workspace,
    case: Mapping[str, object],
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    horizon = max(int(value) for value in workspace.forecast_horizons)
    baseline = case["baseline"]
    instance = case["instance"]
    true_label = int(case["true_label"])
    predicted = baseline["predicted_concepts_by_step"][1][0, -1].detach().cpu().numpy()
    target = calibrated_future_target(
        workspace,
        int(instance["video_index"]),
        int(instance["timestep"]) + 1,
    )
    target = np.clip(target, 0.0, 1.0)
    logits = _forecast_logits_by_horizon(workspace, baseline)[horizon][0, -1].detach()
    competitors = logits.clone()
    competitors[true_label] = -torch.inf
    runner = int(torch.argmax(competitors).item())
    head = workspace.model.activity_head.weight.detach().cpu().numpy()
    score = (target - predicted) * (head[true_label] - head[runner])
    order = np.argsort(-score).tolist()
    specs = []
    for budget in budgets:
        selected = [int(index) for index in order[: int(budget)]]
        specs.append(
            {
                "intervention_type": "concept",
                "policy": "directional_oracle",
                "treatment": "instance_target",
                "budget": int(budget),
                "selected": selected,
                "payload": {
                    "mode": "input",
                    "items": [
                        {
                            "item_type": "concept",
                            "rollout_step": 1,
                            "concept_idx": index,
                            "value": float(target[index]),
                        }
                        for index in selected
                    ],
                },
            }
        )
    return specs


def class_specs(
    workspace,
    case: Mapping[str, object],
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    specs = []
    for budget in budgets:
        payload = activity_intervention(workspace, case, int(budget), "oracle_label")
        specs.append(
            {
                "intervention_type": "class",
                "policy": "temporal_sources",
                "treatment": "oracle_label",
                "budget": int(budget),
                "selected": [int(item["step"]) for item in payload["items"]],
                "payload": payload,
            }
        )
    return specs


def release_workspace() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_task(
    protocol: str,
    seed: int,
    records: Mapping[str, CheckpointRecord],
    args: argparse.Namespace,
) -> Path:
    concept_budgets = parse_unique_ints(args.concept_budgets, maximum=5)
    class_budgets = parse_unique_ints(args.class_budgets, maximum=3)
    performance = [performance_row(arm, records[arm]) for arm in ARM_ORDER]
    selection_workspace = load_workspace(records["full"].checkpoint, device=args.device)
    selected = select_cases(
        selection_workspace,
        args.stride,
        args.errors,
        args.correct,
        seed,
    )
    instances = [row["instance"] for row in selected]
    selection_manifest = [
        {
            "case_index": index,
            "video_id": row["instance"]["video_id"],
            "video_index": row["instance"]["video_index"],
            "timestep": row["instance"]["timestep"],
            "true_label": row["true_label"],
            "full_prediction": row["prediction"],
            "full_correct": row["correct"],
        }
        for index, row in enumerate(selected)
    ]
    del selected
    del selection_workspace
    release_workspace()

    rows: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        workspace = load_workspace(records[arm].checkpoint, device=args.device)
        for case_index, instance in enumerate(instances):
            baseline = forward_outputs(workspace, instance)
            horizon = max(int(value) for value in workspace.forecast_horizons)
            probabilities = probabilities_for_horizon(workspace, baseline, horizon)
            case = {
                "instance": instance,
                "baseline": baseline,
                "true_label": int(instance["future_labels"][horizon]),
                "prediction": int(probabilities.argmax()),
                "correct": int(probabilities.argmax()) == int(instance["future_labels"][horizon]),
            }
            specs = concept_specs(workspace, case, concept_budgets)
            feedback_enabled = getattr(workspace.model, "_activity_feedback_enabled", None)
            if callable(feedback_enabled) and bool(feedback_enabled()):
                specs.extend(class_specs(workspace, case, class_budgets))
            for start in range(0, len(specs), int(args.forward_chunk)):
                for row in score_chunk(workspace, case, specs[start : start + int(args.forward_chunk)]):
                    row.update(
                        {
                            "arm": arm,
                            "arm_order": ARM_ORDER.index(arm),
                            "variant": ARM_LABELS[arm],
                            "protocol": protocol,
                            "seed": seed,
                            "case_index": case_index,
                            "video_id": instance["video_id"],
                            "video_index": instance["video_index"],
                            "timestep": instance["timestep"],
                        }
                    )
                    rows.append(row)
        del workspace
        release_workspace()

    cumulative_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cumulative_groups[
            (str(row["arm"]), int(row["case_index"]), str(row["intervention_type"]))
        ].append(row)
    for selected_rows in cumulative_groups.values():
        corrected = False
        harmed = False
        first_correction_budget = None
        first_harm_budget = None
        for row in sorted(selected_rows, key=lambda value: int(value["budget"])):
            budget = int(row["budget"])
            if not bool(row["baseline_correct"]) and bool(row["after_correct"]):
                corrected = True
                first_correction_budget = first_correction_budget or budget
            if bool(row["baseline_correct"]) and not bool(row["after_correct"]):
                harmed = True
                first_harm_budget = first_harm_budget or budget
            row["wrong_to_correct_by_budget"] = int(corrected)
            row["correct_to_wrong_by_budget"] = int(harmed)
            row["first_correction_budget"] = first_correction_budget
            row["first_harm_budget"] = first_harm_budget
            row["effective_stop_budget"] = (
                min(budget, int(first_correction_budget))
                if first_correction_budget is not None
                else budget
            )

    destination = Path(args.output_root) / f"component_analysis_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{protocol}_seed{seed}"
    write_csv(destination / f"{stem}_performance.csv", performance)
    write_csv(destination / f"{stem}_intervention_rows.csv", rows)
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["intervention_type"]), int(row["budget"]))].append(row)
    summary = []
    for (arm, intervention_type, budget), selected_rows in sorted(
        grouped.items(), key=lambda item: (ARM_ORDER.index(item[0][0]), item[0][1], item[0][2])
    ):
        summary.append(
            {
                "arm": arm,
                "arm_order": ARM_ORDER.index(arm),
                "variant": ARM_LABELS[arm],
                "protocol": protocol,
                "seed": seed,
                "intervention_type": intervention_type,
                "budget": budget,
                "n": len(selected_rows),
                "accuracy_after": mean(float(row["after_correct"]) for row in selected_rows),
                "wrong_to_correct_rate": mean(
                    float(row["wrong_to_correct"])
                    for row in selected_rows
                    if not bool(row["baseline_correct"])
                ),
                "correct_to_wrong_rate": mean(
                    float(row["correct_to_wrong"])
                    for row in selected_rows
                    if bool(row["baseline_correct"])
                ),
                "wrong_to_correct_by_budget_rate": mean(
                    float(row["wrong_to_correct_by_budget"])
                    for row in selected_rows
                    if not bool(row["baseline_correct"])
                ),
                "correct_to_wrong_by_budget_rate": mean(
                    float(row["correct_to_wrong_by_budget"])
                    for row in selected_rows
                    if bool(row["baseline_correct"])
                ),
                "label_flip_rate": mean(float(row["label_flip"]) for row in selected_rows),
                "true_probability_delta": mean(float(row["true_probability_delta"]) for row in selected_rows),
                "future_concept_l1": mean(float(row["future_concept_l1"]) for row in selected_rows),
            }
        )
    write_csv(destination / f"{stem}_intervention_summary.csv", summary)
    write_json(destination / f"{stem}_manifest.json", selection_manifest)
    return destination


def aggregate(output_dir: Path) -> None:
    performance: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*_performance.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            performance.extend(dict(row) for row in csv.DictReader(handle))
    for path in sorted(output_dir.glob("*_intervention_summary.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            interventions.extend(dict(row) for row in csv.DictReader(handle))

    performance_summary = []
    grouped_performance: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in performance:
        grouped_performance[(str(row["protocol"]), str(row["arm"]))].append(row)
    metric_names = [
        f"{endpoint}_{metric}"
        for endpoint, metrics in ENDPOINT_METRICS.items()
        for metric in metrics
    ]
    for (protocol, arm), selected in sorted(
        grouped_performance.items(), key=lambda item: (item[0][0], ARM_ORDER.index(item[0][1]))
    ):
        result: dict[str, object] = {
            "protocol": protocol,
            "arm": arm,
            "arm_order": ARM_ORDER.index(arm),
            "variant": ARM_LABELS[arm],
            "seeds": len(selected),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        performance_summary.append(result)

    intervention_summary = []
    grouped_interventions: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in interventions:
        key = (str(row["protocol"]), str(row["arm"]), str(row["intervention_type"]), int(row["budget"]))
        grouped_interventions[key].append(row)
    intervention_metrics = (
        "accuracy_after",
        "wrong_to_correct_rate",
        "correct_to_wrong_rate",
        "wrong_to_correct_by_budget_rate",
        "correct_to_wrong_by_budget_rate",
        "label_flip_rate",
        "true_probability_delta",
        "future_concept_l1",
    )
    for key, selected in sorted(
        grouped_interventions.items(),
        key=lambda item: (item[0][0], ARM_ORDER.index(item[0][1]), item[0][2], item[0][3]),
    ):
        result = {
            "protocol": key[0],
            "arm": key[1],
            "arm_order": ARM_ORDER.index(key[1]),
            "variant": ARM_LABELS[key[1]],
            "intervention_type": key[2],
            "budget": key[3],
            "seeds": len(selected),
        }
        for metric in intervention_metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) not in (None, "", "nan")]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        intervention_summary.append(result)

    write_csv(output_dir / "performance_mean_std.csv", performance_summary)
    write_csv(output_dir / "intervention_mean_std.csv", intervention_summary)


def main() -> None:
    args = parse_args()
    tasks = task_matrix(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "arms": list(ARM_ORDER),
                    "records": [
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "checkpoints": {arm: str(records[arm].checkpoint) for arm in ARM_ORDER},
                        }
                        for protocol, seed, records in tasks
                    ],
                },
                indent=2,
            )
        )
        return
    if args.record_index is None:
        destination = None
        for task in tasks:
            destination = evaluate_task(*task, args)
        if destination is not None:
            aggregate(destination)
        return
    if args.record_index < 0 or args.record_index >= len(tasks):
        raise IndexError(f"record-index must be in [0, {len(tasks) - 1}]")
    evaluate_task(*tasks[args.record_index], args)


if __name__ == "__main__":
    main()
