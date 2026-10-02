"""Budgeted concept, edge, and activity-belief intervention evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROTOCOLS,
    deterministic_sample,
    discover_checkpoints,
    mean,
    sample_std,
    write_csv,
    write_json,
)
from utils.graph_concept_ui import (
    forward_outputs,
    forward_outputs_batched_interventions,
    load_workspace,
    prediction_concept_contributors,
)
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance
from utils.model import _active_graph_edges


CONCEPT_POLICIES = ("random", "contribution", "graph_guided", "oracle_error", "directional_oracle")
CONCEPT_TREATMENTS = ("no_op", "flip", "oracle_pe_target", "class_delta", "predicted_class_prototype")
EDGE_POLICIES = ("random", "strongest", "trained_fixed")
EDGE_TREATMENTS = ("no_op", "delete", "invert")
ACTIVITY_TREATMENTS = ("no_op", "oracle_label", "wrong_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--budgets", default="1,2,3,4,5")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--errors", type=int, default=30)
    parser.add_argument("--correct", type=int, default=30)
    parser.add_argument("--forward-chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate-runs", action="store_true")
    parser.add_argument(
        "--main-paper-only",
        action="store_true",
        help="Evaluate only the agreed concept, edge, and activity diagnostics.",
    )
    parser.add_argument(
        "--main-paper-edge-treatment",
        choices=("delete", "invert"),
        default="delete",
        help="Edge treatment used by the main-paper diagnostic subset.",
    )
    return parser.parse_args()


def probabilities_for_horizon(workspace, outputs: Mapping[str, object], horizon: int, row: int = 0) -> np.ndarray:
    logits = _forecast_logits_by_horizon(workspace, outputs)[int(horizon)]
    return torch.softmax(logits[row, -1], dim=-1).detach().cpu().numpy()


def calibrated_future_target(workspace, video_index: int, timestep: int) -> np.ndarray:
    split = workspace.standardized_splits["test"]
    key = "concepts_std" if "concepts_std" in split else "concepts"
    raw = torch.as_tensor(split[key][video_index, timestep], dtype=torch.float32, device=workspace.device).unsqueeze(0)
    calibrator = getattr(workspace.model, "calibrator", None)
    with torch.no_grad():
        target = calibrator(raw) if callable(calibrator) else raw
    return target[0].detach().cpu().numpy()


def class_delta_target(workspace, class_idx: int, horizon: int) -> np.ndarray:
    """Train-only mean next-concept state for examples ending in class_idx at horizon."""
    cache_key = f"_class_delta_targets_h{int(horizon)}"
    cached = getattr(workspace, cache_key, None)
    if cached is None:
        split = workspace.standardized_splits["train"]
        concept_key = "concepts_std" if "concepts_std" in split else "concepts"
        concept_rows = []
        label_rows = []
        for video_index, raw_length in enumerate(split["lengths"]):
            length = int(raw_length)
            if length <= int(horizon):
                continue
            concepts = np.asarray(split[concept_key][video_index])
            labels = np.asarray(split["activity_labels"][video_index], dtype=np.int64)
            for timestep in range(0, length - int(horizon)):
                concept_rows.append(concepts[timestep + 1])
                label_rows.append(labels[timestep + int(horizon)])
        raw = torch.as_tensor(np.stack(concept_rows), dtype=torch.float32, device=workspace.device)
        calibrator = getattr(workspace.model, "calibrator", None)
        with torch.no_grad():
            calibrated = calibrator(raw) if callable(calibrator) else raw
        calibrated_np = calibrated.detach().cpu().numpy()
        labels_np = np.asarray(label_rows, dtype=np.int64)
        global_mean = calibrated_np.mean(axis=0)
        targets = []
        for activity_idx in range(len(workspace.activity_names)):
            selected = labels_np == activity_idx
            targets.append(calibrated_np[selected].mean(axis=0) if np.any(selected) else global_mean)
        cached = np.stack(targets)
        setattr(workspace, cache_key, cached)
    return np.asarray(cached[int(class_idx)])


def candidate_instances(workspace, stride: int) -> list[dict[str, object]]:
    split = workspace.standardized_splits["test"]
    horizon = max(int(value) for value in workspace.forecast_horizons)
    candidates: list[dict[str, object]] = []
    for video_index, raw_length in enumerate(split["lengths"]):
        length = int(raw_length)
        start = max(workspace.history_length - 1, 0)
        for timestep in range(start, max(start, length - horizon), int(stride)):
            instance = select_instance(workspace, "test", video_index, timestep)
            outputs = forward_outputs(workspace, instance)
            true_label = instance["future_labels"].get(horizon)
            if true_label is None:
                continue
            probabilities = probabilities_for_horizon(workspace, outputs, horizon)
            prediction = int(probabilities.argmax())
            candidates.append(
                {
                    "instance": instance,
                    "baseline": outputs,
                    "true_label": int(true_label),
                    "prediction": prediction,
                    "correct": prediction == int(true_label),
                }
            )
    return candidates


def select_cases(workspace, stride: int, errors: int, correct: int, seed: int) -> list[dict[str, object]]:
    candidates = candidate_instances(workspace, stride)
    error_rows = [row for row in candidates if not row["correct"]]
    correct_rows = [row for row in candidates if row["correct"]]
    selected = deterministic_sample(error_rows, errors, seed) + deterministic_sample(correct_rows, correct, seed + 10_000)
    return [dict(row) for row in selected]


def complete_order(order: Sequence[int], size: int, seed: int) -> list[int]:
    unique = []
    for value in order:
        value = int(value)
        if 0 <= value < size and value not in unique:
            unique.append(value)
    remaining = [value for value in range(size) if value not in unique]
    random.Random(seed).shuffle(remaining)
    return unique + remaining


def concept_orders(workspace, case: Mapping[str, object], maximum: int, seed: int) -> dict[str, list[int]]:
    instance = case["instance"]
    baseline = case["baseline"]
    horizon = max(workspace.forecast_horizons)
    true_label = int(case["true_label"])
    num_concepts = len(workspace.concept_names)
    random_order = list(range(num_concepts))
    random.Random(seed).shuffle(random_order)

    contribution_rows = prediction_concept_contributors(
        workspace,
        instance,
        baseline,
        target="forecast",
        class_idx=true_label,
        horizon=horizon,
        top_k=max(32, maximum),
    )
    contribution = [int(row["concept_idx"]) for row in contribution_rows]
    edges = _active_graph_edges(workspace.model, max_edges=max(128, maximum * 8))
    graph_guided = [int(edge["source"]) for edge in edges]

    predicted = baseline["predicted_concepts_by_step"][1][0, -1].detach().cpu().numpy()
    target = calibrated_future_target(workspace, int(instance["video_index"]), int(instance["timestep"]) + 1)
    oracle_error = np.argsort(-np.abs(predicted - target)).tolist()
    forecast_logits = _forecast_logits_by_horizon(workspace, baseline)[horizon][0, -1].detach()
    runner_logits = forecast_logits.clone()
    runner_logits[true_label] = -torch.inf
    runner = int(torch.argmax(runner_logits).item())
    head_weight = workspace.model.activity_head.weight.detach().cpu().numpy()
    directional_score = (target - predicted) * (head_weight[true_label] - head_weight[runner])
    directional_oracle = [int(index) for index in np.argsort(-directional_score) if directional_score[index] > 0.0]
    return {
        "random": complete_order(random_order, num_concepts, seed)[:maximum],
        "contribution": complete_order(contribution, num_concepts, seed)[:maximum],
        "graph_guided": complete_order(graph_guided, num_concepts, seed)[:maximum],
        "oracle_error": complete_order(oracle_error, num_concepts, seed)[:maximum],
        "directional_oracle": complete_order(directional_oracle, num_concepts, seed)[:maximum],
    }


def concept_intervention(
    baseline: Mapping[str, object],
    target: np.ndarray,
    order: Sequence[int],
    budget: int,
    treatment: str,
) -> dict[str, object]:
    predicted = baseline["predicted_concepts_by_step"][1][0, -1].detach().cpu().numpy()
    items = []
    for concept_idx in order[:budget]:
        if treatment == "no_op":
            value = float(predicted[concept_idx])
        elif treatment == "flip":
            value = 0.1 if float(predicted[concept_idx]) >= 0.5 else 0.9
        elif treatment == "oracle_pe_target":
            value = float(np.clip(target[concept_idx], 0.0, 1.0))
        elif treatment in {"class_delta", "predicted_class_prototype"}:
            value = float(np.clip(target[concept_idx], 0.0, 1.0))
        else:
            raise ValueError(treatment)
        items.append(
            {
                "item_type": "concept",
                "rollout_step": 1,
                "concept_idx": int(concept_idx),
                "value": value,
            }
        )
    return {"mode": "input", "items": items}


def active_edge_orders(workspace, maximum: int, seed: int) -> dict[str, list[dict[str, object]]]:
    edges = _active_graph_edges(workspace.model, max_edges=max(512, maximum))
    strongest = [dict(edge) for edge in edges]
    random_edges = list(strongest)
    random.Random(seed).shuffle(random_edges)
    orders = {"strongest": strongest[:maximum], "random": random_edges[:maximum]}
    trained_edges = getattr(workspace.model, "trained_intervention_edges", None)
    if isinstance(trained_edges, Sequence) and trained_edges:
        orders["trained_fixed"] = [dict(edge) for edge in trained_edges[:maximum]]
    return orders


def edge_intervention(order: Sequence[Mapping[str, object]], budget: int, treatment: str) -> dict[str, object]:
    scale = {"no_op": 1.0, "delete": 0.0, "invert": -1.0}[treatment]
    items = []
    for edge in order[:budget]:
        items.append(
            {
                "item_type": "edge",
                "edge_kind": str(edge["kind"]),
                "branch": str(edge["branch"]),
                "layer_index": int(edge["layer"]),
                "source_idx": int(edge["source"]),
                "target_idx": int(edge["target"]),
                "edge_scale": scale,
            }
        )
    return {"mode": "input", "items": items}


def activity_probabilities(
    baseline: Mapping[str, object],
    step: int,
    instance: Mapping[str, object],
) -> torch.Tensor | None:
    key = "effective_activity_probs_by_history_step" if step < 0 else "effective_activity_probs_by_step"
    values = baseline.get(key)
    if not isinstance(values, Mapping) or step not in values or not torch.is_tensor(values[step]):
        if step >= 0:
            return None
        logits = baseline.get("activity_logits")
        if not torch.is_tensor(logits) or logits.ndim != 3:
            return None
        valid = ~torch.as_tensor(instance["key_padding_mask"], dtype=torch.bool, device=logits.device)
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        source_offset = abs(int(step)) + 1
        if positions.numel() < source_offset:
            return None
        return torch.softmax(logits[0, int(positions[-source_offset]), :], dim=-1).detach()
    tensor = values[step]
    if tensor.ndim == 2:
        probabilities = tensor[0].detach()
    elif tensor.ndim >= 3:
        probabilities = tensor[0, -1].detach()
    else:
        probabilities = tensor.detach()
    return probabilities if probabilities.ndim == 1 and probabilities.numel() > 0 else None


def activity_ground_truth(workspace, instance: Mapping[str, object], step: int) -> int | None:
    if step == 0:
        return int(instance["current_label"])
    if step > 0:
        value = instance["future_labels"].get(step)
        return None if value is None else int(value)
    labels = workspace.preprocessed_data["test"]["activity_labels"][int(instance["video_index"])]
    index = int(instance["timestep"]) + int(step)
    return int(labels[index]) if 0 <= index < len(labels) else None


def activity_intervention(workspace, case: Mapping[str, object], budget: int, treatment: str) -> dict[str, object]:
    source_steps = (0, 1, 2, -1, -2)
    items = []
    for step in source_steps[:budget]:
        probabilities = activity_probabilities(case["baseline"], step, case["instance"])
        true_label = activity_ground_truth(workspace, case["instance"], step)
        if probabilities is None or true_label is None:
            continue
        if treatment == "wrong_label":
            ranked = torch.argsort(probabilities, descending=True).tolist()
            class_idx = next(int(value) for value in ranked if int(value) != int(true_label))
            probability = 0.9
        else:
            class_idx = int(true_label)
            probability = float(probabilities[class_idx]) if treatment == "no_op" else 0.9
        items.append(
            {
                "item_type": "activity",
                "step": int(step),
                "class_idx": class_idx,
                "probability": probability,
            }
        )
    return {"mode": "input", "items": items}


def intervention_specs(workspace, case: Mapping[str, object], budgets: Sequence[int], case_seed: int) -> list[dict[str, object]]:
    maximum = max(budgets)
    instance = case["instance"]
    target = calibrated_future_target(workspace, int(instance["video_index"]), int(instance["timestep"]) + 1)
    horizon = max(int(value) for value in workspace.forecast_horizons)
    class_target = class_delta_target(workspace, int(case["true_label"]), horizon)
    specs: list[dict[str, object]] = []
    orders = concept_orders(workspace, case, maximum, case_seed)
    for policy, order in orders.items():
        for treatment in CONCEPT_TREATMENTS:
            for budget in budgets:
                specs.append(
                    {
                        "intervention_type": "concept",
                        "policy": policy,
                        "treatment": treatment,
                        "budget": budget,
                        "payload": concept_intervention(
                            case["baseline"],
                            class_target if treatment == "class_delta" else target,
                            order,
                            budget,
                            treatment,
                        ),
                        "selected": list(order[:budget]),
                    }
                )
    for policy, order in active_edge_orders(workspace, maximum, case_seed).items():
        for treatment in EDGE_TREATMENTS:
            for budget in budgets:
                specs.append(
                    {
                        "intervention_type": "edge",
                        "policy": policy,
                        "treatment": treatment,
                        "budget": budget,
                        "payload": edge_intervention(order, budget, treatment),
                        "selected": [
                            f"{edge['branch']}:{edge['layer']}:{edge['kind']}:{edge['source']}->{edge['target']}"
                            for edge in order[:budget]
                        ],
                    }
                )
    for treatment in ACTIVITY_TREATMENTS:
        for budget in budgets:
            payload = activity_intervention(workspace, case, budget, treatment)
            specs.append(
                {
                    "intervention_type": "activity",
                    "policy": "temporal_sources",
                    "treatment": treatment,
                    "budget": budget,
                    "payload": payload,
                    "selected": [int(item["step"]) for item in payload["items"]],
                }
            )
    return [spec for spec in specs if spec["payload"]["items"]]


def main_paper_specs(
    specs: Sequence[Mapping[str, object]],
    edge_treatment: str = "delete",
) -> list[dict[str, object]]:
    """Keep the prespecified diagnostic policies used in the main paper."""

    selected = []
    for spec in specs:
        intervention_type = str(spec["intervention_type"])
        policy = str(spec["policy"])
        treatment = str(spec["treatment"])
        budget = int(spec["budget"])
        keep = (
            intervention_type == "concept"
            and policy in {"random", "directional_oracle"}
            and treatment == "oracle_pe_target"
        ) or (
            intervention_type == "edge"
            and policy in {"random", "strongest"}
            and treatment == edge_treatment
        ) or (
            intervention_type == "activity"
            and treatment == "oracle_label"
            and budget <= 3
        )
        if keep:
            selected.append(dict(spec))
    return selected


def score_chunk(workspace, case: Mapping[str, object], specs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    horizon = max(int(value) for value in workspace.forecast_horizons)
    baseline_prob = probabilities_for_horizon(workspace, case["baseline"], horizon)
    true_label = int(case["true_label"])
    baseline_prediction = int(baseline_prob.argmax())
    outputs = forward_outputs_batched_interventions(workspace, case["instance"], [spec["payload"] for spec in specs])
    logits = _forecast_logits_by_horizon(workspace, outputs)[horizon][:, -1]
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    baseline_future = case["baseline"].get("predicted_concepts_by_step", {})
    after_future = outputs.get("predicted_concepts_by_step", {})
    rows = []
    for index, spec in enumerate(specs):
        actual_budget = len(spec["payload"]["items"])
        if actual_budget != int(spec["budget"]):
            raise RuntimeError(
                f"Intervention budget mismatch: requested={spec['budget']} actual={actual_budget} "
                f"type={spec['intervention_type']} treatment={spec['treatment']}"
            )
        after_prob = probabilities[index]
        after_prediction = int(after_prob.argmax())
        baseline_correct = baseline_prediction == true_label
        after_correct = after_prediction == true_label
        row = {
            "intervention_type": spec["intervention_type"],
            "policy": spec["policy"],
            "treatment": spec["treatment"],
            "budget": spec["budget"],
            "actual_budget": actual_budget,
            "selected": json.dumps(spec["selected"]),
            "payload": json.dumps(spec["payload"], sort_keys=True),
            "baseline_prediction": baseline_prediction,
            "after_prediction": after_prediction,
            "true_label": true_label,
            "baseline_correct": int(baseline_correct),
            "after_correct": int(after_correct),
            "label_flip": int(after_prediction != baseline_prediction),
            "wrong_to_correct": int((not baseline_correct) and after_correct),
            "correct_to_wrong": int(baseline_correct and (not after_correct)),
            "true_probability_before": float(baseline_prob[true_label]),
            "true_probability_after": float(after_prob[true_label]),
            "true_probability_delta": float(after_prob[true_label] - baseline_prob[true_label]),
            "probability_l1": float(np.mean(np.abs(after_prob - baseline_prob))),
        }
        deltas = []
        for step, baseline_state in baseline_future.items():
            after_state = after_future.get(step)
            if torch.is_tensor(baseline_state) and torch.is_tensor(after_state):
                deltas.append(float((after_state[index] - baseline_state[0]).abs().mean().item()))
        row["future_concept_l1"] = max(deltas, default=0.0)
        rows.append(row)
    return rows


def evaluate_record(record, args: argparse.Namespace) -> Path:
    workspace = load_workspace(record.checkpoint, device=args.device)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value.strip())
    if budgets != tuple(sorted(set(budgets))) or min(budgets) < 1 or max(budgets) > 5:
        raise ValueError("budgets must be unique ascending integers in [1, 5]")
    cases = select_cases(workspace, args.stride, args.errors, args.correct, record.seed)
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        instance = case["instance"]
        specs = intervention_specs(workspace, case, budgets, record.seed * 100_000 + case_index)
        if args.main_paper_only:
            specs = main_paper_specs(specs, args.main_paper_edge_treatment)
        for start in range(0, len(specs), int(args.forward_chunk)):
            scored = score_chunk(workspace, case, specs[start : start + int(args.forward_chunk)])
            for row in scored:
                row.update(
                    {
                        "dataset": record.dataset_key,
                        "seed": record.seed,
                        "run_name": record.run_name,
                        "case_index": case_index,
                        "video_id": instance["video_id"],
                        "video_path": instance["video_path"],
                        "video_index": instance["video_index"],
                        "timestep": instance["timestep"],
                    }
                )
                rows.append(row)
        manifest.append(
            {
                "dataset": record.dataset_key,
                "seed": record.seed,
                "case_index": case_index,
                "video_id": instance["video_id"],
                "video_path": instance["video_path"],
                "video_index": instance["video_index"],
                "timestep": instance["timestep"],
                "true_label": int(case["true_label"]),
                "baseline_prediction": int(case["prediction"]),
                "baseline_correct": bool(case["correct"]),
            }
        )

    cumulative_groups: dict[tuple[int, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["case_index"]),
            str(row["intervention_type"]),
            str(row["policy"]),
            str(row["treatment"]),
        )
        cumulative_groups.setdefault(key, []).append(row)
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

    destination = Path(args.output_root) / f"interventions_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}"
    destination.mkdir(parents=True, exist_ok=True)
    safe_run_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in record.run_name)
    stem = (
        f"{record.dataset_key}_seed{record.seed}_{safe_run_name}"
        if args.allow_duplicate_runs
        else f"{record.dataset_key}_seed{record.seed}"
    )
    write_csv(destination / f"{stem}_rows.csv", rows)
    write_json(destination / f"{stem}_manifest.json", manifest)
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["intervention_type"]), str(row["policy"]), str(row["treatment"]), int(row["budget"]))
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, selected in sorted(grouped.items()):
        summary.append(
            {
                "dataset": record.dataset_key,
                "seed": record.seed,
                "run_name": record.run_name,
                "intervention_type": key[0],
                "policy": key[1],
                "treatment": key[2],
                "budget": key[3],
                "n": len(selected),
                "accuracy_after": mean(float(row["after_correct"]) for row in selected),
                "wrong_to_correct_rate": mean(float(row["wrong_to_correct"]) for row in selected if not row["baseline_correct"]),
                "correct_to_wrong_rate": mean(float(row["correct_to_wrong"]) for row in selected if row["baseline_correct"]),
                "wrong_to_correct_by_budget_rate": mean(
                    float(row["wrong_to_correct_by_budget"])
                    for row in selected
                    if not row["baseline_correct"]
                ),
                "correct_to_wrong_by_budget_rate": mean(
                    float(row["correct_to_wrong_by_budget"])
                    for row in selected
                    if row["baseline_correct"]
                ),
                "label_flip_rate": mean(float(row["label_flip"]) for row in selected),
                "true_probability_delta": mean(float(row["true_probability_delta"]) for row in selected),
                "future_concept_l1": mean(float(row["future_concept_l1"]) for row in selected),
            }
        )
    write_csv(destination / f"{stem}_summary.csv", summary)
    write_json(
        destination / f"{stem}_metadata.json",
        {
            "checkpoint": str(record.checkpoint),
            "cases": len(cases),
            "errors_requested": args.errors,
            "correct_requested": args.correct,
            "budgets": budgets,
            "concept_oracle_definition": "PE-L14 target activation at rollout step H1",
            "edge_ground_truth_available": False,
        },
    )
    return destination


def aggregate(output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*_summary.csv")):
        if path.name == "aggregate_summary.csv":
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["intervention_type"]),
            str(row["policy"]),
            str(row["treatment"]),
            int(row["budget"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate_rows = []
    for key, selected in sorted(grouped.items()):
        result: dict[str, object] = {
            "dataset": key[0],
            "intervention_type": key[1],
            "policy": key[2],
            "treatment": key[3],
            "budget": key[4],
            "seeds": len(selected),
        }
        for metric in (
            "accuracy_after",
            "wrong_to_correct_rate",
            "correct_to_wrong_rate",
            "wrong_to_correct_by_budget_rate",
            "correct_to_wrong_by_budget_rate",
            "label_flip_rate",
            "true_probability_delta",
            "future_concept_l1",
        ):
            values = [float(row[metric]) for row in selected if row.get(metric) not in (None, "", "nan")]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        aggregate_rows.append(result)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)


def main() -> None:
    args = parse_args()
    records = discover_checkpoints(
        args.source_batch,
        args.protocols,
        require_complete_matrix=not args.allow_duplicate_runs,
        allow_duplicate_runs=args.allow_duplicate_runs,
    )
    if args.dry_run:
        print(json.dumps({"records": [str(row.checkpoint) for row in records], "budgets": args.budgets}, indent=2))
        return
    if args.record_index is None:
        destination = None
        for record in records:
            destination = evaluate_record(record, args)
        if destination is not None:
            aggregate(destination)
        return
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError(f"record-index must be in [0, {len(records) - 1}]")
    evaluate_record(records[args.record_index], args)


if __name__ == "__main__":
    main()
