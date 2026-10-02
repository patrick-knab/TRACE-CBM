"""Matched H1 concept-clamp evaluation for TRACE and non-relational rollouts.

The evaluator deliberately uses TRACE's saved Table-3 windows for every arm.
Every selected coordinate is clamped only in the H1 rollout state and is then
recursively re-forecast through H3.  Besides arm-specific diagnostic rankings,
it supports common rankings that do not inspect an evaluated arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from paper_expos.common import (
    DEFAULT_OUTPUT_ROOT,
    CheckpointRecord,
    discover_checkpoints,
    mean,
    read_json,
    require_batch_complete,
    sample_std,
    write_csv,
    write_json,
)
from paper_expos.intervention_budget import calibrated_future_target, complete_order
from utils.graph_concept_ui import forward_outputs, forward_outputs_batched_interventions, load_workspace
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance


PROTOCOLS = Path("paper_expos/configs/main_protocols_single_split_128_v1.json")
ARMS = ("trace", "dense", "sparse_linear")
ARM_LABELS = {
    "trace": "TRACE",
    "dense": "Dense non-relational rollout",
    "sparse_linear": "Sparse Linear",
}
SEEDS = (42, 43, 44)
HORIZONS = (1, 2, 3)
CONTROLLED_DENSE_EXCEPTIONS = {"st_forecast_rollout_mode", "st_observed_refiner_mode"}
SELECTION_POLICIES = (
    "arm_directional_oracle",
    "shared_oracle_transition",
    "shared_external_probe_oracle",
    "trace_directional_oracle",
)
EXTERNAL_PROBE_L2 = 1.0
MATCHED_TRACE_DENSE_FIELDS = (
    "backbone", "concept_set", "dataset", "test_split", "seed", "window_size", "history_length", "horizon",
    "learning_rate", "weight_decay", "batch_size", "num_epochs", "patience", "early_stopping_metric",
    "learn_concept_threshold", "teacher_forcing_start_ratio", "teacher_forcing_end_ratio",
    "concept_forecast_loss_weight", "forecast_transition_tolerance_radius", "forecast_transition_tolerance_weight",
    "classifier_l1_weight", "activity_sil_false_positive_penalty", "forecast_sil_false_positive_penalty",
    "activity_label_mode", "activity_label_fill_mode", "embedding_path",
)
MATCHED_SPARSE_FIELDS = (
    "backbone", "concept_set", "dataset", "test_split", "seed", "window_size", "history_length", "horizon",
    "activity_label_mode", "activity_label_fill_mode", "embedding_path",
)


@dataclass(frozen=True)
class ArmRecord:
    arm: str
    record: CheckpointRecord
    args: Mapping[str, Any]


def method_family(method: object) -> str:
    """Normalize the legacy TRACE label to the current graph implementation."""
    value = str(method)
    return "graph_cbm" if value in {"trace", "graph_cbm", "concept_forecast_cbm"} else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-batch", type=Path, required=True)
    parser.add_argument("--dense-batch", type=Path, required=True)
    parser.add_argument("--sparse-batch", type=Path, required=True)
    parser.add_argument(
        "--trace-intervention-dir",
        type=Path,
        required=True,
        help="Directory containing the per-checkpoint Table-3 *_manifest.json files.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--budgets", default="1,2,3,4,5")
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="arm_directional_oracle",
        help="Choose an arm-specific ranking or a common oracle prefix shared by all arms.",
    )
    parser.add_argument(
        "--require-capacity-matched-dense",
        action="store_true",
        help="Require the dense control to reuse its forecast transition rather than add parameters.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not budgets or budgets != tuple(sorted(set(budgets))) or min(budgets) < 1 or max(budgets) > 5:
        raise ValueError("budgets must be unique ascending integers in [1, 5]")
    return budgets


def expected_keys() -> set[tuple[str, int]]:
    return {(dataset, seed) for dataset in ("barista", "breakfast_s1", "mpii_attr") for seed in SEEDS}


def _records_for_batch(batch: Path, arm: str) -> dict[tuple[str, int], ArmRecord]:
    require_batch_complete(batch)
    records: dict[tuple[str, int], ArmRecord] = {}
    for record in discover_checkpoints(batch, PROTOCOLS, require_complete_matrix=False, allow_duplicate_runs=True):
        args = read_json(record.args_path)
        if not isinstance(args, dict):
            continue
        method = str(args.get("base_method", ""))
        hparams = args.get("model_hparams", {})
        if not isinstance(hparams, dict):
            hparams = {}
        if arm == "trace":
            keep = method in {"trace", "graph_cbm", "concept_forecast_cbm"} and str(hparams.get("st_forecast_rollout_mode", "legacy")) != "controlled_dense"
        elif arm == "dense":
            keep = (
                method in {"trace", "graph_cbm", "concept_forecast_cbm"}
                and hparams.get("st_forecast_rollout_mode") == "controlled_dense"
                and hparams.get("st_observed_refiner_mode") == "dense"
            )
        else:
            keep = method == "linear_sparse_dynamics_shared_head"
        if not keep:
            continue
        key = (record.dataset_key, record.seed)
        if key in records:
            raise RuntimeError(f"Duplicate {arm} checkpoint for {key}: {records[key].record.checkpoint} and {record.checkpoint}")
        records[key] = ArmRecord(arm=arm, record=record, args=args)
    if set(records) != expected_keys():
        raise RuntimeError(
            f"Incomplete {arm} matrix in {batch}: missing={sorted(expected_keys() - set(records))}, "
            f"extra={sorted(set(records) - expected_keys())}"
        )
    return records


def validate_trace_dense_pair(trace: ArmRecord, dense: ArmRecord, *, require_capacity_matched: bool = False) -> None:
    mismatched = [field for field in MATCHED_TRACE_DENSE_FIELDS if trace.args.get(field) != dense.args.get(field)]
    if mismatched or method_family(trace.args.get("base_method")) != method_family(dense.args.get("base_method")):
        raise RuntimeError(
            "TRACE and dense controls differ on matched settings "
            f"{mismatched}; methods={trace.args.get('base_method')} versus {dense.args.get('base_method')}: "
            f"{trace.record.checkpoint} versus {dense.record.checkpoint}"
        )
    trace_hparams = trace.args.get("model_hparams", {})
    dense_hparams = dense.args.get("model_hparams", {})
    if not isinstance(trace_hparams, Mapping) or not isinstance(dense_hparams, Mapping):
        raise RuntimeError("TRACE and dense checkpoints must record model_hparams")
    hparam_mismatches = [
        key for key, value in trace_hparams.items()
        if key not in CONTROLLED_DENSE_EXCEPTIONS and dense_hparams.get(key) != value
    ]
    if hparam_mismatches:
        raise RuntimeError(f"TRACE and dense controls differ on model_hparams {hparam_mismatches}")
    dense_hparams = dense.args.get("model_hparams", {})
    if not isinstance(dense_hparams, Mapping) or dense_hparams.get("st_forecast_rollout_mode") != "controlled_dense" or dense_hparams.get("st_observed_refiner_mode") != "dense":
        raise RuntimeError(f"Dense checkpoint is not the approved controlled-dense control: {dense.record.checkpoint}")
    if require_capacity_matched and not bool(dense_hparams.get("st_controlled_dense_reuse_forecast_layer")):
        raise RuntimeError(f"Dense checkpoint does not reuse its forecast transition for capacity matching: {dense.record.checkpoint}")


def validate_trace_sparse_pair(trace: ArmRecord, sparse: ArmRecord) -> None:
    mismatched = [field for field in MATCHED_SPARSE_FIELDS if trace.args.get(field) != sparse.args.get(field)]
    if mismatched:
        raise RuntimeError(f"TRACE and Sparse Linear controls differ on shared inputs {mismatched}: {trace.record.checkpoint} versus {sparse.record.checkpoint}")


def collect_matrix(args: argparse.Namespace) -> dict[tuple[str, int], dict[str, ArmRecord]]:
    by_arm = {
        "trace": _records_for_batch(args.trace_batch, "trace"),
        "dense": _records_for_batch(args.dense_batch, "dense"),
        "sparse_linear": _records_for_batch(args.sparse_batch, "sparse_linear"),
    }
    matrix = {
        key: {arm: by_arm[arm][key] for arm in ARMS}
        for key in sorted(expected_keys())
    }
    for records in matrix.values():
        validate_trace_dense_pair(
            records["trace"],
            records["dense"],
            require_capacity_matched=bool(getattr(args, "require_capacity_matched_dense", False)),
        )
        validate_trace_sparse_pair(records["trace"], records["sparse_linear"])
    return matrix


def manifest_path(root: Path, dataset: str, seed: int) -> Path:
    matches = sorted(root.glob(f"{dataset}_seed{seed}_*manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Table-3 manifest for {dataset} seed {seed} in {root}, found {len(matches)}")
    return matches[0]


def load_trace_cases(root: Path, dataset: str, seed: int, workspace) -> list[dict[str, object]]:
    path = manifest_path(root, dataset, seed)
    payload = read_json(path)
    if not isinstance(payload, list) or len(payload) != 60:
        raise RuntimeError(f"Table-3 manifest must contain 60 cases: {path}")
    if sum(bool(row.get("baseline_correct")) for row in payload if isinstance(row, Mapping)) != 30:
        raise RuntimeError(f"Table-3 manifest must contain 30 initially correct TRACE cases: {path}")
    cases = []
    for case_index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Invalid Table-3 manifest row {case_index}: {path}")
        instance = select_instance(workspace, "test", int(row["video_index"]), int(row["timestep"]))
        cases.append({"case_index": case_index, "manifest": row, "instance": instance})
    return cases


def is_sparse_workspace(workspace) -> bool:
    return isinstance(workspace.model, Mapping) and "forecast_model" in workspace.model


def sparse_model(workspace) -> torch.nn.Module:
    if not is_sparse_workspace(workspace):
        raise TypeError("Expected a Sparse Linear workspace")
    return workspace.model["forecast_model"]


def sparse_h1_target(workspace, instance: Mapping[str, object]) -> torch.Tensor:
    split = workspace.standardized_splits["test"]
    values = split["concepts_std"][int(instance["video_index"]), int(instance["timestep"]) + 1]
    return torch.as_tensor(values, dtype=torch.float32, device=workspace.device)


def h1_target(workspace, instance: Mapping[str, object]) -> np.ndarray:
    if is_sparse_workspace(workspace):
        return sparse_h1_target(workspace, instance).detach().cpu().numpy()
    return np.asarray(
        calibrated_future_target(workspace, int(instance["video_index"]), int(instance["timestep"]) + 1),
        dtype=np.float32,
    )


def horizon_target(workspace, instance: Mapping[str, object], horizon: int) -> np.ndarray:
    if is_sparse_workspace(workspace):
        split = workspace.standardized_splits["test"]
        return np.asarray(
            split["concepts_std"][int(instance["video_index"]), int(instance["timestep"]) + int(horizon)],
            dtype=np.float32,
        )
    return np.asarray(
        calibrated_future_target(
            workspace,
            int(instance["video_index"]),
            int(instance["timestep"]) + int(horizon),
        ),
        dtype=np.float32,
    )


def sparse_forward(workspace, instance: Mapping[str, object], selected: Sequence[int] | None = None, target: np.ndarray | None = None) -> Mapping[str, object]:
    model = sparse_model(workspace)
    current = torch.as_tensor(instance["concepts"][-1], dtype=torch.float32, device=workspace.device).unsqueeze(0)
    if selected is None:
        return model(current)
    if target is None:
        raise ValueError("Sparse H1 clamp requires a target")
    mask = model.transition_mask()
    weight = model.concept_dynamics.weight * mask
    state = F.linear(current, weight, model.concept_dynamics.bias)
    state = state.clone()
    target_tensor = torch.as_tensor(target, dtype=state.dtype, device=state.device)
    state[:, list(selected)] = target_tensor[list(selected)]
    concepts, logits = {1: state}, {1: model.activity_head(state)}
    for horizon in (2, 3):
        state = F.linear(state, weight, model.concept_dynamics.bias)
        concepts[horizon] = state
        logits[horizon] = model.activity_head(state)
    return {
        "forecast_logits_by_step": logits,
        "forecast_logits": logits[3],
        "predicted_concepts_by_step": concepts,
    }


def horizon_logits(workspace, outputs: Mapping[str, object], horizon: int, row: int = 0) -> torch.Tensor:
    if "forecast_logits_by_step" in outputs:
        values = {int(key): value for key, value in outputs["forecast_logits_by_step"].items()}
    else:
        values = _forecast_logits_by_horizon(workspace, outputs)
    tensor = values[int(horizon)]
    if tensor.ndim == 3:
        return tensor[int(row), -1]
    if tensor.ndim == 2:
        return tensor[int(row)]
    raise ValueError(f"Unexpected logits shape {tuple(tensor.shape)} at H{horizon}")


def horizon_concepts(outputs: Mapping[str, object], horizon: int, row: int = 0) -> torch.Tensor:
    values = outputs.get("predicted_concepts_by_step")
    if not isinstance(values, Mapping) or int(horizon) not in values:
        raise RuntimeError(f"Missing predicted concepts at H{horizon}")
    tensor = values[int(horizon)]
    if tensor.ndim == 3:
        return tensor[int(row), -1]
    if tensor.ndim == 2:
        return tensor[int(row)]
    raise ValueError(f"Unexpected concept-state shape {tuple(tensor.shape)} at H{horizon}")


def directional_order(workspace, baseline: Mapping[str, object], instance: Mapping[str, object], seed: int) -> list[int]:
    target = h1_target(workspace, instance)
    predicted = horizon_concepts(baseline, 1).detach().cpu().numpy()
    logits = horizon_logits(workspace, baseline, 3).detach().clone()
    true_label = int(instance["future_labels"][3])
    runner_logits = logits.clone()
    runner_logits[true_label] = -torch.inf
    runner = int(torch.argmax(runner_logits).item())
    model = sparse_model(workspace) if is_sparse_workspace(workspace) else workspace.model
    head = model.activity_head.weight.detach().cpu().numpy()
    score = (target - predicted) * (head[true_label] - head[runner])
    preferred = [int(index) for index in np.argsort(-score) if score[index] > 0.0]
    return complete_order(preferred, len(target), seed)


def shared_oracle_transition_order(trace_workspace, instance: Mapping[str, object], seed: int) -> list[int]:
    """Rank a common prefix from the true next transition, without any arm's decoder."""
    split = trace_workspace.standardized_splits["test"]
    video_index = int(instance["video_index"])
    timestep = int(instance["timestep"])
    current = np.asarray(split["concepts_std"][video_index, timestep], dtype=np.float32)
    target = np.asarray(split["concepts_std"][video_index, timestep + 1], dtype=np.float32)
    order = np.argsort(-np.abs(target - current)).tolist()
    return complete_order(order, len(target), seed)


def fit_shared_external_probe(trace_workspace, *, l2: float = EXTERNAL_PROBE_L2) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a train-only linear H1-concept to H3-class probe shared by all arms.

    This probe deliberately uses common standardized data and ground-truth
    labels, rather than a learned component from TRACE, Dense, or Sparse Linear.
    Its coefficients are only an oracle ranking device; it is not evaluated as
    a forecasting model.
    """
    if l2 < 0.0:
        raise ValueError("External probe L2 penalty must be non-negative")
    split = trace_workspace.standardized_splits["train"]
    raw_split = trace_workspace.preprocessed_data["train"]
    concepts = np.asarray(split["concepts_std"], dtype=np.float64)
    labels = np.asarray(raw_split["activity_labels"], dtype=np.int64)
    lengths = np.asarray(split["lengths"], dtype=np.int64)
    if concepts.ndim != 3 or labels.ndim != 2 or len(concepts) != len(labels) or len(lengths) != len(concepts):
        raise RuntimeError("Unexpected train split layout for shared external probe")
    features, targets = [], []
    for video_index, length in enumerate(lengths.tolist()):
        # An edit is applied at H1 and scored at H3, so pair c_(t+1) with y_(t+3).
        for timestep in range(max(0, int(length) - 3)):
            features.append(concepts[video_index, timestep + 1])
            targets.append(int(labels[video_index, timestep + 3]))
    if not features:
        raise RuntimeError("No valid train H1-to-H3 examples for shared external probe")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    num_classes = len(trace_workspace.activity_names)
    if np.any(y < 0) or np.any(y >= num_classes):
        raise RuntimeError("Shared external probe encountered an out-of-range activity label")
    design = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    one_hot = np.eye(num_classes, dtype=np.float64)[y]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[-1, -1] = 0.0  # Do not regularize the intercept.
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ one_hot)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(design.T @ design + penalty, design.T @ one_hot, rcond=None)[0]
    metadata = {
        "kind": "train_only_ridge_linear_h1_concept_to_h3_class",
        "feature_split": "train",
        "feature_horizon": 1,
        "label_horizon": 3,
        "l2": float(l2),
        "n_train_examples": int(len(x)),
        "n_concepts": int(x.shape[1]),
        "n_classes": int(num_classes),
    }
    return coefficients, metadata


def shared_external_probe_oracle_order(
    trace_workspace,
    instance: Mapping[str, object],
    coefficients: np.ndarray,
    seed: int,
) -> list[int]:
    """Return a common class-aware oracle order without using an evaluated model.

    The H1 target and H3 class are held-out oracle inputs.  The current-to-H1
    ground-truth transition is weighted by a fixed external probe's true-versus-
    runner-up contrast.  Hence all arms receive precisely the same coordinate
    order, while retaining their own semantically corresponding H1 target scale.
    """
    split = trace_workspace.standardized_splits["test"]
    video_index = int(instance["video_index"])
    timestep = int(instance["timestep"])
    current = np.asarray(split["concepts_std"][video_index, timestep], dtype=np.float64)
    target = np.asarray(split["concepts_std"][video_index, timestep + 1], dtype=np.float64)
    if coefficients.shape[0] != len(current) + 1:
        raise RuntimeError("Shared external probe concept dimensionality does not match the held-out case")
    true_label = int(instance["future_labels"][3])
    scores = np.append(current, 1.0) @ coefficients
    if true_label < 0 or true_label >= scores.shape[0]:
        raise RuntimeError("Shared external probe true label is out of range")
    runner_scores = scores.copy()
    runner_scores[true_label] = -np.inf
    runner = int(np.argmax(runner_scores))
    class_contrast = coefficients[:-1, true_label] - coefficients[:-1, runner]
    intervention_score = (target - current) * class_contrast
    # Complete-order resolves exact numerical ties reproducibly without using an arm.
    ordered = [int(index) for index in np.argsort(-intervention_score, kind="stable")]
    return complete_order(ordered, len(target), seed)


def sequence_payload(order: Sequence[int], target: np.ndarray, budget: int) -> dict[str, object]:
    selected = [int(index) for index in order[:budget]]
    return {
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
    }


def score_rows(
    arm: str,
    workspace,
    case: Mapping[str, object],
    baseline: Mapping[str, object],
    after: Mapping[str, object],
    after_row: int,
    order: Sequence[int],
    budget: int,
) -> list[dict[str, object]]:
    instance = case["instance"]
    selected = [int(index) for index in order[:budget]]
    rows = []
    for horizon in HORIZONS:
        true_label = int(instance["future_labels"][horizon])
        baseline_probabilities = torch.softmax(horizon_logits(workspace, baseline, horizon), dim=-1)
        after_probabilities = torch.softmax(horizon_logits(workspace, after, horizon, after_row), dim=-1)
        baseline_prediction = int(torch.argmax(baseline_probabilities).item())
        after_prediction = int(torch.argmax(after_probabilities).item())
        baseline_correct = baseline_prediction == true_label
        after_correct = after_prediction == true_label
        displacement = (horizon_concepts(after, horizon, after_row) - horizon_concepts(baseline, horizon)).abs()
        target_concepts = torch.as_tensor(
            horizon_target(workspace, instance, horizon), dtype=displacement.dtype, device=displacement.device
        )
        baseline_error = (horizon_concepts(baseline, horizon) - target_concepts).abs().mean()
        after_error = (horizon_concepts(after, horizon, after_row) - target_concepts).abs().mean()
        selected_mask = torch.zeros_like(displacement, dtype=torch.bool)
        selected_mask[selected] = True
        total_displacement = displacement.sum()
        selected_displacement = displacement[selected_mask].sum()
        nonselected_displacement = displacement[~selected_mask].mean()
        rows.append(
            {
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "case_index": int(case["case_index"]),
                "video_index": int(instance["video_index"]),
                "timestep": int(instance["timestep"]),
                "budget": int(budget),
                "selected_count": len(selected),
                "selected": json.dumps(selected),
                "horizon": horizon,
                "true_label": true_label,
                "baseline_prediction": baseline_prediction,
                "after_prediction": after_prediction,
                "baseline_correct": int(baseline_correct),
                "after_correct": int(after_correct),
                "label_flip": int(after_prediction != baseline_prediction),
                "wrong_to_correct": int((not baseline_correct) and after_correct),
                "correct_to_wrong": int(baseline_correct and not after_correct),
                "true_probability_delta": float(after_probabilities[true_label].item() - baseline_probabilities[true_label].item()),
                "future_concept_l1": float(displacement.mean().item()),
                "future_concept_error_reduction": float((baseline_error - after_error).item()),
                "future_concept_nonselected_l1": float(nonselected_displacement.item()),
                "future_concept_nontrivial_fraction": float((displacement > 0.01).float().mean().item()),
                "future_concept_selected_displacement_share": float(
                    (selected_displacement / total_displacement).item() if float(total_displacement.item()) > 0.0 else 0.0
                ),
            }
        )
    return rows


def annotate_cumulative(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["case_index"]), int(row["horizon"]))].append(row)
    for group in grouped.values():
        corrected = harmed = False
        first_correction = first_harm = None
        for row in sorted(group, key=lambda value: int(value["budget"])):
            budget = int(row["budget"])
            if not bool(row["baseline_correct"]) and bool(row["after_correct"]):
                corrected = True
                first_correction = first_correction or budget
            if bool(row["baseline_correct"]) and not bool(row["after_correct"]):
                harmed = True
                first_harm = first_harm or budget
            row["wrong_to_correct_by_budget"] = int(corrected)
            row["correct_to_wrong_by_budget"] = int(harmed)
            row["first_correction_budget"] = first_correction
            row["first_harm_budget"] = first_harm


def summarize_task(rows: Sequence[Mapping[str, object]], dataset: str, seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["budget"]), int(row["horizon"]))].append(row)
    summary = []
    for (arm, budget, horizon), values in sorted(grouped.items()):
        wrong = [row for row in values if not bool(row["baseline_correct"])]
        correct = [row for row in values if bool(row["baseline_correct"])]
        summary.append(
            {
                "dataset": dataset,
                "seed": seed,
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "budget": budget,
                "horizon": horizon,
                "n_cases": len(values),
                "baseline_wrong_n": len(wrong),
                "baseline_correct_n": len(correct),
                "true_probability_delta_mean": mean(float(row["true_probability_delta"]) for row in values),
                "future_concept_l1_mean": mean(float(row["future_concept_l1"]) for row in values),
                "future_concept_error_reduction_mean": mean(
                    float(row["future_concept_error_reduction"]) for row in values
                ),
                "future_concept_nonselected_l1_mean": mean(
                    float(row["future_concept_nonselected_l1"]) for row in values
                ),
                "future_concept_nontrivial_fraction_mean": mean(
                    float(row["future_concept_nontrivial_fraction"]) for row in values
                ),
                "future_concept_selected_displacement_share_mean": mean(
                    float(row["future_concept_selected_displacement_share"]) for row in values
                ),
                "label_flip_rate": mean(float(row["label_flip"]) for row in values),
                "wrong_to_correct_rate": mean(float(row["wrong_to_correct"]) for row in wrong),
                "correct_to_wrong_rate": mean(float(row["correct_to_wrong"]) for row in correct),
                "wrong_to_correct_by_budget_rate": mean(float(row["wrong_to_correct_by_budget"]) for row in wrong),
                "correct_to_wrong_by_budget_rate": mean(float(row["correct_to_wrong_by_budget"]) for row in correct),
            }
        )
    return summary


def evaluate_task(
    dataset: str,
    seed: int,
    records: Mapping[str, ArmRecord],
    args: argparse.Namespace,
    budgets: Sequence[int],
) -> None:
    trace_workspace = load_workspace(records["trace"].record.checkpoint, device=args.device, dataset_root=args.dataset_root)
    cases = load_trace_cases(args.trace_intervention_dir, dataset, seed, trace_workspace)
    trace_baselines = [forward_outputs(trace_workspace, case["instance"]) for case in cases]
    for case, baseline in zip(cases, trace_baselines):
        expected_prediction = case["manifest"].get("baseline_prediction")
        if expected_prediction is not None and int(expected_prediction) != int(torch.argmax(torch.softmax(horizon_logits(trace_workspace, baseline, 3), dim=-1)).item()):
            raise RuntimeError(f"TRACE checkpoint does not reproduce its Table-3 manifest at case {case['case_index']}")

    all_rows: list[dict[str, object]] = []
    shared_transition_orders = None
    trace_directional_orders = None
    external_probe_orders = None
    policy_metadata: dict[str, object] = {}
    if args.selection_policy == "shared_oracle_transition":
        shared_transition_orders = [
            shared_oracle_transition_order(trace_workspace, case["instance"], seed * 100_000 + case_index)
            for case_index, case in enumerate(cases)
        ]
        policy_metadata = {"kind": "common_true_h0_to_h1_transition_magnitude"}
    elif args.selection_policy == "shared_external_probe_oracle":
        coefficients, policy_metadata = fit_shared_external_probe(trace_workspace)
        external_probe_orders = [
            shared_external_probe_oracle_order(
                trace_workspace, case["instance"], coefficients, seed * 100_000 + case_index
            )
            for case_index, case in enumerate(cases)
        ]
    elif args.selection_policy == "trace_directional_oracle":
        trace_directional_orders = [
            directional_order(trace_workspace, baseline, case["instance"], seed * 100_000 + case_index)
            for case_index, (case, baseline) in enumerate(zip(cases, trace_baselines))
        ]
        policy_metadata = {"kind": "trace_model_directional_oracle"}
    else:
        policy_metadata = {"kind": "per_arm_model_directional_oracle"}
    manifests: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "selection_policy": args.selection_policy,
        "selection_policy_metadata": policy_metadata,
        "cases": [case["manifest"] for case in cases],
        "arms": {},
    }
    for arm in ARMS:
        workspace = trace_workspace if arm == "trace" else load_workspace(records[arm].record.checkpoint, device=args.device, dataset_root=args.dataset_root)
        arm_rows = []
        for case_index, case in enumerate(cases):
            baseline = trace_baselines[case_index] if arm == "trace" else (sparse_forward(workspace, case["instance"]) if is_sparse_workspace(workspace) else forward_outputs(workspace, case["instance"]))
            target = h1_target(workspace, case["instance"])
            if args.selection_policy == "shared_oracle_transition":
                if shared_transition_orders is None:
                    raise RuntimeError("Shared transition orders were not initialized")
                order = shared_transition_orders[case_index]
            elif args.selection_policy == "shared_external_probe_oracle":
                if external_probe_orders is None:
                    raise RuntimeError("Shared external probe orders were not initialized")
                order = external_probe_orders[case_index]
            elif args.selection_policy == "trace_directional_oracle":
                if trace_directional_orders is None:
                    raise RuntimeError("TRACE directional orders were not initialized")
                order = trace_directional_orders[case_index]
            else:
                order = directional_order(workspace, baseline, case["instance"], seed * 100_000 + case_index)
            if is_sparse_workspace(workspace):
                for budget in budgets:
                    after = sparse_forward(workspace, case["instance"], order[:budget], target)
                    arm_rows.extend(score_rows(arm, workspace, case, baseline, after, 0, order, budget))
            else:
                payloads = [sequence_payload(order, target, budget) for budget in budgets]
                after = forward_outputs_batched_interventions(workspace, case["instance"], payloads)
                for row_index, budget in enumerate(budgets):
                    arm_rows.extend(score_rows(arm, workspace, case, baseline, after, row_index, order, budget))
        annotate_cumulative(arm_rows)
        all_rows.extend(arm_rows)
        manifests["arms"][arm] = {
            "checkpoint": str(records[arm].record.checkpoint),
            "run_name": records[arm].record.run_name,
            "base_method": records[arm].args.get("base_method"),
        }
        if arm != "trace":
            del workspace

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_seed{seed}"
    write_csv(destination / f"{stem}_rows.csv", all_rows)
    write_csv(destination / f"{stem}_summary.csv", summarize_task(all_rows, dataset, seed))
    write_json(destination / f"{stem}_manifest.json", manifests)
    print(f"[matched-rollout] wrote {stem} with {len(all_rows)} rows", flush=True)


def aggregate(output_dir: Path, expected_tasks: int = 9, threshold: float = 0.05) -> None:
    summary_paths = sorted(output_dir.glob("*_summary.csv"))
    if len(summary_paths) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} per-task summaries in {output_dir}, found {len(summary_paths)}")
    rows: list[dict[str, str]] = []
    for path in summary_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["arm"], int(row["budget"]), int(row["horizon"]))].append(row)
    aggregate_rows = []
    metrics = (
        "true_probability_delta_mean",
        "future_concept_l1_mean",
        "future_concept_error_reduction_mean",
        "future_concept_nonselected_l1_mean",
        "future_concept_nontrivial_fraction_mean",
        "future_concept_selected_displacement_share_mean",
        "label_flip_rate",
        "wrong_to_correct_rate",
        "correct_to_wrong_rate",
        "wrong_to_correct_by_budget_rate",
        "correct_to_wrong_by_budget_rate",
    )
    for key, values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f"Expected three seeds for {key}, found {len(values)}")
        result: dict[str, object] = {
            "dataset": key[0], "arm": key[1], "arm_label": ARM_LABELS[key[1]], "budget": key[2], "horizon": key[3], "seeds": len(values)
        }
        for metric in metrics:
            metric_values = [float(row[metric]) for row in values]
            output_name = metric.removesuffix("_mean")
            result[f"{output_name}_mean"] = mean(metric_values)
            result[f"{output_name}_std"] = sample_std(metric_values)
        aggregate_rows.append(result)
    for dataset in ("barista", "breakfast_s1", "mpii_attr"):
        for arm in ARMS:
            for horizon in HORIZONS:
                observed = {
                    int(row["budget"])
                    for row in aggregate_rows
                    if row["dataset"] == dataset and row["arm"] == arm and int(row["horizon"]) == horizon
                }
                if observed != {1, 2, 3, 4, 5}:
                    raise RuntimeError(
                        f"Incomplete budget curve for dataset={dataset}, arm={arm}, H{horizon}: {sorted(observed)}"
                    )
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)

    thresholds = []
    for dataset in ("barista", "breakfast_s1", "mpii_attr"):
        for arm in ARMS:
            h3 = [row for row in aggregate_rows if row["dataset"] == dataset and row["arm"] == arm and row["horizon"] == 3]
            h3 = sorted(h3, key=lambda row: int(row["budget"]))
            reached = next((row for row in h3 if float(row["true_probability_delta_mean"]) >= threshold), None)
            seed_attainment = 0
            if reached is not None:
                matched = [row for row in rows if row["dataset"] == dataset and row["arm"] == arm and int(row["horizon"]) == 3 and int(row["budget"]) == int(reached["budget"])]
                seed_attainment = sum(float(row["true_probability_delta_mean"]) >= threshold for row in matched)
            thresholds.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "threshold_true_probability_gain": threshold,
                    "threshold_true_probability_gain_pp": 100.0 * threshold,
                    "minimum_budget": "not_reached_by_5" if reached is None else int(reached["budget"]),
                    "attainment_seeds_at_minimum": seed_attainment,
                    "total_seeds": len(SEEDS),
                }
            )
    write_csv(output_dir / "efficiency_thresholds.csv", thresholds)
    plot_budget_curves(output_dir, aggregate_rows, threshold)


def plot_budget_curves(output_dir: Path, rows: Sequence[Mapping[str, object]], threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = ("barista", "breakfast_s1", "mpii_attr")
    titles = {"barista": "BARISTA", "breakfast_s1": "Breakfast s1", "mpii_attr": "MPII Attr"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        for arm in ARMS:
            selected = sorted(
                [row for row in rows if row["dataset"] == dataset and row["arm"] == arm and int(row["horizon"]) == 3],
                key=lambda row: int(row["budget"]),
            )
            x = [int(row["budget"]) for row in selected]
            y = [100.0 * float(row["true_probability_delta_mean"]) for row in selected]
            err = [100.0 * float(row["true_probability_delta_std"]) for row in selected]
            axis.errorbar(x, y, yerr=err, marker="o", capsize=2.5, label=ARM_LABELS[arm])
        axis.axhline(100.0 * threshold, color="black", lw=0.8, ls="--")
        axis.set_title(titles[dataset])
        axis.set_xlabel("H1 edit budget")
        axis.set_xticks((1, 2, 3, 4, 5))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("H3 true-label probability gain (pp)")
    axes[-1].legend(fontsize=7, frameon=False, loc="best")
    fig.savefig(output_dir / "h3_probability_gain_by_budget.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "h3_probability_gain_by_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    if args.aggregate_only:
        aggregate(args.output_dir)
        return
    matrix = collect_matrix(args)
    tasks = [(dataset, seed, matrix[(dataset, seed)]) for dataset, seed in sorted(matrix)]
    if args.record_index is None:
        for dataset, seed, records in tasks:
            evaluate_task(dataset, seed, records, args, budgets)
        aggregate(args.output_dir)
        return
    if not 0 <= args.record_index < len(tasks):
        raise IndexError(f"record-index must be in [0, {len(tasks) - 1}]")
    dataset, seed, records = tasks[args.record_index]
    evaluate_task(dataset, seed, records, args, budgets)


if __name__ == "__main__":
    main()
