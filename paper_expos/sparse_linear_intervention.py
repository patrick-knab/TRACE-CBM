"""Matched concept-coordinate interventions for Sparse Linear checkpoints.

This evaluator deliberately omits graph-edge and activity-feedback edits.  A
Sparse Linear checkpoint exposes a concept state and a recursive linear
transition, so its comparable intervention is a coordinate clamp at the
observed state followed by the normal H1--H3 rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from utils.graph_concept_ui import load_workspace
from utils.intervention_notebook import (
    _forecast_logits_by_horizon,
    _forward,
    _instance_tensors,
    select_instance,
)


TARGETS = {
    ("barista", "s1"): "barista",
    ("breakfast", "official_s1"): "breakfast_s1",
    ("mpii_cooking_2", "attr"): "mpii_attr",
}
SEEDS = {42, 43, 44}
POLICIES = ("random", "contribution", "oracle_error")


@dataclass(frozen=True)
class Record:
    checkpoint: Path
    dataset: str
    split: str
    dataset_key: str
    seed: int
    run_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument("--budgets", default="1,2,3,4,5")
    parser.add_argument("--errors", type=int, default=30)
    parser.add_argument("--correct", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


def discover_records(source_batch: Path) -> list[Record]:
    records = []
    for checkpoint in sorted((source_batch / "models").glob("*/model.pt")):
        args_path = checkpoint.parent / "args.json"
        if not args_path.exists():
            continue
        args = json.loads(args_path.read_text())
        dataset = str(args.get("dataset", "")).strip().lower()
        split = str(args.get("test_split", ""))
        method = str(args.get("base_method", ""))
        concept_set = str(args.get("concept_set", ""))
        dataset_key = TARGETS.get((dataset, split))
        if (
            dataset_key is not None
            and int(args.get("seed", -1)) in SEEDS
            and method == "linear_sparse_dynamics_shared_head"
            and concept_set.endswith("_iterative_qwen35_pe_l14_128_v1")
        ):
            records.append(
                Record(
                    checkpoint=checkpoint.resolve(),
                    dataset=dataset,
                    split=split,
                    dataset_key=dataset_key,
                    seed=int(args["seed"]),
                    run_name=str(args.get("run_name", checkpoint.parent.name)),
                )
            )
    unique = {(record.dataset_key, record.seed): record for record in records}
    expected = {(dataset_key, seed) for dataset_key in TARGETS.values() for seed in SEEDS}
    if set(unique) != expected:
        raise RuntimeError(
            f"Expected 9 matched 128-concept Sparse Linear checkpoints; "
            f"missing={sorted(expected - set(unique))}, extra={sorted(set(unique) - expected)}"
        )
    return sorted(unique.values(), key=lambda row: (row.dataset_key, row.seed))


def model_outputs(model: torch.nn.Module, x: torch.Tensor) -> Mapping[str, object]:
    outputs = model(x)
    if not isinstance(outputs, Mapping) or "forecast_logits_by_step" not in outputs:
        raise TypeError("Sparse Linear checkpoint did not return forecast_logits_by_step")
    return outputs


def horizon_logits(outputs: Mapping[str, object], horizon: int) -> torch.Tensor:
    if "forecast_logits_by_step" in outputs:
        values = outputs["forecast_logits_by_step"]
    elif "autoregressive_logits_by_step" in outputs:
        values = outputs["autoregressive_logits_by_step"]
    elif "forecast_logits_by_horizon" in outputs:
        values = outputs["forecast_logits_by_horizon"]
    else:
        values = {int(horizon): outputs["forecast_logits"]}
    logits = values[int(horizon)]
    return logits[:, -1, :] if logits.ndim == 3 else logits


def predicted_concepts(outputs: Mapping[str, object], horizon: int) -> torch.Tensor:
    values = outputs["predicted_concepts_by_step"][int(horizon)]
    return values[:, -1, :] if values.ndim == 3 else values


def workspace_outputs(workspace, instance: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(workspace.model, Mapping):
        current = torch.as_tensor(instance["concepts"][-1], dtype=torch.float32, device=workspace.device).unsqueeze(0)
        return model_outputs(workspace.model["forecast_model"], current)
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    return _forward(workspace, concepts, key_padding_mask)


def candidate_cases(workspace, stride: int) -> list[dict[str, object]]:
    split = workspace.standardized_splits["test"]
    horizon = max(int(value) for value in workspace.forecast_horizons)
    cases = []
    with torch.no_grad():
        for video_index, length_value in enumerate(split["lengths"]):
            length = int(length_value)
            for timestep in range(workspace.history_length - 1, max(workspace.history_length - 1, length - horizon), stride):
                instance = select_instance(workspace, "test", video_index, timestep)
                current = torch.as_tensor(instance["concepts"][-1], dtype=torch.float32, device=workspace.device).unsqueeze(0)
                outputs = workspace_outputs(workspace, instance)
                probabilities = torch.softmax(horizon_logits(outputs, horizon), dim=-1)[0]
                true_label = instance["future_labels"].get(horizon)
                if true_label is None:
                    continue
                cases.append(
                    {
                        "instance": instance,
                        "current": current[0].detach(),
                        "outputs": outputs,
                        "history_concepts": torch.as_tensor(instance["concepts"], dtype=torch.float32, device=workspace.device),
                        "key_padding_mask": torch.as_tensor(instance["key_padding_mask"], dtype=torch.bool, device=workspace.device),
                        "true_label": int(true_label),
                        "prediction": int(probabilities.argmax().item()),
                    }
                )
    return cases


def select_cases(
    workspace,
    errors: int,
    correct: int,
    stride: int,
    seed: int,
    fixed_keys: list[tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    cases = candidate_cases(workspace, stride)
    if fixed_keys is not None:
        indexed = {
            (int(case["instance"]["video_index"]), int(case["instance"]["timestep"])): case
            for case in cases
        }
        missing = [key for key in fixed_keys if key not in indexed]
        if missing:
            raise RuntimeError(f"Fixed case manifest contains missing cases: {missing[:3]}")
        return [indexed[key] for key in fixed_keys]
    wrong = [case for case in cases if case["prediction"] != case["true_label"]]
    right = [case for case in cases if case["prediction"] == case["true_label"]]
    if len(wrong) < errors or len(right) < correct:
        raise RuntimeError(f"Insufficient cases: wrong={len(wrong)}, correct={len(right)}")
    rng = random.Random(seed)
    return rng.sample(wrong, errors) + random.Random(seed + 10_000).sample(right, correct)


def target_concept(workspace, case: Mapping[str, object]) -> torch.Tensor:
    instance = case["instance"]
    split = workspace.standardized_splits["test"]
    values = split["concepts_std"][int(instance["video_index"]), int(instance["timestep"]) + 1]
    return torch.as_tensor(values, dtype=torch.float32, device=workspace.device)


def orders(workspace, case: Mapping[str, object], seed: int) -> dict[str, list[int]]:
    current = case["current"].detach().clone().requires_grad_(True)
    target_label = int(case["true_label"])
    horizon = max(int(value) for value in workspace.forecast_horizons)
    if isinstance(workspace.model, Mapping):
        outputs = model_outputs(workspace.model["forecast_model"], current.unsqueeze(0))
    else:
        history = case["history_concepts"].detach().clone()
        history[-1] = current
        history = history.unsqueeze(0).requires_grad_(True)
        outputs = _forward(workspace, history, case["key_padding_mask"].unsqueeze(0))
    runner_logits = horizon_logits(outputs, horizon)[0].detach().clone()
    runner_logits[target_label] = -torch.inf
    runner = int(runner_logits.argmax().item())
    score = horizon_logits(outputs, horizon)[0, target_label] - horizon_logits(outputs, horizon)[0, runner]
    gradient = torch.autograd.grad(score, current if isinstance(workspace.model, Mapping) else history)[0]
    if not isinstance(workspace.model, Mapping):
        gradient = gradient[0, -1]
    contribution = (gradient * current).abs().detach().cpu().numpy()

    predicted = predicted_concepts(case["outputs"], 1)[0].detach()
    target = target_concept(workspace, case).detach()
    oracle_error = (predicted - target).abs().cpu().numpy()
    random_order = list(range(len(workspace.concept_names)))
    random.Random(seed).shuffle(random_order)
    return {
        "random": random_order,
        "contribution": np.argsort(-contribution).tolist(),
        "oracle_error": np.argsort(-oracle_error).tolist(),
    }


def edit_outputs(workspace, case: Mapping[str, object], target: torch.Tensor, selected: list[int]) -> Mapping[str, object]:
    with torch.no_grad():
        if isinstance(workspace.model, Mapping):
            edited = case["current"].detach().clone()
            edited[selected] = target[selected]
            return model_outputs(workspace.model["forecast_model"], edited.unsqueeze(0))
        edited = case["history_concepts"].detach().clone()
        edited[-1, selected] = target[selected]
        return _forward(workspace, edited.unsqueeze(0), case["key_padding_mask"].unsqueeze(0))


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def intervention_rows(workspace, record: Record, args: argparse.Namespace) -> list[dict[str, object]]:
    if isinstance(workspace.model, Mapping):
        workspace.model["forecast_model"].eval()
    else:
        workspace.model.eval()
    sil_index = workspace.activity_names.index("SIL") if "SIL" in workspace.activity_names else None
    horizons = (1, 2, 3)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value.strip())
    fixed_keys = None
    case_manifest_path = getattr(args, "case_manifest", None)
    if case_manifest_path:
        manifest = json.loads(Path(case_manifest_path).read_text(encoding="utf-8"))
        fixed_keys = [tuple(map(int, pair)) for pair in manifest[record.dataset_key][str(record.seed)]]
    cases = select_cases(workspace, args.errors, args.correct, args.stride, record.seed, fixed_keys=fixed_keys)
    rows = []
    for case_index, case in enumerate(cases):
        target = target_concept(workspace, case)
        policy_orders = orders(workspace, case, record.seed * 100_000 + case_index)
        baseline = case["outputs"]
        baseline_predictions = {h: int(torch.argmax(horizon_logits(baseline, h)[0]).item()) for h in horizons}
        for policy in POLICIES:
            order = policy_orders[policy]
            for budget in budgets:
                selected = order[:budget]
                after = edit_outputs(workspace, case, target, selected)
                for horizon in horizons:
                    base_pred = baseline_predictions[horizon]
                    after_logits = horizon_logits(after, horizon)[0]
                    after_prob = torch.softmax(after_logits, dim=-1)
                    base_prob = torch.softmax(horizon_logits(baseline, horizon)[0], dim=-1)
                    true_label = int(case["instance"]["future_labels"][horizon])
                    base_correct = base_pred == true_label
                    after_pred = int(after_prob.argmax().item())
                    after_correct = after_pred == true_label
                    true_sil = sil_index is not None and true_label == sil_index
                    base_sil = sil_index is not None and base_pred == sil_index
                    after_sil = sil_index is not None and after_pred == sil_index
                    base_future = predicted_concepts(baseline, horizon)[0]
                    after_future = predicted_concepts(after, horizon)[0]
                    row = {
                        "dataset": record.dataset_key,
                        "seed": record.seed,
                        "case_index": case_index,
                        "baseline_correct": int(base_correct),
                        "after_correct": int(after_correct),
                        "policy": policy,
                        "budget": budget,
                        "horizon": horizon,
                        "selected_count": len(selected),
                        "true_label": true_label,
                        "baseline_prediction": base_pred,
                        "after_prediction": after_pred,
                        "label_flip": int(after_pred != base_pred),
                        "wrong_to_correct": int((not base_correct) and after_correct),
                        "correct_to_wrong": int(base_correct and (not after_correct)),
                        "true_probability_delta": float((after_prob[true_label] - base_prob[true_label]).detach().item()),
                        "future_concept_l1": float((after_future - base_future).abs().mean().item()),
                        "sil_target_wrong_n": int(true_sil and not base_correct),
                        "sil_target_w2c": int(true_sil and not base_correct and after_correct),
                        "sil_target_correct_n": int(true_sil and base_correct),
                        "sil_target_c2w": int(true_sil and base_correct and not after_correct),
                        "false_sil_n": int((sil_index is not None) and (not true_sil) and base_sil),
                        "false_sil_correction": int((sil_index is not None) and (not true_sil) and base_sil and after_correct),
                        "non_sil_correct_n": int((sil_index is not None) and (not true_sil) and base_correct),
                        "false_sil_harm": int((sil_index is not None) and (not true_sil) and base_correct and after_sil),
                    }
                    rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["dataset"], row["seed"], row["policy"], row["budget"], row["horizon"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        total = lambda name: sum(int(value[name]) for value in values)
        delta = np.mean([float(value["true_probability_delta"]) for value in values])
        l1 = np.mean([float(value["future_concept_l1"]) for value in values])
        summaries.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "policy": key[2],
                "budget": key[3],
                "horizon": key[4],
                "n_cases": len(values),
                "wrong_to_correct_rate": rate(total("wrong_to_correct"), sum(1 - int(value["baseline_correct"]) for value in values)),
                "correct_to_wrong_rate": rate(total("correct_to_wrong"), sum(int(value["baseline_correct"]) for value in values)),
                "label_flip_rate": rate(total("label_flip"), len(values)),
                "true_probability_delta_mean": float(delta),
                "future_concept_l1_mean": float(l1),
                "sil_target_w2c_rate": rate(total("sil_target_w2c"), total("sil_target_wrong_n")),
                "sil_target_c2w_rate": rate(total("sil_target_c2w"), total("sil_target_correct_n")),
                "false_sil_correction_rate": rate(total("false_sil_correction"), total("false_sil_n")),
                "false_sil_harm_rate": rate(total("false_sil_harm"), total("non_sil_correct_n")),
                "sil_target_wrong_n": total("sil_target_wrong_n"),
                "sil_target_correct_n": total("sil_target_correct_n"),
                "false_sil_n": total("false_sil_n"),
                "non_sil_correct_n": total("non_sil_correct_n"),
            }
        )
    return summaries


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def run_record(record: Record, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    workspace = load_workspace(record.checkpoint, device=args.device, dataset_root=args.dataset_root)
    if workspace.base_method != "linear_sparse_dynamics_shared_head":
        raise RuntimeError(f"Unexpected method {workspace.base_method} for {record.checkpoint}")
    if len(workspace.concept_names) != 128:
        raise RuntimeError(f"Expected 128 concepts, found {len(workspace.concept_names)}")
    rows = intervention_rows(workspace, record, args)
    return rows, summarize(rows)


def main() -> None:
    args = parse_args()
    records = discover_records(args.source_batch)
    selected = records if args.record_index is None else [records[int(args.record_index)]]
    all_rows: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    manifests = []
    for record in selected:
        print(f"[sparse-intervention] {record.dataset_key} seed={record.seed} checkpoint={record.checkpoint}", flush=True)
        rows, summaries = run_record(record, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        manifests.append({"dataset": record.dataset_key, "seed": record.seed, "checkpoint": str(record.checkpoint), "run_name": record.run_name})
        print(f"[sparse-intervention] rows={len(rows)} summaries={len(summaries)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "intervention_rows.csv", all_rows)
    write_rows(args.output_dir / "intervention_summary.csv", all_summaries)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "observed-coordinate clamp to the true t+1 standardized concept target, followed by recursive H1-H3 rollout",
                "policies": list(POLICIES),
                "budgets": [int(value) for value in args.budgets.split(",") if value.strip()],
                "errors": args.errors,
                "correct": args.correct,
                "stride": args.stride,
                "records": manifests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[sparse-intervention] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
