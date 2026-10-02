"""Export compact top-1-versus-runner-up concept ablation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from paper_expos.common import DEFAULT_OUTPUT_ROOT, write_csv, write_json
from paper_expos.qualitative_example_export import _prediction_margin_contributors
from utils.graph_concept_ui import forward_outputs, load_workspace
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance


DEFAULT_DATASETS = ("barista", "breakfast_s1", "mpii_attr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "contrastive_examples_seed42")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--off-value", type=float, default=0.1)
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional ordered comma-separated dataset:case_index selections.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _probabilities(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits[0, -1], dim=-1).detach().cpu().numpy()


def _activity_name(workspace, index: int) -> str:
    return str(workspace.activity_names[int(index)])


def _checkpoint(intervention_dir: Path, dataset: str, seed: int) -> Path:
    metadata_path = intervention_dir / f"{dataset}_seed{seed}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return Path(str(metadata["checkpoint"]))


def _cases(intervention_dir: Path, dataset: str, seed: int) -> list[dict[str, object]]:
    manifest_path = intervention_dir / f"{dataset}_seed{seed}_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {manifest_path}")
    return [dict(row) for row in payload]


def evaluate_dataset(
    *,
    intervention_dir: Path,
    dataset: str,
    dataset_root: Path,
    seed: int,
    off_value: float,
    device: str,
) -> list[dict[str, object]]:
    workspace = load_workspace(
        _checkpoint(intervention_dir, dataset, seed),
        device=device,
        dataset_root=dataset_root,
    )
    horizon = max(int(value) for value in workspace.forecast_horizons)
    rows: list[dict[str, object]] = []
    for case in _cases(intervention_dir, dataset, seed):
        if not bool(case["baseline_correct"]):
            continue
        instance = select_instance(
            workspace,
            "test",
            int(case["video_index"]),
            int(case["timestep"]),
        )
        baseline = forward_outputs(workspace, instance)
        baseline_logits = _forecast_logits_by_horizon(workspace, baseline)[horizon]
        baseline_probabilities = _probabilities(baseline_logits)
        class_order = np.argsort(-baseline_probabilities)
        predicted_idx, runner_up_idx = int(class_order[0]), int(class_order[1])
        true_label = int(case["true_label"])
        if predicted_idx != true_label:
            raise RuntimeError(
                f"Saved baseline changed for {dataset} seed={seed} case={case['case_index']}"
            )
        predicted_name = _activity_name(workspace, predicted_idx)
        runner_up_name = _activity_name(workspace, runner_up_idx)
        if "SIL" in {predicted_name, runner_up_name}:
            continue

        contributors = _prediction_margin_contributors(
            workspace,
            baseline,
            baseline_probabilities,
            horizon,
        )
        concept_states = baseline["predicted_concepts_by_step"][horizon]
        selected = next(
            (
                row
                for row in contributors
                if float(row["contribution"]) > 0.0
                and bool(row["scene_relevant"])
                and float(concept_states[0, -1, int(row["concept_idx"])].item()) > off_value
            ),
            None,
        )
        if selected is None:
            continue
        concept_idx = int(selected["concept_idx"])
        intervention = {
            "mode": "input",
            "items": [
                {
                    "item_type": "concept",
                    "rollout_step": horizon,
                    "concept_idx": concept_idx,
                    "value": float(off_value),
                }
            ],
        }
        intervened = forward_outputs(workspace, instance, intervention=intervention)
        intervened_logits = _forecast_logits_by_horizon(workspace, intervened)[horizon]
        intervened_probabilities = _probabilities(intervened_logits)
        after_prediction = int(np.argmax(intervened_probabilities))

        head = workspace.model.activity_head
        bias_margin = 0.0
        if head.bias is not None:
            bias_margin = float((head.bias[predicted_idx] - head.bias[runner_up_idx]).detach().cpu().item())
        contribution_sum = float(sum(float(row["contribution"]) for row in contributors))
        margin_before = float(
            (baseline_logits[0, -1, predicted_idx] - baseline_logits[0, -1, runner_up_idx])
            .detach()
            .cpu()
            .item()
        )
        margin_after = float(
            (intervened_logits[0, -1, predicted_idx] - intervened_logits[0, -1, runner_up_idx])
            .detach()
            .cpu()
            .item()
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "case_index": int(case["case_index"]),
                "video_id": str(case["video_id"]),
                "video_index": int(case["video_index"]),
                "timestep": int(case["timestep"]),
                "horizon": horizon,
                "predicted_activity": predicted_name,
                "runner_up_activity": runner_up_name,
                "prediction_probability_before": float(baseline_probabilities[predicted_idx]),
                "runner_up_probability_before": float(baseline_probabilities[runner_up_idx]),
                "concept_idx": concept_idx,
                "concept": str(selected["concept"]),
                "concept_activation_before": float(concept_states[0, -1, concept_idx].detach().cpu().item()),
                "concept_contribution": float(selected["contribution"]),
                "bias_margin": bias_margin,
                "contribution_sum": contribution_sum,
                "margin_before": margin_before,
                "reconstructed_margin": contribution_sum + bias_margin,
                "decomposition_error": abs(margin_before - (contribution_sum + bias_margin)),
                "off_value": float(off_value),
                "after_prediction": _activity_name(workspace, after_prediction),
                "prediction_probability_after": float(intervened_probabilities[predicted_idx]),
                "runner_up_probability_after": float(intervened_probabilities[runner_up_idx]),
                "margin_after": margin_after,
                "margin_drop": margin_before - margin_after,
                "runner_up_reversal": int(after_prediction == runner_up_idx),
                "label_flip": int(after_prediction != predicted_idx),
                "intervention": json.dumps(intervention, sort_keys=True),
            }
        )
    return rows


def select_examples(
    rows: Sequence[Mapping[str, object]],
    count: int,
    case_ids: Sequence[tuple[str, int]] = (),
) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("num-examples must be positive")
    if case_ids:
        by_id = {(str(row["dataset"]), int(row["case_index"])): dict(row) for row in rows}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise KeyError(f"Selected cases were not eligible: {missing}")
        if len(case_ids) != count:
            raise ValueError("num-examples must equal the number of explicit case IDs")
        return [by_id[case_id] for case_id in case_ids]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["runner_up_reversal"]), float(row["margin_drop"])),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    used_datasets: set[str] = set()
    for row in ordered:
        dataset = str(row["dataset"])
        if dataset in used_datasets:
            continue
        selected.append(row)
        used_datasets.add(dataset)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} examples from distinct datasets")
    return selected


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.off_value) <= 1.0:
        raise ValueError("off-value must be in [0, 1]")
    datasets = tuple(value.strip() for value in str(args.datasets).split(",") if value.strip())
    case_ids = tuple(
        (dataset.strip(), int(case_index))
        for value in str(args.case_ids).split(",")
        if value.strip()
        for dataset, case_index in [value.split(":", maxsplit=1)]
    )
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(
            evaluate_dataset(
                intervention_dir=args.intervention_dir,
                dataset=dataset,
                dataset_root=args.dataset_root,
                seed=int(args.seed),
                off_value=float(args.off_value),
                device=str(args.device),
            )
        )
    if not rows:
        raise RuntimeError("No eligible contrastive examples were found")
    selected = select_examples(rows, int(args.num_examples), case_ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "all_candidates.csv", rows)
    write_json(
        args.output_dir / "selected_examples.json",
        {
            "selection_rule": (
                "Explicit post-hoc semantic selection from eligible exact runner-up reversals: "
                + ", ".join(f"{dataset}:{case_index}" for dataset, case_index in case_ids)
                if case_ids
                else (
                    "Seed-42 correct matched cases with non-SIL top-1 and runner-up; ablate the strongest "
                    "positive scene-relevant H3 concept to 0.1; prioritize runner-up reversals and then "
                    "largest top-1-versus-runner-up margin drop, with distinct datasets."
                )
            ),
            "examples": selected,
        },
    )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
