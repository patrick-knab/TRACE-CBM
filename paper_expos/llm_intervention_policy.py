"""Evaluate a video-capable Qwen model as a blinded ordered intervention policy."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from paper_expos.common import DEFAULT_OUTPUT_ROOT, DEFAULT_PROTOCOLS, deterministic_sample, discover_checkpoints, mean, sample_std, write_csv, write_json
from utils.graph_concept_ui import (
    forward_outputs,
    forward_outputs_batched_interventions,
    learned_threshold_intervention_value,
    load_workspace,
    prediction_concept_contributors,
)
from utils.intervention_notebook import _forecast_logits_by_horizon, select_instance
from utils.model import _active_graph_edges
from utils.paths import dataset_root as configured_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--intervention-dir", type=Path)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "llm_policy")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--errors-per-seed", type=int, default=30)
    parser.add_argument("--correct-per-seed", type=int, default=30)
    parser.add_argument("--policy-runs", type=int, default=3)
    parser.add_argument("--endpoints", default="activity,h1,h2,h3")
    parser.add_argument("--action-types", default="concept,edge,activity")
    parser.add_argument("--candidate-concepts", type=int, default=20)
    parser.add_argument("--candidate-edges", type=int, default=10)
    parser.add_argument("--candidate-classes", type=int, default=3)
    parser.add_argument("--intervention-mode", choices=("input", "persistent"), default="persistent")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-consecutive-request-errors", type=int, default=3)
    parser.add_argument("--policy-attempts", type=int, default=8)
    parser.add_argument(
        "--oracle-context",
        action="store_true",
        help="Reveal the true endpoint label and single-action counterfactual effects to Qwen.",
    )
    parser.add_argument(
        "--sequential-oracle-context",
        action="store_true",
        help="Recompute conditional counterfactual effects and ask Qwen for one action at each budget step.",
    )
    parser.add_argument(
        "--invalid-next-action-fallback",
        choices=("error", "conditional_best"),
        default="error",
        help="Sequential-policy recovery after exhausted invalid-response retries.",
    )
    parser.add_argument(
        "--target-steering",
        action="store_true",
        help="Evaluate non-oracle steering toward a requested H3 activity target.",
    )
    parser.add_argument("--steering-cases-per-seed", type=int, default=60)
    parser.add_argument("--steering-stride", type=int, default=5)
    parser.add_argument(
        "--steering-max-baseline-confidence",
        type=float,
        default=0.60,
        help="Keep target-steering windows whose unedited H3 top-1 probability is at most this value.",
    )
    parser.add_argument(
        "--steering-exclude-sil-datasets",
        default="",
        help="Comma-separated dataset keys for which SIL is excluded as a steering target when alternatives exist.",
    )
    parser.add_argument(
        "--steering-selection-only",
        action="store_true",
        help="Preflight target-steering case selection without contacting the policy server.",
    )
    parser.add_argument(
        "--steering-catalog-only",
        action="store_true",
        help="Preflight target-steering cases, candidate catalogs, and blinded prompts without contacting Qwen.",
    )
    parser.add_argument(
        "--steering-activity-candidate-rule",
        choices=("legacy", "reverse_transition"),
        default="legacy",
        help="Choose activity candidates from current beliefs or train-only predecessors of the requested H3 target.",
    )
    parser.add_argument(
        "--steering-concept-evidence",
        choices=("legacy", "signed_contribution"),
        default="legacy",
        help="Expose frozen target-head concept sensitivity and its recommended intervention direction.",
    )
    parser.add_argument(
        "--steering-reference-cases",
        type=Path,
        help="Optional prior case catalog or selection preflight whose case identities and targets must match exactly.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifests(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(dict(row) for row in payload)
    return rows


def parse_endpoints(value: str) -> tuple[str, ...]:
    endpoints = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"activity", "h1", "h2", "h3"}
    if not endpoints or len(set(endpoints)) != len(endpoints) or not set(endpoints) <= allowed:
        raise ValueError(f"endpoints must be unique values from {sorted(allowed)}")
    return endpoints


def parse_action_types(value: str) -> tuple[str, ...]:
    action_types = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"concept", "edge", "activity"}
    if not action_types or len(set(action_types)) != len(action_types) or not set(action_types) <= allowed:
        raise ValueError(f"action-types must be unique values from {sorted(allowed)}")
    return action_types


def endpoint_horizon(endpoint: str) -> int:
    return 0 if endpoint == "activity" else int(endpoint.removeprefix("h"))


def endpoint_logits(workspace, outputs: Mapping[str, object], endpoint: str) -> torch.Tensor:
    horizon = endpoint_horizon(endpoint)
    if horizon == 0:
        logits = outputs.get("activity_logits")
        if not torch.is_tensor(logits):
            raise KeyError("activity_logits")
        return logits
    return _forecast_logits_by_horizon(workspace, outputs)[horizon]


def endpoint_true_label(instance: Mapping[str, object], endpoint: str) -> int | None:
    horizon = endpoint_horizon(endpoint)
    if horizon == 0:
        return int(instance["current_label"])
    value = instance["future_labels"].get(horizon)
    return None if value is None else int(value)


def endpoint_display(endpoint: str) -> str:
    return "classification at t" if endpoint == "activity" else f"forecast {endpoint.upper()}"


def activity_probabilities(baseline: Mapping[str, object], step: int) -> torch.Tensor | None:
    values = baseline.get("effective_activity_probs_by_step")
    if not isinstance(values, Mapping) or step not in values or not torch.is_tensor(values[step]):
        return None
    probabilities = values[step][0, -1].detach()
    return probabilities if probabilities.ndim == 1 and probabilities.numel() > 0 else None


def signed_recommended_state(target_head_weight: float) -> str:
    """Choose the intervention direction implied by one frozen target-head coefficient."""

    return "on" if float(target_head_weight) >= 0.0 else "off"


def concept_actions(
    workspace,
    instance,
    baseline,
    endpoint: str,
    maximum: int,
    target_class: int | None = None,
    evidence_rule: str = "legacy",
) -> list[dict[str, object]]:
    horizon = endpoint_horizon(endpoint)
    probabilities = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1)
    predicted_class = int(probabilities.argmax().item())
    reference_class = predicted_class if target_class is None else int(target_class)
    contributors = prediction_concept_contributors(
        workspace,
        instance,
        baseline,
        target="activity" if horizon == 0 else "forecast",
        class_idx=reference_class,
        horizon=None if horizon == 0 else horizon,
        top_k=(
            max(int(maximum), len(workspace.concept_names) * int(workspace.history_length))
            if evidence_rule == "signed_contribution"
            else max(maximum, 20)
        ),
    )
    evidence_by_concept: dict[int, dict[str, object]] = {}
    if evidence_rule == "signed_contribution":
        for row in contributors:
            if "head_weight" not in row:
                continue
            concept_idx = int(row["concept_idx"])
            head_weight = float(row["head_weight"])
            previous = evidence_by_concept.get(concept_idx)
            if previous is None or abs(head_weight) > abs(float(previous["target_head_weight"])):
                evidence_by_concept[concept_idx] = {
                    "target_head_weight": head_weight,
                    "signed_contribution": float(row["contribution"]),
                    "recommended_state": signed_recommended_state(head_weight),
                    "evidence_source": str(row.get("source", "frozen target head")),
                }
        if not evidence_by_concept:
            raise RuntimeError("signed_contribution requires an inspectable frozen linear target head")
        order = sorted(
            evidence_by_concept,
            key=lambda concept_idx: (-abs(float(evidence_by_concept[concept_idx]["target_head_weight"])), concept_idx),
        )
    else:
        order = [int(row["concept_idx"]) for row in contributors]
    order.extend(int(edge["source"]) for edge in _active_graph_edges(workspace.model, max_edges=max(maximum * 8, 80)))
    unique = []
    for concept_idx in order:
        if concept_idx not in unique:
            unique.append(concept_idx)
        if len(unique) >= maximum:
            break
    calibrated = baseline.get("calibrated_concepts")
    current = calibrated[0, -1].detach().cpu().numpy() if torch.is_tensor(calibrated) else np.zeros(len(workspace.concept_names))
    locations: list[tuple[str, int, np.ndarray]] = [("observed_t", 0, current)]
    predicted_by_step = baseline.get("predicted_concepts_by_step")
    if isinstance(predicted_by_step, Mapping):
        for step in range(1, max(horizon, 1)):
            predicted = predicted_by_step.get(step, predicted_by_step.get(str(step)))
            if torch.is_tensor(predicted):
                locations.append((f"forecast_t+{step}", step, predicted[0, -1].detach().cpu().numpy()))
    actions = []
    for location_name, rollout_step, location_state in locations:
        for concept_idx in unique:
            for state_name, state_value in (("off", 0.1), ("on", 0.9)):
                if rollout_step == 0:
                    intervention_value = learned_threshold_intervention_value(workspace, concept_idx, state_value)
                    location_payload = {"time_idx": workspace.history_length - 1}
                else:
                    intervention_value = state_value
                    location_payload = {"rollout_step": rollout_step}
                if intervention_value is None:
                    continue
                action = {
                        "action_id": f"c{concept_idx}_{location_name}_{state_name}",
                        "action_type": "concept",
                        "conflict_key": f"concept:{location_name}:{concept_idx}",
                        "concept_idx": concept_idx,
                        "concept": workspace.concept_names[concept_idx],
                        "location": location_name,
                        "set_state": state_name,
                        "current_activation": float(location_state[concept_idx]),
                        "payload_item": {
                            "item_type": "concept",
                            **location_payload,
                            "concept_idx": concept_idx,
                            "value": float(intervention_value),
                        },
                    }
                if concept_idx in evidence_by_concept:
                    evidence = evidence_by_concept[concept_idx]
                    action.update(evidence)
                    action["matches_recommended_state"] = state_name == str(evidence["recommended_state"])
                actions.append(action)
    return actions


def reverse_transition_predecessors(
    workspace,
    target_class: int,
    remaining_horizon: int,
    maximum_classes: int,
) -> list[dict[str, object]]:
    """Rank requested and train-only predecessor activities for one rollout source step."""

    counts = train_transition_counts(workspace, int(remaining_horizon))
    totals = counts.sum(axis=1)
    probabilities = np.divide(
        counts[:, int(target_class)],
        totals,
        out=np.zeros(counts.shape[0], dtype=np.float64),
        where=totals > 0,
    )
    ranked = sorted(
        range(counts.shape[0]),
        key=lambda class_idx: (-float(probabilities[class_idx]), -int(counts[class_idx, int(target_class)]), class_idx),
    )
    class_indices = [int(target_class)] + [class_idx for class_idx in ranked if class_idx != int(target_class)]
    return [
        {
            "class_idx": int(class_idx),
            "reverse_transition_probability": float(probabilities[class_idx]),
            "reverse_transition_count": int(counts[class_idx, int(target_class)]),
            "reverse_transition_remaining_horizon": int(remaining_horizon),
            "reverse_transition_rank": rank,
        }
        for rank, class_idx in enumerate(class_indices[: int(maximum_classes)], 1)
    ]


def edge_actions(workspace, endpoint: str, maximum: int) -> list[dict[str, object]]:
    actions = []
    candidates = _active_graph_edges(workspace.model, max_edges=max(maximum * 8, 80))
    if endpoint == "activity":
        candidates = [edge for edge in candidates if str(edge["branch"]) != "forecast"]
    for edge_index, edge in enumerate(candidates[:maximum]):
        edge_key = (
            f"{edge['branch']}:{int(edge['layer'])}:{edge['kind']}:"
            f"{int(edge['source'])}->{int(edge['target'])}"
        )
        for treatment, scale in (("delete", 0.0), ("invert", -1.0)):
            actions.append(
                {
                    "action_id": f"e{edge_index}_{treatment}",
                    "action_type": "edge",
                    "conflict_key": f"edge:{edge_key}",
                    "treatment": treatment,
                    "edge": edge_key,
                    "source": workspace.concept_names[int(edge["source"])],
                    "target": workspace.concept_names[int(edge["target"])],
                    "signed_weight": float(edge["score"]) * float(edge["sign"]),
                    "payload_item": {
                        "item_type": "edge",
                        "edge_kind": str(edge["kind"]),
                        "branch": str(edge["branch"]),
                        "layer_index": int(edge["layer"]),
                        "source_idx": int(edge["source"]),
                        "target_idx": int(edge["target"]),
                        "edge_scale": scale,
                    },
                }
            )
    return actions


def activity_actions(
    workspace,
    baseline,
    endpoint: str,
    maximum_classes: int,
    target_class: int | None = None,
    candidate_rule: str = "legacy",
) -> list[dict[str, object]]:
    horizon = endpoint_horizon(endpoint)
    feedback_enabled = getattr(workspace.model, "_activity_feedback_enabled", None)
    if horizon == 0 or not callable(feedback_enabled) or not bool(feedback_enabled()):
        return []
    actions = []
    for step in range(horizon):
        probabilities = activity_probabilities(baseline, step)
        if probabilities is None:
            continue
        transition_evidence: dict[int, dict[str, object]] = {}
        if candidate_rule == "reverse_transition":
            if target_class is None:
                raise ValueError("reverse_transition activity candidates require a steering target")
            predecessor_rows = reverse_transition_predecessors(
                workspace, int(target_class), horizon - step, maximum_classes
            )
            class_indices = [int(row["class_idx"]) for row in predecessor_rows]
            transition_evidence = {int(row["class_idx"]): row for row in predecessor_rows}
        else:
            class_indices = [int(value) for value in torch.argsort(probabilities, descending=True).tolist()]
            if target_class is not None:
                class_indices = [int(target_class)] + [value for value in class_indices if value != int(target_class)]
            class_indices = class_indices[:maximum_classes]
        for class_idx in class_indices:
            class_idx = int(class_idx)
            action = {
                    "action_id": f"a{step}_c{class_idx}",
                    "action_type": "activity",
                    "conflict_key": f"activity:{step}",
                    "source_step": step,
                    "activity": workspace.activity_names[class_idx],
                    "current_probability": float(probabilities[class_idx]),
                    "set_probability": 0.9,
                    "payload_item": {
                        "item_type": "activity",
                        "step": step,
                        "class_idx": class_idx,
                        "probability": 0.9,
                    },
                }
            action.update(transition_evidence.get(class_idx, {}))
            actions.append(action)
    return actions


def candidate_actions(
    workspace,
    instance,
    baseline,
    endpoint: str,
    action_types: Sequence[str],
    maximum_concepts: int,
    maximum_edges: int,
    maximum_classes: int,
    target_class: int | None = None,
    steering_activity_candidate_rule: str = "legacy",
    steering_concept_evidence: str = "legacy",
) -> list[dict[str, object]]:
    actions = []
    if "concept" in action_types:
        actions.extend(
            concept_actions(
                workspace, instance, baseline, endpoint, maximum_concepts, target_class, steering_concept_evidence
            )
        )
    if "edge" in action_types:
        actions.extend(edge_actions(workspace, endpoint, maximum_edges))
    if "activity" in action_types:
        actions.extend(
            activity_actions(
                workspace, baseline, endpoint, maximum_classes, target_class, steering_activity_candidate_rule
            )
        )
    return actions


def graph_context(workspace, maximum: int = 20) -> list[dict[str, object]]:
    rows = []
    for edge in _active_graph_edges(workspace.model, max_edges=maximum):
        rows.append(
            {
                "kind": edge["kind"],
                "source": workspace.concept_names[int(edge["source"])],
                "target": workspace.concept_names[int(edge["target"])],
                "signed_weight": float(edge["score"]) * float(edge["sign"]),
            }
        )
    return rows


def build_prompt(
    workspace,
    instance,
    baseline,
    endpoint: str,
    actions: list[dict[str, object]],
    *,
    true_label: int | None = None,
    oracle_context: bool = False,
    sequential_oracle_context: bool = False,
    selected_actions: Sequence[Mapping[str, object]] = (),
    steering_target: int | None = None,
) -> str:
    probabilities = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1).detach().cpu().numpy()
    top = np.argsort(-probabilities)[:5]
    prediction_rows = [
        {"activity": workspace.activity_names[int(index)], "probability": round(float(probabilities[index]), 4)}
        for index in top
    ]
    visible_actions = []
    for action in actions:
        visible = {
            key: value
            for key, value in action.items()
            if key not in {"payload_item", "conflict_key", "concept_idx"}
        }
        for key in (
            "current_activation",
            "current_probability",
            "signed_weight",
            "target_head_weight",
            "signed_contribution",
            "reverse_transition_probability",
        ):
            if key in visible:
                visible[key] = round(float(visible[key]), 4)
        visible_actions.append(visible)
    selected_action_ids = [str(action["action_id"]) for action in selected_actions]
    if steering_target is not None:
        if oracle_context:
            raise ValueError("steering_target and oracle_context are mutually exclusive")
        target_activity = workspace.activity_names[int(steering_target)]
        privilege = (
            "A user requests that the H3 forecast be steered toward "
            f"the activity '{target_activity}'. The true future activity, future concept targets, "
            "and action-level counterfactual effects are hidden. "
        )
    elif oracle_context:
        if true_label is None:
            raise ValueError("oracle_context requires true_label")
        target_activity = workspace.activity_names[int(true_label)]
        effect_key = "oracle_next_action" if sequential_oracle_context else "oracle_single_action"
        effect_description = (
            "the frozen model's conditional true-target probability change from the current selected set"
            if sequential_oracle_context
            else "the frozen model's true-target probability change from the unedited baseline"
        )
        planning_instruction = (
            "Use the storyboard to reason about compatibility, redundancy, and interactions. "
            "Choose a diverse ordered set of five actions that maximizes the final true-target outcome. "
            if not sequential_oracle_context
            else "Use the storyboard to reason about compatibility, redundancy, and interactions for the next decision. "
        )
        privilege = (
            "Privileged oracle context is enabled for this experiment. "
            f"The true future target activity is '{target_activity}'. "
            f"Each candidate action includes {effect_key}: {effect_description}, post-action predicted activity, "
            "and whether the resulting intervention set corrects the target. "
            "Treat these counterfactual effects as authoritative evidence for individual actions; "
            + planning_instruction
            + "Avoid redundant copies of the same visual fact unless their "
            "counterfactual effects justify them. "
        )
        if sequential_oracle_context:
            privilege += (
                f"The current selected action IDs are {json.dumps(selected_action_ids)}. "
                "The endpoint predictions and candidate effects below already include those selected actions. "
                "Choose only the single next action; it will be applied before the next planning step. "
            )
    else:
        privilege = (
            "The evaluation endpoint is "
            f"{endpoint_display(endpoint)} and its ground-truth activity is hidden. "
        )
    objective = (
        "rank candidate actions that could steer the H3 forecast toward the requested activity. "
        if steering_target is not None
        else "rank candidate actions that could correct this endpoint. "
    )
    suffix = (
        "The supplied four-frame storyboard contains only the final observed window at time t and never includes future frames. "
        + "Inspect the storyboard and model state, then "
        + objective
        + "Concept actions change either the current observed concept state or a causally preceding forecast state; edge actions temporarily delete or invert one learned relation; "
        + "activity actions alter an autoregressive source belief and can only affect later forecast steps. "
        + "Use only listed action_id values, do not choose conflicting alternatives for the same concept, edge, or activity source step, and avoid edits unsupported by visible evidence. "
        + "Return JSON only with keys ordered_action_ids and rationale; keep the rationale below 30 words.\n\n"
        f"Endpoint predictions: {json.dumps(prediction_rows)}\n"
        f"Learned relational edges: {json.dumps(graph_context(workspace))}\n"
        f"Candidate actions: {json.dumps(visible_actions)}"
    )
    return "You are an intervention policy for a concept-based video forecasting model. " + privilege + suffix


def assert_blinded_steering_prompt(prompt: str, actions: Sequence[Mapping[str, object]]) -> None:
    """Fail closed if privileged outcome fields enter a target-steering prompt."""

    forbidden_action_keys = {
        "held_out_h3_label",
        "target_matches_held_out_h3",
        "oracle_single_action",
        "oracle_next_action",
        "target_probability_delta",
        "counterfactual_effect",
    }
    leaked_keys = sorted({key for action in actions for key in action if key in forbidden_action_keys})
    if leaked_keys:
        raise RuntimeError(f"Privileged steering action fields would be visible: {leaked_keys}")
    forbidden_prompt_markers = ("oracle_single_action", "oracle_next_action", "held_out_h3_label")
    leaked_markers = [marker for marker in forbidden_prompt_markers if marker in prompt]
    if leaked_markers:
        raise RuntimeError(f"Privileged steering prompt markers found: {leaked_markers}")


def annotate_oracle_single_actions(
    workspace,
    instance,
    baseline,
    endpoint: str,
    true_label: int,
    actions: list[dict[str, object]],
    intervention_mode: str,
) -> None:
    """Attach privileged one-action counterfactual outcomes for the oracle-informed Qwen test."""

    payloads = [{"mode": intervention_mode, "items": [action["payload_item"]]} for action in actions]
    outputs = forward_outputs_batched_interventions(workspace, instance, payloads)
    baseline_prob = torch.softmax(
        endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1
    ).detach().cpu().numpy()
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1)
    for index, action in enumerate(actions):
        probability = probabilities[index].detach().cpu().numpy()
        prediction = int(probability.argmax())
        action["oracle_single_action"] = {
            "true_target_probability": round(float(probability[true_label]), 4),
            "delta_true_target_probability": round(
                float(probability[true_label] - baseline_prob[true_label]), 4
            ),
            "after_prediction": workspace.activity_names[prediction],
            "becomes_correct": bool(prediction == int(true_label)),
        }


def annotate_oracle_next_actions(
    workspace,
    instance,
    current_outputs,
    endpoint: str,
    true_label: int,
    actions: list[dict[str, object]],
    selected_actions: Sequence[Mapping[str, object]],
    intervention_mode: str,
) -> None:
    """Attach exact conditional effects for one compatible next action."""

    selected_items = [action["payload_item"] for action in selected_actions]
    payloads = [
        {"mode": intervention_mode, "items": selected_items + [action["payload_item"]]}
        for action in actions
    ]
    outputs = forward_outputs_batched_interventions(workspace, instance, payloads)
    current_probability = torch.softmax(
        endpoint_logits(workspace, current_outputs, endpoint)[0, -1], dim=-1
    ).detach().cpu().numpy()
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1)
    for index, action in enumerate(actions):
        probability = probabilities[index].detach().cpu().numpy()
        prediction = int(probability.argmax())
        action["oracle_next_action"] = {
            "true_target_probability": round(float(probability[true_label]), 4),
            "delta_true_target_probability": round(
                float(probability[true_label] - current_probability[true_label]), 4
            ),
            "after_prediction": workspace.activity_names[prediction],
            "becomes_correct": bool(prediction == int(true_label)),
        }


def resolve_media_path(raw_path: str) -> Path | None:
    path = Path(raw_path).expanduser()
    candidates = [path, Path.cwd() / path]
    dataset_root = configured_dataset_root().resolve()
    roots = {
        "Video_data": dataset_root / "Breakfast/Video_data",
        "Image_data": None,
    }
    parts = list(path.parts)
    if "Video_data" in parts:
        candidates.append(roots["Video_data"] / Path(*parts[parts.index("Video_data") + 1 :]))
    if "Image_data" in parts:
        suffix = Path(*parts[parts.index("Image_data") + 1 :])
        candidates.extend(
            root / suffix
            for root in (
                dataset_root / "Barista/Image_data",
                dataset_root / "MPII_Cooking_2/Image_data",
            )
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def observed_storyboard_data_url(workspace, instance: Mapping[str, object]) -> tuple[str, str]:
    media = resolve_media_path(str(instance["video_path"]))
    if media is None:
        raise FileNotFoundError(f"Could not resolve media path: {instance['video_path']}")
    video_index = int(instance["video_index"])
    timestep = int(instance["timestep"])
    num_windows = int(workspace.standardized_splits["test"]["lengths"][video_index])
    timestep = min(max(0, timestep), max(0, num_windows - 1))
    frames: list[np.ndarray] = []
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if media.is_dir():
        paths = [path for path in sorted(media.iterdir()) if path.is_file() and path.suffix.lower() in image_suffixes]
        if not paths:
            raise RuntimeError(f"No image frames under {media}")
        start = int(timestep * len(paths) / max(num_windows, 1))
        stop = max(start, int((timestep + 1) * len(paths) / max(num_windows, 1)) - 1)
        indices = np.linspace(start, min(stop, len(paths) - 1), 4).round().astype(int)
        frames = [frame for frame in (cv2.imread(str(paths[index])) for index in indices) if frame is not None]
    else:
        capture = cv2.VideoCapture(str(media))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Could not read frame count from {media}")
        start = int(timestep * frame_count / max(num_windows, 1))
        stop = max(start, int((timestep + 1) * frame_count / max(num_windows, 1)) - 1)
        for index in np.linspace(start, min(stop, frame_count - 1), 4).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        capture.release()
    if len(frames) != 4:
        raise RuntimeError(f"Expected four storyboard frames from {media}, got {len(frames)}")
    resized = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = 256.0 / max(float(height), 1.0)
        resized.append(cv2.resize(frame, (max(1, int(round(width * scale))), 256)))
    storyboard = cv2.hconcat(resized)
    ok, encoded = cv2.imencode(".jpg", storyboard, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError(f"Could not encode storyboard for {media}")
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return data_url, hashlib.sha256(encoded.tobytes()).hexdigest()


def request_policy(
    args: argparse.Namespace,
    prompt: str,
    storyboard_url: str,
    run_seed: int,
    actions: Sequence[Mapping[str, object]],
    required_count: int,
    retry_feedback: str | None = None,
) -> str:
    content: list[dict[str, object]] = [
        {"type": "image_url", "image_url": {"url": storyboard_url}},
    ]
    if retry_feedback:
        prompt += f"\nPrevious response was invalid: {retry_feedback}. Return a corrected response."
    prompt += f"\nReturn exactly {int(required_count)} action ID(s) from the candidate list in ranked order."
    content.append({"type": "text", "text": prompt})
    action_ids = sorted(str(action["action_id"]) for action in actions)
    response_schema = {
        "type": "object",
        "properties": {
            "ordered_action_ids": {
                "type": "array",
                "items": {"type": "string", "enum": action_ids},
                "minItems": int(required_count),
                "maxItems": int(required_count),
            },
            "rationale": {"type": "string", "maxLength": 400},
        },
        "required": ["ordered_action_ids", "rationale"],
        "additionalProperties": False,
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(args.temperature),
        "max_tokens": 600,
        "seed": int(run_seed),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "intervention_policy",
                "strict": True,
                "schema": response_schema,
            },
        },
    }
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(args.timeout)) as response:
        result = json.loads(response.read().decode("utf-8"))
    return str(result["choices"][0]["message"]["content"])


def parse_response(
    text: str,
    actions: Sequence[Mapping[str, object]],
    maximum: int = 5,
) -> tuple[list[str], str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    payload = json.loads(match.group(0))
    valid = {str(action["action_id"]): action for action in actions}
    ordered = []
    used_conflicts = set()
    raw_ids = payload.get("ordered_action_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("ordered_action_ids must be a list")
    rejected = []
    for raw_id in raw_ids:
        action_id = str(raw_id)
        action = valid.get(action_id)
        if action is None:
            rejected.append(action_id)
            continue
        conflict_key = str(action["conflict_key"])
        if conflict_key in used_conflicts:
            rejected.append(action_id)
            continue
        ordered.append(action_id)
        used_conflicts.add(conflict_key)
        if len(ordered) >= int(maximum):
            break
    if raw_ids and not ordered:
        raise ValueError(f"No permitted action IDs in response; rejected={rejected}")
    return ordered, str(payload.get("rationale", ""))


def compatible_actions(
    actions: Sequence[Mapping[str, object]],
    ordered_action_ids: Sequence[str],
) -> list[dict[str, object]]:
    by_id = {str(action["action_id"]): action for action in actions}
    used_conflicts = {
        str(by_id[action_id]["conflict_key"])
        for action_id in ordered_action_ids
        if action_id in by_id
    }
    return [dict(action) for action in actions if str(action["conflict_key"]) not in used_conflicts]


def evaluate_order(
    workspace,
    instance,
    baseline,
    endpoint: str,
    true_label: int,
    actions: list[dict[str, object]],
    order: list[str],
    intervention_mode: str,
    policy: str,
) -> list[dict[str, object]]:
    if not order:
        return []
    by_id = {str(action["action_id"]): action for action in actions}
    interventions = [
        {"mode": intervention_mode, "items": [by_id[action_id]["payload_item"] for action_id in order[:budget]]}
        for budget in range(1, len(order) + 1)
    ]
    outputs = forward_outputs_batched_interventions(workspace, instance, interventions)
    baseline_prob = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1).detach().cpu().numpy()
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1).detach().cpu().numpy()
    baseline_prediction = int(baseline_prob.argmax())
    baseline_correct = baseline_prediction == int(true_label)
    correctness = [int(probability.argmax()) == int(true_label) for probability in probabilities]
    first_correction_budget = next(
        (index + 1 for index, correct in enumerate(correctness) if not baseline_correct and correct),
        None,
    )
    first_harm_budget = next(
        (index + 1 for index, correct in enumerate(correctness) if baseline_correct and not correct),
        None,
    )
    rows = []
    for index, after_prob in enumerate(probabilities):
        prediction = int(after_prob.argmax())
        rows.append(
            {
                "policy": policy,
                "budget": index + 1,
                "actual_budget": len(order[: index + 1]),
                "endpoint": endpoint,
                "endpoint_horizon": endpoint_horizon(endpoint),
                "ordered_action_ids": json.dumps(order[: index + 1]),
                "selected_action_types": json.dumps(
                    [str(by_id[action_id]["action_type"]) for action_id in order[: index + 1]]
                ),
                "baseline_prediction": baseline_prediction,
                "after_prediction": prediction,
                "true_label": int(true_label),
                "baseline_correct": int(baseline_correct),
                "after_correct": int(prediction == true_label),
                "wrong_to_correct": int(not baseline_correct and prediction == true_label),
                "correct_to_wrong": int(baseline_correct and prediction != true_label),
                "wrong_to_correct_by_budget": int(
                    first_correction_budget is not None and first_correction_budget <= index + 1
                ),
                "correct_to_wrong_by_budget": int(
                    first_harm_budget is not None and first_harm_budget <= index + 1
                ),
                "first_correction_budget": first_correction_budget,
                "first_harm_budget": first_harm_budget,
                "effective_stop_budget": min(index + 1, first_correction_budget)
                if first_correction_budget is not None
                else index + 1,
                "label_flip": int(prediction != baseline_prediction),
                "true_probability_before": float(baseline_prob[true_label]),
                "true_probability_after": float(after_prob[true_label]),
                "true_probability_delta": float(after_prob[true_label] - baseline_prob[true_label]),
            }
        )
    return rows


def conflict_free_order(actions: Sequence[Mapping[str, object]], ranked_ids: Sequence[str], maximum: int = 5) -> list[str]:
    by_id = {str(action["action_id"]): action for action in actions}
    order = []
    used_conflicts = set()
    for action_id in ranked_ids:
        action = by_id.get(str(action_id))
        if action is None:
            continue
        conflict = str(action["conflict_key"])
        if conflict in used_conflicts:
            continue
        order.append(str(action_id))
        used_conflicts.add(conflict)
        if len(order) >= int(maximum):
            break
    return order


def ranked_counterfactual_order(
    workspace,
    instance,
    baseline,
    endpoint: str,
    true_label: int,
    actions: list[dict[str, object]],
    intervention_mode: str,
    objective: str,
) -> list[str]:
    if not actions:
        return []
    interventions = [
        {"mode": intervention_mode, "items": [action["payload_item"]]}
        for action in actions
    ]
    outputs = forward_outputs_batched_interventions(workspace, instance, interventions)
    baseline_prob = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1).detach().cpu().numpy()
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1).detach().cpu().numpy()
    predicted_class = int(baseline_prob.argmax())
    if objective == "oracle_true_class":
        scores = probabilities[:, int(true_label)] - baseline_prob[int(true_label)]
    elif objective == "predicted_class_reduction":
        scores = baseline_prob[predicted_class] - probabilities[:, predicted_class]
    else:
        raise ValueError(objective)
    ranked = [str(actions[index]["action_id"]) for index in np.argsort(-scores)]
    return conflict_free_order(actions, ranked)


def greedy_true_class_order(
    workspace,
    instance,
    baseline,
    endpoint: str,
    true_label: int,
    actions: list[dict[str, object]],
    intervention_mode: str,
    maximum: int = 5,
) -> list[str]:
    """Greedily optimize the true target after each selected action.

    This is stronger than the static one-action ranking, but it remains a
    greedy oracle rather than a globally optimized five-action upper bound.
    """

    selected: list[str] = []
    for _ in range(int(maximum)):
        candidates = compatible_actions(actions, selected)
        if not candidates:
            break
        by_id = {str(action["action_id"]): action for action in actions}
        payloads = [
            {
                "mode": intervention_mode,
                "items": [by_id[action_id]["payload_item"] for action_id in selected]
                + [candidate["payload_item"]],
            }
            for candidate in candidates
        ]
        outputs = forward_outputs_batched_interventions(workspace, instance, payloads)
        probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1)
        winner = int(torch.argmax(probabilities[:, int(true_label)]).item())
        selected.append(str(candidates[winner]["action_id"]))
    return selected


def random_order(actions: list[dict[str, object]], seed: int) -> list[str]:
    action_ids = [str(action["action_id"]) for action in actions]
    random.Random(int(seed)).shuffle(action_ids)
    return conflict_free_order(actions, action_ids)


def prepare_endpoint_cases(
    manifest_rows: list[dict[str, object]],
    record_map: Mapping[tuple[str, int], object],
    endpoints: Sequence[str],
    errors_per_seed: int,
    correct_per_seed: int,
    device: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, int], object], dict[str, object]]:
    workspace_cache: dict[tuple[str, int], object] = {}
    grouped: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    candidate_counts: dict[tuple[str, int, str, str], int] = defaultdict(int)
    seen = set()
    for row in manifest_rows:
        key = (str(row["dataset"]), int(row["seed"]))
        instance_key = (key[0], key[1], int(row["video_index"]), int(row["timestep"]))
        if instance_key in seen or key not in record_map:
            continue
        seen.add(instance_key)
        if key not in workspace_cache:
            workspace_cache[key] = load_workspace(record_map[key].checkpoint, device=device)
        workspace = workspace_cache[key]
        instance = select_instance(workspace, "test", int(row["video_index"]), int(row["timestep"]))
        baseline = forward_outputs(workspace, instance)
        for endpoint in endpoints:
            horizon = endpoint_horizon(endpoint)
            if horizon > 0 and horizon not in set(int(value) for value in workspace.forecast_horizons):
                continue
            true_label = endpoint_true_label(instance, endpoint)
            if true_label is None:
                continue
            probabilities = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1)
            prediction = int(probabilities.argmax().item())
            case_kind = "correct" if prediction == int(true_label) else "wrong"
            group_key = (key[0], key[1], endpoint, case_kind)
            candidate_counts[group_key] += 1
            grouped[group_key].append(
                {
                    **dict(row),
                    "endpoint": endpoint,
                    "true_label": int(true_label),
                    "baseline_prediction": prediction,
                    "instance": instance,
                    "baseline": baseline,
                }
            )

    selected = []
    selection_summary = {}
    for offset, group_key in enumerate(sorted(candidate_counts)):
        candidates = grouped.get(group_key, [])
        requested = int(correct_per_seed if group_key[3] == "correct" else errors_per_seed)
        chosen = deterministic_sample(candidates, requested, 20260804 + offset)
        selected.extend(dict(row) for row in chosen)
        selection_summary[f"{group_key[0]}/seed{group_key[1]}/{group_key[2]}/{group_key[3]}"] = {
            "candidate_pool": candidate_counts[group_key],
            "requested": requested,
            "selected": len(chosen),
        }
    return selected, workspace_cache, selection_summary


def train_transition_counts(workspace, horizon: int) -> np.ndarray:
    """Count train-only activity transitions for label-free steering targets."""

    cache_key = f"_steering_transition_counts_h{int(horizon)}"
    cached = getattr(workspace, cache_key, None)
    if isinstance(cached, np.ndarray):
        return cached
    split = workspace.standardized_splits["train"]
    count = np.zeros((len(workspace.activity_names), len(workspace.activity_names)), dtype=np.int64)
    for labels, raw_length in zip(split["activity_labels"], split["lengths"]):
        length = int(raw_length)
        values = np.asarray(labels, dtype=np.int64)
        for timestep in range(max(0, length - int(horizon))):
            source, target = int(values[timestep]), int(values[timestep + int(horizon)])
            if 0 <= source < count.shape[0] and 0 <= target < count.shape[1]:
                count[source, target] += 1
    setattr(workspace, cache_key, count)
    return count


def steering_target(
    workspace,
    baseline: Mapping[str, object],
    horizon: int,
    seed: int,
    exclude_sil: bool = False,
) -> dict[str, object]:
    """Choose a plausible requested target from model beliefs and train transitions only."""

    source_probabilities = torch.softmax(
        endpoint_logits(workspace, baseline, "activity")[0, -1], dim=-1
    ).detach().cpu().numpy()
    forecast_probabilities = torch.softmax(endpoint_logits(workspace, baseline, f"h{int(horizon)}")[0, -1], dim=-1)
    source_class = int(np.argmax(source_probabilities))
    baseline_forecast = int(forecast_probabilities.argmax().item())
    baseline_forecast_probability = float(forecast_probabilities.max().item())
    transition_counts = train_transition_counts(workspace, horizon)
    row_totals = transition_counts.sum(axis=1, keepdims=True)
    transition_probabilities = np.divide(
        transition_counts,
        row_totals,
        out=np.zeros_like(transition_counts, dtype=np.float64),
        where=row_totals > 0,
    )
    target_probabilities = source_probabilities @ transition_probabilities
    if not np.any(target_probabilities > 0):
        total = int(transition_counts.sum())
        if total:
            target_probabilities = transition_counts.sum(axis=0) / total
    ranked = sorted(
        range(len(workspace.activity_names)),
        key=lambda index: (-float(target_probabilities[index]), int(index)),
    )
    candidates = [index for index in ranked if index != baseline_forecast and target_probabilities[index] > 0]
    if exclude_sil:
        sil_indices = {
            index for index, name in enumerate(workspace.activity_names)
            if str(name).strip().lower() == "sil"
        }
        non_sil_candidates = [index for index in candidates if index not in sil_indices]
        if non_sil_candidates:
            candidates = non_sil_candidates
    if not candidates:
        candidates = [index for index in ranked if index != baseline_forecast]
    if not candidates:
        raise RuntimeError("No activity class is available as a steering target")
    target_rank = int(seed) % min(3, len(candidates))
    target_class = int(candidates[target_rank])
    argmax_counts = transition_counts[source_class]
    argmax_total = int(argmax_counts.sum())
    return {
        "target_label": target_class,
        "target_activity": workspace.activity_names[target_class],
        "source_activity_prediction": workspace.activity_names[source_class],
        "source_activity_prediction_index": source_class,
        "source_activity_prediction_probability": float(source_probabilities[source_class]),
        "baseline_h3_prediction": workspace.activity_names[baseline_forecast],
        "baseline_h3_prediction_index": baseline_forecast,
        "baseline_h3_top1_probability": baseline_forecast_probability,
        "target_plausibility": float(target_probabilities[target_class]),
        "target_rank_among_plausible_alternatives": target_rank + 1,
        "train_transition_count_given_argmax": int(argmax_counts[target_class]),
        "train_transition_probability_given_argmax": (
            float(argmax_counts[target_class] / argmax_total) if argmax_total else 0.0
        ),
        "selection_rule": (
            "train_transition_mixture_given_current_model_activity_distribution_excluding_baseline_h3"
            + ("_and_sil" if exclude_sil else "")
        ),
    }


def prepare_steering_cases(
    records: Sequence[object],
    cases_per_seed: int,
    stride: int,
    device: str,
    max_baseline_confidence: float,
    exclude_sil_datasets: Sequence[str] = (),
) -> tuple[list[dict[str, object]], dict[tuple[str, int], object], dict[str, object]]:
    """Sample low-confidence H3 windows and targets without test future labels."""

    if int(cases_per_seed) < 1 or int(stride) < 1:
        raise ValueError("steering cases per seed and stride must be positive")
    if not 0.0 < float(max_baseline_confidence) <= 1.0:
        raise ValueError("steering baseline confidence must be in (0, 1]")
    exclude_sil_datasets = {str(value).strip() for value in exclude_sil_datasets if str(value).strip()}
    horizon = 3
    workspace_cache: dict[tuple[str, int], object] = {}
    selected: list[dict[str, object]] = []
    selection_summary: dict[str, object] = {}
    for record_index, record in enumerate(sorted(records, key=lambda row: (row.dataset_key, row.seed))):
        key = (str(record.dataset_key), int(record.seed))
        workspace = load_workspace(record.checkpoint, device=device)
        workspace_cache[key] = workspace
        if horizon not in {int(value) for value in workspace.forecast_horizons}:
            raise ValueError(f"Checkpoint lacks H{horizon}: {record.checkpoint}")
        split = workspace.standardized_splits["test"]
        window_refs = []
        for video_index, raw_length in enumerate(split["lengths"]):
            start = max(int(workspace.history_length) - 1, 0)
            stop = max(start, int(raw_length) - horizon)
            window_refs.extend((int(video_index), int(timestep)) for timestep in range(start, stop, int(stride)))
        baseline_cache: dict[tuple[int, int], tuple[object, float]] = {}
        eligible_refs = []
        eligible_confidences = []
        for video_index, timestep in window_refs:
            ref = (video_index, timestep)
            instance = select_instance(workspace, "test", video_index, timestep)
            baseline = forward_outputs(workspace, instance)
            forecast_probabilities = torch.softmax(
                endpoint_logits(workspace, baseline, f"h{horizon}")[0, -1], dim=-1
            )
            baseline_confidence = float(forecast_probabilities.max().item())
            baseline_cache[ref] = (baseline, baseline_confidence)
            if baseline_confidence <= float(max_baseline_confidence):
                eligible_refs.append(ref)
                eligible_confidences.append(baseline_confidence)
        chosen = deterministic_sample(eligible_refs, int(cases_per_seed), 20260819 + record_index)
        if not chosen:
            raise RuntimeError(
                f"No H3 steering windows with baseline confidence <= {float(max_baseline_confidence):.3f} for {key}"
            )
        selected_confidences = []
        for case_offset, (video_index, timestep) in enumerate(chosen):
            instance = select_instance(workspace, "test", video_index, timestep)
            baseline, baseline_confidence = baseline_cache[(video_index, timestep)]
            target = steering_target(
                workspace,
                baseline,
                horizon,
                20260819 + record_index * 10_000 + case_offset,
                exclude_sil=key[0] in exclude_sil_datasets,
            )
            held_out_label = endpoint_true_label(instance, "h3")
            selected.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "endpoint": "h3",
                    "video_id": instance["video_id"],
                    "video_path": instance["video_path"],
                    "video_index": video_index,
                    "timestep": timestep,
                    "instance": instance,
                    "baseline": baseline,
                    "steering_eligible": True,
                    "baseline_h3_top1_probability": baseline_confidence,
                    "held_out_h3_label": held_out_label,
                    "target_matches_held_out_h3": bool(held_out_label == int(target["target_label"])),
                    **target,
                }
            )
            selected_confidences.append(baseline_confidence)
        selection_summary[f"{key[0]}/seed{key[1]}"] = {
            "candidate_windows": len(window_refs),
            "eligible_windows": len(eligible_refs),
            "selected": len(chosen),
            "max_baseline_h3_top1_probability": float(max_baseline_confidence),
            "eligible_baseline_h3_top1_probability": {
                "min": min(eligible_confidences),
                "mean": mean(eligible_confidences),
                "max": max(eligible_confidences),
            },
            "selected_baseline_h3_top1_probability": {
                "min": min(selected_confidences),
                "mean": mean(selected_confidences),
                "max": max(selected_confidences),
            },
            "selection_uses_held_out_future_labels": False,
            "selection_rule": "baseline H3 top-1 probability at most the configured threshold; deterministic cap per dataset and seed",
            "target_rule": (
                "top-three train-only transition alternatives weighted by the current model activity distribution and excluding baseline H3"
                + ("; SIL excluded when non-SIL alternatives exist" if key[0] in exclude_sil_datasets else "")
            ),
        }
    return selected, workspace_cache, selection_summary


def target_heuristic_order(actions: Sequence[Mapping[str, object]], target_activity: str) -> list[str]:
    """Non-oracle baseline: inject the requested activity, then enable target-ranked concepts."""

    ranked = sorted(
        actions,
        key=lambda action: (
            0 if str(action.get("action_type")) == "activity" and str(action.get("activity")) == target_activity else 1,
            0 if str(action.get("action_type")) == "concept" and str(action.get("set_state")) == "on" else 1,
            str(action["action_id"]),
        ),
    )
    return conflict_free_order(actions, [str(action["action_id"]) for action in ranked])


def signed_transition_heuristic_order(actions: Sequence[Mapping[str, object]]) -> list[str]:
    """Rank frozen signed sensitivities and train-only predecessor evidence on a common scale."""

    concept_scale = max(
        (abs(float(action["target_head_weight"])) for action in actions if "target_head_weight" in action),
        default=0.0,
    )
    transition_scale = max(
        (
            float(action["reverse_transition_probability"])
            for action in actions
            if "reverse_transition_probability" in action
        ),
        default=0.0,
    )

    def score(action: Mapping[str, object]) -> tuple[float, int, str]:
        if str(action.get("action_type")) == "concept" and bool(action.get("matches_recommended_state")):
            strength = abs(float(action.get("target_head_weight", 0.0))) / max(concept_scale, 1e-12)
            return (-strength, 0, str(action["action_id"]))
        if str(action.get("action_type")) == "activity" and "reverse_transition_probability" in action:
            strength = float(action["reverse_transition_probability"]) / max(transition_scale, 1e-12)
            return (-strength, 1, str(action["action_id"]))
        return (1.0, 2, str(action["action_id"]))

    ranked_ids = [str(action["action_id"]) for action in sorted(actions, key=score)]
    return conflict_free_order(actions, ranked_ids)


def validate_reference_steering_cases(cases: Sequence[Mapping[str, object]], reference_path: Path) -> None:
    """Require the selected windows and requested targets to match a prior artifact exactly."""

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        reference = payload.get("selected_cases")
    else:
        reference = payload
    if not isinstance(reference, list):
        raise ValueError(f"Reference cases must be a list or contain selected_cases: {reference_path}")
    if len(reference) != len(cases):
        raise RuntimeError(f"Reference case count differs: current={len(cases)} reference={len(reference)}")
    identity_fields = ("dataset", "seed", "video_id", "video_index", "timestep", "target_activity")
    for case_index, (case, expected) in enumerate(zip(cases, reference)):
        if not isinstance(expected, Mapping):
            raise ValueError(f"Reference case {case_index} is not an object")
        fields = identity_fields + (("target_label",) if "target_label" in expected else ())
        differences = {
            field: {"current": case.get(field), "reference": expected.get(field)}
            for field in fields
            if case.get(field) != expected.get(field)
        }
        if differences:
            raise RuntimeError(f"Reference case mismatch at index {case_index}: {differences}")


def single_action_target_diagnostics(
    workspace,
    instance,
    baseline: Mapping[str, object],
    endpoint: str,
    target_label: int,
    actions: Sequence[Mapping[str, object]],
    intervention_mode: str,
) -> tuple[dict[str, float], list[str]]:
    """Measure blinded-candidate capacity for diagnostics; these effects are never prompt-visible."""

    payloads = [
        {"mode": intervention_mode, "items": [action["payload_item"]]}
        for action in actions
    ]
    outputs = forward_outputs_batched_interventions(workspace, instance, payloads)
    baseline_probability = float(
        torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1)[int(target_label)].item()
    )
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1)
    deltas = {
        str(action["action_id"]): float(probabilities[index, int(target_label)].item() - baseline_probability)
        for index, action in enumerate(actions)
    }
    ranked_ids = sorted(deltas, key=lambda action_id: (-deltas[action_id], action_id))
    return deltas, conflict_free_order(actions, ranked_ids)


def first_action_diagnostics(
    actions: Sequence[Mapping[str, object]],
    ordered: Sequence[str],
    single_action_deltas: Mapping[str, float],
) -> dict[str, object]:
    by_id = {str(action["action_id"]): action for action in actions}
    first_id = str(ordered[0])
    first_concept_id = next(
        (action_id for action_id in ordered if str(by_id[str(action_id)]["action_type"]) == "concept"),
        None,
    )
    return {
        "first_action_id": first_id,
        "first_action_type": str(by_id[first_id]["action_type"]),
        "first_action_single_delta": float(single_action_deltas[first_id]),
        "first_concept_action_id": first_concept_id,
        "first_concept_single_delta": (
            float(single_action_deltas[str(first_concept_id)]) if first_concept_id is not None else None
        ),
    }


def evaluate_steering_order(
    workspace,
    instance,
    baseline: Mapping[str, object],
    endpoint: str,
    target_label: int,
    actions: Sequence[Mapping[str, object]],
    order: Sequence[str],
    intervention_mode: str,
    policy: str,
) -> list[dict[str, object]]:
    if not order:
        return []
    by_id = {str(action["action_id"]): action for action in actions}
    interventions = [
        {"mode": intervention_mode, "items": [by_id[action_id]["payload_item"] for action_id in order[:budget]]}
        for budget in range(1, len(order) + 1)
    ]
    outputs = forward_outputs_batched_interventions(workspace, instance, interventions)
    baseline_probabilities = torch.softmax(endpoint_logits(workspace, baseline, endpoint)[0, -1], dim=-1).detach().cpu().numpy()
    probabilities = torch.softmax(endpoint_logits(workspace, outputs, endpoint)[:, -1], dim=-1).detach().cpu().numpy()
    baseline_rank = 1 + int(np.sum(baseline_probabilities > baseline_probabilities[int(target_label)]))
    rows = []
    for index, after_probabilities in enumerate(probabilities):
        after_rank = 1 + int(np.sum(after_probabilities > after_probabilities[int(target_label)]))
        rows.append(
            {
                "policy": policy,
                "budget": index + 1,
                "actual_budget": len(order[: index + 1]),
                "endpoint": endpoint,
                "endpoint_horizon": endpoint_horizon(endpoint),
                "ordered_action_ids": json.dumps(order[: index + 1]),
                "selected_action_types": json.dumps(
                    [str(by_id[action_id]["action_type"]) for action_id in order[: index + 1]]
                ),
                "target_label": int(target_label),
                "baseline_prediction": int(baseline_probabilities.argmax()),
                "after_prediction": int(after_probabilities.argmax()),
                "target_probability_before": float(baseline_probabilities[int(target_label)]),
                "target_probability_after": float(after_probabilities[int(target_label)]),
                "target_probability_delta": float(
                    after_probabilities[int(target_label)] - baseline_probabilities[int(target_label)]
                ),
                "target_probability_increase": int(
                    after_probabilities[int(target_label)] > baseline_probabilities[int(target_label)]
                ),
                "target_top1": int(int(after_probabilities.argmax()) == int(target_label)),
                "target_rank_before": baseline_rank,
                "target_rank_after": after_rank,
                "target_rank_improvement": baseline_rank - after_rank,
            }
        )
    return rows


def run_target_steering(
    args: argparse.Namespace,
    records: Sequence[object],
    action_types: Sequence[str],
) -> None:
    if args.oracle_context or args.sequential_oracle_context:
        raise ValueError("Target steering must not enable oracle context")
    if set(action_types) != {"concept", "activity"}:
        raise ValueError("Target steering requires exactly concept and activity action types")
    exclude_sil_datasets = tuple(
        value.strip() for value in str(args.steering_exclude_sil_datasets).split(",") if value.strip()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases, workspace_cache, selection_summary = prepare_steering_cases(
        records,
        args.steering_cases_per_seed,
        args.steering_stride,
        args.device,
        args.steering_max_baseline_confidence,
        exclude_sil_datasets,
    )
    if args.steering_reference_cases is not None:
        validate_reference_steering_cases(cases, args.steering_reference_cases)
    rows: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    case_catalogs: list[dict[str, object]] = []
    diagnostic_cases: list[dict[str, object]] = []
    consecutive_request_errors = 0

    def persist_partial() -> None:
        write_csv(args.output_dir / "policy_rows.csv", rows)
        write_json(args.output_dir / "policy_responses.json", responses)
        write_json(args.output_dir / "case_action_catalogs.json", case_catalogs)

    for case_index, case in enumerate(cases):
        key = (str(case["dataset"]), int(case["seed"]))
        workspace = workspace_cache[key]
        instance = case["instance"]
        baseline = case["baseline"]
        target_label = int(case["target_label"])
        actions = candidate_actions(
            workspace,
            instance,
            baseline,
            "h3",
            action_types,
            args.candidate_concepts,
            0,
            args.candidate_classes,
            target_label,
            args.steering_activity_candidate_rule,
            args.steering_concept_evidence,
        )
        if not actions:
            raise RuntimeError(f"No steering actions for case={case_index}")
        prompt = build_prompt(
            workspace,
            instance,
            baseline,
            "h3",
            actions,
            steering_target=target_label,
        )
        assert_blinded_steering_prompt(prompt, actions)
        single_action_deltas, diagnostic_oracle_order = single_action_target_diagnostics(
            workspace,
            instance,
            baseline,
            "h3",
            target_label,
            actions,
            args.intervention_mode,
        )
        by_id = {str(action["action_id"]): action for action in actions}
        concept_deltas = {
            action_id: delta
            for action_id, delta in single_action_deltas.items()
            if str(by_id[action_id]["action_type"]) == "concept"
        }
        activity_deltas = {
            action_id: delta
            for action_id, delta in single_action_deltas.items()
            if str(by_id[action_id]["action_type"]) == "activity"
        }
        target_activity_deltas = {
            action_id: delta
            for action_id, delta in activity_deltas.items()
            if int(by_id[action_id]["payload_item"]["class_idx"]) == target_label
        }
        target_persistence = {}
        for transition_horizon in (1, 2, 3):
            transition_counts = train_transition_counts(workspace, transition_horizon)
            row = transition_counts[target_label]
            target_persistence[f"target_self_persistence_h{transition_horizon}"] = (
                float(row[target_label] / row.sum()) if int(row.sum()) else 0.0
            )
        diagnostic_case = {
            "dataset": case["dataset"],
            "seed": case["seed"],
            "case_index": case_index,
            "video_id": case["video_id"],
            "video_index": case["video_index"],
            "timestep": case["timestep"],
            "target_label": target_label,
            "target_activity": case["target_activity"],
            **target_persistence,
            "candidate_count": len(actions),
            "concept_candidate_count": len(concept_deltas),
            "activity_candidate_count": len(activity_deltas),
            "best_candidate_action_id": max(single_action_deltas, key=single_action_deltas.get),
            "best_candidate_type": str(by_id[max(single_action_deltas, key=single_action_deltas.get)]["action_type"]),
            "best_candidate_single_delta": max(single_action_deltas.values()),
            "best_concept_single_delta": max(concept_deltas.values()) if concept_deltas else None,
            "best_activity_single_delta": max(activity_deltas.values()) if activity_deltas else None,
            "best_requested_activity_single_delta": (
                max(target_activity_deltas.values()) if target_activity_deltas else None
            ),
            "requested_activity_is_best_activity": bool(
                activity_deltas
                and target_activity_deltas
                and max(target_activity_deltas.values()) == max(activity_deltas.values())
            ),
        }
        storyboard_url, storyboard_sha256 = observed_storyboard_data_url(workspace, instance)
        case_catalogs.append(
            {
                **{key: value for key, value in case.items() if key not in {"instance", "baseline"}},
                "candidate_actions": actions,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "storyboard_sha256": storyboard_sha256,
                "storyboard_observation_limit": "final observed window at t",
            }
        )
        deterministic_orders = [
            ("target_heuristic", 0, target_heuristic_order(actions, str(case["target_activity"]))),
            ("signed_transition_heuristic", 0, signed_transition_heuristic_order(actions)),
            ("diagnostic_oracle_candidate_upper_bound", 0, diagnostic_oracle_order),
            *[
                ("random", policy_run, random_order(actions, 20260819 + case_index * 100 + policy_run))
                for policy_run in range(int(args.policy_runs))
            ],
        ]
        for policy, policy_run, ordered in deterministic_orders:
            if len(ordered) != 5:
                raise RuntimeError(
                    f"{policy} could not construct five compatible steering actions for case={case_index}"
                )
            for row in evaluate_steering_order(
                workspace,
                instance,
                baseline,
                "h3",
                target_label,
                actions,
                ordered,
                args.intervention_mode,
                policy,
            ):
                row.update(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "case_index": case_index,
                        "policy_run": policy_run,
                        "video_id": case["video_id"],
                        "target_activity": case["target_activity"],
                        "baseline_h3_top1_probability": case["baseline_h3_top1_probability"],
                        "target_matches_held_out_h3": int(case["target_matches_held_out_h3"]),
                        "media_included": 1,
                        "model": "deterministic",
                    }
                )
                rows.append(row)
            diagnostic_case.update(
                {
                    f"{policy}_{key}": value
                    for key, value in first_action_diagnostics(actions, ordered, single_action_deltas).items()
                }
            )

        for policy_run in range(int(args.policy_runs)):
            ordered: list[str] = []
            response_text = ""
            rationale = ""
            attempt_errors = []
            raw_responses = []
            for attempt in range(int(args.policy_attempts)):
                allowed_actions = compatible_actions(actions, ordered)
                required_count = 5 - len(ordered)
                if required_count < 1 or not allowed_actions:
                    break
                try:
                    response_text = request_policy(
                        args,
                        prompt,
                        storyboard_url,
                        20260819 + case_index * 100 + policy_run * 10 + attempt,
                        allowed_actions,
                        required_count,
                        attempt_errors[-1] if attempt_errors else None,
                    )
                    raw_responses.append(response_text)
                    consecutive_request_errors = 0
                    additions, rationale = parse_response(response_text, allowed_actions, maximum=required_count)
                    ordered.extend(additions)
                    if len(additions) == required_count:
                        break
                    attempt_errors.append(
                        f"Expected {required_count} conflict-free additions, got {len(additions)}"
                    )
                except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                    attempt_errors.append(f"{type(exc).__name__}: {exc}")
                    consecutive_request_errors += 1
                    if consecutive_request_errors >= int(args.max_consecutive_request_errors):
                        persist_partial()
                        raise RuntimeError("LLM server failed repeatedly during target steering") from exc
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    attempt_errors.append(f"{type(exc).__name__}: {exc}")
            if len(ordered) != 5:
                persist_partial()
                raise RuntimeError(
                    f"LLM failed to return five valid steering actions for case={case_index}: {attempt_errors}"
                )
            for row in evaluate_steering_order(
                workspace,
                instance,
                baseline,
                "h3",
                target_label,
                actions,
                ordered,
                args.intervention_mode,
                "llm_target_steering",
            ):
                row.update(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "case_index": case_index,
                        "policy_run": policy_run,
                        "video_id": case["video_id"],
                        "target_activity": case["target_activity"],
                        "baseline_h3_top1_probability": case["baseline_h3_top1_probability"],
                        "target_matches_held_out_h3": int(case["target_matches_held_out_h3"]),
                        "media_included": 1,
                        "model": args.model,
                    }
                )
                rows.append(row)
            responses.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "case_index": case_index,
                    "policy_run": policy_run,
                    "video_id": case["video_id"],
                    "target_activity": case["target_activity"],
                    "ordered_action_ids": ordered,
                    "rationale": rationale,
                    "raw_response": response_text,
                    "raw_responses": raw_responses,
                    "attempt_errors": attempt_errors,
                    "media_included": True,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "storyboard_sha256": storyboard_sha256,
                }
            )
            qwen_diagnostics = first_action_diagnostics(actions, ordered, single_action_deltas)
            diagnostic_cases.append(
                {
                    **diagnostic_case,
                    "policy_run": policy_run,
                    "qwen_ordered_action_ids": json.dumps(ordered),
                    "qwen_selected_action_types": json.dumps(
                        [str(by_id[action_id]["action_type"]) for action_id in ordered]
                    ),
                    **{f"qwen_{key}": value for key, value in qwen_diagnostics.items()},
                }
            )
        if (case_index + 1) % max(1, int(args.checkpoint_every)) == 0:
            persist_partial()
    persist_partial()

    expected_llm_rows = len(cases) * int(args.policy_runs) * 5
    if len([row for row in rows if row["policy"] == "llm_target_steering"]) != expected_llm_rows:
        raise RuntimeError("Incomplete target-steering LLM rows")
    expected_rows_by_policy = {
        "llm_target_steering": expected_llm_rows,
        "random": expected_llm_rows,
        "target_heuristic": len(cases) * 5,
        "signed_transition_heuristic": len(cases) * 5,
        "diagnostic_oracle_candidate_upper_bound": len(cases) * 5,
    }
    actual_rows_by_policy = {
        policy: sum(str(row["policy"]) == policy for row in rows)
        for policy in expected_rows_by_policy
    }
    if actual_rows_by_policy != expected_rows_by_policy:
        raise RuntimeError(
            f"Incomplete target-steering policy rows: actual={actual_rows_by_policy} expected={expected_rows_by_policy}"
        )
    seed_summary = []
    metrics = ("target_probability_delta", "target_probability_increase", "target_top1", "target_rank_improvement")
    for policy in sorted({str(row["policy"]) for row in rows}):
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            for budget in range(1, 6):
                selected = [
                    row for row in rows
                    if row["policy"] == policy and row["dataset"] == dataset and int(row["budget"]) == budget
                ]
                for seed in sorted({int(row["seed"]) for row in selected}):
                    seed_rows = [row for row in selected if int(row["seed"]) == seed]
                    if not seed_rows:
                        continue
                    result = {"policy": policy, "dataset": dataset, "seed": seed, "endpoint": "h3", "budget": budget, "n": len(seed_rows)}
                    result.update({metric: mean(float(row[metric]) for row in seed_rows) for metric in metrics})
                    seed_summary.append(result)
    summary = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_summary:
        grouped[(str(row["policy"]), str(row["dataset"]), int(row["budget"]))].append(row)
    for (policy, dataset, budget), selected in sorted(grouped.items()):
        result = {"policy": policy, "dataset": dataset, "endpoint": "h3", "budget": budget, "seeds": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        summary.append(result)
    write_csv(args.output_dir / "policy_seed_summary.csv", seed_summary)
    write_csv(args.output_dir / "policy_summary.csv", summary)
    write_csv(args.output_dir / "diagnostic_case_summary.csv", diagnostic_cases)
    target_summary = []
    target_groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        target_groups[
            (str(row["dataset"]), str(row["target_activity"]), str(row["policy"]), int(row["budget"]))
        ].append(row)
    for (dataset, target_activity, policy, budget), selected in sorted(target_groups.items()):
        target_summary.append(
            {
                "dataset": dataset,
                "target_activity": target_activity,
                "policy": policy,
                "budget": budget,
                "n": len(selected),
                **{metric: mean(float(row[metric]) for row in selected) for metric in metrics},
            }
        )
    write_csv(args.output_dir / "diagnostic_target_summary.csv", target_summary)
    write_json(
        args.output_dir / "metadata.json",
        {
            "experiment": "target_conditioned_non_oracle_h3_steering",
            "model": args.model,
            "ground_truth_visible_to_policy": False,
            "future_concept_targets_visible_to_policy": False,
            "counterfactual_effects_visible_to_policy": False,
            "diagnostic_oracle_visible_to_policy": False,
            "planning_mode": "one_shot",
            "cases": len(cases),
            "cases_per_seed": args.steering_cases_per_seed,
            "steering_max_baseline_confidence": args.steering_max_baseline_confidence,
            "selection": selection_summary,
            "case_selection": "unmodified H3 top-1 probability at most the configured confidence threshold; at most the configured number of cases per dataset and seed",
            "target_selection": (
                "top-three train-only transition alternatives weighted by the current model activity distribution; baseline H3 prediction excluded"
                + (
                    "; SIL excluded for " + ", ".join(exclude_sil_datasets)
                    + " when non-SIL alternatives exist"
                    if exclude_sil_datasets
                    else ""
                )
            ),
            "steering_exclude_sil_datasets": list(exclude_sil_datasets),
            "steering_activity_candidate_rule": args.steering_activity_candidate_rule,
            "steering_concept_evidence": args.steering_concept_evidence,
            "steering_reference_cases": (
                str(args.steering_reference_cases) if args.steering_reference_cases is not None else None
            ),
            "target_matches_held_out_h3_count": sum(int(case["target_matches_held_out_h3"]) for case in cases),
            "action_types": list(action_types),
            "intervention_mode": args.intervention_mode,
            "media_definition": "four-frame storyboard from final observed window t only; future frames excluded",
            "objective": "increase frozen-model H3 probability and rank of the requested target activity",
            "baselines": ["random", "target_heuristic", "signed_transition_heuristic"],
            "diagnostic_oracle": {
                "policy": "diagnostic_oracle_candidate_upper_bound",
                "definition": "static ranking by measured single-candidate requested-target probability change, followed by conflict removal",
                "deployable": False,
                "visible_to_qwen": False,
            },
            "expected_rows_by_policy": expected_rows_by_policy,
            "actual_rows_by_policy": actual_rows_by_policy,
        },
    )


def run_steering_selection_only(
    args: argparse.Namespace,
    records: Sequence[object],
) -> None:
    exclude_sil_datasets = tuple(
        value.strip() for value in str(args.steering_exclude_sil_datasets).split(",") if value.strip()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases, _workspace_cache, selection_summary = prepare_steering_cases(
        records,
        args.steering_cases_per_seed,
        args.steering_stride,
        args.device,
        args.steering_max_baseline_confidence,
        exclude_sil_datasets,
    )
    if args.steering_reference_cases is not None:
        validate_reference_steering_cases(cases, args.steering_reference_cases)
    selected_cases = [
        {
            "dataset": case["dataset"],
            "seed": case["seed"],
            "video_id": case["video_id"],
            "video_index": case["video_index"],
            "timestep": case["timestep"],
            "baseline_h3_prediction": case["baseline_h3_prediction"],
            "baseline_h3_top1_probability": case["baseline_h3_top1_probability"],
            "target_activity": case["target_activity"],
            "target_plausibility": case["target_plausibility"],
        }
        for case in cases
    ]
    write_json(
        args.output_dir / "selection_preflight.json",
        {
            "experiment": "target_conditioned_non_oracle_h3_steering",
            "model": args.model,
            "ground_truth_visible_to_policy": False,
            "future_concept_targets_visible_to_policy": False,
            "counterfactual_effects_visible_to_policy": False,
            "steering_max_baseline_confidence": args.steering_max_baseline_confidence,
            "steering_cases_per_seed": args.steering_cases_per_seed,
            "steering_exclude_sil_datasets": list(exclude_sil_datasets),
            "steering_activity_candidate_rule": args.steering_activity_candidate_rule,
            "steering_concept_evidence": args.steering_concept_evidence,
            "steering_reference_cases": (
                str(args.steering_reference_cases) if args.steering_reference_cases is not None else None
            ),
            "reference_cases_match": args.steering_reference_cases is not None,
            "selection": selection_summary,
            "selected_cases": selected_cases,
        },
    )
    print(json.dumps(selection_summary, indent=2, sort_keys=True))


def run_steering_catalog_only(
    args: argparse.Namespace,
    records: Sequence[object],
    action_types: Sequence[str],
) -> None:
    """Build and validate the exact repaired candidate catalogs without policy requests."""

    exclude_sil_datasets = tuple(
        value.strip() for value in str(args.steering_exclude_sil_datasets).split(",") if value.strip()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases, workspace_cache, selection_summary = prepare_steering_cases(
        records,
        args.steering_cases_per_seed,
        args.steering_stride,
        args.device,
        args.steering_max_baseline_confidence,
        exclude_sil_datasets,
    )
    if args.steering_reference_cases is not None:
        validate_reference_steering_cases(cases, args.steering_reference_cases)
    catalogs = []
    activity_counts = []
    for case_index, case in enumerate(cases):
        workspace = workspace_cache[(str(case["dataset"]), int(case["seed"]))]
        actions = candidate_actions(
            workspace,
            case["instance"],
            case["baseline"],
            "h3",
            action_types,
            args.candidate_concepts,
            0,
            args.candidate_classes,
            int(case["target_label"]),
            args.steering_activity_candidate_rule,
            args.steering_concept_evidence,
        )
        prompt = build_prompt(
            workspace,
            case["instance"],
            case["baseline"],
            "h3",
            actions,
            steering_target=int(case["target_label"]),
        )
        assert_blinded_steering_prompt(prompt, actions)
        per_step = {
            step: sum(
                str(action["action_type"]) == "activity" and int(action.get("source_step", -1)) == step
                for action in actions
            )
            for step in range(3)
        }
        if args.steering_activity_candidate_rule == "reverse_transition" and any(
            count != int(args.candidate_classes) for count in per_step.values()
        ):
            raise RuntimeError(f"Case {case_index} has incomplete reverse-transition candidates: {per_step}")
        if args.steering_concept_evidence == "signed_contribution" and not any(
            "target_head_weight" in action for action in actions if str(action["action_type"]) == "concept"
        ):
            raise RuntimeError(f"Case {case_index} has no signed concept evidence")
        activity_counts.append(per_step)
        catalogs.append(
            {
                **{key: value for key, value in case.items() if key not in {"instance", "baseline"}},
                "candidate_actions": actions,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_blinding_validated": True,
            }
        )
    write_json(args.output_dir / "case_action_catalogs.json", catalogs)
    preflight = {
        "experiment": "target_conditioned_non_oracle_h3_steering_catalog_preflight",
        "cases": len(cases),
        "reference_cases_match": args.steering_reference_cases is not None,
        "steering_reference_cases": (
            str(args.steering_reference_cases) if args.steering_reference_cases is not None else None
        ),
        "steering_activity_candidate_rule": args.steering_activity_candidate_rule,
        "steering_concept_evidence": args.steering_concept_evidence,
        "candidate_classes_per_step": args.candidate_classes,
        "all_prompts_blinded": True,
        "ground_truth_visible_to_policy": False,
        "future_concept_targets_visible_to_policy": False,
        "counterfactual_effects_visible_to_policy": False,
        "activity_candidate_counts": {
            f"step{step}": sorted({counts[step] for counts in activity_counts}) for step in range(3)
        },
        "selection": selection_summary,
    }
    write_json(args.output_dir / "catalog_preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.sequential_oracle_context and not args.oracle_context:
        raise ValueError("--sequential-oracle-context requires --oracle-context")
    endpoints = parse_endpoints(args.endpoints)
    action_types = parse_action_types(args.action_types)
    if args.target_steering:
        if args.oracle_context or args.sequential_oracle_context:
            raise ValueError("--target-steering cannot use oracle context")
        if endpoints != ("h3",):
            raise ValueError("--target-steering requires --endpoints h3")
        if set(action_types) != {"concept", "activity"}:
            raise ValueError("--target-steering requires --action-types concept,activity")
    elif args.intervention_dir is None:
        raise ValueError("--intervention-dir is required unless --target-steering is set")
    records = discover_checkpoints(args.source_batch, args.protocols)
    record_map = {(record.dataset_key, record.seed): record for record in records}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "endpoints": endpoints,
                    "action_types": action_types,
                    "model": args.model,
                    "records": len(records),
                    "target_steering": bool(args.target_steering),
                    "steering_cases_per_seed": args.steering_cases_per_seed if args.target_steering else None,
                    "steering_max_baseline_confidence": (
                        args.steering_max_baseline_confidence if args.target_steering else None
                    ),
                    "steering_exclude_sil_datasets": (
                        [value.strip() for value in str(args.steering_exclude_sil_datasets).split(",") if value.strip()]
                        if args.target_steering
                        else []
                    ),
                    "steering_selection_only": bool(args.steering_selection_only),
                    "steering_activity_candidate_rule": args.steering_activity_candidate_rule,
                    "steering_concept_evidence": args.steering_concept_evidence,
                    "steering_reference_cases": (
                        str(args.steering_reference_cases) if args.steering_reference_cases is not None else None
                    ),
                },
                indent=2,
            )
        )
        return
    if args.target_steering:
        if args.steering_selection_only and args.steering_catalog_only:
            raise ValueError("Choose only one of --steering-selection-only and --steering-catalog-only")
        if args.steering_catalog_only:
            run_steering_catalog_only(args, records, action_types)
            return
        if args.steering_selection_only:
            run_steering_selection_only(args, records)
            return
        run_target_steering(args, records, action_types)
        return
    assert args.intervention_dir is not None
    manifest_rows = read_manifests(args.intervention_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases, workspace_cache, selection_summary = prepare_endpoint_cases(
        manifest_rows,
        record_map,
        endpoints,
        args.errors_per_seed,
        args.correct_per_seed,
        args.device,
    )
    rows = []
    responses = []
    case_catalogs = []
    consecutive_request_errors = 0
    llm_policy = (
        "llm_oracle_context_sequential"
        if args.sequential_oracle_context
        else "llm_oracle_context"
        if args.oracle_context
        else "llm"
    )

    def persist_partial() -> None:
        write_csv(args.output_dir / "policy_rows.csv", rows)
        write_json(args.output_dir / "policy_responses.json", responses)
        write_json(args.output_dir / "case_action_catalogs.json", case_catalogs)

    for case_index, case in enumerate(cases):
        key = (str(case["dataset"]), int(case["seed"]))
        workspace = workspace_cache[key]
        instance = case["instance"]
        baseline = case["baseline"]
        endpoint = str(case["endpoint"])
        actions = candidate_actions(
            workspace,
            instance,
            baseline,
            endpoint,
            action_types,
            args.candidate_concepts,
            args.candidate_edges,
            args.candidate_classes,
        )
        if not actions:
            raise RuntimeError(f"No candidate actions for case={case_index} endpoint={endpoint}")
        if args.oracle_context and not args.sequential_oracle_context:
            annotate_oracle_single_actions(
                workspace,
                instance,
                baseline,
                endpoint,
                int(case["true_label"]),
                actions,
                args.intervention_mode,
            )
        prompt = ""
        if not args.sequential_oracle_context:
            prompt = build_prompt(
                workspace,
                instance,
                baseline,
                endpoint,
                actions,
                true_label=int(case["true_label"]) if args.oracle_context else None,
                oracle_context=args.oracle_context,
            )
        storyboard_url, storyboard_sha256 = observed_storyboard_data_url(workspace, instance)
        case_catalog = {
                "dataset": case["dataset"],
                "seed": case["seed"],
                "case_index": case_index,
                "endpoint": endpoint,
                "video_id": case["video_id"],
                "video_path": case["video_path"],
                "video_index": case["video_index"],
                "timestep": case["timestep"],
                "true_label": case["true_label"],
                "baseline_prediction": case["baseline_prediction"],
                "candidate_actions": actions,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None,
                "storyboard_sha256": storyboard_sha256,
                "storyboard_observation_limit": "final observed window at t",
                "planning_mode": "sequential_oracle_context" if args.sequential_oracle_context else "one_shot",
                "planning_steps": [],
            }
        case_catalogs.append(case_catalog)

        deterministic_orders = [
            (
                "predicted_class_reduction",
                0,
                ranked_counterfactual_order(
                    workspace,
                    instance,
                    baseline,
                    endpoint,
                    int(case["true_label"]),
                    actions,
                    args.intervention_mode,
                    "predicted_class_reduction",
                ),
            ),
            (
                "oracle_true_class_static",
                0,
                ranked_counterfactual_order(
                    workspace,
                    instance,
                    baseline,
                    endpoint,
                    int(case["true_label"]),
                    actions,
                    args.intervention_mode,
                    "oracle_true_class",
                ),
            ),
        ]
        if args.oracle_context:
            deterministic_orders.append(
                (
                    "oracle_true_class_greedy_sequential",
                    0,
                    greedy_true_class_order(
                        workspace,
                        instance,
                        baseline,
                        endpoint,
                        int(case["true_label"]),
                        actions,
                        args.intervention_mode,
                    ),
                )
            )
        deterministic_orders.extend(
            (
                "random",
                policy_run,
                random_order(actions, 20260805 + case_index * 100 + policy_run),
            )
            for policy_run in range(int(args.policy_runs))
        )
        for policy, policy_run, ordered in deterministic_orders:
            scored = evaluate_order(
                workspace,
                instance,
                baseline,
                endpoint,
                int(case["true_label"]),
                actions,
                ordered,
                args.intervention_mode,
                policy,
            )
            for row in scored:
                row.update(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "case_index": case_index,
                        "policy_run": policy_run,
                        "endpoint": endpoint,
                        "video_id": case["video_id"],
                        "media_included": 1,
                        "model": args.model if policy.startswith("llm") else "deterministic",
                    }
                )
                rows.append(row)

        for policy_run in range(int(args.policy_runs)):
            run_seed = 20260803 + case_index * 100 + policy_run
            response_text = ""
            error = None
            ordered: list[str] = []
            rationale = ""
            attempt_errors = []
            raw_responses = []
            rationale_parts = []
            planning_steps = []
            if args.sequential_oracle_context:
                by_id = {str(action["action_id"]): action for action in actions}
                for step in range(5):
                    allowed_actions = compatible_actions(actions, ordered)
                    if not allowed_actions:
                        attempt_errors.append("No conflict-compatible candidate actions remain")
                        break
                    selected_actions = [by_id[action_id] for action_id in ordered]
                    current_outputs = (
                        baseline
                        if not selected_actions
                        else forward_outputs_batched_interventions(
                            workspace,
                            instance,
                            [{"mode": args.intervention_mode, "items": [action["payload_item"] for action in selected_actions]}],
                        )
                    )
                    annotate_oracle_next_actions(
                        workspace,
                        instance,
                        current_outputs,
                        endpoint,
                        int(case["true_label"]),
                        allowed_actions,
                        selected_actions,
                        args.intervention_mode,
                    )
                    prompt = build_prompt(
                        workspace,
                        instance,
                        current_outputs,
                        endpoint,
                        allowed_actions,
                        true_label=int(case["true_label"]),
                        oracle_context=True,
                        sequential_oracle_context=True,
                        selected_actions=selected_actions,
                    )
                    step_errors = []
                    step_response = ""
                    step_rationale = ""
                    addition: list[str] = []
                    selection_source = "qwen"
                    for attempt in range(int(args.policy_attempts)):
                        try:
                            step_response = request_policy(
                                args,
                                prompt,
                                storyboard_url,
                                run_seed + step * 100 + attempt,
                                allowed_actions,
                                1,
                                step_errors[-1] if step_errors else None,
                            )
                            raw_responses.append(step_response)
                            consecutive_request_errors = 0
                            addition, step_rationale = parse_response(
                                step_response,
                                allowed_actions,
                                maximum=1,
                            )
                            if len(addition) == 1:
                                error = None
                                break
                            error = f"Expected one conflict-free addition, got {len(addition)}"
                            step_errors.append(error)
                        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            step_errors.append(error)
                            consecutive_request_errors += 1
                            if consecutive_request_errors >= int(args.max_consecutive_request_errors):
                                persist_partial()
                                raise RuntimeError(
                                    f"LLM server failed {consecutive_request_errors} consecutive requests; last error: {error}"
                                ) from exc
                        except (ValueError, KeyError, json.JSONDecodeError) as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            step_errors.append(error)
                    if len(addition) != 1 and args.invalid_next_action_fallback == "conditional_best":
                        fallback = max(
                            allowed_actions,
                            key=lambda action: float(action["oracle_next_action"]["true_target_probability"]),
                        )
                        addition = [str(fallback["action_id"])]
                        selection_source = "conditional_best_fallback"
                        error = None
                    attempt_errors.extend(step_errors)
                    planning_steps.append(
                        {
                            "step": step + 1,
                            "selected_before": list(ordered),
                            "candidate_action_ids": [str(action["action_id"]) for action in allowed_actions],
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "chosen_action_ids": addition,
                            "selection_source": selection_source,
                            "rationale": step_rationale,
                            "raw_response": step_response,
                            "errors": step_errors,
                        }
                    )
                    response_text = step_response
                    if len(addition) != 1:
                        break
                    ordered.extend(addition)
                    if step_rationale:
                        rationale_parts.append(step_rationale)
                case_catalog["planning_steps"] = [
                    {key: value for key, value in step.items() if key not in {"raw_response"}}
                    for step in planning_steps
                ]
            else:
                for attempt in range(int(args.policy_attempts)):
                    allowed_actions = compatible_actions(actions, ordered)
                    required_count = 5 if not ordered else 1
                    if not allowed_actions:
                        attempt_errors.append("No conflict-compatible candidate actions remain")
                        break
                    try:
                        response_text = request_policy(
                            args,
                            prompt,
                            storyboard_url,
                            run_seed + attempt,
                            allowed_actions,
                            required_count,
                            attempt_errors[-1] if attempt_errors else None,
                        )
                        raw_responses.append(response_text)
                        consecutive_request_errors = 0
                        additions, addition_rationale = parse_response(
                            response_text,
                            allowed_actions,
                            maximum=required_count,
                        )
                        ordered.extend(additions)
                        if addition_rationale:
                            rationale_parts.append(addition_rationale)
                        if len(additions) != required_count:
                            error = f"Expected {required_count} conflict-free additions, got {len(additions)}"
                            attempt_errors.append(error)
                        else:
                            error = None
                        if len(ordered) == 5:
                            break
                    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_errors.append(error)
                        consecutive_request_errors += 1
                        if consecutive_request_errors >= int(args.max_consecutive_request_errors):
                            persist_partial()
                            raise RuntimeError(
                                f"LLM server failed {consecutive_request_errors} consecutive requests; last error: {error}"
                            ) from exc
                    except (ValueError, KeyError, json.JSONDecodeError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        attempt_errors.append(error)
            if len(ordered) != 5:
                responses.append(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "case_index": case_index,
                        "policy_run": policy_run,
                        "endpoint": endpoint,
                        "video_id": case["video_id"],
                        "video_path": case["video_path"],
                        "media_included": True,
                        "model": args.model,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "storyboard_sha256": storyboard_sha256,
                        "ordered_action_ids": ordered,
                        "rationale": " ".join(rationale_parts),
                        "raw_response": response_text,
                        "raw_responses": raw_responses,
                        "planning_steps": planning_steps,
                        "error": " | ".join(attempt_errors),
                    }
                )
                persist_partial()
                raise RuntimeError(
                    f"LLM failed to return five valid actions after {args.policy_attempts} attempts "
                    f"for case={case_index}, policy_run={policy_run}"
                )
            rationale = " ".join(rationale_parts)
            scored = evaluate_order(
                workspace,
                instance,
                baseline,
                endpoint,
                int(case["true_label"]),
                actions,
                ordered,
                args.intervention_mode,
                llm_policy,
            )
            for row in scored:
                row.update(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "case_index": case_index,
                        "policy_run": policy_run,
                        "endpoint": endpoint,
                        "video_id": case["video_id"],
                        "media_included": 1,
                        "model": args.model,
                    }
                )
                rows.append(row)
            responses.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "case_index": case_index,
                    "policy_run": policy_run,
                    "endpoint": endpoint,
                    "video_id": case["video_id"],
                    "video_path": case["video_path"],
                    "media_included": True,
                    "model": args.model,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "storyboard_sha256": storyboard_sha256,
                    "ordered_action_ids": ordered,
                    "rationale": rationale,
                    "raw_response": response_text,
                    "raw_responses": raw_responses,
                    "planning_steps": planning_steps,
                    "error": error,
                    "attempt_errors": attempt_errors,
                }
            )
        if (case_index + 1) % max(1, int(args.checkpoint_every)) == 0:
            persist_partial()
    persist_partial()
    llm_rows = [row for row in rows if row.get("policy") == llm_policy]
    expected_llm_rows = len(cases) * int(args.policy_runs) * 5
    if len(llm_rows) != expected_llm_rows:
        raise RuntimeError(f"Expected {expected_llm_rows} LLM score rows, got {len(llm_rows)}")
    if any(not bool(response.get("media_included")) for response in responses):
        raise RuntimeError("At least one LLM policy request omitted its observed storyboard")
    seed_summary = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            for endpoint in endpoints:
                for budget in range(1, 6):
                    selected = [
                        row
                        for row in rows
                        if row["policy"] == policy
                        and row["dataset"] == dataset
                        and row["endpoint"] == endpoint
                        and int(row["budget"]) == budget
                    ]
                    if not selected:
                        continue
                    for seed in sorted({int(row["seed"]) for row in selected}):
                        seed_rows = [row for row in selected if int(row["seed"]) == seed]
                        wrong_rows = [row for row in seed_rows if not bool(row["baseline_correct"])]
                        correct_rows = [row for row in seed_rows if bool(row["baseline_correct"])]
                        seed_summary.append(
                            {
                                "policy": policy,
                                "dataset": dataset,
                                "seed": seed,
                                "endpoint": endpoint,
                                "budget": budget,
                                "n": len(seed_rows),
                                "wrong_to_correct_by_budget_rate": mean(
                                    float(row["wrong_to_correct_by_budget"]) for row in wrong_rows
                                ),
                                "correct_to_wrong_by_budget_rate": mean(
                                    float(row["correct_to_wrong_by_budget"]) for row in correct_rows
                                ),
                                "true_probability_delta": mean(
                                    float(row["true_probability_delta"]) for row in seed_rows
                                ),
                            }
                        )
    summary = []
    grouped_seed_rows: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_summary:
        grouped_seed_rows[(str(row["policy"]), str(row["dataset"]), str(row["endpoint"]), int(row["budget"]))].append(row)
    for key, selected in sorted(grouped_seed_rows.items()):
        result = {
            "policy": key[0],
            "dataset": key[1],
            "endpoint": key[2],
            "budget": key[3],
            "seeds": len(selected),
        }
        for metric in (
            "wrong_to_correct_by_budget_rate",
            "correct_to_wrong_by_budget_rate",
            "true_probability_delta",
        ):
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        summary.append(result)
    write_csv(args.output_dir / "policy_seed_summary.csv", seed_summary)
    write_csv(args.output_dir / "policy_summary.csv", summary)
    write_json(
        args.output_dir / "metadata.json",
        {
            "model": args.model,
            "ground_truth_visible_to_policy": bool(args.oracle_context),
            "single_action_counterfactuals_visible_to_policy": bool(args.oracle_context),
            "conditional_counterfactuals_visible_to_policy": bool(args.sequential_oracle_context),
            "planning_mode": "sequential" if args.sequential_oracle_context else "one_shot",
            "invalid_next_action_fallback": args.invalid_next_action_fallback,
            "conditional_best_fallback_steps": sum(
                int(step.get("selection_source") == "conditional_best_fallback")
                for response in responses
                for step in response.get("planning_steps", [])
            ),
            "policy_runs": args.policy_runs,
            "policy_attempts": args.policy_attempts,
            "cases": len(cases),
            "endpoints": list(endpoints),
            "action_types": list(action_types),
            "intervention_mode": args.intervention_mode,
            "selection": selection_summary,
            "candidate_pool_definition": "fixed H3 manifests with correct and wrong cases sampled separately within each dataset and model seed",
            "activity_action_semantics": "source step s may affect forecast steps greater than s; direct activity-distribution overwrite is excluded for classification at t",
            "objective_evaluator": "true endpoint activity label; no LLM judge",
            "media_definition": "four-frame storyboard from final observed window t only; future frames excluded",
            "baselines": [
                "random",
                "predicted_class_reduction",
                "oracle_true_class_static",
                *( ["oracle_true_class_greedy_sequential"] if args.oracle_context else [] ),
            ],
        },
    )


if __name__ == "__main__":
    main()
