from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from web_ui import app as legacy_ui
from utils.graph_concept_ui import (
    _classifier_tensor_and_head,
    _prediction_tensor_for_head,
    forward_outputs,
    forward_outputs_batched_interventions,
    graph_branch_options,
    graph_layer_options,
    graph_matrix,
    graph_temporal_vector,
    learned_threshold_intervention_value,
)
from utils.intervention_notebook import select_instance


MAX_ROLLOUT_HORIZON = 3
MAX_REPAIR_ROWS = 10
MAX_EDGE_CANDIDATES = 12
REPAIR_SEARCH_KEY = "diagnosis_repair_search"
ACTIVE_REPAIRS_KEY = "diagnosis_active_repairs"


@dataclass(frozen=True)
class Target:
    step: int
    truth_idx: int
    predicted_idx: int
    probabilities: np.ndarray

    @property
    def wrong(self) -> bool:
        return int(self.predicted_idx) != int(self.truth_idx)


def probability_matrix(
    workspace,
    outputs: Mapping[str, object],
    step: int,
) -> np.ndarray | None:
    step = int(step)
    if step == 0:
        logits = outputs.get("activity_logits")
    else:
        logits = None
        for key in ("autoregressive_logits_by_step", "forecast_logits_by_horizon"):
            by_step = outputs.get(key)
            if isinstance(by_step, Mapping):
                logits = by_step.get(step, by_step.get(str(step)))
                if torch.is_tensor(logits):
                    break
    if not torch.is_tensor(logits):
        return None
    selected = logits[:, -1, :] if logits.ndim == 3 else logits
    return torch.softmax(selected, dim=-1).detach().cpu().numpy()


def activity_name(workspace, class_idx: int) -> str:
    index = int(class_idx)
    if 0 <= index < len(workspace.activity_names):
        return str(workspace.activity_names[index])
    return str(index)


def concept_name(workspace, concept_idx: int) -> str:
    index = int(concept_idx)
    if 0 <= index < len(workspace.concept_names):
        return str(workspace.concept_names[index])
    return str(index)


def target_options(
    workspace,
    instance: Mapping[str, Any],
    outputs: Mapping[str, object],
) -> list[Target]:
    future_labels = instance.get("future_labels", {})
    max_horizon = min(
        MAX_ROLLOUT_HORIZON,
        max(int(value) for value in workspace.forecast_horizons),
    )
    targets: list[Target] = []
    for step in range(max_horizon + 1):
        truth = (
            instance.get("current_label")
            if step == 0
            else (
                future_labels.get(step, future_labels.get(str(step)))
                if isinstance(future_labels, Mapping)
                else None
            )
        )
        probabilities = probability_matrix(workspace, outputs, step)
        if truth is None or probabilities is None or probabilities.shape[0] < 1:
            continue
        targets.append(
            Target(
                step=int(step),
                truth_idx=int(truth),
                predicted_idx=int(np.argmax(probabilities[0])),
                probabilities=probabilities[0],
            )
        )
    return targets


def target_label(workspace, target: Target) -> str:
    time_label = "t" if target.step == 0 else f"t+{target.step}"
    marker = "WRONG" if target.wrong else "correct"
    return (
        f"{time_label}: {activity_name(workspace, target.predicted_idx)} "
        f"vs truth {activity_name(workspace, target.truth_idx)} ({marker})"
    )


def render_paused_prediction_pane(
    workspace,
    targets: list[Target],
    timestep: int,
    length: int,
) -> None:
    st.markdown("### Predictions")
    st.caption(f"Window {timestep} / {max(length - 1, 0)}")
    current = next((target for target in targets if target.step == 0), targets[0])
    cols = st.columns(3)
    cols[0].metric("Prediction", activity_name(workspace, current.predicted_idx))
    cols[1].metric("Ground truth", activity_name(workspace, current.truth_idx))
    cols[2].metric(
        "Confidence",
        f"{float(current.probabilities[current.predicted_idx]):.3f}",
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "time": "t" if target.step == 0 else f"t+{target.step}",
                    "prediction": activity_name(workspace, target.predicted_idx),
                    "confidence": float(target.probabilities[target.predicted_idx]),
                    "ground truth": activity_name(workspace, target.truth_idx),
                    "truth p": float(target.probabilities[target.truth_idx]),
                    "correct": "yes" if not target.wrong else "no",
                }
                for target in targets
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Play or seek to navigate. Pause at a window to diagnose and build "
        "a multi-intervention repair."
    )


def strongest_other_probability(probabilities: np.ndarray, truth_idx: int) -> float:
    others = [
        float(value)
        for index, value in enumerate(probabilities)
        if int(index) != int(truth_idx)
    ]
    return max(others, default=0.0)


def repair_metrics(
    workspace,
    probabilities: np.ndarray,
    target: Target,
) -> dict[str, object]:
    truth_probability = float(probabilities[target.truth_idx])
    margin = truth_probability - strongest_other_probability(
        probabilities,
        target.truth_idx,
    )
    baseline_margin = float(target.probabilities[target.truth_idx]) - strongest_other_probability(
        target.probabilities,
        target.truth_idx,
    )
    predicted_idx = int(np.argmax(probabilities))
    return {
        "truth_probability": truth_probability,
        "margin": margin,
        "margin_improvement": margin - baseline_margin,
        "new_prediction_idx": predicted_idx,
        "new_prediction": activity_name(workspace, predicted_idx),
        "success": predicted_idx == int(target.truth_idx),
    }


def contrastive_reason_rows(
    workspace,
    outputs: Mapping[str, object],
    target: Target,
) -> list[dict[str, object]]:
    target_name = "activity" if target.step == 0 else "forecast"
    horizon = None if target.step == 0 else target.step
    tensor, head = _classifier_tensor_and_head(
        workspace,
        outputs,
        target_name,
        horizon,
    )
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        return []
    if head is None or not hasattr(head, "weight"):
        return []
    head_input = _prediction_tensor_for_head(workspace, outputs, tensor)
    if not torch.is_tensor(head_input):
        return []
    weights = head.weight.detach()
    if (
        weights.ndim != 2
        or target.predicted_idx >= weights.shape[0]
        or target.truth_idx >= weights.shape[0]
    ):
        return []
    values = head_input[0, -1, :].detach().cpu().numpy()
    contrastive_weights = (
        weights[target.predicted_idx] - weights[target.truth_idx]
    ).detach().cpu().numpy()
    contributions = values * contrastive_weights
    order = np.argsort(-np.abs(contributions))
    rows: list[dict[str, object]] = []
    for concept_idx in order[: min(MAX_REPAIR_ROWS, len(order))]:
        contribution = float(contributions[int(concept_idx)])
        rows.append(
            {
                "concept_idx": int(concept_idx),
                "concept": concept_name(workspace, int(concept_idx)),
                "head_input": float(values[int(concept_idx)]),
                "contrastive_weight": float(contrastive_weights[int(concept_idx)]),
                "contrastive_contribution": contribution,
                "favors": "wrong prediction" if contribution >= 0.0 else "ground truth",
                "abs_contribution": abs(contribution),
            }
        )
    return rows


def render_contrastive_reasons(
    workspace,
    target: Target,
    rows: list[Mapping[str, object]],
) -> None:
    st.markdown("### 1. Why is this prediction wrong?")
    st.caption(
        "Positive bars push the wrong prediction above the ground truth. Negative bars "
        "support the ground truth. This is one contrastive view, not two separate class plots."
    )
    if not rows:
        st.info("This checkpoint does not expose a compatible final linear head.")
        return
    frame = pd.DataFrame(rows)
    max_abs = max(float(frame["abs_contribution"].max()), 1e-6)
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "contrastive_contribution:Q",
                title="Push toward wrong prediction  ←  0  →  push toward ground truth",
                scale=alt.Scale(domain=[-1.08 * max_abs, 1.08 * max_abs], reverse=True),
            ),
            y=alt.Y(
                "concept:N",
                title=None,
                sort=alt.SortField(field="abs_contribution", order="descending"),
                axis=alt.Axis(labelLimit=320),
            ),
            color=alt.condition(
                alt.datum.contrastive_contribution >= 0,
                alt.value("#c2414b"),
                alt.value("#2f855a"),
            ),
            tooltip=[
                alt.Tooltip("concept:N", title="Concept"),
                alt.Tooltip(
                    "contrastive_contribution:Q",
                    title="Wrong-vs-truth contribution",
                    format="+.4f",
                ),
                alt.Tooltip("head_input:Q", title="Head input", format="+.4f"),
                alt.Tooltip("contrastive_weight:Q", title="Weight difference", format="+.4f"),
                alt.Tooltip("favors:N", title="Favors"),
            ],
        )
        .properties(height=max(220, 36 * len(rows)))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Wrong class: {activity_name(workspace, target.predicted_idx)} · "
        f"ground truth: {activity_name(workspace, target.truth_idx)}"
    )


def intervention_probability_rows(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
    interventions: list[Mapping[str, object]],
    metadata: list[dict[str, object]],
    chunk_size: int = 256,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(interventions), max(int(chunk_size), 1)):
        stop = min(len(interventions), start + max(int(chunk_size), 1))
        outputs = forward_outputs_batched_interventions(
            workspace,
            instance,
            interventions[start:stop],
        )
        probabilities = probability_matrix(workspace, outputs, target.step)
        if probabilities is None or probabilities.shape[0] != stop - start:
            continue
        for offset, probability_row in enumerate(probabilities):
            metadata_row = dict(metadata[start + offset])
            metadata_row.update(repair_metrics(workspace, probability_row, target))
            metadata_row["payload"] = dict(interventions[start + offset])
            rows.append(metadata_row)
    return rows


def rank_repairs(rows: list[dict[str, object]], top_k: int = MAX_REPAIR_ROWS) -> list[dict[str, object]]:
    rows.sort(
        key=lambda row: (
            bool(row.get("success", False)),
            float(row.get("margin_improvement", float("-inf"))),
        ),
        reverse=True,
    )
    return rows[: max(int(top_k), 0)]


def concept_repairs(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
) -> list[dict[str, object]]:
    valid_history = legacy_ui.valid_history_time_indices(instance)
    if not valid_history:
        return []
    current_time = max(valid_history)
    source_steps = list(range(target.step + 1))
    interventions: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for source_step in source_steps:
        for concept_idx in range(len(workspace.concept_names)):
            for setting, state in (("false", 0.1), ("true", 0.9)):
                if source_step == 0:
                    value = learned_threshold_intervention_value(
                        workspace,
                        concept_idx,
                        state,
                    )
                    if value is None:
                        continue
                    item = {
                        "item_type": "concept",
                        "time_idx": int(current_time),
                        "concept_idx": int(concept_idx),
                        "value": float(value),
                    }
                    source_label = "t"
                    scope = "observed node clamp"
                else:
                    value = state
                    item = {
                        "item_type": "concept",
                        "rollout_step": int(source_step),
                        "concept_idx": int(concept_idx),
                        "value": float(value),
                    }
                    source_label = f"t+{source_step}"
                    scope = "rollout-state edit"
                payload = {"mode": "persistent", "items": [item]}
                interventions.append(payload)
                metadata.append(
                    {
                        "kind": "concept",
                        "source": source_label,
                        "change": f"{concept_name(workspace, concept_idx)} = {state:.1f}",
                        "concept_idx": int(concept_idx),
                        "setting": setting,
                        "state": state,
                        "scope": scope,
                    }
                )
    return rank_repairs(
        intervention_probability_rows(
            workspace,
            instance,
            target,
            interventions,
            metadata,
        )
    )


def class_repairs(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
) -> list[dict[str, object]]:
    enabled_fn = getattr(workspace.model, "_activity_feedback_enabled", None)
    if not callable(enabled_fn) or not bool(enabled_fn()):
        return []
    interventions: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    future_labels = instance.get("future_labels", {})
    max_source_step = min(
        int(target.step),
        min(
            MAX_ROLLOUT_HORIZON,
            max(int(value) for value in workspace.forecast_horizons),
        )
        - 1,
    )
    source_steps = [max_source_step, *range(max_source_step)]
    for source_step in source_steps:
        source_truth = (
            instance.get("current_label")
            if source_step == 0
            else (
                future_labels.get(source_step, future_labels.get(str(source_step)))
                if isinstance(future_labels, Mapping)
                else None
            )
        )
        if source_truth is None:
            continue
        source_truth = int(source_truth)
        class_indices = [
            source_truth,
            *(
                class_idx
                for class_idx in range(len(workspace.activity_names))
                if int(class_idx) != source_truth
            ),
        ]
        for class_idx in class_indices:
            is_ground_truth = int(class_idx) == source_truth
            item = {
                "item_type": "activity",
                "step": int(source_step),
                "class_idx": int(class_idx),
                "probability": 0.9,
            }
            interventions.append({"mode": "input", "items": [item]})
            metadata.append(
                {
                    "kind": "class belief",
                    "source": "t" if source_step == 0 else f"t+{source_step}",
                    "change": (
                        f"{'ground truth' if is_ground_truth else 'override'}: "
                        f"{activity_name(workspace, class_idx)} belief = 0.9"
                    ),
                    "class_idx": int(class_idx),
                    "is_source_ground_truth": is_ground_truth,
                    "is_selected_source": int(source_step) == int(target.step),
                    "scope": (
                        f"does not rewrite {'t' if source_step == 0 else f't+{source_step}'}; "
                        f"propagates through concepts from t+{source_step + 1} onward"
                    ),
                }
            )
    rows = rank_repairs(
        intervention_probability_rows(
            workspace,
            instance,
            target,
            interventions,
            metadata,
        ),
        top_k=len(interventions),
    )
    ground_truth_rows = sorted(
        (row for row in rows if bool(row.get("is_source_ground_truth", False))),
        key=lambda row: bool(row.get("is_selected_source", False)),
        reverse=True,
    )
    override_rows = [
        row for row in rows if not bool(row.get("is_source_ground_truth", False))
    ]
    return (ground_truth_rows + override_rows)[:MAX_REPAIR_ROWS]


def edge_candidates(
    workspace,
    target: Target,
    anchor_indices: set[int],
) -> list[dict[str, object]]:
    available = graph_branch_options(workspace)
    branches: list[str] = []
    for branch in ("shared", "window" if target.step == 0 else "forecast"):
        if branch in available and branch not in branches:
            branches.append(branch)
    if not branches and "legacy" in available:
        branches = ["legacy"]

    candidates: list[dict[str, object]] = []
    for branch in branches:
        layer_options = graph_layer_options(workspace, branch)
        layer = "mean" if "mean" in layer_options else layer_options[0]
        for kind, matrix_kind in (("spatial", "spatial"), ("cross_temporal", "cross temporal")):
            matrix = graph_matrix(workspace, branch, layer, edge_kind=matrix_kind)
            if matrix.ndim != 2:
                continue
            for raw_source, raw_target in np.argwhere(np.abs(matrix) > 0.0):
                source_idx = int(raw_source)
                target_idx = int(raw_target)
                if source_idx == target_idx and kind == "cross_temporal":
                    continue
                if anchor_indices and source_idx not in anchor_indices and target_idx not in anchor_indices:
                    continue
                candidates.append(
                    {
                        "branch": branch,
                        "kind": kind,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "weight": float(matrix[source_idx, target_idx]),
                    }
                )
        temporal = graph_temporal_vector(workspace, branch, layer)
        for concept_idx, weight in enumerate(np.asarray(temporal).reshape(-1)):
            if abs(float(weight)) <= 0.0:
                continue
            if anchor_indices and int(concept_idx) not in anchor_indices:
                continue
            candidates.append(
                {
                    "branch": branch,
                    "kind": "temporal",
                    "source_idx": int(concept_idx),
                    "target_idx": int(concept_idx),
                    "weight": float(weight),
                }
            )
    candidates.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return candidates[:MAX_EDGE_CANDIDATES]


def edge_repairs(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
    anchor_indices: set[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in edge_candidates(workspace, target, anchor_indices):
        for setting, scale in (("zero", 0.0), ("invert", -1.0)):
            item = {
                "item_type": "edge",
                "edge_kind": str(candidate["kind"]),
                "branch": str(candidate["branch"]),
                "source_idx": int(candidate["source_idx"]),
                "target_idx": int(candidate["target_idx"]),
                "edge_scale": float(scale),
            }
            payload = {"mode": "input", "items": [item]}
            try:
                outputs = forward_outputs(workspace, instance, intervention=payload)
                probabilities = probability_matrix(workspace, outputs, target.step)
            except (IndexError, RuntimeError, TypeError, ValueError):
                continue
            if probabilities is None or probabilities.shape[0] < 1:
                continue
            metrics = repair_metrics(workspace, probabilities[0], target)
            source_name = concept_name(workspace, int(candidate["source_idx"]))
            target_name = concept_name(workspace, int(candidate["target_idx"]))
            rows.append(
                {
                    "kind": "edge",
                    "source": "shared across eligible times",
                    "change": (
                        f"{candidate['branch']} {candidate['kind']}: "
                        f"{source_name} -> {target_name} ({setting})"
                    ),
                    "scope": "edge parameter is not occurrence-specific",
                    "payload": payload,
                    **metrics,
                }
            )
    return rank_repairs(rows)


def search_context(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
) -> tuple[object, ...]:
    return (
        id(workspace.model),
        str(instance.get("split")),
        int(instance.get("video_index", -1)),
        int(instance.get("timestep", -1)),
        int(target.step),
        int(target.truth_idx),
        int(target.predicted_idx),
    )


def run_repair_search(
    workspace,
    instance: Mapping[str, Any],
    target: Target,
    reason_rows: list[Mapping[str, object]],
) -> dict[str, object]:
    concepts = concept_repairs(workspace, instance, target)
    classes = class_repairs(workspace, instance, target)
    anchors = {
        int(row["concept_idx"])
        for row in reason_rows[:5]
        if row.get("concept_idx") is not None
    }
    anchors.update(
        int(row["concept_idx"])
        for row in concepts[:5]
        if row.get("concept_idx") is not None
    )
    edges = edge_repairs(workspace, instance, target, anchors)
    return {
        "context": search_context(workspace, instance, target),
        "concept": concepts,
        "class belief": classes,
        "edge": edges,
    }


def repair_summary_rows(search: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in ("concept", "class belief", "edge"):
        candidates = search.get(kind, [])
        if not isinstance(candidates, list) or not candidates:
            rows.append(
                {
                    "type": kind,
                    "source": "n/a",
                    "recommended change": "No applicable candidate",
                    "new prediction": "n/a",
                    "truth p": np.nan,
                    "margin improvement": np.nan,
                    "flips to truth": "no",
                }
            )
            continue
        best = candidates[0]
        rows.append(
            {
                "type": kind,
                "source": best["source"],
                "recommended change": best["change"],
                "new prediction": best["new_prediction"],
                "truth p": float(best["truth_probability"]),
                "margin improvement": float(best["margin_improvement"]),
                "flips to truth": "yes" if best["success"] else "no",
            }
        )
    return rows


def repair_choice_rows(search: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in ("concept", "class belief", "edge"):
        candidates = search.get(kind, [])
        if not isinstance(candidates, list):
            continue
        for rank, row in enumerate(candidates[:5], 1):
            rows.append({"choice": f"{kind} #{rank}: {row['source']} · {row['change']}", **dict(row)})
    rows.sort(
        key=lambda row: (
            bool(row.get("success", False)),
            float(row.get("margin_improvement", float("-inf"))),
        ),
        reverse=True,
    )
    return rows


def render_candidate_details(search: Mapping[str, object]) -> None:
    with st.expander("Candidate details", expanded=False):
        for kind in ("concept", "class belief", "edge"):
            st.markdown(f"**{kind.title()} candidates**")
            candidates = search.get(kind, [])
            if not isinstance(candidates, list) or not candidates:
                st.caption("No applicable candidates.")
                continue
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "source": row["source"],
                            "change": row["change"],
                            "new prediction": row["new_prediction"],
                            "truth p": float(row["truth_probability"]),
                            "margin improvement": float(row["margin_improvement"]),
                            "flips to truth": "yes" if row["success"] else "no",
                            "scope": row.get("scope", ""),
                        }
                        for row in candidates
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def rollout_comparison_rows(
    workspace,
    instance: Mapping[str, Any],
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> list[dict[str, object]]:
    before_targets = {
        target.step: target
        for target in target_options(workspace, instance, before)
    }
    after_targets = {
        target.step: target
        for target in target_options(workspace, instance, after)
    }
    rows: list[dict[str, object]] = []
    for step in sorted(set(before_targets) & set(after_targets)):
        baseline = before_targets[step]
        intervened = after_targets[step]
        truth_idx = baseline.truth_idx
        rows.append(
            {
                "time": "t" if step == 0 else f"t+{step}",
                "ground truth": activity_name(workspace, truth_idx),
                "prediction before": activity_name(workspace, baseline.predicted_idx),
                "truth p before": float(baseline.probabilities[truth_idx]),
                "prediction after": activity_name(workspace, intervened.predicted_idx),
                "truth p after": float(intervened.probabilities[truth_idx]),
                "correct after": "yes" if intervened.predicted_idx == truth_idx else "no",
            }
        )
    return rows


def active_repair_entries(context: tuple[object, ...]) -> list[dict[str, object]]:
    stored = st.session_state.get(ACTIVE_REPAIRS_KEY)
    if not isinstance(stored, dict) or stored.get("context") != context:
        return []
    return [
        dict(entry)
        for entry in stored.get("entries", [])
        if isinstance(entry, Mapping)
    ]


def store_active_repair_entries(
    context: tuple[object, ...],
    entries: list[Mapping[str, object]],
) -> None:
    st.session_state[ACTIVE_REPAIRS_KEY] = {
        "context": context,
        "entries": [dict(entry) for entry in entries],
    }


def combined_repair_payload(
    entries: list[Mapping[str, object]],
) -> dict[str, object] | None:
    if not entries:
        return None
    items: list[dict[str, object]] = []
    persistent = False
    for entry in entries:
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            continue
        persistent = persistent or str(payload.get("mode", "input")) == "persistent"
        raw_items = payload.get("items")
        if raw_items is None:
            raw_items = [payload]
        items.extend(
            dict(item)
            for item in raw_items
            if isinstance(item, Mapping)
        )
    if not items:
        return None
    return {
        "mode": "persistent" if persistent else "input",
        "items": items,
    }


def render_active_repair_cart(
    context: tuple[object, ...],
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    st.markdown("### 3. Combined intervention set")
    if not entries:
        st.caption("No interventions added yet. Add several recommendations to combine them.")
        return entries
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "#": index + 1,
                    "type": entry.get("kind", ""),
                    "source": entry.get("source", ""),
                    "change": entry.get("change", ""),
                }
                for index, entry in enumerate(entries)
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    cols = st.columns([0.55, 0.2, 0.25])
    selected_index = int(
        cols[0].selectbox(
            "Intervention to remove",
            list(range(len(entries))),
            format_func=lambda index: (
                f"{int(index) + 1}: {entries[int(index)].get('change', '')}"
            ),
            key="diagnosis_remove_repair",
        )
    )
    if cols[1].button("Remove", width="stretch"):
        updated = [
            entry
            for index, entry in enumerate(entries)
            if index != selected_index
        ]
        store_active_repair_entries(context, updated)
        st.rerun()
    if cols[2].button("Reset all", width="stretch"):
        store_active_repair_entries(context, [])
        st.rerun()
    return entries


def graph_driver_rows(
    instance: Mapping[str, Any],
    reason_rows: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    valid_history = legacy_ui.valid_history_time_indices(instance)
    if not valid_history:
        return []
    current_time = max(valid_history)
    return [
        {
            "concept_idx": int(row["concept_idx"]),
            "history_t": int(current_time),
            "contribution": float(row.get("contrastive_contribution", 0.0)),
        }
        for row in reason_rows
        if row.get("concept_idx") is not None
    ]


def render_repair_panel(
    workspace,
    instance: Mapping[str, Any],
    baseline_outputs: Mapping[str, object],
    target: Target,
    reason_rows: list[Mapping[str, object]],
) -> None:
    st.markdown("### 2. What intervention repairs it?")
    st.caption(
        "Success means the model predicts the ground-truth activity. If no candidate flips "
        "the decision, candidates are ranked by improvement in the ground-truth decision margin."
    )
    context = search_context(workspace, instance, target)
    cached = st.session_state.get(REPAIR_SEARCH_KEY)
    if st.button(
        "Find concept, class-belief, and edge repairs",
        type="primary",
        width="stretch",
        disabled=not target.wrong,
    ):
        with st.spinner("Testing interventions through the real model forward path..."):
            cached = run_repair_search(workspace, instance, target, reason_rows)
        st.session_state[REPAIR_SEARCH_KEY] = cached
    if not isinstance(cached, dict) or cached.get("context") != context:
        if target.wrong:
            st.info("Run the repair search for this error.")
        else:
            st.success("The selected horizon is already classified correctly.")
        return

    st.dataframe(
        pd.DataFrame(repair_summary_rows(cached)),
        width="stretch",
        hide_index=True,
    )
    render_candidate_details(cached)
    choices = repair_choice_rows(cached)
    if not choices:
        return
    selected_choice = st.selectbox(
        "Recommended intervention to inspect",
        [row["choice"] for row in choices],
        key="diagnosis_repair_choice",
    )
    selected = next(row for row in choices if row["choice"] == selected_choice)
    cols = st.columns([0.7, 0.3])
    cols[0].caption(
        f"Expected prediction: {selected['new_prediction']} · "
        f"margin improvement: {float(selected['margin_improvement']):+.3f}"
    )
    if cols[1].button("Add to intervention set", width="stretch"):
        entries = active_repair_entries(context)
        if not any(str(entry.get("choice")) == selected_choice for entry in entries):
            entries.append(
                {
                    "choice": selected_choice,
                    "kind": selected.get("kind", ""),
                    "source": selected.get("source", ""),
                    "change": selected.get("change", ""),
                    "payload": selected["payload"],
                }
            )
            store_active_repair_entries(context, entries)
        st.rerun()

    entries = render_active_repair_cart(
        context,
        active_repair_entries(context),
    )
    combined_payload = combined_repair_payload(entries)
    if combined_payload is None:
        return
    intervened_outputs = forward_outputs(
        workspace,
        instance,
        intervention=combined_payload,
    )
    st.markdown("### 4. Combined before and after across the rollout")
    st.caption(
        f"{len(entries)} active intervention{'s' if len(entries) != 1 else ''}; "
        "the result below is one joint forward pass."
    )
    st.dataframe(
        pd.DataFrame(
            rollout_comparison_rows(
                workspace,
                instance,
                baseline_outputs,
                intervened_outputs,
            )
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown("### 5. Combined intervention propagation graph")
    drivers = graph_driver_rows(instance, reason_rows)
    target_name = "activity" if target.step == 0 else "forecast"
    legacy_ui.render_scene_graph_overview(
        workspace,
        instance,
        baseline_outputs,
        intervened_outputs,
        drivers,
        drivers,
        list(combined_payload["items"]),
        target_name,
    )


def main() -> None:
    st.set_page_config(page_title="Forecast Error Diagnosis and Repair", layout="wide")
    legacy_ui.inject_page_style()
    st.title("Forecast Error Diagnosis and Repair")
    st.caption(
        "Select one incorrect window and horizon. The app explains why the wrong class "
        "beat the ground truth, then searches for interventions that correct it."
    )

    record, device = legacy_ui.checkpoint_picker()
    if record is None:
        return
    try:
        workspace = legacy_ui.cached_workspace(
            str(record.path),
            device,
            str(legacy_ui.ARGS.dataset_root),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load checkpoint: {exc}")
        return

    split, video_index = legacy_ui.instance_picker(workspace)
    length = int(workspace.standardized_splits[split]["lengths"][video_index])
    raw_path = workspace.preprocessed_data[split]["video_paths"][video_index]
    video_id = workspace.preprocessed_data[split]["video_ids"][video_index]
    path = legacy_ui.resolve_video_path(
        raw_path,
        dataset=record.dataset,
        video_id=video_id,
        args=record.args,
        dataset_root=legacy_ui.ARGS.dataset_root,
    )
    timing = legacy_ui.video_timing_metadata(
        workspace,
        raw_path,
        path,
        video_id,
    )
    source = legacy_ui.video_source_info(
        path,
        length,
        window_spans=timing.get("window_spans"),
        video_meta=timing.get("video_meta"),
    )
    legacy_ui.ensure_video_state(f"diagnosis:{split}:{video_index}:{path}", source)
    legacy_ui.apply_pending_slider_position()
    playing = bool(st.session_state.get(legacy_ui.PLAYING_STATE_KEY, False))
    if not playing:
        legacy_ui.apply_slider_seek_if_needed(source)
    position_seconds = legacy_ui.current_position_seconds(source)

    video_col, context_col = st.columns([1.25, 0.75], gap="large")
    with video_col:
        video_slot = st.empty()
        position_seconds = legacy_ui.video_position_picker(source)
        legacy_ui.render_video_pane(
            video_slot,
            path,
            source,
            position_seconds,
            playing=playing,
        )
        legacy_ui.render_playback_status(
            source,
            position_seconds,
            legacy_ui.source_position_to_timestep(position_seconds, source),
        )
    timestep = legacy_ui.source_position_to_timestep(position_seconds, source)
    if playing:
        with context_col:
            legacy_ui.render_playing_prediction_pane(
                workspace,
                split,
                video_index,
                source,
            )
            st.info("Pause playback to run diagnosis and intervention search.")
        return

    instance = select_instance(
        workspace,
        split=split,
        video_index=video_index,
        timestep=timestep,
    )
    baseline_outputs = forward_outputs(workspace, instance)
    targets = target_options(workspace, instance, baseline_outputs)
    if not targets:
        st.error("No current or future predictions are available for this window.")
        return
    with context_col:
        render_paused_prediction_pane(
            workspace,
            targets,
            timestep,
            length,
        )

    wrong_indices = [index for index, target in enumerate(targets) if target.wrong]
    default_target = wrong_indices[0] if wrong_indices else 0
    selected_index = int(
        st.selectbox(
            "Error to diagnose",
            list(range(len(targets))),
            index=default_target,
            format_func=lambda index: target_label(workspace, targets[int(index)]),
        )
    )
    target = targets[selected_index]
    if st.session_state.get("diagnosis_context") != search_context(workspace, instance, target):
        st.session_state["diagnosis_context"] = search_context(workspace, instance, target)
        st.session_state.pop(REPAIR_SEARCH_KEY, None)
        st.session_state.pop(ACTIVE_REPAIRS_KEY, None)

    time_label = "t" if target.step == 0 else f"t+{target.step}"
    predicted_probability = float(target.probabilities[target.predicted_idx])
    truth_probability = float(target.probabilities[target.truth_idx])
    margin = truth_probability - strongest_other_probability(
        target.probabilities,
        target.truth_idx,
    )
    st.markdown(f"## Selected error: {time_label}")
    cols = st.columns(4)
    cols[0].metric("Prediction", activity_name(workspace, target.predicted_idx))
    cols[1].metric("Ground truth", activity_name(workspace, target.truth_idx))
    cols[2].metric("Predicted-class p", f"{predicted_probability:.3f}")
    cols[3].metric("Truth decision margin", f"{margin:+.3f}")
    st.caption(
        f"Video: {instance['video_id']} · window {timestep} of {max(length - 1, 0)}. "
        "A positive truth margin means the ground truth beats every competing class."
    )

    reason_rows = contrastive_reason_rows(workspace, baseline_outputs, target)
    render_contrastive_reasons(workspace, target, reason_rows)
    render_repair_panel(
        workspace,
        instance,
        baseline_outputs,
        target,
        reason_rows,
    )


if __name__ == "__main__":
    main()
