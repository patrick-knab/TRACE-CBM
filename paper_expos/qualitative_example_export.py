"""Manual, Figma-friendly qualitative exports for forecasting examples.

The public workflow is intentionally small:

1. load a checkpoint with :func:`load_workspace` from ``utils.graph_concept_ui``;
2. use :func:`candidate_table` to find interesting held-out moments;
3. call :func:`prepare_case` and :func:`preview_case` in a notebook; and
4. call :func:`export_case` once the example is worth keeping.

Exports keep the source frames as raster images, but use SVG/PDF for the learned
graph and activity distributions so those elements remain editable in Figma.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils.graph_concept_ui import (
    forward_outputs,
    graph_branch_options,
    graph_matrix,
    graph_temporal_vector,
)
from utils.paths import dataset_root as configured_dataset_root
from utils.intervention_notebook import (
    InterventionWorkspace,
    _forecast_logits_by_horizon,
    prediction_summary,
    select_instance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "paper_expos" / "qualitative_manual"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DATASET_ROOT = configured_dataset_root()
DATASET_MEDIA_ROOTS = {
    "barista": (DATASET_ROOT / "Barista/Image_data",),
    "breakfast": (DATASET_ROOT / "Breakfast/Video_data",),
    "gtea_gaze": (
        DATASET_ROOT / "GTEA_Gaze/Image_data",
        DATASET_ROOT / "GTEA_Gaze/cropped_clips_mp4",
    ),
    "mpii_cooking_2": (DATASET_ROOT / "MPII_Cooking_2/Image_data",),
}

# Exact accents used in runs/paper_expos/GDCD.pdf.
COLORS = {
    "ink": "#111111",
    "muted": "#666666",
    "line": "#B3B3B3",
    "observed": "#3F67D6",
    "observed_light": "#C8D8FF",
    "forecast": "#8E76D0",
    "forecast_light": "#C5B4ED",
    "positive": "#5F9838",
    "positive_light": "#C7ECB0",
    "negative": "#D29D11",
    "negative_light": "#FFE7A7",
}
AUTO_CONCEPT_MIN_ACTIVATION = 0.50
AUTO_CONCEPT_MIN_CHANGE_ACTIVATION = 0.30
AUTO_CONCEPT_MIN_RANGE = 0.15


@dataclass
class QualitativeCase:
    workspace: InterventionWorkspace
    instance: dict[str, object]
    outputs: Mapping[str, object]
    distributions: dict[int, np.ndarray]
    frame_rows: list[dict[str, object]]
    concept_indices: list[int]
    contributor_rows: list[dict[str, object]]
    graph_edges: list[dict[str, object]]
    target_horizon: int
    concept_selection_strategy: str


def canonical_dataset(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "gtea": "gtea_gaze",
        "mpii": "mpii_cooking_2",
        "mpii_cooking": "mpii_cooking_2",
    }
    return aliases.get(key, key)


def candidate_table(
    workspace: InterventionWorkspace,
    *,
    split: str = "test",
    horizon: int | None = None,
    video_indices: Sequence[int] | None = None,
    stride: int = 1,
    max_rows: int | None = 250,
    require_three_observed_frames: bool = True,
) -> pd.DataFrame:
    """Score held-out moments and return a compact table for manual selection.

    This is deliberately a browsing aid, not an automatic exemplar-selection
    policy. Restrict ``video_indices`` while exploring, then expand the scan if
    needed. Rows are returned in dataset order and can be filtered/sorted in the
    notebook without changing the frozen predictions.
    """

    split_data = workspace.standardized_splits[split]
    horizon = int(horizon or max(workspace.forecast_horizons))
    if horizon not in {int(value) for value in workspace.forecast_horizons}:
        raise ValueError(f"Checkpoint does not expose horizon {horizon}.")
    count = len(split_data["lengths"])
    selected_videos = list(range(count)) if video_indices is None else [int(value) for value in video_indices]
    rows: list[dict[str, object]] = []
    for video_index in selected_videos:
        length = int(split_data["lengths"][video_index])
        first_t = 2 if require_three_observed_frames else 0
        for timestep in range(first_t, max(first_t, length - horizon), max(1, int(stride))):
            instance = select_instance(workspace, split, video_index, timestep)
            outputs = forward_outputs(workspace, instance)
            summary = prediction_summary(workspace, instance, outputs)
            forecast = summary["forecasts"][horizon]
            probabilities = np.asarray(forecast["probs"], dtype=float)
            true_idx = forecast["true_idx"]
            if true_idx is None:
                continue
            rows.append(
                {
                    "video_index": video_index,
                    "video_id": instance["video_id"],
                    "timestep": timestep,
                    "current_ground_truth": summary["activity_true_label"],
                    "current_prediction": summary["activity_pred_label"],
                    f"h{horizon}_ground_truth": forecast["true_label"],
                    f"h{horizon}_prediction": forecast["pred_label"],
                    f"h{horizon}_correct": bool(int(forecast["pred_idx"]) == int(true_idx)),
                    f"h{horizon}_confidence": float(probabilities.max()),
                    f"h{horizon}_ground_truth_probability": float(probabilities[int(true_idx)]),
                }
            )
            if max_rows is not None and len(rows) >= int(max_rows):
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def prepare_case(
    workspace: InterventionWorkspace,
    *,
    split: str,
    video_index: int,
    timestep: int,
    target_horizon: int | None = None,
    concepts: Sequence[int | str] | None = None,
    max_concepts: int = 6,
) -> QualitativeCase:
    """Run one frozen example and collect frames, predictions, and learned graph."""

    target_horizon = int(target_horizon or max(workspace.forecast_horizons))
    instance = select_instance(workspace, split, int(video_index), int(timestep))
    outputs = forward_outputs(workspace, instance)
    distributions = _activity_distributions(workspace, outputs)
    if target_horizon not in distributions:
        raise ValueError(f"No activity distribution for horizon {target_horizon}.")
    media_path = resolve_media_path(workspace, instance)
    frame_rows = _observed_frame_rows(workspace, instance, media_path, offsets=(-2, -1, 0))

    contributor_rows = _prediction_margin_contributors(
        workspace,
        outputs,
        distributions[target_horizon],
        target_horizon,
    )
    concept_indices = _resolve_concepts(
        workspace,
        concepts,
        contributor_rows,
        outputs,
        target_horizon,
        max_concepts=max_concepts,
    )
    graph_edges = _selected_graph_edges(workspace, concept_indices, max_edges=16)
    return QualitativeCase(
        workspace=workspace,
        instance=instance,
        outputs=outputs,
        distributions=distributions,
        frame_rows=frame_rows,
        concept_indices=concept_indices,
        contributor_rows=contributor_rows,
        graph_edges=graph_edges,
        target_horizon=target_horizon,
        concept_selection_strategy=(
            "manual_exact" if concepts else "prediction_margin_vs_runner_up_with_scene_relevant_connectors"
        ),
    )


def preview_case(case: QualitativeCase) -> tuple[plt.Figure, str, plt.Figure]:
    """Return the observed-frame figure, graph SVG text, and activity figure."""

    return plot_observed_frames(case), render_graph_svg(case), plot_activity_distributions(case)


def export_case(
    case: QualitativeCase,
    *,
    case_name: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
) -> Path:
    """Export independent Figma-ready assets and exact numeric source tables."""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_name).strip()).strip("._")
    if not safe_name:
        raise ValueError("case_name must contain at least one letter or number.")
    output_dir = Path(output_root) / safe_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)

    for row in case.frame_rows:
        Image.fromarray(np.asarray(row["rgb"], dtype=np.uint8)).save(frames_dir / f"{row['file_stem']}.png")

    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        frames_figure = plot_observed_frames(case)
        _save_figure_bundle(frames_figure, output_dir / "observed_frames", dpi=240)
        plt.close(frames_figure)
        activity_figure = plot_activity_distributions(case)
        _save_figure_bundle(activity_figure, output_dir / "activity_distributions", dpi=240)
        plt.close(activity_figure)

    dot_source = graph_dot(case)
    (output_dir / "learned_concept_graph.dot").write_text(dot_source + "\n", encoding="utf-8")
    for suffix in ("svg", "pdf", "png"):
        command = ["dot", f"-T{suffix}"]
        if suffix == "png":
            command.append("-Gdpi=240")
        result = subprocess.run(
            command,
            input=dot_source.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"Graphviz failed to render {suffix}.")
        destination = output_dir / f"learned_concept_graph.{suffix}"
        destination.write_bytes(result.stdout)

    _write_activity_csv(case, output_dir / "activity_distributions.csv")
    _write_concept_csv(case, output_dir / "concept_nodes.csv")
    _write_rows_csv(case.graph_edges, output_dir / "learned_graph_edges.csv")
    _write_manifest(case, output_dir / "case_manifest.json")
    (output_dir / "FIGMA_LAYOUT.txt").write_text(_figma_layout_text(case), encoding="utf-8")
    return output_dir


def plot_observed_frames(case: QualitativeCase) -> plt.Figure:
    figure, axes = plt.subplots(1, len(case.frame_rows), figsize=(9.0, 3.0), squeeze=False)
    for axis, row in zip(axes[0], case.frame_rows):
        axis.imshow(row["rgb"])
        axis.set_title(str(row["relative_time"]), fontsize=12, fontweight="bold", color=COLORS["ink"])
        axis.set_xlabel(f"GT: {row['ground_truth']}", fontsize=9, color=COLORS["ink"])
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(COLORS["observed"])
            spine.set_linewidth(2.2 if int(row["offset"]) == 0 else 1.2)
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.8)
    return figure


def plot_activity_distributions(case: QualitativeCase, top_k: int = 3) -> plt.Figure:
    steps = [step for step in range(0, case.target_horizon + 1) if step in case.distributions]
    figure, axes = plt.subplots(1, len(steps), figsize=(3.1 * len(steps), 3.2), squeeze=False)
    for axis, step in zip(axes[0], steps):
        probabilities = case.distributions[step]
        true_idx = _ground_truth_index(case, step)
        shown = list(np.argsort(-probabilities)[: int(top_k)])
        if true_idx is not None and int(true_idx) not in shown:
            shown.append(int(true_idx))
        shown = sorted(set(shown), key=lambda idx: float(probabilities[idx]))
        names = [_activity_name(case.workspace, index) for index in shown]
        values = [float(probabilities[index]) for index in shown]
        predicted_idx = int(np.argmax(probabilities))
        base_color = COLORS["observed"] if step == 0 else COLORS["forecast"]
        bars = axis.barh(range(len(shown)), values, color=base_color, alpha=0.88)
        for bar, class_idx, probability in zip(bars, shown, values):
            if true_idx is not None and int(class_idx) == int(true_idx):
                bar.set_edgecolor(COLORS["positive"])
                bar.set_linewidth(2.5)
            tag_parts = []
            if int(class_idx) == predicted_idx:
                tag_parts.append("Pred")
            if true_idx is not None and int(class_idx) == int(true_idx):
                tag_parts.append("GT")
            tag = f"  {' = '.join(tag_parts)}" if tag_parts else ""
            axis.text(min(probability + 0.015, 0.98), bar.get_y() + bar.get_height() / 2, f"{probability:.2f}{tag}", va="center", fontsize=8)
        axis.set_yticks(range(len(shown)), labels=names, fontsize=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_title("t" if step == 0 else f"t+{step}", fontsize=12, fontweight="bold", color=base_color)
        axis.set_xlabel("probability", fontsize=8)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.8)
    return figure


def render_graph_svg(case: QualitativeCase) -> str:
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=graph_dot(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Graphviz failed to render the case graph.")
    return result.stdout


def graph_dot(case: QualitativeCase) -> str:
    observed_offsets = [int(row["offset"]) for row in case.frame_rows]
    times = observed_offsets + list(range(1, case.target_horizon + 1))
    lines = [
        "digraph QualitativeForecastGraph {",
        '  graph [rankdir=LR, bgcolor="white", pad="0.12", nodesep="0.14", ranksep="0.62", splines=polyline, newrank=true];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.07,0.05", width=0.62, height=0.40];',
        '  edge [arrowsize=0.48, color="#B3B3B3"];',
    ]
    for time in times:
        title = _relative_time_label(time)
        color = COLORS["observed"] if time <= 0 else COLORS["forecast"]
        lines.append(
            f'  h_{_time_id(time)} [shape=plaintext, style="", label=<<B><FONT COLOR="{color}">{title}</FONT></B>>];'
        )
    for concept_idx in case.concept_indices:
        name = _dot_escape(_short_label(case.workspace.concept_names[concept_idx], 24))
        lines.append(
            f'  label_{concept_idx} [shape=plaintext, style="", label="{name}", '
            f'fontname="Helvetica", fontsize=10, fontcolor="{COLORS["ink"]}"];'
        )
        for time in times:
            value = _concept_value(case, concept_idx, time)
            base = COLORS["observed"] if time <= 0 else COLORS["forecast"]
            light = COLORS["observed_light"] if time <= 0 else COLORS["forecast_light"]
            fill = _mix_hex("#FFFFFF", base, min(1.0, max(0.0, value)))
            font = "#FFFFFF" if value >= 0.58 else COLORS["ink"]
            border = base if value >= 0.5 else light
            lines.append(
                f'  n_{concept_idx}_{_time_id(time)} [label="{value:.2f}", fillcolor="{fill}", color="{border}", fontcolor="{font}", penwidth=1.4];'
            )
        first = times[0]
        lines.append(f'  label_{concept_idx} -> n_{concept_idx}_{_time_id(first)} [style=invis, weight=30];')
    for time in times:
        rank_nodes = " ".join([f"h_{_time_id(time)}"] + [f"n_{idx}_{_time_id(time)}" for idx in case.concept_indices])
        lines.append(f"  {{ rank=same; {rank_nodes}; }}")
    for left, right in zip(times, times[1:]):
        lines.append(f'  h_{_time_id(left)} -> h_{_time_id(right)} [style=invis, weight=40];')
        for concept_idx in case.concept_indices:
            temporal_weight = _temporal_weight(case.graph_edges, concept_idx, left, right)
            width = 1.0 + min(2.3, 12.0 * abs(temporal_weight))
            lines.append(
                f'  n_{concept_idx}_{_time_id(left)} -> n_{concept_idx}_{_time_id(right)} '
                f'[color="#A3A3A3", penwidth="{width:.2f}", weight=12];'
            )
    spatial = [row for row in case.graph_edges if row["kind"] == "spatial"][:6]
    cross = [row for row in case.graph_edges if row["kind"] == "cross_temporal"][:8]
    for row in spatial:
        display_times = [0] + ([1] if case.target_horizon >= 1 else [])
        for time in display_times:
            if (time <= 0 and row["branch"] != "shared") or (time > 0 and row["branch"] == "shared"):
                continue
            lines.append(_dot_edge(row, time, time, constraint=False, style="dotted"))
    for row in cross:
        for left, right in zip(times, times[1:]):
            if left <= 0 and right <= 0 and row["branch"] != "shared":
                continue
            if right > 0 and row["branch"] == "shared":
                continue
            lines.append(_dot_edge(row, left, right, constraint=False, style="dashed"))
    lines.append("}")
    return "\n".join(lines)


def resolve_media_path(workspace: InterventionWorkspace, instance: Mapping[str, object]) -> Path:
    raw = Path(str(instance["video_path"])).expanduser()
    if raw.exists():
        return raw.resolve()
    args_path = _workspace_checkpoint_path(workspace).parent / "args.json"
    args = json.loads(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    candidates: list[Path] = [PROJECT_ROOT / raw]
    for key in ("data_root", "embedding_path"):
        if args.get(key):
            anchor = Path(str(args[key])).expanduser()
            anchor = anchor if key == "data_root" else anchor.parent
            candidates.extend((anchor / raw, anchor.parent / raw))
    dataset = canonical_dataset(args.get("dataset"))
    roots = DATASET_MEDIA_ROOTS.get(dataset, ())
    for marker in ("Image_data", "Video_data"):
        if marker in raw.parts:
            suffix = Path(*raw.parts[raw.parts.index(marker) + 1 :])
            candidates.extend(root / suffix for root in roots if root.name == marker)
    for root in roots:
        candidates.extend((root / str(instance["video_id"]), root / raw.name))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve media path: {raw}")


def _activity_distributions(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    effective = outputs.get("effective_activity_probs_by_step")
    if isinstance(effective, Mapping):
        for raw_step, tensor in effective.items():
            if torch.is_tensor(tensor):
                result[int(raw_step)] = _last_vector(tensor)
    if 0 not in result and torch.is_tensor(outputs.get("activity_logits")):
        result[0] = _softmax_last(outputs["activity_logits"])
    for horizon, logits in _forecast_logits_by_horizon(workspace, outputs).items():
        if int(horizon) not in result:
            result[int(horizon)] = _softmax_last(logits)
    return result


def _resolve_concepts(
    workspace: InterventionWorkspace,
    concepts: Sequence[int | str] | None,
    contributor_rows: list[dict[str, object]],
    outputs: Mapping[str, object],
    target_horizon: int,
    *,
    max_concepts: int,
) -> list[int]:
    if concepts:
        resolved = [_concept_index(workspace, value) for value in concepts]
        return list(dict.fromkeys(resolved))[: int(max_concepts)]

    seed_count = max(2, int(max_concepts) - 2)
    relevant_rows = [row for row in contributor_rows if bool(row.get("scene_relevant"))]
    supporting = sorted(
        (row for row in relevant_rows if float(row["contribution"]) > 0.0),
        key=lambda row: float(row["contribution"]),
        reverse=True,
    )
    opposing = sorted(
        (row for row in relevant_rows if float(row["contribution"]) <= 0.0),
        key=lambda row: abs(float(row["contribution"])),
        reverse=True,
    )
    ranked_rows = supporting + opposing
    selected = list(dict.fromkeys(int(row["concept_idx"]) for row in ranked_rows))[:seed_count]
    contributor_by_idx = {int(row["concept_idx"]): row for row in relevant_rows}
    strongest_support = float(supporting[0]["contribution"]) if supporting else 0.0
    connector_margin_floor = max(1e-6, 0.15 * strongest_support)

    edge_pool = _all_graph_edges(workspace)
    for row in edge_pool:
        source, target = int(row["source_idx"]), int(row["target_idx"])
        candidate = None
        if source in selected and target not in selected and source != target:
            candidate = target
        elif target in selected and source not in selected and source != target:
            candidate = source
        candidate_row = contributor_by_idx.get(int(candidate)) if candidate is not None else None
        if (
            candidate is not None
            and candidate_row is not None
            and float(candidate_row["contribution"]) >= connector_margin_floor
            and _concept_is_scene_relevant(
                outputs,
                int(candidate),
                target_horizon,
                workspace.history_length,
            )
        ):
            selected.append(int(candidate))
        if len(selected) >= int(max_concepts):
            break

    for row in ranked_rows:
        concept_idx = int(row["concept_idx"])
        if concept_idx not in selected:
            selected.append(concept_idx)
        if len(selected) >= int(max_concepts):
            break
    if not selected:
        states = outputs.get("predicted_concepts_by_step", {}).get(int(target_horizon))
        if torch.is_tensor(states):
            selected = np.argsort(-_last_vector(states))[: int(max_concepts)].tolist()
    return selected[: int(max_concepts)]


def _prediction_margin_contributors(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    probabilities: np.ndarray,
    target_horizon: int,
) -> list[dict[str, object]]:
    """Rank concepts by predicted-vs-runner-up logit-margin contribution."""

    predicted_states = outputs.get("predicted_concepts_by_step", {}).get(int(target_horizon))
    head = getattr(workspace.model, "activity_head", None)
    if not torch.is_tensor(predicted_states) or head is None or not hasattr(head, "weight"):
        return []
    weights = head.weight.detach()
    if weights.ndim != 2 or weights.shape[1] != len(workspace.concept_names):
        return []

    class_order = np.argsort(-np.asarray(probabilities, dtype=float))
    if len(class_order) < 2:
        return []
    predicted_idx, runner_up_idx = int(class_order[0]), int(class_order[1])
    classifier_input = _prediction_input_tensor(workspace, outputs, predicted_states)
    values = _last_vector(classifier_input)
    weight_difference = (weights[predicted_idx] - weights[runner_up_idx]).detach().cpu().numpy()
    contributions = values * weight_difference

    rows = []
    for concept_idx in np.argsort(-np.abs(contributions)):
        scene_max, scene_range = _concept_scene_stats(
            outputs,
            int(concept_idx),
            target_horizon,
            workspace.history_length,
        )
        contribution = float(contributions[concept_idx])
        rows.append(
            {
                "rank": len(rows) + 1,
                "concept_idx": int(concept_idx),
                "concept": workspace.concept_names[int(concept_idx)],
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": "supports prediction" if contribution > 0.0 else "supports runner-up",
                "predicted_class_idx": predicted_idx,
                "predicted_class": _activity_name(workspace, predicted_idx),
                "runner_up_class_idx": runner_up_idx,
                "runner_up_class": _activity_name(workspace, runner_up_idx),
                "classifier_input": float(values[concept_idx]),
                "margin_weight": float(weight_difference[concept_idx]),
                "scene_max_activation": scene_max,
                "scene_activation_range": scene_range,
                "scene_relevant": _scene_stats_are_relevant(scene_max, scene_range),
            }
        )
    return rows


def _prediction_input_tensor(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    tensor: torch.Tensor,
) -> torch.Tensor:
    transform = str(
        outputs.get("st_prediction_transform")
        or getattr(workspace.model, "st_prediction_transform", "identity")
    )
    if transform == "logit":
        epsilon = torch.finfo(tensor.dtype).eps
        bounded = tensor.clamp(min=epsilon, max=1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)
    if transform == "centered":
        return (2.0 * tensor) - 1.0
    return tensor


def _concept_scene_stats(
    outputs: Mapping[str, object],
    concept_idx: int,
    target_horizon: int,
    history_length: int,
) -> tuple[float, float]:
    values: list[float] = []
    observed = outputs.get("shared_refined_concepts", outputs.get("concept_states"))
    if torch.is_tensor(observed) and observed.ndim == 3:
        start = max(0, int(history_length) - 3)
        values.extend(
            float(value)
            for value in observed[0, start:int(history_length), int(concept_idx)].detach().cpu().tolist()
        )
    predicted = outputs.get("predicted_concepts_by_step", {})
    if isinstance(predicted, Mapping):
        for step in range(1, int(target_horizon) + 1):
            states = predicted.get(step, predicted.get(str(step)))
            if torch.is_tensor(states):
                values.append(float(_last_vector(states)[int(concept_idx)]))
    if not values:
        return 0.0, 0.0
    return max(values), max(values) - min(values)


def _concept_is_scene_relevant(
    outputs: Mapping[str, object],
    concept_idx: int,
    target_horizon: int,
    history_length: int,
) -> bool:
    scene_max, scene_range = _concept_scene_stats(
        outputs,
        concept_idx,
        target_horizon,
        history_length,
    )
    return _scene_stats_are_relevant(scene_max, scene_range)


def _scene_stats_are_relevant(scene_max: float, scene_range: float) -> bool:
    return bool(
        scene_max >= AUTO_CONCEPT_MIN_ACTIVATION
        or (
            scene_max >= AUTO_CONCEPT_MIN_CHANGE_ACTIVATION
            and scene_range >= AUTO_CONCEPT_MIN_RANGE
        )
    )


def _all_graph_edges(workspace: InterventionWorkspace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch in graph_branch_options(workspace):
        if branch not in {"shared", "forecast"}:
            continue
        for kind, matrix_kind in (("spatial", "spatial"), ("cross_temporal", "cross temporal")):
            matrix = graph_matrix(workspace, branch, "mean", edge_kind=matrix_kind)
            for source, target in np.argwhere(np.abs(matrix) > 0.0):
                rows.append(
                    {
                        "branch": branch,
                        "kind": kind,
                        "source_idx": int(source),
                        "source": workspace.concept_names[int(source)],
                        "target_idx": int(target),
                        "target": workspace.concept_names[int(target)],
                        "weight": float(matrix[source, target]),
                    }
                )
        temporal = graph_temporal_vector(workspace, branch, "mean")
        for concept_idx in np.flatnonzero(np.abs(temporal) > 0.0):
            rows.append(
                {
                    "branch": branch,
                    "kind": "temporal",
                    "source_idx": int(concept_idx),
                    "source": workspace.concept_names[int(concept_idx)],
                    "target_idx": int(concept_idx),
                    "target": workspace.concept_names[int(concept_idx)],
                    "weight": float(temporal[concept_idx]),
                }
            )
    rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return rows


def _selected_graph_edges(
    workspace: InterventionWorkspace,
    concept_indices: Sequence[int],
    *,
    max_edges: int,
) -> list[dict[str, object]]:
    selected = {int(value) for value in concept_indices}
    rows = [
        row for row in _all_graph_edges(workspace)
        if int(row["source_idx"]) in selected and int(row["target_idx"]) in selected
    ]
    return rows[: int(max_edges)]


def _observed_frame_rows(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    media_path: Path,
    *,
    offsets: Sequence[int],
) -> list[dict[str, object]]:
    metadata = workspace.preprocessed_data.get("metadata", {})
    timing = _lookup_metadata(metadata, "video_window_spans", instance, media_path)
    video_meta = _lookup_metadata(metadata, "video_meta", instance, media_path)
    labels = np.asarray(
        workspace.preprocessed_data[str(instance["split"])]["activity_labels"][int(instance["video_index"])],
        dtype=np.int64,
    )
    rows = []
    for offset in offsets:
        source_timestep = int(instance["timestep"]) + int(offset)
        if source_timestep < 0:
            continue
        rgb, source_frame = _read_window_midpoint(
            media_path,
            source_timestep,
            int(instance["length"]),
            window_spans=timing,
            video_meta=video_meta,
        )
        rows.append(
            {
                "offset": int(offset),
                "relative_time": _relative_time_label(int(offset)),
                "source_timestep": source_timestep,
                "source_frame": source_frame,
                "ground_truth_index": int(labels[source_timestep]),
                "ground_truth": _activity_name(workspace, int(labels[source_timestep])),
                "file_stem": { -2: "t_minus_2", -1: "t_minus_1", 0: "t" }.get(int(offset), f"t_{offset:+d}"),
                "rgb": rgb,
            }
        )
    return rows


def _read_window_midpoint(
    path: Path,
    timestep: int,
    num_windows: int,
    *,
    window_spans: object,
    video_meta: object,
) -> tuple[np.ndarray, int]:
    meta = dict(video_meta) if isinstance(video_meta, Mapping) else {}
    spans = window_spans if isinstance(window_spans, (list, tuple)) and len(window_spans) >= num_windows else None
    if path.is_dir():
        frames = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=_natural_key,
        )
        if not frames:
            raise RuntimeError(f"No image frames found under {path}")
        fps = float(meta.get("fps", 0.0) or 0.0)
        if spans and fps > 0.0:
            start = int(round(float(spans[timestep][0]) * fps))
            end = int(round(float(spans[timestep][1]) * fps)) - 1
        else:
            start = int(round(timestep * len(frames) / num_windows))
            end = int(round((timestep + 1) * len(frames) / num_windows)) - 1
        frame_index = min(len(frames) - 1, max(0, int(round((start + max(start, end)) / 2))))
        bgr = cv2.imread(str(frames[frame_index]))
    elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Could not read video frames from {path}")
        if spans and fps > 0.0:
            midpoint_seconds = 0.5 * (float(spans[timestep][0]) + float(spans[timestep][1]))
            frame_index = int(round(midpoint_seconds * fps))
        else:
            frame_index = int(round((timestep + 0.5) * frame_count / num_windows))
        frame_index = min(frame_count - 1, max(0, frame_index))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            bgr = None
    else:
        raise FileNotFoundError(f"Unsupported media path: {path}")
    if bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(frame_index)


def _lookup_metadata(
    metadata: object,
    key: str,
    instance: Mapping[str, object],
    media_path: Path,
) -> object | None:
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get(key), Mapping):
        return None
    values = metadata[key]
    candidates = {
        str(instance["video_path"]),
        str(instance["video_id"]),
        str(media_path),
        Path(str(instance["video_path"])).name,
        media_path.name,
    }
    for stored_key, value in values.items():
        if str(stored_key) in candidates or Path(str(stored_key)).name in candidates:
            return value
    return None


def _concept_value(case: QualitativeCase, concept_idx: int, relative_time: int) -> float:
    if relative_time > 0:
        states = case.outputs.get("predicted_concepts_by_step", {}).get(int(relative_time))
        return float(_last_vector(states)[int(concept_idx)]) if torch.is_tensor(states) else 0.0
    tensor = case.outputs.get("shared_refined_concepts", case.outputs.get("concept_states"))
    history_idx = case.workspace.history_length - 1 + int(relative_time)
    if torch.is_tensor(tensor) and 0 <= history_idx < tensor.shape[1]:
        return float(tensor[0, history_idx, int(concept_idx)].detach().cpu().item())
    return 0.0


def _ground_truth_index(case: QualitativeCase, step: int) -> int | None:
    if step == 0:
        return int(case.instance["current_label"])
    value = case.instance["future_labels"].get(int(step))
    return None if value is None else int(value)


def _write_activity_csv(case: QualitativeCase, path: Path) -> None:
    rows = []
    for step, probabilities in sorted(case.distributions.items()):
        if step > case.target_horizon:
            continue
        predicted = int(np.argmax(probabilities))
        ground_truth = _ground_truth_index(case, step)
        for class_idx, probability in enumerate(probabilities):
            rows.append(
                {
                    "relative_time": _relative_time_label(step),
                    "step": step,
                    "class_idx": class_idx,
                    "activity": _activity_name(case.workspace, class_idx),
                    "probability": float(probability),
                    "is_prediction": class_idx == predicted,
                    "is_ground_truth": ground_truth is not None and class_idx == ground_truth,
                }
            )
    _write_rows_csv(rows, path)


def _write_concept_csv(case: QualitativeCase, path: Path) -> None:
    rows = []
    for relative_time in [int(row["offset"]) for row in case.frame_rows] + list(range(1, case.target_horizon + 1)):
        for concept_idx in case.concept_indices:
            rows.append(
                {
                    "relative_time": _relative_time_label(relative_time),
                    "step": relative_time,
                    "concept_idx": concept_idx,
                    "concept": case.workspace.concept_names[concept_idx],
                    "activation": _concept_value(case, concept_idx, relative_time),
                }
            )
    _write_rows_csv(rows, path)


def _write_rows_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(str(key) for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(case: QualitativeCase, path: Path) -> None:
    payload = {
        "checkpoint": str(_workspace_checkpoint_path(case.workspace)),
        "split": case.instance["split"],
        "video_index": case.instance["video_index"],
        "video_id": case.instance["video_id"],
        "video_path": case.instance["video_path"],
        "timestep": case.instance["timestep"],
        "target_horizon": case.target_horizon,
        "concept_selection_strategy": case.concept_selection_strategy,
        "concept_indices": case.concept_indices,
        "concept_names": [case.workspace.concept_names[index] for index in case.concept_indices],
        "top_margin_contributors": [
            {
                **row,
                "selected_for_graph": int(row["concept_idx"]) in set(case.concept_indices),
            }
            for row in case.contributor_rows[:20]
        ],
        "predictions": [],
        "frames": [
            {key: value for key, value in row.items() if key != "rgb"}
            for row in case.frame_rows
        ],
        "files": {
            "individual_frames": "frames/*.png",
            "observed_frame_strip": "observed_frames.svg",
            "learned_graph": "learned_concept_graph.svg",
            "activity_distributions": "activity_distributions.svg",
            "numeric_sources": ["activity_distributions.csv", "concept_nodes.csv", "learned_graph_edges.csv"],
        },
    }
    for step, probabilities in sorted(case.distributions.items()):
        if step > case.target_horizon:
            continue
        predicted = int(np.argmax(probabilities))
        ground_truth = _ground_truth_index(case, step)
        payload["predictions"].append(
            {
                "relative_time": _relative_time_label(step),
                "prediction_idx": predicted,
                "prediction": _activity_name(case.workspace, predicted),
                "prediction_probability": float(probabilities[predicted]),
                "ground_truth_idx": ground_truth,
                "ground_truth": None if ground_truth is None else _activity_name(case.workspace, ground_truth),
                "ground_truth_probability": None if ground_truth is None else float(probabilities[ground_truth]),
                "correct": ground_truth is not None and predicted == ground_truth,
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _figma_layout_text(case: QualitativeCase) -> str:
    return (
        "Recommended two-row paper layout (wide, white background)\n"
        "\n"
        "TOP: observed_frames.svg on the left; activity_distributions.svg on the right.\n"
        "BOTTOM: learned_concept_graph.svg aligned to the same t / t+n columns.\n"
        "\n"
        "Keep the three source frames as raster images. Import the SVG graph and distributions\n"
        "as vectors, then ungroup in Figma for final typography and spacing. Preserve the color\n"
        "semantics: blue=observed, purple=forecast, green=ground truth/positive relation,\n"
        "gold=negative relation. Solid gray arrows are same-concept temporal carryover; dotted\n"
        "arrows are same-time learned relations; dashed arrows are learned cross-time relations.\n"
        "\n"
        f"Selected target: t+{case.target_horizon}; selected concepts: "
        + ", ".join(case.workspace.concept_names[index] for index in case.concept_indices)
        + "\n"
    )


def _save_figure_bundle(figure: plt.Figure, stem: Path, *, dpi: int) -> None:
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")


def _softmax_last(tensor: torch.Tensor) -> np.ndarray:
    selected = tensor[0, -1, :] if tensor.ndim == 3 else tensor[0, :]
    return torch.softmax(selected, dim=-1).detach().cpu().numpy()


def _last_vector(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 3:
        selected = tensor[0, -1, :]
    elif tensor.ndim == 2:
        selected = tensor[0, :]
    else:
        selected = tensor
    return selected.detach().cpu().numpy()


def _activity_name(workspace: InterventionWorkspace, index: int) -> str:
    return str(workspace.activity_names[int(index)]) if 0 <= int(index) < len(workspace.activity_names) else str(index)


def _workspace_checkpoint_path(workspace: InterventionWorkspace) -> Path:
    checkpoint_path = workspace.config.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("The workspace does not record a source checkpoint path.")
    return Path(checkpoint_path)


def _concept_index(workspace: InterventionWorkspace, value: int | str) -> int:
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < len(workspace.concept_names):
            return index
        raise IndexError(f"Concept index out of range: {index}")
    lowered = str(value).strip().lower()
    exact = [index for index, name in enumerate(workspace.concept_names) if str(name).lower() == lowered]
    if exact:
        return exact[0]
    partial = [index for index, name in enumerate(workspace.concept_names) if lowered in str(name).lower()]
    if len(partial) == 1:
        return partial[0]
    raise ValueError(f"Concept name must have one exact or unique partial match: {value!r}")


def _relative_time_label(value: int) -> str:
    return "t" if int(value) == 0 else f"t{int(value):+d}"


def _time_id(value: int) -> str:
    return f"m{abs(int(value))}" if int(value) < 0 else f"p{int(value)}"


def _short_label(value: object, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: max(1, width - 1)].rstrip() + "…"


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _mix_hex(left: str, right: str, amount: float) -> str:
    amount = min(1.0, max(0.0, float(amount)))
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round((1.0 - amount) * a + amount * b) for a, b in zip(left_rgb, right_rgb))
    return "#" + "".join(f"{value:02X}" for value in mixed)


def _temporal_weight(edges: Sequence[Mapping[str, object]], concept_idx: int, left: int, right: int) -> float:
    wanted_branch = "shared" if right <= 0 else "forecast"
    for row in edges:
        if row["kind"] == "temporal" and row["branch"] == wanted_branch and int(row["source_idx"]) == int(concept_idx):
            return float(row["weight"])
    return 0.0


def _dot_edge(
    row: Mapping[str, object],
    left: int,
    right: int,
    *,
    constraint: bool,
    style: str,
) -> str:
    weight = float(row["weight"])
    color = COLORS["positive"] if weight >= 0.0 else COLORS["negative"]
    width = 1.0 + min(3.0, 24.0 * abs(weight))
    constraint_text = "true" if constraint else "false"
    return (
        f'  n_{int(row["source_idx"])}_{_time_id(left)} -> n_{int(row["target_idx"])}_{_time_id(right)} '
        f'[color="{color}", penwidth="{width:.2f}", style="{style}", constraint={constraint_text}];'
    )


def _dot_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
