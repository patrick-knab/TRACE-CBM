"""Utilities for the interactive graph-concept intervention UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from utils.model import canonical_method
from utils.intervention_notebook import (
    InterventionNotebookConfig,
    InterventionWorkspace,
    _forecast_logits_by_horizon,
    _forward,
    _instance_tensors,
    load_or_train_workspace,
)


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    label: str
    dataset: str
    backbone: str
    base_method: str
    run_name: str
    modified: str
    args: Dict[str, Any]
    metrics: Dict[str, Any]


def discover_checkpoints(model_root: str | Path) -> tuple[List[CheckpointRecord], List[str]]:
    root = Path(model_root).expanduser()
    warnings: List[str] = []
    if not root.exists():
        return [], [f"Model root does not exist: {root}"]

    records: List[CheckpointRecord] = []
    for checkpoint_path in sorted(root.glob("**/model.pt"), key=lambda path: path.stat().st_mtime, reverse=True):
        run_dir = checkpoint_path.parent
        args, args_warning = _read_json(run_dir / "args.json")
        metrics, metrics_warning = _read_json(run_dir / "metrics.json")
        for warning in (args_warning, metrics_warning):
            if warning:
                warnings.append(warning)

        dataset = str(args.get("dataset") or _infer_dataset(run_dir))
        backbone = str(args.get("backbone") or "unknown")
        raw_base_method = str(args.get("base_method") or "unknown")
        base_method = canonical_method(raw_base_method) if raw_base_method != "unknown" else raw_base_method
        run_name = str(args.get("run_name") or run_dir.name)
        modified_dt = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        modified = modified_dt.strftime("%Y-%m-%d %H:%M")
        label = f"{dataset} / {backbone} / {base_method} / {run_name} / {modified}"
        records.append(
            CheckpointRecord(
                path=checkpoint_path,
                label=label,
                dataset=dataset,
                backbone=backbone,
                base_method=base_method,
                run_name=run_name,
                modified=modified,
                args=args,
                metrics=metrics,
            )
        )
    return records, warnings


def load_workspace(
    checkpoint_path: str | Path,
    device: str = "cpu",
    dataset_root: str | Path | None = None,
) -> InterventionWorkspace:
    config = InterventionNotebookConfig(
        checkpoint_path=Path(checkpoint_path),
        checkpoint_selection="latest",
        device=device,
        generate_embeddings=False,
    )
    if dataset_root is not None:
        config.dataset_root = Path(dataset_root)
    return load_or_train_workspace(config)


def split_names(workspace: InterventionWorkspace) -> List[str]:
    return [name for name in ("train", "val", "test") if name in workspace.preprocessed_data]


def video_options(workspace: InterventionWorkspace, split: str) -> List[tuple[str, int]]:
    split_data = workspace.preprocessed_data[split]
    video_ids = list(split_data.get("video_ids", []))
    lengths = list(workspace.standardized_splits[split].get("lengths", []))
    options = []
    for index, video_id in enumerate(video_ids):
        length_text = f"{int(lengths[index])} steps" if index < len(lengths) else "unknown length"
        options.append((f"{index}: {video_id} ({length_text})", index))
    return options


def timestep_bounds(workspace: InterventionWorkspace, split: str, video_index: int) -> tuple[int, int, int]:
    lengths = workspace.standardized_splits[split]["lengths"]
    length = int(lengths[int(video_index)])
    default = max(0, length - max(workspace.forecast_horizons) - 1)
    return 0, max(0, length - 1), default


def forward_outputs(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    intervention: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    with torch.no_grad():
        if workspace.base_method == "linear":
            return _forward_linear(workspace, concepts, key_padding_mask, intervention)
        return _forward(workspace, concepts, key_padding_mask, intervention=intervention)


def forward_outputs_batched_interventions(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    interventions: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Run one independent intervention per batch row."""

    if not interventions:
        raise ValueError("At least one intervention is required.")
    if workspace.base_method in {"linear", "motif"} or isinstance(workspace.model, Mapping):
        raise ValueError("Batched node interventions require a sequence graph model.")

    modes = {
        str(intervention.get("mode", intervention.get("intervention_mode", "input"))).lower()
        for intervention in interventions
    }
    if len(modes) != 1:
        raise ValueError("All batched interventions must use the same intervention mode.")

    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    batch_size = len(interventions)
    concepts = concepts.expand(batch_size, -1, -1).clone()
    key_padding_mask = key_padding_mask.expand(batch_size, -1).clone()
    items: List[Dict[str, object]] = []
    for batch_idx, intervention in enumerate(interventions):
        raw_items = intervention.get("items")
        if raw_items is None:
            raw_items = [intervention]
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            item = dict(raw_item)
            item["batch_idx"] = int(batch_idx)
            items.append(item)
    if not items:
        raise ValueError("Batched interventions contain no valid items.")

    with torch.no_grad():
        return _forward(
            workspace,
            concepts,
            key_padding_mask,
            intervention={"mode": modes.pop(), "items": items},
        )


def prediction_tables(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    top_k: int = 3,
) -> tuple[List[Dict[str, object]], Dict[int, List[Dict[str, object]]]]:
    current: List[Dict[str, object]] = []
    current_probabilities = _effective_activity_probabilities(outputs, 0)
    if torch.is_tensor(current_probabilities):
        current = top_probability_rows(
            current_probabilities,
            workspace.activity_names,
            true_idx=instance.get("current_label"),
            top_k=top_k,
        )
    elif torch.is_tensor(outputs.get("activity_logits")):
        current = top_class_rows(
            outputs["activity_logits"],
            workspace.activity_names,
            true_idx=instance.get("current_label"),
            top_k=top_k,
        )

    forecasts: Dict[int, List[Dict[str, object]]] = {}
    for horizon, logits in _forecast_logits_by_horizon(workspace, outputs).items():
        probabilities = _effective_activity_probabilities(outputs, int(horizon))
        if torch.is_tensor(probabilities):
            forecasts[int(horizon)] = top_probability_rows(
                probabilities,
                workspace.activity_names,
                true_idx=instance.get("future_labels", {}).get(int(horizon)),
                top_k=top_k,
            )
        else:
            forecasts[int(horizon)] = top_class_rows(
                logits,
                workspace.activity_names,
                true_idx=instance.get("future_labels", {}).get(int(horizon)),
                top_k=top_k,
            )
    return current, forecasts


def top_class_rows(
    logits: torch.Tensor,
    names: Sequence[str],
    true_idx: int | None = None,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    if logits.ndim == 3:
        selected = logits[0, -1, :]
    elif logits.ndim == 2:
        selected = logits[0, :]
    else:
        raise ValueError(f"Expected 2D or 3D logits, got shape {tuple(logits.shape)}")
    probs = torch.softmax(selected, dim=-1).detach().cpu().numpy()
    top = np.argsort(-probs)[: int(top_k)]
    true_label = _name_at(names, true_idx)
    return [
        {
            "rank": rank,
            "class_idx": int(index),
            "class": _name_at(names, int(index)),
            "probability": float(probs[index]),
            "true": true_label,
            "is_true": bool(true_idx is not None and int(index) == int(true_idx)),
        }
        for rank, index in enumerate(top, 1)
    ]


def top_probability_rows(
    probabilities: torch.Tensor,
    names: Sequence[str],
    true_idx: int | None = None,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    if probabilities.ndim == 3:
        selected = probabilities[0, -1, :]
    elif probabilities.ndim == 2:
        selected = probabilities[0, :]
    else:
        raise ValueError(
            f"Expected 2D or 3D probabilities, got shape {tuple(probabilities.shape)}"
        )
    probs = selected.detach().cpu().numpy()
    top = np.argsort(-probs)[: int(top_k)]
    true_label = _name_at(names, true_idx)
    return [
        {
            "rank": rank,
            "class_idx": int(index),
            "class": _name_at(names, int(index)),
            "probability": float(probs[index]),
            "true": true_label,
            "is_true": bool(true_idx is not None and int(index) == int(true_idx)),
        }
        for rank, index in enumerate(top, 1)
    ]


def probability_delta_rows(
    workspace: InterventionWorkspace,
    before: Mapping[str, object],
    after: Mapping[str, object],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if torch.is_tensor(before.get("activity_logits")) and torch.is_tensor(after.get("activity_logits")):
        rows.extend(_delta_rows_for_logits("activity", before["activity_logits"], after["activity_logits"], workspace.activity_names, top_k))
    before_forecasts = _forecast_logits_by_horizon(workspace, before)
    after_forecasts = _forecast_logits_by_horizon(workspace, after)
    for horizon in sorted(set(before_forecasts) & set(after_forecasts)):
        rows.extend(
            _delta_rows_for_logits(
                f"forecast +{horizon}",
                before_forecasts[horizon],
                after_forecasts[horizon],
                workspace.activity_names,
                top_k,
            )
        )
    return rows


def concept_driver_rows(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    target: str,
    class_idx: int,
    horizon: int | None = None,
    intervention: Mapping[str, object] | None = None,
    top_k: int = 10,
) -> List[Dict[str, object]]:
    """Rank history concept nodes by signed contribution to one selected class."""

    if workspace.base_method == "linear":
        return _linear_driver_rows(workspace, instance, target, class_idx, intervention, top_k)

    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    concepts = concepts.detach().clone().requires_grad_(True)
    outputs = _forward(workspace, concepts, key_padding_mask, intervention=intervention)
    if target == "activity":
        logits = outputs.get("activity_logits")
        if not torch.is_tensor(logits):
            return []
        score = logits[0, -1, int(class_idx)]
    else:
        selected_horizon = int(horizon or workspace.horizon)
        logits_by_horizon = _forecast_logits_by_horizon(workspace, outputs)
        if selected_horizon not in logits_by_horizon:
            return []
        score = logits_by_horizon[selected_horizon][0, -1, int(class_idx)]

    workspace.model.zero_grad(set_to_none=True)
    score.backward()
    if concepts.grad is None:
        return []
    attribution = (concepts.grad.detach()[0] * concepts.detach()[0]).cpu().numpy()
    values = concepts.detach()[0].cpu().numpy()
    return _rank_history_concepts(workspace, instance, attribution, values, top_k=top_k)


def prediction_concept_contributors(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    target: str,
    class_idx: int,
    horizon: int | None = None,
    top_k: int = 5,
    history_time_idx: int | None = None,
) -> List[Dict[str, object]]:
    """Return the most important concept-time nodes for one selected prediction."""

    direct_rows = _direct_classifier_contributors(
        workspace,
        instance,
        outputs,
        target=target,
        class_idx=class_idx,
        horizon=horizon,
        top_k=top_k,
        history_time_idx=history_time_idx,
    )
    if direct_rows:
        return direct_rows

    if history_time_idx is not None:
        return []

    fallback_rows = concept_driver_rows(
        workspace,
        instance,
        target=target,
        class_idx=class_idx,
        horizon=horizon,
        top_k=top_k,
    )
    return _augment_contributor_rows(workspace, instance, outputs, fallback_rows, target, horizon)


def learned_threshold_intervention_value(
    workspace: InterventionWorkspace,
    concept_idx: int,
    target_z: float,
) -> float | None:
    """Solve the learned-threshold calibrator equation for the raw standardized input."""

    calibrator = getattr(workspace.model, "calibrator", None)
    if getattr(calibrator, "activation", None) != "learned_threshold":
        return None
    target_z = min(1.0 - 1e-4, max(1e-4, float(target_z)))
    threshold = calibrator.threshold.detach().cpu().numpy()
    sharpness = torch.nn.functional.softplus(calibrator.log_sharpness).detach().cpu().numpy() + 1e-4
    idx = int(concept_idx)
    logit = np.log(target_z / (1.0 - target_z))
    return float(threshold[idx] + (logit / max(float(sharpness[idx]), 1e-6)))


def has_learned_threshold_calibrator(workspace: InterventionWorkspace) -> bool:
    return getattr(getattr(workspace.model, "calibrator", None), "activation", None) == "learned_threshold"


def selected_class_probability(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    target: str,
    class_idx: int,
    horizon: int | None = None,
) -> float | None:
    step = 0 if target == "activity" else int(horizon or workspace.horizon)
    effective = _effective_activity_probabilities(outputs, step)
    if torch.is_tensor(effective):
        probs = _selected_probabilities(effective)
        idx = int(class_idx)
        if idx < 0 or idx >= probs.shape[0]:
            return None
        return float(probs[idx])
    if target == "activity":
        logits = outputs.get("activity_logits")
    else:
        logits = _forecast_logits_by_horizon(workspace, outputs).get(int(horizon or workspace.horizon))
    if not torch.is_tensor(logits):
        return None
    probs = _probabilities(logits)
    idx = int(class_idx)
    if idx < 0 or idx >= probs.shape[0]:
        return None
    return float(probs[idx])


def prediction_option_rows(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    target: str,
    horizon: int | None = None,
    top_k: int = 4,
) -> List[Dict[str, object]]:
    step = 0 if target == "activity" else int(horizon or workspace.horizon)
    effective = _effective_activity_probabilities(outputs, step)
    if torch.is_tensor(effective):
        return top_probability_rows(effective, workspace.activity_names, top_k=top_k)
    if target == "activity":
        logits = outputs.get("activity_logits")
    else:
        logits = _forecast_logits_by_horizon(workspace, outputs).get(int(horizon or workspace.horizon))
    if not torch.is_tensor(logits):
        return []
    return top_class_rows(logits, workspace.activity_names, top_k=top_k)


def _effective_activity_probabilities(
    outputs: Mapping[str, object],
    step: int,
) -> torch.Tensor | None:
    values = outputs.get("effective_activity_probs_by_step")
    if not isinstance(values, Mapping):
        return None
    probabilities = values.get(int(step), values.get(str(int(step))))
    return probabilities if torch.is_tensor(probabilities) else None


def _selected_probabilities(probabilities: torch.Tensor) -> np.ndarray:
    if probabilities.ndim == 3:
        selected = probabilities[0, -1, :]
    elif probabilities.ndim == 2:
        selected = probabilities[0, :]
    else:
        raise ValueError(
            f"Expected 2D or 3D probabilities, got shape {tuple(probabilities.shape)}"
        )
    return selected.detach().cpu().numpy()


def graph_edges_for_concepts(
    workspace: InterventionWorkspace,
    concept_indices: Sequence[int],
    branch: str = "shared",
    layer_option: str = "mean",
    top_k: int = 20,
) -> List[Dict[str, object]]:
    """Return readable graph edges that touch selected concepts."""

    selected = {int(index) for index in concept_indices}
    if not selected:
        return []
    matrix = graph_matrix(workspace, branch, layer_option, edge_kind="spatial")
    if matrix.size == 0:
        return []
    rows = []
    for source, target in np.argwhere(np.abs(matrix) > 0.0):
        if int(source) not in selected and int(target) not in selected:
            continue
        rows.append(
            {
                "source_idx": int(source),
                "source": _name_at(workspace.concept_names, int(source)),
                "target_idx": int(target),
                "target": _name_at(workspace.concept_names, int(target)),
                "weight": float(matrix[source, target]),
                "touches_driver_as": _edge_touch_label(source, target, selected),
            }
        )
    rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return [
        {"rank": rank, **row}
        for rank, row in enumerate(rows[: int(top_k)], 1)
    ]


def concept_activation_tables(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    top_k: int = 20,
) -> Dict[str, List[Dict[str, object]]]:
    tables: Dict[str, List[Dict[str, object]]] = {}
    sources = {
        "1 raw calibrated evidence": outputs.get("calibrated_concepts"),
        "2 after temporal attention": outputs.get("temporalized_concepts"),
        "3 bounded shared graph state": outputs.get("shared_refined_concepts", outputs.get("concept_states")),
        "4 bounded current-window branch": outputs.get("window_refined_concepts"),
        "5 bounded forecast branch": outputs.get("forecast_refined_concepts", outputs.get("forecast_repr")),
        "6 classifier current-window input": outputs.get("activity_repr"),
        "7 classifier forecast input": outputs.get("forecast_repr"),
    }
    for name, tensor in sources.items():
        rows = _concept_rows(workspace, tensor, top_k=top_k)
        if rows:
            tables[name] = rows

    for horizon, tensor in dict(outputs.get("predicted_concepts_by_step", {})).items():
        rows = _concept_rows(workspace, tensor, top_k=top_k)
        if rows:
            tables[f"forecast rollout +{int(horizon)}"] = rows
    return tables


def graph_branch_options(workspace: InterventionWorkspace) -> List[str]:
    model = workspace.model
    if hasattr(model, "shared_graph_layers"):
        return ["shared", "window", "forecast"]
    if hasattr(model, "effective_same_matrix"):
        return ["legacy"]
    return []


def graph_layer_options(workspace: InterventionWorkspace, branch: str) -> List[str]:
    layers = _branch_layers(workspace, branch)
    if not layers:
        return ["mean"]
    return ["mean"] + [name for name, _ in layers]


def graph_matrix(
    workspace: InterventionWorkspace,
    branch: str,
    layer_option: str,
    edge_kind: str = "spatial",
) -> np.ndarray:
    model = workspace.model
    if branch == "legacy" and hasattr(model, "effective_same_matrix"):
        return model.effective_same_matrix().detach().cpu().numpy()

    layers = _branch_layers(workspace, branch)
    selected = [layer for name, layer in layers if layer_option == "mean" or name == layer_option]
    matrices = []
    for layer in selected:
        if edge_kind == "cross temporal" and hasattr(layer, "effective_cross_temporal_matrix"):
            matrices.append(layer.effective_cross_temporal_matrix().detach().cpu().numpy())
        elif edge_kind == "spatial" and hasattr(layer, "effective_spatial_matrix"):
            matrices.append(layer.effective_spatial_matrix().detach().cpu().numpy())
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)
    return np.mean(np.stack(matrices, axis=0), axis=0)


def graph_temporal_vector(
    workspace: InterventionWorkspace,
    branch: str,
    layer_option: str,
) -> np.ndarray:
    """Return the selected same-concept temporal edge vector."""

    layers = _branch_layers(workspace, branch)
    selected = [layer for name, layer in layers if layer_option == "mean" or name == layer_option]
    vectors = []
    for layer in selected:
        if hasattr(layer, "effective_temporal_vector"):
            vectors.append(layer.effective_temporal_vector().detach().cpu().numpy())
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def temporal_edge_rows(
    workspace: InterventionWorkspace,
    branch: str,
    layer_option: str,
    top_k: int = 20,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    vector = graph_temporal_vector(workspace, branch, layer_option)
    if vector.size == 0:
        return rows
    top = np.argsort(-np.abs(vector))[: int(top_k)]
    for rank, index in enumerate(top, 1):
        rows.append(
            {
                "rank": rank,
                "concept_idx": int(index),
                "concept": _name_at(workspace.concept_names, int(index)),
                "weight": float(vector[index]),
            }
        )
    return rows


def top_edge_rows(
    workspace: InterventionWorkspace,
    matrix: np.ndarray,
    top_k: int = 20,
) -> List[Dict[str, object]]:
    if matrix.size == 0:
        return []
    flat = np.argsort(-np.abs(matrix.reshape(-1)))[: int(top_k)]
    rows: List[Dict[str, object]] = []
    for rank, flat_idx in enumerate(flat, 1):
        source, target = np.unravel_index(flat_idx, matrix.shape)
        weight = float(matrix[source, target])
        if abs(weight) <= 0.0:
            continue
        rows.append(
            {
                "rank": rank,
                "source_idx": int(source),
                "source": _name_at(workspace.concept_names, int(source)),
                "target_idx": int(target),
                "target": _name_at(workspace.concept_names, int(target)),
                "weight": weight,
            }
        )
    return rows


def graph_summary(workspace: InterventionWorkspace) -> Dict[str, float]:
    if hasattr(workspace.model, "graph_metrics"):
        return dict(workspace.model.graph_metrics())
    return {}


def _forward_linear(
    workspace: InterventionWorkspace,
    concepts: torch.Tensor,
    key_padding_mask: torch.Tensor,
    intervention: Mapping[str, object] | None,
) -> Dict[str, object]:
    if intervention is not None:
        concepts = concepts.clone()
        items = intervention.get("items") if isinstance(intervention, Mapping) else None
        if items is None:
            items = [intervention]
        for item in items:
            time_idx = int(item["time_idx"])
            concept_idx = int(item["concept_idx"])
            if item.get("value") is not None:
                concepts[:, time_idx, concept_idx] = float(item["value"])
            else:
                concepts[:, time_idx, concept_idx] += float(item["delta"])

    model_bundle = workspace.model
    forecast_model = model_bundle["forecast_model"].eval()
    features = concepts.detach().cpu().numpy().reshape(concepts.shape[0], -1)
    standardizer = model_bundle.get("standardizer")
    if standardizer is not None:
        features = standardizer.transform(features)
    x = torch.as_tensor(features, dtype=torch.float32, device=workspace.device)
    logits = forecast_model(x)
    logits_by_time = logits[:, None, :].repeat(1, concepts.size(1), 1)
    valid_mask = (~key_padding_mask).float()
    concept_view = concepts * valid_mask.unsqueeze(-1)
    return {
        "calibrated_concepts": concept_view,
        "concept_states": concept_view,
        "forecast_logits_by_horizon": {int(workspace.horizon): logits_by_time},
    }


def _linear_driver_rows(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    target: str,
    class_idx: int,
    intervention: Mapping[str, object] | None,
    top_k: int,
) -> List[Dict[str, object]]:
    if target == "activity":
        return []
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    if intervention is not None:
        concepts = concepts.clone()
        time_idx = int(intervention["time_idx"])
        concept_idx = int(intervention["concept_idx"])
        if intervention.get("value") is not None:
            concepts[:, time_idx, concept_idx] = float(intervention["value"])
        else:
            concepts[:, time_idx, concept_idx] += float(intervention["delta"])

    model_bundle = workspace.model
    forecast_model = model_bundle["forecast_model"]
    standardizer = model_bundle.get("standardizer")
    features = concepts.detach().cpu().numpy().reshape(1, -1)
    standardized = standardizer.transform(features) if standardizer is not None else features
    linear = getattr(forecast_model, "linear", None)
    if linear is None:
        return []
    weights = linear.weight.detach().cpu().numpy()[int(class_idx)].reshape(standardized.shape[-1])
    num_concepts = len(workspace.concept_names)
    attribution = (standardized.reshape(workspace.history_length, num_concepts) * weights.reshape(
        workspace.history_length,
        num_concepts,
    ))
    values = concepts.detach()[0].cpu().numpy()
    mask = key_padding_mask.detach()[0].cpu().numpy().astype(bool)
    attribution[mask, :] = 0.0
    return _rank_history_concepts(workspace, instance, attribution, values, top_k=top_k)


def _direct_classifier_contributors(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    target: str,
    class_idx: int,
    horizon: int | None,
    top_k: int,
    history_time_idx: int | None,
) -> List[Dict[str, object]]:
    tensor, head = _classifier_tensor_and_head(workspace, outputs, target, horizon)
    if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[-1] != len(workspace.concept_names):
        return []
    if head is None or not hasattr(head, "weight"):
        return []
    weight = head.weight.detach()
    if weight.ndim != 2 or int(class_idx) >= weight.shape[0] or weight.shape[1] != tensor.shape[-1]:
        return []

    classifier_tensor = _prediction_tensor_for_head(workspace, outputs, tensor)
    classifier_input = classifier_tensor.detach()[0].cpu().numpy()
    weights = weight[int(class_idx)].detach().cpu().numpy()
    contributions = classifier_input * weights.reshape(1, -1)
    if history_time_idx is not None:
        selected_time = int(history_time_idx)
        if selected_time < 0 or selected_time >= contributions.shape[0]:
            return []
    rows = _rank_contributors(
        workspace,
        instance,
        outputs,
        contributions,
        classifier_input,
        target=target,
        horizon=horizon,
        source="linear head",
        top_k=top_k,
        history_time_idx=history_time_idx,
    )
    for row in rows:
        row["head_weight"] = float(weights[int(row["concept_idx"])])
    return rows


def _classifier_tensor_and_head(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    target: str,
    horizon: int | None,
) -> tuple[object, torch.nn.Module | None]:
    model = workspace.model
    if target == "activity":
        return outputs.get("window_refined_concepts", outputs.get("activity_context_repr")), getattr(model, "activity_head", None)

    selected_horizon = int(horizon or workspace.horizon)
    predicted = dict(outputs.get("predicted_concepts_by_step", {}))
    if selected_horizon in predicted:
        return predicted[selected_horizon], getattr(model, "activity_head", None)

    forecast_heads = getattr(model, "forecast_heads", None)
    if forecast_heads is not None and str(selected_horizon) in forecast_heads:
        return outputs.get("forecast_refined_concepts", outputs.get("forecast_repr")), forecast_heads[str(selected_horizon)]

    return outputs.get("forecast_refined_concepts", outputs.get("forecast_repr")), getattr(model, "shared_forecast_head", None)


def _prediction_tensor_for_head(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
    tensor: object,
) -> object:
    if not torch.is_tensor(tensor):
        return tensor
    transform = str(outputs.get("st_prediction_transform") or getattr(workspace.model, "st_prediction_transform", "identity"))
    if transform == "logit":
        eps = torch.finfo(tensor.dtype).eps
        bounded = tensor.clamp(min=eps, max=1.0 - eps)
        return torch.log(bounded) - torch.log1p(-bounded)
    if transform == "centered":
        return (2.0 * tensor) - 1.0
    return tensor


def _augment_contributor_rows(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    rows: List[Dict[str, object]],
    target: str,
    horizon: int | None,
) -> List[Dict[str, object]]:
    tensor, _ = _classifier_tensor_and_head(workspace, outputs, target, horizon)
    classifier_tensor = _prediction_tensor_for_head(workspace, outputs, tensor)
    classifier_input = (
        classifier_tensor.detach()[0].cpu().numpy()
        if torch.is_tensor(classifier_tensor) and classifier_tensor.ndim == 3
        else None
    )
    augmented = []
    for row in rows:
        time_idx = int(row["history_t"])
        concept_idx = int(row["concept_idx"])
        enriched = dict(row)
        enriched.update(_concept_value_columns(workspace, instance, outputs, time_idx, concept_idx))
        if classifier_input is not None and time_idx < classifier_input.shape[0] and concept_idx < classifier_input.shape[1]:
            enriched["classifier_input"] = float(classifier_input[time_idx, concept_idx])
        enriched.setdefault("source", "gradient fallback")
        augmented.append(enriched)
    return augmented


def _rank_contributors(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    contributions: np.ndarray,
    classifier_input: np.ndarray,
    target: str,
    horizon: int | None,
    source: str,
    top_k: int,
    history_time_idx: int | None = None,
) -> List[Dict[str, object]]:
    del target, horizon
    valid = ~np.asarray(instance["key_padding_mask"], dtype=bool)
    scores = np.abs(contributions).copy()
    scores[~valid, :] = -np.inf
    if history_time_idx is not None:
        time_mask = np.ones(scores.shape[0], dtype=bool)
        time_mask[int(history_time_idx)] = False
        scores[time_mask, :] = -np.inf
    flat = np.argsort(-scores.reshape(-1))[: int(top_k)]
    rows: List[Dict[str, object]] = []
    for rank, flat_idx in enumerate(flat, 1):
        time_idx, concept_idx = np.unravel_index(flat_idx, scores.shape)
        if not np.isfinite(scores[time_idx, concept_idx]):
            continue
        contribution = float(contributions[time_idx, concept_idx])
        row = {
            "rank": rank,
            "history_t": int(time_idx),
            "original_t": _original_timestep(instance, int(time_idx)),
            "concept_idx": int(concept_idx),
            "concept": _name_at(workspace.concept_names, int(concept_idx)),
            "classifier_input": float(classifier_input[time_idx, concept_idx]),
            "contribution": contribution,
            "abs_contribution": float(abs(contribution)),
            "direction": "supports" if contribution >= 0.0 else "opposes",
            "source": source,
        }
        row.update(_concept_value_columns(workspace, instance, outputs, int(time_idx), int(concept_idx)))
        rows.append(row)
    return rows


def _concept_value_columns(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
    time_idx: int,
    concept_idx: int,
) -> Dict[str, object]:
    raw_cosine = _raw_cosine_value(workspace, instance, time_idx, concept_idx)
    standardized = np.asarray(instance["concepts"], dtype=np.float32)
    values: Dict[str, object] = {
        "cosine_activation": raw_cosine,
        "standardized_input": float(standardized[time_idx, concept_idx]),
        "z": _tensor_value(outputs.get("calibrated_concepts"), time_idx, concept_idx),
    }
    calibrator = getattr(workspace.model, "calibrator", None)
    if getattr(calibrator, "activation", None) == "learned_threshold":
        threshold = calibrator.threshold.detach().cpu().numpy()
        values["learned_threshold"] = float(threshold[int(concept_idx)])
        z_value = values.get("z")
        values["soft_binary"] = None if z_value is None else bool(float(z_value) >= 0.5)
    return values


def _raw_cosine_value(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    time_idx: int,
    concept_idx: int,
) -> float | None:
    original_t = _original_timestep(instance, int(time_idx))
    if original_t is None:
        return None
    split = str(instance["split"])
    video_index = int(instance["video_index"])
    concepts = workspace.preprocessed_data.get(split, {}).get("concepts")
    if concepts is None:
        return None
    array = np.asarray(concepts)
    if video_index >= array.shape[0] or original_t >= array.shape[1] or concept_idx >= array.shape[2]:
        return None
    return float(array[video_index, original_t, concept_idx])


def _tensor_value(tensor: object, time_idx: int, concept_idx: int) -> float | None:
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        return None
    if time_idx >= tensor.shape[1] or concept_idx >= tensor.shape[2]:
        return None
    return float(tensor[0, time_idx, concept_idx].detach().cpu().item())


def _concept_rows(
    workspace: InterventionWorkspace,
    tensor: object,
    top_k: int,
) -> List[Dict[str, object]]:
    if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[-1] != len(workspace.concept_names):
        return []
    values = tensor[0, -1, :].detach().cpu().numpy()
    top = np.argsort(-np.abs(values))[: int(top_k)]
    return [
        {
            "rank": rank,
            "concept_idx": int(index),
            "concept": workspace.concept_names[int(index)],
            "score": float(values[index]),
        }
        for rank, index in enumerate(top, 1)
    ]


def _rank_history_concepts(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    attribution: np.ndarray,
    values: np.ndarray,
    top_k: int,
) -> List[Dict[str, object]]:
    valid = ~np.asarray(instance["key_padding_mask"], dtype=bool)
    scores = np.abs(attribution).copy()
    scores[~valid, :] = -np.inf
    flat = np.argsort(-scores.reshape(-1))[: int(top_k)]
    rows: List[Dict[str, object]] = []
    for rank, flat_idx in enumerate(flat, 1):
        time_idx, concept_idx = np.unravel_index(flat_idx, scores.shape)
        if not np.isfinite(scores[time_idx, concept_idx]):
            continue
        rows.append(
            {
                "rank": rank,
                "history_t": int(time_idx),
                "original_t": _original_timestep(instance, int(time_idx)),
                "concept_idx": int(concept_idx),
                "concept": _name_at(workspace.concept_names, int(concept_idx)),
                "contribution": float(attribution[time_idx, concept_idx]),
                "abs_contribution": float(abs(attribution[time_idx, concept_idx])),
                "concept_score": float(values[time_idx, concept_idx]),
                "direction": "supports" if float(attribution[time_idx, concept_idx]) >= 0.0 else "opposes",
            }
        )
    return rows


def _delta_rows_for_logits(
    label: str,
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    names: Sequence[str],
    top_k: int,
) -> List[Dict[str, object]]:
    before_probs = _probabilities(before_logits)
    after_probs = _probabilities(after_logits)
    delta = after_probs - before_probs
    top = np.argsort(-np.abs(delta))[: int(top_k)]
    return [
        {
            "target": label,
            "class_idx": int(index),
            "class": _name_at(names, int(index)),
            "before": float(before_probs[index]),
            "after": float(after_probs[index]),
            "delta": float(delta[index]),
        }
        for index in top
    ]


def _probabilities(logits: torch.Tensor) -> np.ndarray:
    selected = logits[0, -1, :] if logits.ndim == 3 else logits[0, :]
    return torch.softmax(selected, dim=-1).detach().cpu().numpy()


def _branch_layers(workspace: InterventionWorkspace, branch: str) -> List[tuple[str, torch.nn.Module]]:
    model = workspace.model
    attr_by_branch = {
        "shared": "shared_graph_layers",
        "window": "window_graph_layers",
        "forecast": "forecast_graph_layers",
    }
    attr = attr_by_branch.get(branch)
    if attr is None or not hasattr(model, attr):
        return []
    return [(f"{branch}.{idx}", layer) for idx, layer in enumerate(getattr(model, attr))]


def _original_timestep(instance: Mapping[str, object], history_t: int) -> int | None:
    if np.asarray(instance["key_padding_mask"], dtype=bool)[int(history_t)]:
        return None
    return int(instance["history_start"]) + int(history_t) - int(instance["history_offset"])


def _edge_touch_label(source: int, target: int, selected: set[int]) -> str:
    source_in = int(source) in selected
    target_in = int(target) in selected
    if source_in and target_in:
        return "driver to driver"
    if source_in:
        return "driver as source"
    return "driver as target"


def _read_json(path: Path) -> tuple[Dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Could not read {path}: {exc}"
    return payload if isinstance(payload, dict) else {}, None


def _infer_dataset(path: Path) -> str:
    for part in reversed(path.parts):
        lowered = part.lower()
        if "barista" in lowered:
            return "barista"
        if "breakfast" in lowered:
            return "Breakfast"
        if "gtea" in lowered:
            return "gtea_gaze"
        if "mpii" in lowered:
            return "mpii_cooking_2"
    return "unknown"


def _name_at(names: Sequence[str], index: int | None) -> str | None:
    if index is None:
        return None
    idx = int(index)
    if 0 <= idx < len(names):
        return names[idx]
    return str(idx)
