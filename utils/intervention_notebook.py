"""Notebook utilities for ST-CBM forecasting explanations and interventions."""

from __future__ import annotations

import os
import pickle
import sys
import json
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_models import default_embedding_path, resolve_device, write_json
from utils.dataset_handling import process_dataset
from utils.model import (
    TrainedModel,
    _classification_metrics,
    _forward_outputs,
    _prepared_splits_with_forecast,
    _sil_metrics,
    _sliding_window_examples,
    _standardized_splits,
    canonical_method,
    train_model,
)
from utils.prepare_datasets import _canonical_dataset_key, _load_concept_set, prepare_data
from utils.paths import dataset_root, embedding_root


@dataclass
class InterventionNotebookConfig:
    dataset: str = "Breakfast"
    test_split: str = "s1"
    window_size: int = 32
    concept_set: str = "breakfast_simple_v1"
    backbone: str = "pe-l14"
    data_root: Path = embedding_root()
    dataset_root: Path = dataset_root()
    checkpoint_path: Path | None = None
    checkpoint_selection: str = "best"
    output_dir: Path = PROJECT_ROOT / "runs" / "intervention_notebook"
    train_if_missing: bool = False
    random_windows: bool = True
    generate_embeddings: bool = True
    embedding_path: Path | None = None
    embedding_batch_size: int = 256
    embedding_num_gpus: int = 0
    device: str = "auto"
    horizon: int = 3
    history_length: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 100
    patience: int = 10
    seed: int = 42
    base_method: str = "graph_cbm"
    binary: bool = True
    activity_label_mode: str = "action"
    activity_label_fill_mode: str = "sil"
    learn_concept_threshold: bool = True
    teacher_forcing_start_ratio: float = 1.0
    teacher_forcing_end_ratio: float = 0.8
    forecast_class_weighting: bool = False
    forecast_class_weight_cap: float = 5.0
    forecast_sil_false_positive_penalty: float = 0.0
    model_hparams: Mapping[str, object] | None = None
    wandb_project: str | None = None
    wandb_mode: str = "disabled"


@dataclass
class InterventionWorkspace:
    config: InterventionNotebookConfig
    model: torch.nn.Module
    base_method: str
    horizon: int
    history_length: int
    preprocessed_data: Dict[str, object]
    standardized_splits: Dict[str, Dict[str, np.ndarray]]
    activity_names: List[str]
    concept_names: List[str]
    checkpoint: Dict[str, object] | None = None

    @property
    def device(self) -> torch.device:
        if isinstance(self.model, Mapping) and "forecast_model" in self.model:
            return next(self.model["forecast_model"].parameters()).device
        return next(self.model.parameters()).device

    @property
    def forecast_horizons(self) -> List[int]:
        if hasattr(self.model, "forecast_horizons"):
            configured = sorted(int(h) for h in self.model.forecast_horizons)
            if self.base_method != "motif" and configured:
                return list(range(1, max(configured) + 1))
            return configured
        return [int(self.horizon)]


def latest_checkpoint(root: str | Path = PROJECT_ROOT / "runs" / "train_models") -> Path | None:
    checkpoints = sorted(Path(root).glob("**/model.pt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _select_checkpoint(config: InterventionNotebookConfig) -> Path | None:
    selection = str(config.checkpoint_selection).lower()
    if selection == "latest":
        selected = latest_checkpoint(config.output_dir)
        return selected or latest_checkpoint(PROJECT_ROOT / "runs" / "train_models")
    if selection != "best":
        raise ValueError("checkpoint_selection must be 'best' or 'latest'.")
    selected = best_checkpoint(config.output_dir)
    return selected or best_checkpoint(PROJECT_ROOT / "runs" / "train_models")


def best_checkpoint(root: str | Path = PROJECT_ROOT / "runs" / "train_models") -> Path | None:
    scored = []
    for checkpoint_path in Path(root).glob("**/model.pt"):
        score = _checkpoint_validation_score(checkpoint_path)
        if score is not None:
            scored.append((float(score), checkpoint_path.stat().st_mtime, checkpoint_path))
    if not scored:
        return latest_checkpoint(root)
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[-1][2]


def _checkpoint_validation_score(checkpoint_path: Path) -> float | None:
    history_path = checkpoint_path.parent / "history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
        scores = [float(row["val_score"]) for row in history if isinstance(row, dict) and "val_score" in row]
        if scores:
            return max(scores)

    metrics_path = checkpoint_path.parent / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    val = metrics.get("val", {})
    activity = val.get("activity", {})
    forecast = val.get("forecast", {})
    required = [
        activity.get("accuracy"),
        activity.get("macro_f1"),
        forecast.get("accuracy"),
        forecast.get("macro_f1"),
    ]
    if any(value is None for value in required):
        return None
    return 0.25 * sum(float(value) for value in required)


def load_or_train_workspace(config: InterventionNotebookConfig | None = None) -> InterventionWorkspace:
    config = config or InterventionNotebookConfig()
    os.environ["WANDB_MODE"] = config.wandb_mode
    device = torch.device(resolve_device(config.device))
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else _select_checkpoint(config)

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", {})
        config = _config_from_checkpoint_args(config, ckpt_args)
        config = _resolve_checkpoint_concept_set(config, checkpoint)
        preprocessed = _prepare_preprocessed_data(config, device)
        model = checkpoint["model"]
        if isinstance(model, ABCMapping) and "forecast_model" in model:
            model = dict(model)
            model["forecast_model"] = model["forecast_model"].to(device).eval()
        else:
            model = model.to(device).eval()
        metadata = preprocessed.get("metadata", {})
        base_method = canonical_method(str(checkpoint.get("base_method", config.base_method)))
        standardized_splits = (
            _standardize_preprocessed_splits(preprocessed)
            if base_method != "linear"
            else _raw_preprocessed_splits(preprocessed, int(checkpoint.get("horizon", config.horizon)))
        )
        return InterventionWorkspace(
            config=config,
            model=model,
            base_method=base_method,
            horizon=int(checkpoint.get("horizon", config.horizon)),
            history_length=int(checkpoint.get("history_length", config.history_length)),
            preprocessed_data=preprocessed,
            standardized_splits=standardized_splits,
            activity_names=list(checkpoint.get("activity_names") or metadata.get("activity_names", [])),
            concept_names=list(metadata.get("concept_names", checkpoint.get("metadata", {}).get("concept_names", []))),
            checkpoint=checkpoint,
        )

    if not config.train_if_missing:
        raise FileNotFoundError(
            "No checkpoint was found. Set config.checkpoint_path or config.train_if_missing=True."
        )

    preprocessed = _prepare_preprocessed_data(config, device)
    trained = train_model(
        preprocessed,
        config.horizon,
        config.history_length,
        config.learning_rate,
        config.weight_decay,
        config.batch_size,
        config.num_epochs,
        config.patience,
        config.wandb_project,
        config.seed,
        config.base_method,
        device=device,
        teacher_forcing_start_ratio=config.teacher_forcing_start_ratio,
        teacher_forcing_end_ratio=config.teacher_forcing_end_ratio,
        learn_concept_threshold=config.learn_concept_threshold,
        forecast_class_weighting=config.forecast_class_weighting,
        forecast_class_weight_cap=config.forecast_class_weight_cap,
        forecast_sil_false_positive_penalty=config.forecast_sil_false_positive_penalty,
        model_hparams=dict(config.model_hparams or {}),
        run_metadata=_run_metadata(config),
    )
    checkpoint = _save_trained_checkpoint(config, trained, preprocessed)
    return InterventionWorkspace(
        config=config,
        model=trained.model.to(device).eval(),
        base_method=trained.base_method,
        horizon=trained.horizon,
        history_length=trained.history_length,
        preprocessed_data=preprocessed,
        standardized_splits=_standardize_preprocessed_splits(preprocessed),
        activity_names=trained.activity_names,
        concept_names=list(preprocessed["metadata"]["concept_names"]),
        checkpoint=checkpoint,
    )


def select_instance(
    workspace: InterventionWorkspace,
    split: str = "test",
    video_index: int = 0,
    timestep: int | None = None,
) -> Dict[str, object]:
    split_data = workspace.standardized_splits[split]
    raw_split = workspace.preprocessed_data[split]
    length = int(split_data["lengths"][video_index])
    if timestep is None:
        timestep = max(0, length - max(workspace.forecast_horizons) - 1)
    if timestep < 0 or timestep >= length:
        raise IndexError(f"timestep must be in [0, {length - 1}], got {timestep}")

    start = max(0, int(timestep) - workspace.history_length + 1)
    concept_key = "concepts_std" if "concepts_std" in split_data else "concepts"
    concepts = np.asarray(split_data[concept_key][video_index, start : int(timestep) + 1], dtype=np.float32)
    valid = np.asarray(split_data["mask"][video_index, start : int(timestep) + 1] > 0.0)
    padded = np.zeros((workspace.history_length, concepts.shape[-1]), dtype=np.float32)
    key_padding_mask = np.ones(workspace.history_length, dtype=bool)
    offset = workspace.history_length - concepts.shape[0]
    padded[offset:] = concepts
    key_padding_mask[offset:] = ~valid

    labels = np.asarray(raw_split["activity_labels"][video_index], dtype=np.int64)
    max_horizon = max(int(horizon) for horizon in workspace.forecast_horizons)
    future_labels = {
        horizon: int(labels[timestep + horizon]) if timestep + horizon < length else None
        for horizon in range(1, max_horizon + 1)
    }
    return {
        "split": split,
        "video_index": int(video_index),
        "timestep": int(timestep),
        "history_start": int(start),
        "history_offset": int(offset),
        "length": length,
        "video_id": str(raw_split["video_ids"][video_index]),
        "video_path": str(raw_split["video_paths"][video_index]),
        "concepts": padded,
        "key_padding_mask": key_padding_mask,
        "current_label": int(labels[timestep]),
        "future_labels": future_labels,
    }


def run_instance(workspace: InterventionWorkspace, instance: Mapping[str, object]) -> Dict[str, object]:
    model = workspace.model.eval()
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    with torch.no_grad():
        outputs = _forward(workspace, concepts, key_padding_mask)
    return prediction_summary(workspace, instance, outputs)


def prediction_summary(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    outputs: Mapping[str, object],
) -> Dict[str, object]:
    activity_logits = outputs["activity_logits"][:, -1, :]
    effective = outputs.get("effective_activity_probs_by_step")
    current_effective = effective.get(0) if isinstance(effective, Mapping) else None
    activity_prob = (
        current_effective[:, -1, :][0].detach().cpu().numpy()
        if torch.is_tensor(current_effective)
        else torch.softmax(activity_logits, dim=-1)[0].detach().cpu().numpy()
    )
    forecasts = {}
    for horizon, logits in _forecast_logits_by_horizon(workspace, outputs).items():
        horizon_effective = effective.get(int(horizon)) if isinstance(effective, Mapping) else None
        prob = (
            horizon_effective[:, -1, :][0].detach().cpu().numpy()
            if torch.is_tensor(horizon_effective)
            else torch.softmax(logits[:, -1, :], dim=-1)[0].detach().cpu().numpy()
        )
        forecasts[int(horizon)] = {
            "probs": prob,
            "pred_idx": int(prob.argmax()),
            "pred_label": _activity_name(workspace, int(prob.argmax())),
            "true_idx": instance["future_labels"].get(int(horizon)),
            "true_label": _activity_name(workspace, instance["future_labels"].get(int(horizon))),
        }
    pred_idx = int(activity_prob.argmax())
    return {
        "outputs": outputs,
        "activity_probs": activity_prob,
        "activity_pred_idx": pred_idx,
        "activity_pred_label": _activity_name(workspace, pred_idx),
        "activity_true_idx": int(instance["current_label"]),
        "activity_true_label": _activity_name(workspace, int(instance["current_label"])),
        "forecasts": forecasts,
    }


def explain_instance(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    target: str = "activity",
    horizon: int | None = None,
    class_index: int | None = None,
    top_k: int = 15,
) -> Dict[str, object]:
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    concepts = concepts.detach().clone().requires_grad_(True)
    outputs = _forward(workspace, concepts, key_padding_mask)
    if target == "activity":
        logits = outputs["activity_logits"][:, -1, :]
    else:
        horizon = int(horizon or max(workspace.forecast_horizons))
        logits = _forecast_logits_by_horizon(workspace, outputs)[horizon][:, -1, :]
    class_index = int(logits.argmax(dim=-1).item() if class_index is None else class_index)
    score = logits[0, class_index]
    workspace.model.zero_grad(set_to_none=True)
    score.backward()
    attribution = (concepts.grad.detach()[0] * concepts.detach()[0]).cpu().numpy()
    rows = _top_history_concepts(workspace, instance, attribution, top_k=top_k)
    return {
        "target": target,
        "horizon": horizon,
        "class_index": class_index,
        "class_label": _activity_name(workspace, class_index),
        "score": float(score.detach().cpu().item()),
        "attribution": attribution,
        "top_rows": rows,
    }


def print_explanation(explanation: Mapping[str, object]) -> None:
    print(f"Target: {explanation['target']} -> {explanation['class_label']} (logit={explanation['score']:.3f})")
    print("rank | window_t | original_t | concept | contribution")
    for i, row in enumerate(explanation["top_rows"], 1):
        print(
            f"{i:>4} | {row['history_t']:>8} | {row['original_t']:>10} | "
            f"{row['concept_name']} | {row['contribution']:+.4f}"
        )


def plot_instance_overview(workspace: InterventionWorkspace, instance: Mapping[str, object]):
    summary = run_instance(workspace, instance)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _bar_top_classes(
        axes[0],
        summary["activity_probs"],
        workspace.activity_names,
        title=f"Current: {summary['activity_pred_label']} | true: {summary['activity_true_label']}",
    )
    horizon = max(summary["forecasts"])
    forecast = summary["forecasts"][horizon]
    _bar_top_classes(
        axes[1],
        forecast["probs"],
        workspace.activity_names,
        title=f"+{horizon}: {forecast['pred_label']} | true: {forecast['true_label']}",
    )
    fig.tight_layout()
    plt.show()
    return fig


def plot_forecast_timeline(workspace: InterventionWorkspace, instance: Mapping[str, object]):
    summary = run_instance(workspace, instance)
    horizons = sorted(summary["forecasts"])
    labels = [summary["forecasts"][h]["pred_label"] for h in horizons]
    confidence = [float(summary["forecasts"][h]["probs"].max()) for h in horizons]
    fig, ax = plt.subplots(figsize=(max(7, len(horizons) * 2), 3.5))
    ax.plot(horizons, confidence, marker="o")
    for horizon, label, conf in zip(horizons, labels, confidence):
        ax.annotate(label, (horizon, conf), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.set_title("Autoregressive forecast up to max horizon")
    ax.grid(True, alpha=0.25)
    plt.show()
    return fig


def forecast_diagnostics(
    workspace: InterventionWorkspace,
    split: str = "test",
    batch_size: int = 512,
    top_k: int = 10,
) -> Dict[str, object]:
    examples = _workspace_examples(workspace, split)
    sil_index = workspace.activity_names.index("SIL") if "SIL" in workspace.activity_names else None
    true_by_horizon: Dict[int, List[np.ndarray]] = {h: [] for h in workspace.forecast_horizons}
    pred_by_horizon: Dict[int, List[np.ndarray]] = {h: [] for h in workspace.forecast_horizons}

    workspace.model.eval()
    with torch.no_grad():
        for start in range(0, int(examples["concepts"].shape[0]), int(batch_size)):
            stop = start + int(batch_size)
            concepts = torch.as_tensor(examples["concepts"][start:stop], dtype=torch.float32, device=workspace.device)
            mask = torch.as_tensor(examples["key_padding_mask"][start:stop], dtype=torch.bool, device=workspace.device)
            outputs = _forward_outputs(workspace.model, workspace.base_method, concepts, mask)
            logits_by_horizon = _forecast_logits_by_horizon(workspace, outputs)
            for horizon in workspace.forecast_horizons:
                if horizon not in logits_by_horizon:
                    continue
                target = (
                    examples["teacher_forcing_labels"][start:stop, horizon]
                    if horizon < workspace.horizon
                    else examples["forecast_labels"][start:stop]
                )
                pred = logits_by_horizon[horizon][:, -1, :].argmax(dim=-1).detach().cpu().numpy()
                true_by_horizon[horizon].append(np.asarray(target, dtype=np.int64))
                pred_by_horizon[horizon].append(pred.astype(np.int64, copy=False))

    summary = []
    distributions = {}
    for horizon in workspace.forecast_horizons:
        labels = np.concatenate(true_by_horizon[horizon]) if true_by_horizon[horizon] else np.zeros(0, dtype=np.int64)
        preds = np.concatenate(pred_by_horizon[horizon]) if pred_by_horizon[horizon] else np.zeros(0, dtype=np.int64)
        metrics = _classification_metrics(labels, preds, len(workspace.activity_names))
        metrics.update(_sil_metrics(labels, preds, sil_index))
        row = {"horizon": int(horizon), **metrics}
        summary.append(row)
        distributions[horizon] = _distribution_diagnostics(
            workspace,
            labels,
            preds,
            sil_index=sil_index,
            top_k=top_k,
        )
    return {"split": split, "summary": summary, "distributions": distributions}


def print_forecast_diagnostics(diagnostics: Mapping[str, object]) -> None:
    print(f"Forecast diagnostics: {diagnostics['split']}")
    print("h | acc | macro_f1 | SIL true | SIL pred | SIL false+")
    for row in diagnostics["summary"]:
        print(
            f"{row['horizon']:>1} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['sil_true_rate']:.4f} | {row['sil_pred_rate']:.4f} | "
            f"{row['sil_false_positive_rate']:.4f}"
        )
    for horizon, dist in diagnostics["distributions"].items():
        print(f"\nTop true labels when horizon +{horizon} was predicted as SIL:")
        for row in dist["actual_when_predicted_sil"]:
            print(f"{row['label']}: {row['count']}")


def plot_forecast_diagnostics(diagnostics: Mapping[str, object], top_k: int = 10) -> None:
    distributions = diagnostics["distributions"]
    fig, axes = plt.subplots(len(distributions), 1, figsize=(11, max(3.5, 3.2 * len(distributions))))
    axes = np.atleast_1d(axes)
    for ax, (horizon, dist) in zip(axes, distributions.items()):
        labels = _diagnostic_bar_labels(dist, top_k=top_k)
        true_values = np.asarray([dist["true_rate_by_label"].get(label, 0.0) for label in labels])
        pred_values = np.asarray([dist["pred_rate_by_label"].get(label, 0.0) for label in labels])
        y = np.arange(len(labels))
        ax.barh(y + 0.18, true_values, height=0.35, label="true")
        ax.barh(y - 0.18, pred_values, height=0.35, label="pred")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, max(float(true_values.max(initial=0)), float(pred_values.max(initial=0)), 0.01) * 1.15)
        ax.set_title(f"Forecast +{horizon}: true vs predicted class distribution")
        ax.set_xlabel("Rate")
        ax.legend()
    fig.tight_layout()
    plt.show()


def concept_scores(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    top_k: int = 20,
) -> Dict[str, List[Dict[str, object]]]:
    """Return the concept scores used by the current and forecast branches.

    ``calibrated`` is the post-calibrator model input. For learned-threshold
    checkpoints, this is the soft binary concept value in [0, 1].
    """

    summary = run_instance(workspace, instance)
    outputs = summary["outputs"]
    tables = {}
    score_sources = {
        "calibrated": outputs.get("calibrated_concepts"),
        "classification": outputs.get("window_refined_concepts", outputs.get("concept_states")),
        "forecast": outputs.get("forecast_refined_concepts", outputs.get("forecast_repr")),
    }
    for name, tensor in score_sources.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[-1] != len(workspace.concept_names):
            continue
        values = tensor[0, -1, :].detach().cpu().numpy()
        top = np.argsort(-np.abs(values))[: int(top_k)]
        tables[name] = [
            {
                "rank": rank,
                "concept_idx": int(index),
                "concept_name": workspace.concept_names[int(index)],
                "score": float(values[index]),
            }
            for rank, index in enumerate(top, 1)
        ]
    return tables


def print_concept_scores(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    top_k: int = 20,
) -> None:
    mode = getattr(getattr(workspace.model, "calibrator", None), "activation", "unknown")
    print(f"Calibrator: {mode}")
    if mode == "learned_threshold":
        print("calibrated scores are soft binary concept activations in [0, 1]")
    for source, rows in concept_scores(workspace, instance, top_k=top_k).items():
        print(f"\n{source}")
        print("rank | concept | score")
        for row in rows:
            print(f"{row['rank']:>4} | {row['concept_name']} | {row['score']:+.4f}")


def plot_concept_scores(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    top_k: int = 20,
):
    tables = concept_scores(workspace, instance, top_k=top_k)
    if not tables:
        print("No concept score tensors are available for this model.")
        return
    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), max(4, 0.28 * top_k)))
    axes = np.atleast_1d(axes)
    for ax, (source, rows) in zip(axes, tables.items()):
        labels = [row["concept_name"] for row in rows][::-1]
        values = np.asarray([row["score"] for row in rows], dtype=np.float32)[::-1]
        ax.barh(labels, values)
        ax.axvline(0.0, color="black", linewidth=0.8)
        if source == "calibrated" and values.size and values.min() >= -1e-6 and values.max() <= 1.0 + 1e-6:
            ax.set_xlim(0, 1)
        ax.set_title(source)
        ax.set_xlabel("Concept score")
    fig.tight_layout()
    plt.show()
    return fig


def plot_explanation_heatmap(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    explanation: Mapping[str, object],
    top_k: int = 20,
) -> None:
    attribution = np.asarray(explanation["attribution"])
    concept_scores = np.abs(attribution).sum(axis=0)
    top = np.argsort(-concept_scores)[: int(top_k)]
    history_labels = [
        str(instance["history_start"] + i - instance["history_offset"]) if not instance["key_padding_mask"][i] else "pad"
        for i in range(workspace.history_length)
    ]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(top))))
    im = ax.imshow(attribution[:, top].T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([workspace.concept_names[i] for i in top], fontsize=8)
    ax.set_xticks(np.arange(workspace.history_length))
    ax.set_xticklabels(history_labels)
    ax.set_xlabel("Original timestep")
    ax.set_title(f"Signed input-gradient contributions for {explanation['class_label']}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.show()


def plot_learned_graphs(
    workspace: InterventionWorkspace,
    top_k: int = 15,
    compact: bool = True,
    include_cross_temporal: bool = False,
) -> None:
    model = workspace.model
    if hasattr(model, "shared_graph_layers"):
        _plot_st_graph_layers(
            workspace,
            top_k=top_k,
            compact=compact,
            include_cross_temporal=include_cross_temporal,
        )
    elif hasattr(model, "effective_same_matrix"):
        _plot_legacy_graph(workspace, top_k=top_k)
    else:
        print("This model does not expose learned concept graph weights.")


def intervene_on_history(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    concept: int | str,
    history_t: int = -1,
    value: float | None = None,
    delta: float | None = None,
) -> Dict[str, object]:
    concept_idx = _concept_index(workspace, concept)
    time_idx = _history_index(workspace, history_t)
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    with torch.no_grad():
        baseline = _forward(workspace, concepts, key_padding_mask)
    intervention = {"concept_idx": concept_idx, "time_idx": time_idx, "value": value, "delta": delta}
    with torch.no_grad():
        intervened = _forward(workspace, concepts, key_padding_mask, intervention=intervention)
    before = prediction_summary(workspace, instance, baseline)
    after = prediction_summary(workspace, instance, intervened)
    return {
        "concept_idx": concept_idx,
        "concept_name": workspace.concept_names[concept_idx],
        "history_t": time_idx,
        "original_t": _original_timestep(instance, time_idx),
        "before": before,
        "after": after,
        "baseline_outputs": baseline,
        "intervened_outputs": intervened,
    }


def intervene_on_activity(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    activity: int | str,
    step: int = 0,
    probability: float = 0.9,
) -> Dict[str, object]:
    """Set one activity belief and propagate it through later concept forecasts."""

    model = workspace.model
    if not callable(getattr(model, "effective_activity_feedback_matrix", None)):
        raise ValueError("Selected model does not expose activity-to-concept feedback.")
    if not bool(getattr(model, "_activity_feedback_enabled", lambda: False)()):
        raise ValueError("Selected checkpoint has activity feedback disabled.")
    class_idx = _activity_index(workspace, activity)
    step = int(step)
    probability = float(probability)
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    with torch.no_grad():
        baseline = _forward(workspace, concepts, key_padding_mask)
    intervention = {
        "item_type": "activity",
        "step": step,
        "class_idx": class_idx,
        "probability": probability,
    }
    with torch.no_grad():
        intervened = _forward(
            workspace,
            concepts,
            key_padding_mask,
            intervention=intervention,
        )
    before = prediction_summary(workspace, instance, baseline)
    after = prediction_summary(workspace, instance, intervened)

    before_probs = baseline["effective_activity_probs_by_step"][step][0, -1, :]
    after_probs = intervened["effective_activity_probs_by_step"][step][0, -1, :]
    matrix = model.effective_activity_feedback_matrix().detach()
    concept_delta = torch.matmul(after_probs - before_probs, matrix).detach().cpu().numpy()
    top = np.argsort(-np.abs(concept_delta))[: min(10, len(workspace.concept_names))]
    edge_contributions = [
        {
            "concept_idx": int(index),
            "concept": workspace.concept_names[int(index)],
            "message_delta": float(concept_delta[index]),
        }
        for index in top
    ]
    future_concept_deltas = {}
    for horizon in sorted(set(baseline.get("predicted_concepts_by_step", {}))):
        if int(horizon) <= step:
            continue
        delta = (
            intervened["predicted_concepts_by_step"][horizon][0, -1, :]
            - baseline["predicted_concepts_by_step"][horizon][0, -1, :]
        )
        future_concept_deltas[int(horizon)] = delta.detach().cpu().numpy()
    return {
        "step": step,
        "class_idx": class_idx,
        "activity": workspace.activity_names[class_idx],
        "probability": probability,
        "before": before,
        "after": after,
        "baseline_outputs": baseline,
        "intervened_outputs": intervened,
        "edge_contributions": edge_contributions,
        "future_concept_deltas": future_concept_deltas,
    }


def intervene_on_future_concept(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    concept: int | str,
    horizon: int,
    value: float | None = None,
    delta: float | None = None,
) -> Dict[str, object]:
    """Edit one predicted concept state at a future forecast horizon."""

    horizon = int(horizon)
    valid_horizons = sorted(workspace.forecast_horizons)
    if horizon not in valid_horizons:
        raise ValueError(f"Forecast horizon must be one of {valid_horizons}, got {horizon}.")
    if (value is None) == (delta is None):
        raise ValueError("Provide exactly one of value or delta.")
    concept_idx = _concept_index(workspace, concept)
    concepts, key_padding_mask = _instance_tensors(workspace, instance)
    with torch.no_grad():
        baseline = _forward(workspace, concepts, key_padding_mask)
        intervened = _forward(
            workspace,
            concepts,
            key_padding_mask,
            intervention={
                "mode": "pulse",
                "items": [
                    {
                        "item_type": "concept",
                        "rollout_step": horizon,
                        "concept_idx": concept_idx,
                        "value": value,
                        "delta": delta,
                    }
                ],
            },
        )
    return {
        "concept_idx": concept_idx,
        "concept_name": workspace.concept_names[concept_idx],
        "horizon": horizon,
        "before": prediction_summary(workspace, instance, baseline),
        "after": prediction_summary(workspace, instance, intervened),
        "baseline_outputs": baseline,
        "intervened_outputs": intervened,
    }


def print_intervention_result(result: Mapping[str, object]) -> None:
    before = result["before"]
    after = result["after"]
    print(
        f"Intervened on {result['concept_name']} at history index {result['history_t']} "
        f"(original t={result['original_t']})."
    )
    print(f"Activity: {before['activity_pred_label']} -> {after['activity_pred_label']}")
    for horizon in sorted(before["forecasts"]):
        b = before["forecasts"][horizon]
        a = after["forecasts"][horizon]
        print(f"+{horizon}: {b['pred_label']} -> {a['pred_label']}")


def make_intervention_widget(workspace: InterventionWorkspace, instance: Mapping[str, object]):
    import ipywidgets as widgets
    from IPython.display import display

    concept_options = [(name, i) for i, name in enumerate(workspace.concept_names)]
    concept_dd = widgets.Dropdown(options=concept_options, description="Concept")
    time_slider = widgets.IntSlider(
        value=workspace.history_length - 1,
        min=0,
        max=workspace.history_length - 1,
        step=1,
        description="History t",
    )
    mode = widgets.ToggleButtons(options=["value", "delta"], value="delta", description="Mode")
    value = widgets.FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.1, description="Amount")
    button = widgets.Button(description="Run intervention", button_style="primary")
    output = widgets.Output()

    def on_click(_):
        with output:
            output.clear_output()
            kwargs = {"delta": value.value} if mode.value == "delta" else {"value": value.value}
            result = intervene_on_history(
                workspace,
                instance,
                concept=concept_dd.value,
                history_t=time_slider.value,
                **kwargs,
            )
            print_intervention_result(result)
            plot_intervention_delta(workspace, result)

    button.on_click(on_click)
    display(widgets.VBox([concept_dd, time_slider, mode, value, button, output]))


def plot_intervention_delta(
    workspace: InterventionWorkspace,
    result: Mapping[str, object],
    horizon: int | None = None,
    show: bool = True,
):
    before = result["before"]
    after = result["after"]
    horizons = sorted(before["forecasts"])
    selected_horizon = max(horizons) if horizon is None else int(horizon)
    if selected_horizon not in before["forecasts"]:
        raise ValueError(f"Forecast horizon must be one of {horizons}, got {selected_horizon}.")
    rows = [
        (
            f"+{selected_horizon}",
            before["forecasts"][selected_horizon]["probs"],
            after["forecasts"][selected_horizon]["probs"],
        )
    ]
    fig, axes = plt.subplots(1, 1, figsize=(10, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (title, p0, p1) in zip(axes, rows):
        delta = np.asarray(p1) - np.asarray(p0)
        top = np.argsort(-np.abs(delta))[:10]
        ax.barh([workspace.activity_names[i] for i in top][::-1], delta[top][::-1])
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{title} probability delta")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _prepare_preprocessed_data(config: InterventionNotebookConfig, device: torch.device) -> Dict[str, object]:
    embedding_path = config.embedding_path or default_embedding_path(config)
    if embedding_path.exists():
        with embedding_path.open("rb") as handle:
            embeddings = pickle.load(handle)
    elif config.generate_embeddings:
        generated_path = Path(
            process_dataset(
                config.dataset,
                config.backbone,
                config.window_size,
                random=config.random_windows,
                batch_size=config.embedding_batch_size,
                seed=config.seed,
                num_gpus=config.embedding_num_gpus,
            )
        )
        with generated_path.open("rb") as handle:
            embeddings = pickle.load(handle)
    else:
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    preprocessed = prepare_data(
        embeddings,
        config.concept_set,
        config.test_split,
        config.backbone,
        device,
        dataset=config.dataset,
        annotation_root=_annotation_root(config.dataset_root, config.dataset),
        binary=config.binary and not config.learn_concept_threshold,
        activity_label_mode=config.activity_label_mode,
        activity_label_fill_mode=config.activity_label_fill_mode,
    )
    metadata = preprocessed.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("video_window_spans", "video_meta"):
            value = embeddings.get(key, {}) if isinstance(embeddings, ABCMapping) else getattr(embeddings, key, {})
            if value:
                metadata[key] = value
    return preprocessed


def _annotation_root(dataset_root: Path, dataset: str) -> Path | None:
    relative_roots = {
        "breakfast": Path("Breakfast/breakfast_segmentation_coarse"),
        "gtea_gaze": Path("GTEA_Gaze/action_annotation/raw_annotations"),
        "mpii_cooking_2": Path("MPII_Cooking_2/annotations"),
        "barista": Path("Barista/labels"),
        "epic_kitchens_100": Path("EPIC-KITCHENS-100"),
    }
    relative = relative_roots.get(_canonical_dataset_key(dataset))
    return None if relative is None else Path(dataset_root).expanduser() / relative


def _standardize_preprocessed_splits(preprocessed: Dict[str, object]) -> Dict[str, Dict[str, np.ndarray]]:
    return _standardized_splits({name: preprocessed[name] for name in ("train", "val", "test")})


def _raw_preprocessed_splits(preprocessed: Dict[str, object], horizon: int) -> Dict[str, Dict[str, np.ndarray]]:
    return _prepared_splits_with_forecast(preprocessed, horizon)


def _workspace_examples(workspace: InterventionWorkspace, split: str) -> Dict[str, np.ndarray]:
    raw = _prepared_splits_with_forecast(workspace.preprocessed_data, workspace.horizon)
    standardized = _standardized_splits(raw)
    return _sliding_window_examples(
        standardized[split],
        horizon=workspace.horizon,
        history_length=workspace.history_length,
    )


def _config_from_checkpoint_args(
    config: InterventionNotebookConfig,
    args: Mapping[str, object],
) -> InterventionNotebookConfig:
    if not args:
        return config
    fields = {field.name for field in InterventionNotebookConfig.__dataclass_fields__.values()}
    keep_from_user = {"checkpoint_path", "checkpoint_selection", "device", "dataset_root"}
    updates = {}
    for key, value in args.items():
        if key in fields and key not in keep_from_user and value is not None:
            updates[key] = Path(value) if key.endswith("_dir") or key.endswith("_root") or key.endswith("_path") else value
    return replace(config, **updates)


def _resolve_checkpoint_concept_set(
    config: InterventionNotebookConfig,
    checkpoint: Mapping[str, object],
) -> InterventionNotebookConfig:
    try:
        _load_concept_set(config.concept_set)
        return config
    except FileNotFoundError:
        pass

    metadata = checkpoint.get("metadata", {})
    expected = metadata.get("concept_names") if isinstance(metadata, Mapping) else None
    if not isinstance(expected, list) or not expected:
        return config
    expected = [str(name) for name in expected]

    matches: List[Path] = []
    for candidate in sorted((PROJECT_ROOT / "concepts").glob("*.json")):
        try:
            _, names = _load_concept_set(candidate)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        if names == expected:
            matches.append(candidate)

    count_token = f"_{len(expected)}_"
    count_bearing = [path for path in matches if count_token in path.stem]
    if len(count_bearing) == 1:
        return replace(config, concept_set=count_bearing[0].stem)
    if len(matches) == 1:
        return replace(config, concept_set=matches[0].stem)
    return config


def _distribution_diagnostics(
    workspace: InterventionWorkspace,
    labels: np.ndarray,
    preds: np.ndarray,
    *,
    sil_index: int | None,
    top_k: int,
) -> Dict[str, object]:
    true_counts = np.bincount(labels, minlength=len(workspace.activity_names)).astype(np.int64)
    pred_counts = np.bincount(preds, minlength=len(workspace.activity_names)).astype(np.int64)
    true_total = max(int(labels.shape[0]), 1)
    pred_total = max(int(preds.shape[0]), 1)
    true_rate_by_label = {
        workspace.activity_names[index]: float(true_counts[index] / true_total)
        for index in range(len(workspace.activity_names))
    }
    pred_rate_by_label = {
        workspace.activity_names[index]: float(pred_counts[index] / pred_total)
        for index in range(len(workspace.activity_names))
    }
    actual_when_predicted_sil = []
    if sil_index is not None:
        selected = labels[preds == int(sil_index)]
        counts = np.bincount(selected, minlength=len(workspace.activity_names)) if selected.size else np.zeros(
            len(workspace.activity_names),
            dtype=np.int64,
        )
        for index in np.argsort(-counts)[: int(top_k)]:
            if int(counts[index]) <= 0:
                continue
            actual_when_predicted_sil.append(
                {"label": workspace.activity_names[int(index)], "count": int(counts[index])}
            )
    return {
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "true_rate_by_label": true_rate_by_label,
        "pred_rate_by_label": pred_rate_by_label,
        "actual_when_predicted_sil": actual_when_predicted_sil,
    }


def _diagnostic_bar_labels(distribution: Mapping[str, object], top_k: int) -> List[str]:
    true_rates = distribution["true_rate_by_label"]
    pred_rates = distribution["pred_rate_by_label"]
    labels = sorted(
        true_rates,
        key=lambda label: max(float(true_rates.get(label, 0.0)), float(pred_rates.get(label, 0.0))),
        reverse=True,
    )
    selected = labels[: int(top_k)]
    if "SIL" in true_rates and "SIL" not in selected:
        selected = ["SIL"] + selected[:-1]
    return selected[::-1]


def _run_metadata(config: InterventionNotebookConfig) -> Dict[str, object]:
    return {
        "dataset": config.dataset,
        "test_split": config.test_split,
        "window_size": config.window_size,
        "concept_set": config.concept_set,
        "backbone": config.backbone,
        "activity_label_mode": config.activity_label_mode,
        "activity_label_fill_mode": config.activity_label_fill_mode,
        "random_windows": config.random_windows,
    }


def _save_trained_checkpoint(
    config: InterventionNotebookConfig,
    trained: TrainedModel,
    preprocessed_data: Dict[str, object],
) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": trained.model,
        "base_method": trained.base_method,
        "horizon": trained.horizon,
        "history_length": trained.history_length,
        "activity_names": trained.activity_names,
        "num_concepts": trained.num_concepts,
        "num_activities": trained.num_activities,
        "info": trained.info,
        "metadata": preprocessed_data.get("metadata", {}),
        "args": config.__dict__,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    write_json(run_dir / "metrics.json", trained.metrics)
    write_json(run_dir / "history.json", trained.history)
    return checkpoint


def _instance_tensors(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = workspace.device
    concepts = torch.as_tensor(instance["concepts"], dtype=torch.float32, device=device).unsqueeze(0)
    key_padding_mask = torch.as_tensor(instance["key_padding_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    return concepts, key_padding_mask


def _forward(
    workspace: InterventionWorkspace,
    concepts: torch.Tensor,
    key_padding_mask: torch.Tensor,
    intervention: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if workspace.base_method == "motif":
        if intervention is not None:
            concepts = concepts.clone()
            time_idx = int(intervention["time_idx"])
            concept_idx = int(intervention["concept_idx"])
            if intervention.get("value") is not None:
                concepts[:, time_idx, concept_idx] = float(intervention["value"])
            else:
                concepts[:, time_idx, concept_idx] += float(intervention["delta"])
        return workspace.model(concepts, key_padding_mask)
    if isinstance(workspace.model, ABCMapping) and "forecast_model" in workspace.model:
        if intervention is not None:
            concepts = concepts.clone()
            items = intervention.get("items") if isinstance(intervention, ABCMapping) else None
            if items is None:
                items = [intervention]
            for item in items:
                time_idx = int(item["time_idx"])
                concept_idx = int(item["concept_idx"])
                if item.get("value") is not None:
                    concepts[:, time_idx, concept_idx] = float(item["value"])
                else:
                    concepts[:, time_idx, concept_idx] += float(item["delta"])

        forecast_model = workspace.model["forecast_model"].eval()
        current_concepts = concepts[:, -1, :]
        outputs = forecast_model(current_concepts)
        valid_mask = (~key_padding_mask).float()
        concept_view = concepts * valid_mask.unsqueeze(-1)
        if "concept_states" not in outputs:
            outputs["concept_states"] = concept_view
        if "calibrated_concepts" not in outputs:
            outputs["calibrated_concepts"] = concept_view
        return outputs
    return workspace.model(concepts, key_padding_mask, intervention=intervention)


def _forecast_logits_by_horizon(
    workspace: InterventionWorkspace,
    outputs: Mapping[str, object],
) -> Dict[int, torch.Tensor]:
    if "autoregressive_logits_by_step" in outputs:
        return {int(k): v for k, v in outputs["autoregressive_logits_by_step"].items()}
    if "forecast_logits_by_horizon" in outputs:
        return {int(k): v for k, v in outputs["forecast_logits_by_horizon"].items()}
    return {int(workspace.horizon): outputs["forecast_logits"]}


def _activity_name(workspace: InterventionWorkspace, index: int | None) -> str | None:
    if index is None:
        return None
    if 0 <= int(index) < len(workspace.activity_names):
        return workspace.activity_names[int(index)]
    return str(index)


def _concept_index(workspace: InterventionWorkspace, concept: int | str) -> int:
    if isinstance(concept, int):
        return concept
    lowered = concept.lower()
    matches = [i for i, name in enumerate(workspace.concept_names) if lowered in name.lower()]
    if not matches:
        raise ValueError(f"Unknown concept: {concept}")
    return matches[0]


def _activity_index(workspace: InterventionWorkspace, activity: int | str) -> int:
    if isinstance(activity, int):
        if 0 <= activity < len(workspace.activity_names):
            return int(activity)
        raise IndexError(f"activity index out of range: {activity}")
    try:
        return workspace.activity_names.index(str(activity))
    except ValueError as exc:
        raise ValueError(f"Unknown activity: {activity}") from exc


def _history_index(workspace: InterventionWorkspace, history_t: int) -> int:
    idx = int(history_t)
    if idx < 0:
        idx = workspace.history_length + idx
    if idx < 0 or idx >= workspace.history_length:
        raise IndexError(f"history_t must be in [0, {workspace.history_length - 1}], got {history_t}")
    return idx


def _original_timestep(instance: Mapping[str, object], history_t: int) -> int | None:
    if instance["key_padding_mask"][history_t]:
        return None
    return int(instance["history_start"] + history_t - instance["history_offset"])


def _top_history_concepts(
    workspace: InterventionWorkspace,
    instance: Mapping[str, object],
    attribution: np.ndarray,
    top_k: int,
) -> List[Dict[str, object]]:
    valid = ~np.asarray(instance["key_padding_mask"], dtype=bool)
    scores = np.abs(attribution).copy()
    scores[~valid, :] = -np.inf
    flat = np.argsort(-scores.reshape(-1))[: int(top_k)]
    rows = []
    for flat_idx in flat:
        time_idx, concept_idx = np.unravel_index(flat_idx, scores.shape)
        if not np.isfinite(scores[time_idx, concept_idx]):
            continue
        rows.append(
            {
                "history_t": int(time_idx),
                "original_t": _original_timestep(instance, int(time_idx)),
                "concept_idx": int(concept_idx),
                "concept_name": workspace.concept_names[int(concept_idx)],
                "contribution": float(attribution[time_idx, concept_idx]),
            }
        )
    return rows


def _bar_top_classes(ax, probs: np.ndarray, names: Iterable[str], title: str, top_k: int = 10) -> None:
    names = list(names)
    top = np.argsort(-np.asarray(probs))[:top_k]
    ax.barh([names[i] for i in top][::-1], np.asarray(probs)[top][::-1])
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Probability")


def _plot_st_graph_layers(
    workspace: InterventionWorkspace,
    top_k: int,
    compact: bool,
    include_cross_temporal: bool,
) -> None:
    layers = []
    model = workspace.model
    for branch_name, branch_layers in (
        ("shared", model.shared_graph_layers),
        ("window", model.window_graph_layers),
        ("forecast", model.forecast_graph_layers),
    ):
        for layer_idx, layer in enumerate(branch_layers):
            layers.append((f"{branch_name}.{layer_idx}", layer))
    if not layers:
        print("No graph layers are configured.")
        return
    if compact:
        _plot_compact_st_graphs(workspace, layers, top_k=top_k, include_cross_temporal=include_cross_temporal)
        return
    cols = min(3, len(layers))
    rows = int(np.ceil(len(layers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax, (name, layer) in zip(axes.ravel(), layers):
        matrix = layer.effective_spatial_matrix().detach().cpu().numpy()
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_title(f"{name} spatial graph")
        ax.set_xlabel("target concept")
        ax.set_ylabel("source concept")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(layers) :]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    _print_top_edges(workspace, layers, top_k=top_k, include_cross_temporal=include_cross_temporal)


def _plot_compact_st_graphs(
    workspace: InterventionWorkspace,
    layers: List[tuple[str, torch.nn.Module]],
    top_k: int,
    include_cross_temporal: bool,
) -> None:
    groups = {}
    for name, layer in layers:
        branch = name.split(".", 1)[0]
        groups.setdefault(branch, []).append(layer)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    axes = axes.ravel()
    for ax, (branch, branch_layers) in zip(axes, groups.items()):
        matrix = _mean_layer_matrix(branch_layers, "spatial")
        vmax = max(float(np.abs(matrix).max()), 1e-8)
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{branch} spatial graph")
        ax.set_xlabel("target")
        ax.set_ylabel("source")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plt.show()

    metrics = workspace.model.graph_metrics() if hasattr(workspace.model, "graph_metrics") else {}
    if metrics:
        print(
            "Graph summary: "
            f"same active={metrics.get('active_same_time_edges', 0):.0f}, "
            f"lagged active={metrics.get('active_lagged_edges', 0):.0f}, "
            f"same density={metrics.get('same_time_density', 0):.4f}, "
            f"lagged density={metrics.get('lagged_density', 0):.4f}"
        )

    for branch, branch_layers in groups.items():
        _print_top_matrix_edges(
            workspace,
            f"{branch}.spatial.mean_over_layers",
            _mean_layer_matrix(branch_layers, "spatial"),
            top_k,
        )
        if include_cross_temporal:
            cross = _mean_layer_matrix(branch_layers, "cross_temporal")
            if np.any(np.abs(cross) > 0):
                _print_top_matrix_edges(workspace, f"{branch}.cross_temporal.mean_over_layers", cross, top_k)


def _mean_layer_matrix(layers: Iterable[torch.nn.Module], kind: str) -> np.ndarray:
    matrices = []
    for layer in layers:
        if kind == "spatial":
            matrix = layer.effective_spatial_matrix()
        elif kind == "cross_temporal" and hasattr(layer, "effective_cross_temporal_matrix"):
            matrix = layer.effective_cross_temporal_matrix()
        else:
            continue
        matrices.append(matrix.detach().cpu().numpy())
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)
    return np.mean(np.stack(matrices, axis=0), axis=0)


def _plot_legacy_graph(workspace: InterventionWorkspace, top_k: int) -> None:
    model = workspace.model
    same = model.effective_same_matrix().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(same, cmap="coolwarm", aspect="auto")
    ax.set_title("Same-time concept graph")
    ax.set_xlabel("target concept")
    ax.set_ylabel("source concept")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    _print_top_matrix_edges(workspace, "same", same, top_k)


def _print_top_edges(
    workspace: InterventionWorkspace,
    layers: List[tuple[str, torch.nn.Module]],
    top_k: int,
    include_cross_temporal: bool,
) -> None:
    for name, layer in layers:
        matrix = layer.effective_spatial_matrix().detach().cpu().numpy()
        _print_top_matrix_edges(workspace, f"{name}.spatial", matrix, top_k)
        if include_cross_temporal and hasattr(layer, "effective_cross_temporal_matrix"):
            cross = layer.effective_cross_temporal_matrix().detach().cpu().numpy()
            if np.any(np.abs(cross) > 0):
                _print_top_matrix_edges(workspace, f"{name}.cross_temporal", cross, top_k)


def _print_top_matrix_edges(
    workspace: InterventionWorkspace,
    name: str,
    matrix: np.ndarray,
    top_k: int,
) -> None:
    if matrix.size == 0:
        return
    flat = np.argsort(-np.abs(matrix.reshape(-1)))[: int(top_k)]
    print(f"\nTop {min(top_k, len(flat))} edges for {name}:")
    for rank, flat_idx in enumerate(flat, 1):
        source, target = np.unravel_index(flat_idx, matrix.shape)
        weight = float(matrix[source, target])
        if abs(weight) <= 0:
            continue
        print(
            f"{rank:>2}. {workspace.concept_names[source]} -> "
            f"{workspace.concept_names[target]}: {weight:+.4f}"
        )
