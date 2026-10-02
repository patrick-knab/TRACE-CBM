from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .motif_activity_forecast import (
        MotifActivityForecastModel,
        compute_valid_mean_std,
        standardize_concepts,
    )
    from .temporal_relational_cbm import (
        ActivityAutoregressiveGraphCBM,
        GraphCBM,
    )
except ImportError:
    from motif_activity_forecast import (
        MotifActivityForecastModel,
        compute_valid_mean_std,
        standardize_concepts,
    )
    from temporal_relational_cbm import (
        ActivityAutoregressiveGraphCBM,
        GraphCBM,
    )


GRAPH_CBM_METHODS = {"trace"}


@dataclass
class TrainedModel:
    model: object
    base_method: str
    horizon: int
    history_length: int
    activity_names: List[str]
    num_concepts: int
    num_activities: int
    info: Dict[str, object]

    @property
    def history(self) -> List[Dict[str, object]]:
        return list(self.info.get("history", []))

    @property
    def metrics(self) -> Dict[str, object]:
        metrics = {
            "train": self.info.get("train_metrics", {}),
            "val": self.info.get("val_metrics", {}),
            "test": self.info.get("test_metrics", {}),
        }
        if "test_intervention_metrics" in self.info:
            metrics["test_intervention"] = self.info["test_intervention_metrics"]
        if "test_edge_guided_intervention_metrics" in self.info:
            metrics["test_edge_guided_intervention"] = self.info["test_edge_guided_intervention_metrics"]
        if "test_graph_disabled_metrics" in self.info:
            metrics["test_graph_disabled"] = self.info["test_graph_disabled_metrics"]
        if self.info.get("test_graph_corrupted_metrics"):
            metrics["test_graph_corrupted"] = self.info["test_graph_corrupted_metrics"]
        if self.info.get("test_graph_corruption_delta_metrics"):
            metrics["test_graph_corruption_delta"] = self.info["test_graph_corruption_delta_metrics"]
        if "graph_corruption" in self.info:
            metrics["graph_corruption"] = self.info["graph_corruption"]
        if "graph_metrics" in self.info:
            metrics["graph_metrics"] = self.info["graph_metrics"]
        if "synthetic_edge_recovery_metrics" in self.info:
            metrics["synthetic_edge_recovery"] = self.info["synthetic_edge_recovery_metrics"]
        if "forecast_baselines" in self.info:
            metrics["forecast_baselines"] = self.info["forecast_baselines"]
        history_rows = self.info.get("history", [])
        last_history = history_rows[-1] if isinstance(history_rows, list) and history_rows else {}
        metrics["experiment"] = {
            "forecast_rollout_mode": self.info.get("forecast_rollout_mode", "legacy"),
            "gcssbm_transition_mode": self.info.get("gcssbm_transition_mode", "none"),
            "observed_refiner_mode": self.info.get("observed_refiner_mode", "graph"),
            "topk_training_mode": self.info.get("topk_training_mode", "hard"),
            "topk_schedule_completion_epoch": int(self.info.get("topk_schedule_completion_epoch", 1)),
            "topk_minimum_training_epoch": int(self.info.get("topk_minimum_training_epoch", 1)),
            "topk_final_state": self.info.get("topk_final_state", {}),
            "forecast_horizon_loss_weights": self.info.get("forecast_horizon_loss_weights", {}),
            "forecast_transition_tolerance_radius": int(
                self.info.get("forecast_transition_tolerance_radius", 0)
            ),
            "forecast_transition_tolerance_weight": float(
                self.info.get("forecast_transition_tolerance_weight", 0.0)
            ),
            "concept_intervention_task_loss_weight": float(
                self.info.get("concept_intervention_task_loss_weight", 0.0)
            ),
            "concept_intervention_mask_ratio": float(
                self.info.get("concept_intervention_mask_ratio", 1.0)
            ),
            "graph_intervention_sample_fraction": float(
                self.info.get("graph_intervention_sample_fraction", 1.0)
            ),
            "persistent_intervention_task_loss_weight": float(
                self.info.get("persistent_intervention_task_loss_weight", 0.0)
            ),
            "graph_task_intervention_loss_weight": float(
                self.info.get("graph_task_intervention_loss_weight", 0.0)
            ),
            "graph_necessity_loss_weight": float(
                self.info.get("graph_necessity_loss_weight", 0.0)
            ),
            "edge_edit_response_loss_weight": float(
                self.info.get("edge_edit_response_loss_weight", 0.0)
            ),
            "edge_edit_response_mode": self.info.get(
                "edge_edit_response_mode", "probability_l1"
            ),
            "gcssbm_state_transition_loss_weight": float(
                self.info.get("gcssbm_state_transition_loss_weight", 0.0)
            ),
            "train_gcssbm_state_transition_loss": float(
                last_history.get("train_gcssbm_state_transition_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "train_concept_intervention_task_loss": float(
                last_history.get("train_concept_intervention_task_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "train_persistent_intervention_task_loss": float(
                last_history.get("train_persistent_intervention_task_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "train_graph_task_intervention_loss": float(
                last_history.get("train_graph_task_intervention_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "train_graph_necessity_loss": float(
                last_history.get("train_graph_necessity_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "train_edge_edit_response_loss": float(
                last_history.get("train_edge_edit_response_loss", 0.0)
                if isinstance(last_history, Mapping)
                else 0.0
            ),
            "best_epoch": int(self.info.get("best_epoch", 0)),
            "rollout_trainable_parameters": int(self.info.get("rollout_trainable_parameters", 0)),
            "total_trainable_parameters": int(self.info.get("total_trainable_parameters", 0)),
        }
        return metrics

    def __repr__(self) -> str:
        score = self.info.get("best_val_score", None)
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        loss = self.info.get("best_val_loss", None)
        loss_text = "n/a" if loss is None else f"{float(loss):.4f}"
        return (
            f"TrainedModel(method={self.base_method!r}, horizon={self.horizon}, "
            f"history_length={self.history_length}, best_val_loss={loss_text}, "
            f"best_val_score={score_text})"
        )


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiTaskLinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, horizon: int) -> None:
        super().__init__()
        self.activity_head = nn.Linear(input_dim, num_classes)
        self.forecast_heads = nn.ModuleDict(
            {str(step): nn.Linear(input_dim, num_classes) for step in range(1, int(horizon) + 1)}
        )
        self.linear = self.forecast_heads[str(int(horizon))]

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        forecast_logits = {int(step): head(x) for step, head in self.forecast_heads.items()}
        return {
            "activity_logits": self.activity_head(x),
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
        }


class LinearDynamicsSharedHeadClassifier(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, horizon: int) -> None:
        super().__init__()
        self.activity_head = nn.Linear(num_concepts, num_classes)
        self.concept_dynamics = nn.Linear(num_concepts, num_concepts)
        self.horizon = int(horizon)
        with torch.no_grad():
            self.concept_dynamics.weight.copy_(torch.eye(num_concepts))
            self.concept_dynamics.bias.zero_()

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        activity_logits = self.activity_head(x)
        current = x
        predicted_concepts_by_step = {}
        forecast_logits = {}
        for step in range(1, self.horizon + 1):
            current = self.concept_dynamics(current)
            predicted_concepts_by_step[step] = current
            forecast_logits[step] = self.activity_head(current)
        return {
            "activity_logits": activity_logits,
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
            "predicted_concepts_by_step": predicted_concepts_by_step,
        }


class LinearSparseDynamicsSharedHeadClassifier(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, horizon: int, top_k: int) -> None:
        super().__init__()
        self.activity_head = nn.Linear(num_concepts, num_classes)
        self.concept_dynamics = nn.Linear(num_concepts, num_concepts)
        self.horizon = int(horizon)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("LinearSparseDynamicsSharedHeadClassifier requires top_k >= 1.")
        with torch.no_grad():
            self.concept_dynamics.weight.copy_(0.02 * torch.randn(num_concepts, num_concepts))
            self.concept_dynamics.weight.add_(torch.eye(num_concepts))
            self.concept_dynamics.bias.zero_()

    def transition_mask(self) -> torch.Tensor:
        weight = self.concept_dynamics.weight
        keep_count = min(self.top_k, int(weight.shape[1]))
        if keep_count >= int(weight.shape[1]):
            return torch.ones_like(weight)
        keep = torch.topk(weight.abs(), k=keep_count, dim=1, largest=True).indices
        mask = torch.zeros_like(weight)
        mask.scatter_(1, keep, 1.0)
        return mask

    def active_edge_count(self) -> int:
        return int(self.transition_mask().sum().item())

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        activity_logits = self.activity_head(x)
        current = x
        predicted_concepts_by_step = {}
        forecast_logits = {}
        mask = self.transition_mask()
        sparse_weight = self.concept_dynamics.weight * mask
        for step in range(1, self.horizon + 1):
            current = F.linear(current, sparse_weight, self.concept_dynamics.bias)
            predicted_concepts_by_step[step] = current
            forecast_logits[step] = self.activity_head(current)
        return {
            "activity_logits": activity_logits,
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
            "predicted_concepts_by_step": predicted_concepts_by_step,
        }


class PlainProbeSharedHeadClassifier(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, horizon: int) -> None:
        super().__init__()
        self.activity_head = nn.Linear(num_concepts, num_classes)
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        logits = self.activity_head(x)
        forecast_logits = {step: logits for step in range(1, self.horizon + 1)}
        return {
            "activity_logits": logits,
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
        }


def _last_valid_sequence_state(states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    valid = ~key_padding_mask
    lengths = valid.long().sum(dim=1).clamp(min=1)
    gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, states.size(-1))
    return states.gather(dim=1, index=gather_index).squeeze(1)


def _masked_sequence_mean(states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~key_padding_mask).to(dtype=states.dtype).unsqueeze(-1)
    return (states * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class FeatureSlowFastTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        horizon: int,
        history_length: int,
        hidden_dim: int = 256,
        fast_layers: int = 2,
        kernel_size: int = 3,
        slow_stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        padding = max(int(kernel_size) // 2, 0)
        blocks = []
        for _ in range(max(int(fast_layers), 1)):
            blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=int(kernel_size), padding=padding),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.fast_tcn = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.activity_head = nn.Linear(hidden_dim * 2, num_classes)
        self.forecast_heads = nn.ModuleDict(
            {str(step): nn.Linear(hidden_dim * 2, num_classes) for step in range(1, self.horizon + 1)}
        )
        self.slow_stride = max(int(slow_stride), 1)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> Dict[str, object]:
        valid = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)
        states = self.input_projection(x) * valid
        fast_states = self.fast_tcn(states.transpose(1, 2)).transpose(1, 2) * valid
        fast_repr = _last_valid_sequence_state(fast_states, key_padding_mask)
        slow_states = states[:, :: self.slow_stride, :]
        slow_mask = key_padding_mask[:, :: self.slow_stride]
        slow_repr = _masked_sequence_mean(slow_states, slow_mask)
        representation = self.norm(torch.cat([fast_repr, slow_repr], dim=-1))
        forecast_logits = {int(step): head(representation) for step, head in self.forecast_heads.items()}
        return {
            "activity_logits": self.activity_head(representation),
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
        }


class FeatureTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        horizon: int,
        history_length: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.history_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activity_head = nn.Linear(hidden_dim, num_classes)
        self.forecast_heads = nn.ModuleDict(
            {str(step): nn.Linear(hidden_dim, num_classes) for step in range(1, self.horizon + 1)}
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> Dict[str, object]:
        states = self.input_projection(x) + self.position[:, : x.size(1), :]
        states = self.encoder(states, src_key_padding_mask=key_padding_mask)
        representation = self.norm(_last_valid_sequence_state(states, key_padding_mask))
        forecast_logits = {int(step): head(representation) for step, head in self.forecast_heads.items()}
        return {
            "activity_logits": self.activity_head(representation),
            "forecast_logits_by_step": forecast_logits,
            "forecast_logits": forecast_logits[max(forecast_logits)],
        }


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        mean = features.mean(axis=0).astype(np.float32)
        std = np.clip(features.std(axis=0).astype(np.float32), 1e-6, None)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean[None, :]) / self.std[None, :]).astype(np.float32)


def train_model(
    preprocessed_data: Dict[str, object],
    horizon: int,
    history_length: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    wandb_project: str | None,
    seed: int,
    base_method: str,
    *,
    device: str | torch.device | None = None,
    teacher_forcing_start_ratio: float = 1.0,
    teacher_forcing_end_ratio: float = 0.8,
    concept_forecast_loss_weight: float = 1.0,
    concept_forecast_loss_deadzone_std: float = 0.0,
    gcssbm_state_transition_loss_weight: float = 0.0,
    concept_intervention_task_loss_weight: float = 0.0,
    concept_intervention_mask_ratio: float = 1.0,
    persistent_intervention_task_loss_weight: float = 0.0,
    persistent_intervention_sample_fraction: float = 0.25,
    persistent_intervention_budgets: str | Sequence[int] = "1,3,5",
    persistent_intervention_margin: float = 0.05,
    persistent_intervention_target_mode: str = "class_prototype",
    activity_intervention_task_loss_weight: float = 0.0,
    activity_intervention_sample_fraction: float = 0.25,
    activity_intervention_budgets: str | Sequence[int] = "1,3",
    activity_intervention_margin: float = 0.05,
    forecast_horizon_loss_weights: Mapping[int | str, float] | Sequence[float] | None = None,
    forecast_transition_tolerance_radius: int = 0,
    forecast_transition_tolerance_weight: float = 0.0,
    graph_intervention_loss_weight: float = 0.0,
    graph_task_intervention_loss_weight: float = 0.0,
    graph_task_intervention_margin: float = 0.02,
    graph_intervention_margin: float = 0.01,
    graph_intervention_edges_per_batch: int = 2,
    graph_intervention_sample_fraction: float = 1.0,
    graph_edge_regularization_weight: float = 1e-5,
    graph_necessity_loss_weight: float = 0.0,
    graph_necessity_margin: float = 0.02,
    graph_necessity_edges_per_batch: int = 2,
    graph_necessity_sample_fraction: float = 0.25,
    edge_edit_response_loss_weight: float = 0.0,
    edge_edit_response_mode: str = "probability_l1",
    edge_edit_response_margin: float = 0.0005,
    edge_edit_response_edges_per_batch: int = 2,
    edge_edit_response_sample_fraction: float = 0.25,
    graph_disabled_eval: bool = False,
    graph_corruption_eval: bool = True,
    graph_corruption_seed: int | None = None,
    classifier_l1_weight: float = 0.0,
    activity_only: bool = False,
    learn_concept_threshold: bool = False,
    activity_class_weighting: bool = False,
    activity_class_weight_cap: float = 5.0,
    activity_sil_false_positive_penalty: float = 0.0,
    forecast_class_weighting: bool = False,
    forecast_class_weight_cap: float = 5.0,
    forecast_sil_false_positive_penalty: float = 0.0,
    train_sampling_strategy: str = "uniform",
    transition_sampler_boundary_radius: int = 2,
    transition_sampler_strength: float = 3.0,
    transition_sampler_rare_alpha: float = 0.5,
    early_stopping_metric: str = "val_selection_loss",
    model_hparams: Dict[str, object] | None = None,
    run_metadata: Dict[str, object] | None = None,
) -> TrainedModel:
    _seed_everything(seed)
    method = canonical_method(base_method)
    horizon = int(horizon)
    history_length = int(history_length)
    concept_forecast_loss_weight = float(concept_forecast_loss_weight)
    if concept_forecast_loss_weight < 0.0:
        raise ValueError("concept_forecast_loss_weight must be >= 0.")
    concept_forecast_loss_deadzone_std = float(concept_forecast_loss_deadzone_std)
    if concept_forecast_loss_deadzone_std < 0.0:
        raise ValueError("concept_forecast_loss_deadzone_std must be >= 0.")
    gcssbm_state_transition_loss_weight = float(gcssbm_state_transition_loss_weight)
    if gcssbm_state_transition_loss_weight < 0.0:
        raise ValueError("gcssbm_state_transition_loss_weight must be >= 0.")
    if gcssbm_state_transition_loss_weight > 0.0 and method != "gcssbm":
        raise ValueError("gcssbm_state_transition_loss_weight > 0 requires base_method=gcssbm.")
    concept_intervention_task_loss_weight = float(concept_intervention_task_loss_weight)
    if concept_intervention_task_loss_weight < 0.0:
        raise ValueError("concept_intervention_task_loss_weight must be >= 0.")
    concept_intervention_mask_ratio = float(concept_intervention_mask_ratio)
    if not 0.0 <= concept_intervention_mask_ratio <= 1.0:
        raise ValueError("concept_intervention_mask_ratio must be in [0, 1].")
    if concept_intervention_task_loss_weight > 0.0 and method not in GRAPH_CBM_METHODS:
        raise ValueError(
            "concept_intervention_task_loss_weight > 0 requires a concept-forecast base method."
        )
    persistent_intervention_task_loss_weight = float(persistent_intervention_task_loss_weight)
    persistent_intervention_sample_fraction = float(persistent_intervention_sample_fraction)
    persistent_intervention_budgets = _parse_positive_ints(persistent_intervention_budgets)
    persistent_intervention_margin = float(persistent_intervention_margin)
    if persistent_intervention_task_loss_weight < 0.0:
        raise ValueError("persistent_intervention_task_loss_weight must be >= 0.")
    if not 0.0 < persistent_intervention_sample_fraction <= 1.0:
        raise ValueError("persistent_intervention_sample_fraction must be in (0, 1].")
    if persistent_intervention_margin < 0.0:
        raise ValueError("persistent_intervention_margin must be >= 0.")
    if persistent_intervention_task_loss_weight > 0.0 and method not in GRAPH_CBM_METHODS:
        raise ValueError("persistent intervention task loss requires a concept-forecast method.")
    persistent_intervention_target_mode = str(persistent_intervention_target_mode)
    if persistent_intervention_target_mode not in {"class_prototype", "instance_oracle"}:
        raise ValueError(
            "persistent_intervention_target_mode must be class_prototype or instance_oracle."
        )
    activity_intervention_task_loss_weight = float(activity_intervention_task_loss_weight)
    activity_intervention_sample_fraction = float(activity_intervention_sample_fraction)
    activity_intervention_budgets = _parse_positive_ints(activity_intervention_budgets)
    activity_intervention_margin = float(activity_intervention_margin)
    if activity_intervention_task_loss_weight < 0.0 or activity_intervention_margin < 0.0:
        raise ValueError("activity intervention task loss weight and margin must be >= 0.")
    if not 0.0 < activity_intervention_sample_fraction <= 1.0:
        raise ValueError("activity_intervention_sample_fraction must be in (0, 1].")
    if activity_intervention_task_loss_weight > 0.0 and method not in GRAPH_CBM_METHODS:
        raise ValueError("activity intervention task loss requires a concept-forecast method.")
    forecast_horizon_loss_weights = _normalized_horizon_loss_weights(
        forecast_horizon_loss_weights,
        horizon,
    )
    forecast_transition_tolerance_radius = int(forecast_transition_tolerance_radius)
    if forecast_transition_tolerance_radius < 0:
        raise ValueError("forecast_transition_tolerance_radius must be >= 0.")
    forecast_transition_tolerance_weight = float(forecast_transition_tolerance_weight)
    if not 0.0 <= forecast_transition_tolerance_weight <= 1.0:
        raise ValueError("forecast_transition_tolerance_weight must be in [0, 1].")
    if forecast_transition_tolerance_weight > 0.0 and forecast_transition_tolerance_radius == 0:
        raise ValueError(
            "forecast_transition_tolerance_radius must be > 0 when its weight is enabled."
        )
    transition_tolerance_unsupported = {
        "linear",
        "plain_probe_shared_head",
        "linear_dynamics_shared_head",
        "linear_sparse_dynamics_shared_head",
        "feature_slowfast_tcn",
        "feature_transformer",
    }
    if (
        forecast_transition_tolerance_weight > 0.0
        and method in transition_tolerance_unsupported
    ):
        raise ValueError(
            "forecast transition tolerance currently requires a sequence forecasting method."
        )
    graph_intervention_loss_weight = float(graph_intervention_loss_weight)
    if graph_intervention_loss_weight < 0.0:
        raise ValueError("graph_intervention_loss_weight must be >= 0.")
    graph_task_intervention_loss_weight = float(graph_task_intervention_loss_weight)
    graph_task_intervention_margin = float(graph_task_intervention_margin)
    if graph_task_intervention_loss_weight < 0.0 or graph_task_intervention_margin < 0.0:
        raise ValueError("graph task intervention loss weight and margin must be >= 0.")
    graph_intervention_margin = float(graph_intervention_margin)
    if graph_intervention_margin < 0.0:
        raise ValueError("graph_intervention_margin must be >= 0.")
    graph_intervention_edges_per_batch = int(graph_intervention_edges_per_batch)
    if graph_intervention_edges_per_batch < 0:
        raise ValueError("graph_intervention_edges_per_batch must be >= 0.")
    graph_intervention_sample_fraction = float(graph_intervention_sample_fraction)
    if not 0.0 < graph_intervention_sample_fraction <= 1.0:
        raise ValueError("graph_intervention_sample_fraction must be in (0, 1].")
    graph_edge_regularization_weight = float(graph_edge_regularization_weight)
    if graph_edge_regularization_weight < 0.0:
        raise ValueError("graph_edge_regularization_weight must be >= 0.")
    graph_necessity_loss_weight = float(graph_necessity_loss_weight)
    graph_necessity_margin = float(graph_necessity_margin)
    graph_necessity_edges_per_batch = int(graph_necessity_edges_per_batch)
    graph_necessity_sample_fraction = float(graph_necessity_sample_fraction)
    if graph_necessity_loss_weight < 0.0 or graph_necessity_margin < 0.0:
        raise ValueError("graph necessity loss weight and margin must be >= 0.")
    if graph_necessity_edges_per_batch < 0:
        raise ValueError("graph_necessity_edges_per_batch must be >= 0.")
    if not 0.0 < graph_necessity_sample_fraction <= 1.0:
        raise ValueError("graph_necessity_sample_fraction must be in (0, 1].")
    edge_edit_response_loss_weight = float(edge_edit_response_loss_weight)
    edge_edit_response_mode = str(edge_edit_response_mode)
    edge_edit_response_margin = float(edge_edit_response_margin)
    edge_edit_response_edges_per_batch = int(edge_edit_response_edges_per_batch)
    edge_edit_response_sample_fraction = float(edge_edit_response_sample_fraction)
    if edge_edit_response_loss_weight < 0.0 or edge_edit_response_margin < 0.0:
        raise ValueError("edge edit response loss weight and margin must be >= 0.")
    if edge_edit_response_mode not in {"probability_l1", "oracle_margin"}:
        raise ValueError("edge_edit_response_mode must be probability_l1 or oracle_margin.")
    if edge_edit_response_edges_per_batch < 0:
        raise ValueError("edge_edit_response_edges_per_batch must be >= 0.")
    if not 0.0 < edge_edit_response_sample_fraction <= 1.0:
        raise ValueError("edge_edit_response_sample_fraction must be in (0, 1].")
    graph_corruption_seed = int(seed if graph_corruption_seed is None else graph_corruption_seed)
    classifier_l1_weight = float(classifier_l1_weight)
    if classifier_l1_weight < 0.0:
        raise ValueError("classifier_l1_weight must be >= 0.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    raw_splits = _prepared_splits_with_forecast(preprocessed_data, horizon)
    num_concepts = int(raw_splits["train"]["concepts"].shape[-1])
    num_raw_features = int(raw_splits["train"].get("raw_features", raw_splits["train"]["concepts"]).shape[-1])
    num_activities = _num_activities(preprocessed_data, raw_splits)
    metadata = preprocessed_data.get("metadata", {})
    activity_names = list(metadata.get("activity_names", []))
    sil_index = activity_names.index("SIL") if "SIL" in activity_names else None
    activity_components = _activity_component_values(metadata, num_activities)
    concept_activation_mode = str(metadata.get("concept_activation_mode", "continuous"))
    model_concept_activation = "learned_threshold" if learn_concept_threshold else "affine"
    wandb_config = {
        "base_method": method,
        "horizon": horizon,
        "history_length": history_length,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "patience": int(patience),
        "seed": int(seed),
        "device": str(device),
        "concept_activation_mode": concept_activation_mode,
        "binary_concepts": bool(metadata.get("binary_concepts", False)),
        "learn_concept_threshold": bool(learn_concept_threshold),
        "model_concept_activation": model_concept_activation,
        "teacher_forcing_start_ratio": float(teacher_forcing_start_ratio),
        "teacher_forcing_end_ratio": float(teacher_forcing_end_ratio),
        "concept_forecast_loss_weight": float(concept_forecast_loss_weight),
        "concept_forecast_loss_deadzone_std": float(concept_forecast_loss_deadzone_std),
        "gcssbm_state_transition_loss_weight": float(gcssbm_state_transition_loss_weight),
        "concept_intervention_task_loss_weight": float(concept_intervention_task_loss_weight),
        "concept_intervention_mask_ratio": float(concept_intervention_mask_ratio),
        "persistent_intervention_task_loss_weight": float(persistent_intervention_task_loss_weight),
        "persistent_intervention_sample_fraction": float(persistent_intervention_sample_fraction),
        "persistent_intervention_budgets": list(persistent_intervention_budgets),
        "persistent_intervention_margin": float(persistent_intervention_margin),
        "persistent_intervention_target_mode": persistent_intervention_target_mode,
        "activity_intervention_task_loss_weight": float(activity_intervention_task_loss_weight),
        "activity_intervention_sample_fraction": float(activity_intervention_sample_fraction),
        "activity_intervention_budgets": list(activity_intervention_budgets),
        "activity_intervention_margin": float(activity_intervention_margin),
        "forecast_horizon_loss_weights": dict(forecast_horizon_loss_weights or {}),
        "forecast_transition_tolerance_radius": int(forecast_transition_tolerance_radius),
        "forecast_transition_tolerance_weight": float(forecast_transition_tolerance_weight),
        "graph_intervention_loss_weight": float(graph_intervention_loss_weight),
        "graph_task_intervention_loss_weight": float(graph_task_intervention_loss_weight),
        "graph_task_intervention_margin": float(graph_task_intervention_margin),
        "graph_intervention_margin": float(graph_intervention_margin),
        "graph_intervention_edges_per_batch": int(graph_intervention_edges_per_batch),
        "graph_intervention_sample_fraction": float(graph_intervention_sample_fraction),
        "graph_edge_regularization_weight": float(graph_edge_regularization_weight),
        "graph_necessity_loss_weight": float(graph_necessity_loss_weight),
        "graph_necessity_margin": float(graph_necessity_margin),
        "graph_necessity_edges_per_batch": int(graph_necessity_edges_per_batch),
        "graph_necessity_sample_fraction": float(graph_necessity_sample_fraction),
        "edge_edit_response_loss_weight": float(edge_edit_response_loss_weight),
        "edge_edit_response_mode": edge_edit_response_mode,
        "edge_edit_response_margin": float(edge_edit_response_margin),
        "edge_edit_response_edges_per_batch": int(edge_edit_response_edges_per_batch),
        "edge_edit_response_sample_fraction": float(edge_edit_response_sample_fraction),
        "graph_disabled_eval": bool(graph_disabled_eval),
        "graph_corruption_eval": bool(graph_corruption_eval),
        "graph_corruption_seed": int(graph_corruption_seed),
        "classifier_l1_weight": float(classifier_l1_weight),
        "activity_only": bool(activity_only),
        "activity_class_weighting": bool(activity_class_weighting),
        "activity_class_weight_cap": float(activity_class_weight_cap),
        "activity_sil_false_positive_penalty": float(activity_sil_false_positive_penalty),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "train_sampling_strategy": str(train_sampling_strategy),
        "transition_sampler_boundary_radius": int(transition_sampler_boundary_radius),
        "transition_sampler_strength": float(transition_sampler_strength),
        "transition_sampler_rare_alpha": float(transition_sampler_rare_alpha),
        "early_stopping_metric": str(early_stopping_metric),
        "model_hparams": dict(model_hparams or {}),
    }
    if activity_components:
        wandb_config["activity_component_names"] = sorted(activity_components)
    wandb_config.update(dict(run_metadata or {}))

    run = _maybe_init_wandb(
        wandb_project,
        wandb_config,
    )

    if method == "linear":
        model, info = _train_linear_multitask(
            raw_splits,
            horizon=horizon,
            history_length=history_length,
            num_activities=num_activities,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            seed=seed,
            device=device,
            wandb_run=run,
            activity_class_weighting=activity_class_weighting,
            activity_class_weight_cap=activity_class_weight_cap,
            activity_sil_false_positive_penalty=activity_sil_false_positive_penalty,
            forecast_class_weighting=forecast_class_weighting,
            forecast_class_weight_cap=forecast_class_weight_cap,
            forecast_sil_false_positive_penalty=forecast_sil_false_positive_penalty,
            classifier_l1_weight=classifier_l1_weight,
            early_stopping_metric=early_stopping_metric,
            sil_index=sil_index,
            transition_sampler_boundary_radius=transition_sampler_boundary_radius,
            activity_components=activity_components,
        )
    elif method in {"plain_probe_shared_head", "linear_dynamics_shared_head", "linear_sparse_dynamics_shared_head"}:
        model, info = _train_linear_dynamics_shared_head(
            raw_splits,
            horizon=horizon,
            history_length=history_length,
            num_concepts=num_concepts,
            num_activities=num_activities,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            seed=seed,
            device=device,
            wandb_run=run,
            activity_class_weighting=activity_class_weighting,
            activity_class_weight_cap=activity_class_weight_cap,
            activity_sil_false_positive_penalty=activity_sil_false_positive_penalty,
            forecast_class_weighting=forecast_class_weighting,
            forecast_class_weight_cap=forecast_class_weight_cap,
            forecast_sil_false_positive_penalty=forecast_sil_false_positive_penalty,
            concept_forecast_loss_weight=concept_forecast_loss_weight,
            concept_forecast_loss_deadzone_std=concept_forecast_loss_deadzone_std,
            classifier_l1_weight=classifier_l1_weight,
            early_stopping_metric=early_stopping_metric,
            sil_index=sil_index,
            transition_sampler_boundary_radius=transition_sampler_boundary_radius,
            use_concept_dynamics=method == "linear_dynamics_shared_head",
            use_sparse_concept_dynamics=method == "linear_sparse_dynamics_shared_head",
            sparse_top_k=int((model_hparams or {}).get("st_spatial_top_k", 0)),
            activity_components=activity_components,
        )
    elif method in {"feature_slowfast_tcn", "feature_transformer"}:
        model, info = _train_feature_sequence_multitask(
            raw_splits,
            method=method,
            horizon=horizon,
            history_length=history_length,
            input_dim=num_raw_features,
            num_activities=num_activities,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            seed=seed,
            device=device,
            wandb_run=run,
            activity_class_weighting=activity_class_weighting,
            activity_class_weight_cap=activity_class_weight_cap,
            activity_sil_false_positive_penalty=activity_sil_false_positive_penalty,
            forecast_class_weighting=forecast_class_weighting,
            forecast_class_weight_cap=forecast_class_weight_cap,
            forecast_sil_false_positive_penalty=forecast_sil_false_positive_penalty,
            classifier_l1_weight=classifier_l1_weight,
            early_stopping_metric=early_stopping_metric,
            sil_index=sil_index,
            transition_sampler_boundary_radius=transition_sampler_boundary_radius,
            train_sampling_strategy=train_sampling_strategy,
            transition_sampler_strength=transition_sampler_strength,
            transition_sampler_rare_alpha=transition_sampler_rare_alpha,
            model_hparams=model_hparams,
            activity_components=activity_components,
        )
    else:
        splits = _standardized_splits(raw_splits)
        model = _build_sequence_model(
            method=method,
            num_concepts=num_concepts,
            num_activities=num_activities,
            horizon=horizon,
            history_length=history_length,
            max_sequence_length=history_length,
            concept_activation=model_concept_activation,
            model_hparams=model_hparams,
        ).to(device)
        info = _train_sequence_model(
            model=model,
            splits=splits,
            metadata=metadata,
            method=method,
            horizon=horizon,
            num_concepts=num_concepts,
            num_activities=num_activities,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
            wandb_run=run,
            teacher_forcing_start_ratio=teacher_forcing_start_ratio,
            teacher_forcing_end_ratio=teacher_forcing_end_ratio,
            concept_forecast_loss_weight=concept_forecast_loss_weight,
            concept_forecast_loss_deadzone_std=concept_forecast_loss_deadzone_std,
            gcssbm_state_transition_loss_weight=gcssbm_state_transition_loss_weight,
            concept_intervention_task_loss_weight=concept_intervention_task_loss_weight,
            concept_intervention_mask_ratio=concept_intervention_mask_ratio,
            persistent_intervention_task_loss_weight=persistent_intervention_task_loss_weight,
            persistent_intervention_sample_fraction=persistent_intervention_sample_fraction,
            persistent_intervention_budgets=persistent_intervention_budgets,
            persistent_intervention_margin=persistent_intervention_margin,
            persistent_intervention_target_mode=persistent_intervention_target_mode,
            activity_intervention_task_loss_weight=activity_intervention_task_loss_weight,
            activity_intervention_sample_fraction=activity_intervention_sample_fraction,
            activity_intervention_budgets=activity_intervention_budgets,
            activity_intervention_margin=activity_intervention_margin,
            forecast_horizon_loss_weights=forecast_horizon_loss_weights,
            forecast_transition_tolerance_radius=forecast_transition_tolerance_radius,
            forecast_transition_tolerance_weight=forecast_transition_tolerance_weight,
            graph_intervention_loss_weight=graph_intervention_loss_weight,
            graph_task_intervention_loss_weight=graph_task_intervention_loss_weight,
            graph_task_intervention_margin=graph_task_intervention_margin,
            graph_intervention_margin=graph_intervention_margin,
            graph_intervention_edges_per_batch=graph_intervention_edges_per_batch,
            graph_intervention_sample_fraction=graph_intervention_sample_fraction,
            graph_edge_regularization_weight=graph_edge_regularization_weight,
            graph_necessity_loss_weight=graph_necessity_loss_weight,
            graph_necessity_margin=graph_necessity_margin,
            graph_necessity_edges_per_batch=graph_necessity_edges_per_batch,
            graph_necessity_sample_fraction=graph_necessity_sample_fraction,
            edge_edit_response_loss_weight=edge_edit_response_loss_weight,
            edge_edit_response_mode=edge_edit_response_mode,
            edge_edit_response_margin=edge_edit_response_margin,
            edge_edit_response_edges_per_batch=edge_edit_response_edges_per_batch,
            edge_edit_response_sample_fraction=edge_edit_response_sample_fraction,
            graph_disabled_eval=graph_disabled_eval,
            graph_corruption_eval=graph_corruption_eval,
            graph_corruption_seed=graph_corruption_seed,
            classifier_l1_weight=classifier_l1_weight,
            activity_only=activity_only,
            activity_class_weighting=activity_class_weighting,
            activity_class_weight_cap=activity_class_weight_cap,
            activity_sil_false_positive_penalty=activity_sil_false_positive_penalty,
            forecast_class_weighting=forecast_class_weighting,
            forecast_class_weight_cap=forecast_class_weight_cap,
            forecast_sil_false_positive_penalty=forecast_sil_false_positive_penalty,
            train_sampling_strategy=train_sampling_strategy,
            transition_sampler_boundary_radius=transition_sampler_boundary_radius,
            transition_sampler_strength=transition_sampler_strength,
            transition_sampler_rare_alpha=transition_sampler_rare_alpha,
            early_stopping_metric=early_stopping_metric,
            sil_index=sil_index,
            activity_components=activity_components,
            model_hparams=model_hparams or {},
        )
    if "graph_corruption" not in info:
        info["graph_corruption"] = _graph_corruption_metadata(
            enabled=bool(graph_corruption_eval),
            seed=graph_corruption_seed,
            applied=False,
            skip_reason="disabled" if not bool(graph_corruption_eval) else "non_graph_method",
        )

    info["concept_activation_mode"] = concept_activation_mode
    info["binary_concepts"] = bool(metadata.get("binary_concepts", False))
    info["learn_concept_threshold"] = bool(learn_concept_threshold)
    info["model_concept_activation"] = model_concept_activation
    info["classifier_l1_weight"] = float(classifier_l1_weight)
    info["activity_only"] = bool(activity_only)

    if run is not None:
        if "best_val_loss" in info:
            run.summary["best_val_loss"] = float(info["best_val_loss"])
        run.summary["best_val_score"] = float(info.get("best_val_score", 0.0))
        if "best_epoch" in info:
            run.summary["best_epoch"] = int(info["best_epoch"])
        if "best_selection_metric" in info:
            run.summary["best_selection_metric"] = str(info["best_selection_metric"])
        if "best_selection_value" in info:
            run.summary["best_selection_value"] = float(info["best_selection_value"])
        summary_metrics = _wandb_summary_metrics(info)
        for key, value in summary_metrics.items():
            run.summary[key] = value
            run.summary[f"selected_checkpoint/{key}"] = value
        run.finish()

    return TrainedModel(
        model=model,
        base_method=method,
        horizon=horizon,
        history_length=history_length,
        activity_names=activity_names,
        num_concepts=num_concepts,
        num_activities=num_activities,
        info=info,
    )


def canonical_method(base_method: str) -> str:
    """Return the public canonical name while accepting historical inputs."""

    key = str(base_method).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "linear": "linear",
        "linear_model": "linear",
        "linear_head": "linear",
        "plain_probe_shared_head": "plain_probe_shared_head",
        "plain_probe": "plain_probe_shared_head",
        "current_concept_probe": "plain_probe_shared_head",
        "current_probe_shared_head": "plain_probe_shared_head",
        "linear_dynamics_shared_head": "linear_dynamics_shared_head",
        "linear_dynamics": "linear_dynamics_shared_head",
        "shared_head_linear_dynamics": "linear_dynamics_shared_head",
        "linear_sparse_dynamics_shared_head": "linear_sparse_dynamics_shared_head",
        "linear_sparse_dynamics": "linear_sparse_dynamics_shared_head",
        "shared_head_linear_sparse_dynamics": "linear_sparse_dynamics_shared_head",
        "feature_slowfast_tcn": "feature_slowfast_tcn",
        "slowfast_tcn": "feature_slowfast_tcn",
        "feature_slowfast": "feature_slowfast_tcn",
        "feature_transformer": "feature_transformer",
        "blackbox_transformer": "feature_transformer",
        "feature_timesformer": "feature_transformer",
        "motif": "motif",
        "motif_standalone": "motif",
        "standalone_motif": "motif",
        "activity_autoregressive_graph_cbm": "activity_autoregressive_graph_cbm",
        "st_cg_cbm": "activity_autoregressive_graph_cbm",
        "st_motif": "activity_autoregressive_graph_cbm",
        "motif_st": "activity_autoregressive_graph_cbm",
        "st_motif_cbm": "activity_autoregressive_graph_cbm",
        # Legacy method IDs normalize to TRACE while preserving checkpoint compatibility.
        "graph_cbm": "trace",
        "trace": "trace",
        "st_cg_cbm_concept_ar": "trace",
        "concept_ar": "trace",
        "concept_forecast_cbm": "trace",
    }
    if key not in aliases:
        raise ValueError(
            "base_method must be one of: linear, plain_probe_shared_head, "
            "linear_dynamics_shared_head, linear_sparse_dynamics_shared_head, feature_slowfast_tcn, "
            "feature_transformer, motif, activity_autoregressive_graph_cbm, trace"
        )
    return aliases[key]


def _parse_positive_ints(values: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(int(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(int(value) for value in values)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("Intervention budgets must be positive integers.")
    return tuple(sorted(set(parsed)))


def _normalized_horizon_loss_weights(
    values: Mapping[int | str, float] | Sequence[float] | None,
    horizon: int,
) -> Dict[int, float] | None:
    if values is None:
        return None
    horizon = int(horizon)
    if isinstance(values, Mapping):
        parsed = {int(step): float(weight) for step, weight in values.items()}
        expected = set(range(1, horizon + 1))
        if set(parsed) != expected:
            raise ValueError(f"forecast_horizon_loss_weights must define exactly horizons 1..{horizon}.")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != horizon:
            raise ValueError(f"forecast_horizon_loss_weights must contain {horizon} values.")
        parsed = {step: float(weight) for step, weight in enumerate(values, start=1)}
    else:
        raise TypeError("forecast_horizon_loss_weights must be a mapping, sequence, or None.")
    if any(weight < 0.0 for weight in parsed.values()):
        raise ValueError("forecast_horizon_loss_weights must be non-negative.")
    total = sum(parsed.values())
    if total <= 0.0:
        raise ValueError("forecast_horizon_loss_weights must have a positive sum.")
    return {step: weight / total for step, weight in parsed.items()}


def _classifier_l1_penalty(model: nn.Module) -> torch.Tensor:
    heads: List[nn.Module] = []
    if hasattr(model, "activity_head"):
        heads.append(model.activity_head)
    if hasattr(model, "linear"):
        heads.append(model.linear)
    if hasattr(model, "forecast_heads"):
        heads.extend(getattr(model, "forecast_heads").values())
    if not heads:
        return next(model.parameters()).new_tensor(0.0)
    seen: set[int] = set()
    penalty = next(model.parameters()).new_tensor(0.0)
    for head in heads:
        weight = getattr(head, "weight", None)
        if not torch.is_tensor(weight) or id(weight) in seen:
            continue
        seen.add(id(weight))
        penalty = penalty + weight.abs().sum()
    return penalty


def forecast_key(horizon: int) -> str:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return f"forecast_labels_h{horizon}"


def _prepared_splits_with_forecast(
    preprocessed_data: Dict[str, object],
    horizon: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        name: _with_forecast_label(preprocessed_data[name], horizon)
        for name in ("train", "val", "test")
    }


def _with_forecast_label(split: Dict[str, np.ndarray], horizon: int) -> Dict[str, np.ndarray]:
    result = {key: np.asarray(value) for key, value in split.items()}
    key = forecast_key(horizon)
    if key in result:
        return result
    labels = np.asarray(result["activity_labels"], dtype=np.int64)
    forecast = np.full_like(labels, -1)
    if horizon < labels.shape[1]:
        target = forecast[:, :-horizon]
        source = labels[:, horizon:]
        valid = source >= 0
        target[valid] = source[valid]
    result[key] = forecast
    return result


def _standardized_splits(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    mean, std = compute_valid_mean_std(splits["train"]["concepts"], splits["train"]["mask"])
    return {
        name: {
            **split,
            "concepts_std": standardize_concepts(split["concepts"], split["mask"], mean, std),
        }
        for name, split in splits.items()
    }


def _standardized_raw_feature_splits(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    if any("raw_features" not in split for split in splits.values()):
        raise ValueError("Feature black-box baselines require raw_features in every split.")
    train_features = np.asarray(splits["train"]["raw_features"], dtype=np.float32)
    mean, std = compute_valid_mean_std(train_features, splits["train"]["mask"])
    standardized = {}
    for name, split in splits.items():
        raw_features = np.asarray(split["raw_features"], dtype=np.float32)
        standardized[name] = {
            **split,
            "raw_features_std": standardize_concepts(raw_features, split["mask"], mean, std),
        }
    return standardized


def _num_activities(
    preprocessed_data: Dict[str, object],
    splits: Dict[str, Dict[str, np.ndarray]],
) -> int:
    metadata = preprocessed_data.get("metadata", {})
    if "num_activities" in metadata:
        return int(metadata["num_activities"])
    return max(int(split["activity_labels"].max()) for split in splits.values()) + 1


def _activity_component_values(
    metadata: Mapping[str, object],
    num_activities: int,
) -> Dict[str, np.ndarray]:
    components = metadata.get("activity_label_components") if isinstance(metadata, Mapping) else None
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return {}
    if len(components) < int(num_activities):
        return {}
    names = metadata.get("activity_component_names") if isinstance(metadata, Mapping) else None
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        component_names = [str(name) for name in names]
    else:
        component_names = sorted(
            {
                str(key)
                for item in components[: int(num_activities)]
                if isinstance(item, Mapping)
                for key in item
            }
        )
    values: Dict[str, np.ndarray] = {}
    for name in component_names:
        series = []
        for item in components[: int(num_activities)]:
            if not isinstance(item, Mapping):
                return {}
            series.append(str(item.get(name, "")))
        if any(value for value in series):
            values[_metric_name_fragment(name)] = np.asarray(series, dtype=object)
    return values


def _metric_name_fragment(value: object) -> str:
    fragment = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    fragment = "_".join(part for part in fragment.split("_") if part)
    return fragment or "component"


def _component_accuracy_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    activity_components: Mapping[str, np.ndarray] | None,
) -> Dict[str, float]:
    if not activity_components:
        return {}
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    metrics: Dict[str, float] = {}
    for name, values in activity_components.items():
        values = np.asarray(values, dtype=object)
        valid = (
            (labels >= 0)
            & (preds >= 0)
            & (labels < values.shape[0])
            & (preds < values.shape[0])
        )
        if not np.any(valid):
            continue
        true_values = values[labels[valid]]
        pred_values = values[preds[valid]]
        present = (true_values != "") & (pred_values != "")
        if not np.any(present):
            continue
        metrics[f"{name}_accuracy"] = float(np.mean(true_values[present] == pred_values[present]))
        metrics[f"{name}_num_examples"] = int(np.sum(present))
    return metrics


def _build_sequence_model(
    *,
    method: str,
    num_concepts: int,
    num_activities: int,
    horizon: int,
    history_length: int,
    max_sequence_length: int,
    concept_activation: str = "affine",
    model_hparams: Dict[str, object] | None = None,
) -> nn.Module:
    overrides = dict(model_hparams or {})
    if method == "motif":
        params = _merge_checked_hparams(
            {
                "transformer_layers": 2,
                "dropout": 0.1,
                "dimension": 4,
                "shared_activity_head": False,
            },
            overrides,
            "motif",
        )
        return MotifActivityForecastModel(
            num_concepts=num_concepts,
            num_activities=num_activities,
            history_length=history_length,
            transformer_layers=int(params["transformer_layers"]),
            dropout=float(params["dropout"]),
            dimension=int(params["dimension"]),
            max_sequence_length=max_sequence_length,
            shared_activity_head=bool(params["shared_activity_head"]),
        )
    # Hyperparameters:
    # - edge_threshold: gate value above which an edge counts as active in graph metrics.
    # - edge_gate_init: initial logit for learned concept graph edge gates; negative starts sparse.
    # - forecast/activity_context="flat": use the ordered causal history, not a pooled mean.
    # - concept_activation: calibrate standardized concept scores; "learned_threshold"
    #   gives differentiable per-concept soft binary activations.
    # - representation_mode="s_only": predict from graph-refined concept states s_t by default.
    # - future_concept_horizons: optional auxiliary future-concept prediction heads; off here.
    # - motif_z_attention_*: causal per-concept temporal attention before the ST graph.
    # - st_graph_layers: shared spatio-temporal concept graph depth.
    # - st_task_graph_layers: task-specific graph depth for activity/forecast branches.
    # - st_spatial_top_k: keep top-k spatial concept neighbors per target in soft graph construction.
    # - st_spatial_soft_threshold: optional soft threshold for spatial edges; 0 disables thresholding.
    # - st_enable_spatial: disable within-window spatial messages for one-hot relation-family ablations.
    # - st_temporal_top_k: keep top-k same-concept temporal channels globally; 0 disables pruning.
    # - st_temporal_soft_threshold: optional soft threshold for same-concept temporal edges.
    # - st_enable_same_concept_temporal: disable same-concept temporal memory for strict cross-edge tests.
    # - st_residual_gate_init: initial logit for residual graph blending; negative favors identity early.
    # - st_enable_cross_temporal: allow temporal messages across windows inside the graph layers.
    # - st_cross_temporal_top_k: keep top-k cross-temporal source concepts per target; 0 disables pruning.
    # - st_cross_temporal_soft_threshold: optional soft threshold for cross-temporal edges.
    # - st_message_scale: multiplies graph messages before the residual state update.
    # - st_video_pooling="none": do not add a video-level pooled prediction head.
    # - st_state_activation="identity": keep legacy unbounded graph states; "bounded_logit" keeps states in [0, 1].
    # - st_prediction_transform="identity": feed graph states directly to heads; "logit" feeds bounded
    #   states as evidence while keeping the exported concept states probability-like.
    # - st_past_context_*: opt-in long-past context experiments before the ST graph; "none" preserves baseline.
    # - st_memory_prefix_length: older observed prefix length used only by prefix_* past-context modes.
    # - st_observed_refiner_mode: graph (default) or an exactly capacity-matched dense refiner.
    # - st_topk_training_mode: hard, none, gradual, or soft_train_hard_eval.
    # - st_activity_feedback_*: optional sparse activity-label -> next-concept
    #   feedback for TRACE; disabled leaves the graph rollout unchanged.
    params = _merge_checked_hparams(
        {
            "edge_threshold": 0.2,
            "edge_gate_init": -1.0,
            "forecast_context": "flat",
            "activity_context": "flat",
            "representation_mode": "s_only",
            "future_concept_horizons": [],
            "motif_z_attention_layers": 1,
            "motif_z_attention_width": 1,
            "motif_z_attention_dropout": 0.1,
            "motif_z_attention_gate_init": -2.0,
            "st_graph_layers": 2,
            "st_task_graph_layers": 1,
            "st_spatial_top_k": 20,
            "st_spatial_soft_threshold": 0.0,
            "st_enable_spatial": True,
            "st_temporal_top_k": 0,
            "st_temporal_soft_threshold": 0.0,
            "st_enable_same_concept_temporal": True,
            "st_residual_gate_init": -2.0,
            "st_enable_cross_temporal": True,
            "st_cross_temporal_top_k": 0,
            "st_cross_temporal_soft_threshold": 0.0,
            "st_message_scale": 1.0,
            "st_video_pooling": "none",
            "st_state_activation": "identity",
            "st_prediction_transform": "identity",
            "st_past_context_mode": "none",
            "st_past_context_gate_init": -2.0,
            "st_memory_prefix_length": 0,
            "st_summary_short_alpha": 0.5,
            "st_summary_long_alpha": 0.1,
            "st_summary_seen_threshold": 0.5,
            "st_forecast_rollout_mode": "legacy",
            "st_controlled_dense_reuse_forecast_layer": False,
            "st_observed_refiner_mode": "graph",
            "st_topk_training_mode": "hard",
            "st_topk_warmup_epochs": 20,
            "st_topk_ramp_epochs": 30,
            "st_activity_feedback_mode": "none",
            "st_activity_feedback_top_k": 10,
            "st_activity_feedback_gate_init": -2.0,
            "st_activity_feedback_history_steps": 0,
            "st_activity_feedback_history_gate_init": 0.0,
            "st_activity_feedback_hidden_dim": 64,
            "st_activity_feedback_ridge_lambda": 0.1,
            "st_activity_feedback_probability_eps": 1e-4,
            "st_activity_feedback_prototype_smoothing": 1.0,
            "transition_prior_enabled": False,
            "transition_prior_weight": 0.0,
            "transition_prior_smoothing": 1.0,
            "transition_prior_detach_activity": True,
            "use_concept_calibrator": True,
        },
        overrides,
        method,
    )
    if method in GRAPH_CBM_METHODS:
        model_cls = GraphCBM
    else:
        model_cls = ActivityAutoregressiveGraphCBM
    model_kwargs = dict(
        num_concepts=num_concepts,
        num_activities=num_activities,
        history_length=history_length,
        forecast_horizons=[horizon],
        edge_threshold=float(params["edge_threshold"]),
        edge_gate_init=float(params["edge_gate_init"]),
        forecast_context=str(params["forecast_context"]),
        activity_context=str(params["activity_context"]),
        concept_activation=concept_activation,
        representation_mode=str(params["representation_mode"]),
        future_concept_horizons=[int(value) for value in params["future_concept_horizons"]],
        motif_z_attention_layers=int(params["motif_z_attention_layers"]),
        motif_z_attention_width=int(params["motif_z_attention_width"]),
        motif_z_attention_dropout=float(params["motif_z_attention_dropout"]),
        motif_z_attention_gate_init=float(params["motif_z_attention_gate_init"]),
        st_graph_layers=int(params["st_graph_layers"]),
        st_task_graph_layers=int(params["st_task_graph_layers"]),
        st_spatial_top_k=int(params["st_spatial_top_k"]),
        st_spatial_soft_threshold=float(params["st_spatial_soft_threshold"]),
        st_enable_spatial=bool(params["st_enable_spatial"]),
        st_temporal_top_k=int(params["st_temporal_top_k"]),
        st_temporal_soft_threshold=float(params["st_temporal_soft_threshold"]),
        st_enable_same_concept_temporal=bool(params["st_enable_same_concept_temporal"]),
        st_residual_gate_init=float(params["st_residual_gate_init"]),
        st_enable_cross_temporal=bool(params["st_enable_cross_temporal"]),
        st_cross_temporal_top_k=int(params["st_cross_temporal_top_k"]),
        st_cross_temporal_soft_threshold=float(params["st_cross_temporal_soft_threshold"]),
        st_message_scale=float(params["st_message_scale"]),
        st_video_pooling=str(params["st_video_pooling"]),
        st_state_activation=str(params["st_state_activation"]),
        st_prediction_transform=str(params["st_prediction_transform"]),
        st_past_context_mode=str(params["st_past_context_mode"]),
        st_past_context_gate_init=float(params["st_past_context_gate_init"]),
        st_memory_prefix_length=int(params["st_memory_prefix_length"]),
        st_summary_short_alpha=float(params["st_summary_short_alpha"]),
        st_summary_long_alpha=float(params["st_summary_long_alpha"]),
        st_summary_seen_threshold=float(params["st_summary_seen_threshold"]),
        st_forecast_rollout_mode=str(params["st_forecast_rollout_mode"]),
        st_controlled_dense_reuse_forecast_layer=bool(
            params["st_controlled_dense_reuse_forecast_layer"]
        ),
        st_observed_refiner_mode=str(params["st_observed_refiner_mode"]),
        st_topk_training_mode=str(params["st_topk_training_mode"]),
        st_topk_warmup_epochs=int(params["st_topk_warmup_epochs"]),
        st_topk_ramp_epochs=int(params["st_topk_ramp_epochs"]),
        use_concept_calibrator=bool(params["use_concept_calibrator"]),
    )
    if method in GRAPH_CBM_METHODS:
        model_kwargs.update(
            st_activity_feedback_mode=str(params["st_activity_feedback_mode"]),
            st_activity_feedback_top_k=int(params["st_activity_feedback_top_k"]),
            st_activity_feedback_gate_init=float(params["st_activity_feedback_gate_init"]),
            st_activity_feedback_history_steps=int(params["st_activity_feedback_history_steps"]),
            st_activity_feedback_history_gate_init=float(
                params["st_activity_feedback_history_gate_init"]
            ),
            st_activity_feedback_hidden_dim=int(
                params["st_activity_feedback_hidden_dim"]
            ),
            st_activity_feedback_ridge_lambda=float(
                params["st_activity_feedback_ridge_lambda"]
            ),
            st_activity_feedback_probability_eps=float(
                params["st_activity_feedback_probability_eps"]
            ),
            st_activity_feedback_prototype_smoothing=float(
                params["st_activity_feedback_prototype_smoothing"]
            ),
        )
    return model_cls(**model_kwargs)


def _merge_checked_hparams(defaults: Dict[str, object], overrides: Dict[str, object], model_name: str) -> Dict[str, object]:
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        valid = ", ".join(sorted(defaults))
        raise ValueError(f"Unknown {model_name} model_hparams: {unknown}. Valid keys: {valid}")
    return {**defaults, **overrides}


def _build_feature_sequence_model(
    *,
    method: str,
    input_dim: int,
    num_activities: int,
    horizon: int,
    history_length: int,
    model_hparams: Dict[str, object] | None = None,
) -> nn.Module:
    overrides = dict(model_hparams or {})
    if method == "feature_slowfast_tcn":
        params = _merge_checked_hparams(
            {
                "hidden_dim": 256,
                "fast_layers": 2,
                "kernel_size": 3,
                "slow_stride": 2,
                "dropout": 0.1,
            },
            overrides,
            method,
        )
        return FeatureSlowFastTCNClassifier(
            input_dim=input_dim,
            num_classes=num_activities,
            horizon=horizon,
            history_length=history_length,
            hidden_dim=int(params["hidden_dim"]),
            fast_layers=int(params["fast_layers"]),
            kernel_size=int(params["kernel_size"]),
            slow_stride=int(params["slow_stride"]),
            dropout=float(params["dropout"]),
        )
    if method == "feature_transformer":
        params = _merge_checked_hparams(
            {
                "hidden_dim": 256,
                "num_layers": 2,
                "num_heads": 4,
                "feedforward_dim": 512,
                "dropout": 0.1,
            },
            overrides,
            method,
        )
        return FeatureTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_activities,
            horizon=horizon,
            history_length=history_length,
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            num_heads=int(params["num_heads"]),
            feedforward_dim=int(params["feedforward_dim"]),
            dropout=float(params["dropout"]),
        )
    raise ValueError(f"Unsupported feature sequence method: {method}")


def _selection_metric_value(row: Dict[str, object], metric: str) -> Tuple[float, str, str]:
    aliases = {
        "loss": "val_selection_loss",
        "val_loss": "val_selection_loss",
        "val_selection": "val_selection_loss",
        "score": "val_score",
        "val": "val_score",
        "val_top3": "val_forecast_top3_accuracy",
        "forecast_top3": "val_forecast_top3_accuracy",
        "top3": "val_forecast_top3_accuracy",
        "activity_top3": "val_activity_top3_accuracy",
        "accuracy": "val_activity_forecast_accuracy",
        "val_accuracy": "val_activity_forecast_accuracy",
        "activity_forecast_accuracy": "val_activity_forecast_accuracy",
    }
    key = aliases.get(str(metric).strip(), str(metric).strip())
    if key not in row:
        valid = ", ".join(sorted(row))
        raise ValueError(f"Unknown early_stopping_metric {metric!r}; available epoch metrics: {valid}")
    mode = "min" if key.endswith("_loss") or key == "val_selection_loss" else "max"
    return float(row[key]), key, mode


def _train_sequence_model(
    *,
    model: nn.Module,
    splits: Dict[str, Dict[str, np.ndarray]],
    metadata: Mapping[str, object] | None = None,
    method: str,
    horizon: int,
    num_concepts: int,
    num_activities: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    device: torch.device,
    wandb_run,
    teacher_forcing_start_ratio: float,
    teacher_forcing_end_ratio: float,
    concept_forecast_loss_weight: float,
    concept_forecast_loss_deadzone_std: float,
    gcssbm_state_transition_loss_weight: float,
    concept_intervention_task_loss_weight: float,
    concept_intervention_mask_ratio: float,
    persistent_intervention_task_loss_weight: float,
    persistent_intervention_sample_fraction: float,
    persistent_intervention_budgets: Sequence[int],
    persistent_intervention_margin: float,
    persistent_intervention_target_mode: str,
    activity_intervention_task_loss_weight: float,
    activity_intervention_sample_fraction: float,
    activity_intervention_budgets: Sequence[int],
    activity_intervention_margin: float,
    forecast_horizon_loss_weights: Mapping[int, float] | None,
    forecast_transition_tolerance_radius: int,
    forecast_transition_tolerance_weight: float,
    graph_intervention_loss_weight: float,
    graph_task_intervention_loss_weight: float,
    graph_task_intervention_margin: float,
    graph_intervention_margin: float,
    graph_intervention_edges_per_batch: int,
    graph_intervention_sample_fraction: float,
    graph_edge_regularization_weight: float,
    graph_necessity_loss_weight: float,
    graph_necessity_margin: float,
    graph_necessity_edges_per_batch: int,
    graph_necessity_sample_fraction: float,
    edge_edit_response_loss_weight: float,
    edge_edit_response_mode: str,
    edge_edit_response_margin: float,
    edge_edit_response_edges_per_batch: int,
    edge_edit_response_sample_fraction: float,
    graph_disabled_eval: bool,
    graph_corruption_eval: bool,
    graph_corruption_seed: int,
    classifier_l1_weight: float,
    activity_only: bool,
    activity_class_weighting: bool,
    activity_class_weight_cap: float,
    activity_sil_false_positive_penalty: float,
    forecast_class_weighting: bool,
    forecast_class_weight_cap: float,
    forecast_sil_false_positive_penalty: float,
    train_sampling_strategy: str,
    transition_sampler_boundary_radius: int,
    transition_sampler_strength: float,
    transition_sampler_rare_alpha: float,
    early_stopping_metric: str,
    sil_index: int | None,
    activity_components: Mapping[str, np.ndarray] | None,
    model_hparams: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable model parameters remain.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    example_builder = _activity_window_examples if activity_only else _sliding_window_examples
    examples = {
        name: example_builder(
            split,
            horizon=horizon,
            history_length=model.history_length,
            memory_prefix_length=int(getattr(model, "st_memory_prefix_length", 0)),
            transition_boundary_radius=transition_sampler_boundary_radius,
        )
        for name, split in splits.items()
    }
    transition_prior_info = _install_transition_prior_from_hparams(
        model,
        model_hparams or {},
        examples=examples,
        horizon=horizon,
        num_activities=num_activities,
        device=device,
    )
    activity_feedback_prototype_info = _install_activity_feedback_prototypes(
        model,
        examples=examples,
        num_activities=num_activities,
        device=device,
    )
    class_concept_prototypes = _class_horizon_concept_prototypes(
        model,
        examples["train"],
        horizon=horizon,
        num_activities=num_activities,
        device=device,
    )
    train_sampling_info = _train_sampling_info(
        examples["train"],
        strategy=train_sampling_strategy,
        num_classes=num_activities,
        boundary_radius=transition_sampler_boundary_radius,
        strength=transition_sampler_strength,
        rare_alpha=transition_sampler_rare_alpha,
    )
    forecast_class_weights = {} if activity_only else _forecast_class_weights_by_step(
        examples["train"],
        horizon=horizon,
        num_classes=num_activities,
        cap=forecast_class_weight_cap,
        device=device,
    ) if forecast_class_weighting else {}
    activity_class_weight = (
        torch.as_tensor(
            _balanced_class_weights(
                examples["train"]["activity_labels"],
                num_activities,
                cap=activity_class_weight_cap,
            ),
            dtype=torch.float32,
            device=device,
        )
        if activity_class_weighting
        else None
    )
    forecast_baselines = {} if activity_only else _forecast_baseline_diagnostics(
        examples,
        horizon=horizon,
        num_classes=num_activities,
        sil_index=sil_index,
    )
    best_state = _state_to_cpu(model)
    best_selection_value: float | None = None
    best_val_loss = float("inf")
    best_score = -1.0
    best_selection_metric = str(early_stopping_metric)
    best_epoch = 0
    wait = 0
    history: List[Dict[str, object]] = []
    fixed_edge_edit_edges: List[Dict[str, object]] | None = None
    if (
        float(edge_edit_response_loss_weight) > 0.0
        and edge_edit_response_mode == "oracle_margin"
        and int(edge_edit_response_edges_per_batch) > 0
    ):
        fixed_edge_edit_edges = _active_graph_edges(
            model, max_edges=int(edge_edit_response_edges_per_batch)
        )
        model.trained_intervention_edges = [dict(edge) for edge in fixed_edge_edit_edges]
    topk_mode = str(getattr(model, "st_topk_training_mode", "hard"))
    topk_completion_epoch = int(
        model.topk_schedule_completion_epoch()
        if hasattr(model, "topk_schedule_completion_epoch")
        else 1
    )
    topk_minimum_epoch = int(
        model.topk_minimum_training_epoch()
        if hasattr(model, "topk_minimum_training_epoch")
        else topk_completion_epoch
    )
    if topk_mode == "gradual" and int(num_epochs) < topk_completion_epoch:
        raise ValueError(
            f"num_epochs={num_epochs} must reach gradual Top-K completion epoch "
            f"{topk_completion_epoch}."
        )

    for epoch in range(1, int(num_epochs) + 1):
        train_topk_state = (
            model.configure_topk_for_training(epoch)
            if hasattr(model, "configure_topk_for_training")
            else {
                "phase": "fixed",
                "checkpoint_eligible": True,
                "spatial_top_k": int(getattr(model, "st_spatial_top_k", 0)),
                "temporal_top_k": int(getattr(model, "st_temporal_top_k", 0)),
                "cross_temporal_top_k": int(getattr(model, "st_cross_temporal_top_k", 0)),
            }
        )
        model.train()
        teacher_forcing_ratio = _scheduled_sampling_ratio(
            epoch,
            int(num_epochs),
            teacher_forcing_start_ratio,
            teacher_forcing_end_ratio,
        )
        permutation = _epoch_train_indices(examples["train"], train_sampling_info)
        totals = {
            "total": 0.0,
            "activity": 0.0,
            "forecast": 0.0,
            "concept_forecast": 0.0,
            "concept_forecast_weighted": 0.0,
            "gcssbm_state_transition": 0.0,
            "gcssbm_state_transition_weighted": 0.0,
            "concept_intervention_task": 0.0,
            "concept_intervention_task_weighted": 0.0,
            "persistent_intervention_task": 0.0,
            "persistent_intervention_task_weighted": 0.0,
            "graph_intervention": 0.0,
            "graph_intervention_weighted": 0.0,
            "graph_task_intervention": 0.0,
            "graph_task_intervention_weighted": 0.0,
            "graph_necessity": 0.0,
            "graph_necessity_weighted": 0.0,
            "edge_edit_response": 0.0,
            "edge_edit_response_weighted": 0.0,
            "activity_intervention_task": 0.0,
            "activity_intervention_task_weighted": 0.0,
            "graph_edge_regularization": 0.0,
            "classifier_l1": 0.0,
            "examples": 0,
        }
        for indices in _iter_example_batches(permutation, batch_size):
            batch = _example_batch_to_torch(examples["train"], indices, device)
            outputs = _forward_outputs(
                model,
                method,
                batch["concepts"],
                batch["key_padding_mask"],
                memory_prefix_concepts=batch.get("memory_prefix_concepts"),
                memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
                previous_activity_labels=batch.get("teacher_forcing_labels"),
                teacher_forcing=True,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            activity_logits = outputs["activity_logits"][:, -1, :]
            activity_loss = F.cross_entropy(activity_logits, batch["activity_labels"], weight=activity_class_weight)
            activity_loss = activity_loss + _sil_false_positive_loss(
                activity_logits,
                batch["activity_labels"],
                sil_index,
                activity_sil_false_positive_penalty,
            )
            state_transition_loss = activity_loss.new_zeros(())
            if method == "gcssbm":
                state_transition_loss = _gcssbm_state_transition_loss(outputs)
            weighted_state_transition_loss = (
                float(gcssbm_state_transition_loss_weight) * state_transition_loss
            )
            if activity_only:
                forecast_loss = activity_loss.new_zeros(())
                concept_forecast_loss = activity_loss.new_zeros(())
                weighted_concept_forecast_loss = activity_loss.new_zeros(())
                concept_intervention_task_loss = activity_loss.new_zeros(())
                weighted_concept_intervention_task_loss = activity_loss.new_zeros(())
                graph_intervention_loss = activity_loss.new_zeros(())
                weighted_graph_intervention_loss = activity_loss.new_zeros(())
                graph_task_intervention_loss = activity_loss.new_zeros(())
                weighted_graph_task_intervention_loss = activity_loss.new_zeros(())
                graph_necessity_loss = activity_loss.new_zeros(())
                weighted_graph_necessity_loss = activity_loss.new_zeros(())
                edge_edit_response_loss = activity_loss.new_zeros(())
                weighted_edge_edit_response_loss = activity_loss.new_zeros(())
                activity_intervention_task_loss = activity_loss.new_zeros(())
                weighted_activity_intervention_task_loss = activity_loss.new_zeros(())
                persistent_intervention_task_loss = activity_loss.new_zeros(())
                weighted_persistent_intervention_task_loss = activity_loss.new_zeros(())
            else:
                forecast_loss = _sequence_forecast_loss(
                    outputs,
                    method,
                    horizon,
                    batch,
                    class_weights_by_step=forecast_class_weights,
                    sil_index=sil_index,
                    sil_false_positive_penalty=forecast_sil_false_positive_penalty,
                    horizon_loss_weights=forecast_horizon_loss_weights,
                    transition_tolerance_radius=forecast_transition_tolerance_radius,
                    transition_tolerance_weight=forecast_transition_tolerance_weight,
                )
                concept_forecast_loss = _concept_forecast_loss(
                    model,
                    outputs,
                    batch,
                    deadzone_std=concept_forecast_loss_deadzone_std,
                    horizon_loss_weights=forecast_horizon_loss_weights,
                )
                weighted_concept_forecast_loss = float(concept_forecast_loss_weight) * concept_forecast_loss
                concept_intervention_task_loss = activity_loss.new_zeros(())
                if float(concept_intervention_task_loss_weight) > 0.0:
                    concept_intervention_task_loss = _concept_intervention_task_loss(
                        model,
                        outputs,
                        method,
                        horizon,
                        batch,
                        mask_ratio=concept_intervention_mask_ratio,
                        class_weights_by_step=forecast_class_weights,
                        sil_index=sil_index,
                        sil_false_positive_penalty=forecast_sil_false_positive_penalty,
                        horizon_loss_weights=forecast_horizon_loss_weights,
                    )
                weighted_concept_intervention_task_loss = (
                    float(concept_intervention_task_loss_weight) * concept_intervention_task_loss
                )
                persistent_intervention_task_loss = activity_loss.new_zeros(())
                if float(persistent_intervention_task_loss_weight) > 0.0:
                    persistent_intervention_task_loss = _persistent_intervention_task_loss(
                        model,
                        outputs,
                        method,
                        horizon,
                        batch,
                        class_concept_prototypes=class_concept_prototypes,
                        budgets=persistent_intervention_budgets,
                        sample_fraction=persistent_intervention_sample_fraction,
                        gain_margin=persistent_intervention_margin,
                        target_mode=persistent_intervention_target_mode,
                    )
                weighted_persistent_intervention_task_loss = (
                    float(persistent_intervention_task_loss_weight)
                    * persistent_intervention_task_loss
                )
                graph_intervention_loss = activity_loss.new_zeros(())
                graph_task_intervention_loss = activity_loss.new_zeros(())
                if (
                    (float(graph_intervention_loss_weight) > 0.0
                    or float(graph_task_intervention_loss_weight) > 0.0)
                    and int(graph_intervention_edges_per_batch) > 0
                ):
                    graph_losses = _graph_intervention_training_losses(
                        model,
                        method,
                        batch,
                        horizon=horizon,
                        max_edges=graph_intervention_edges_per_batch,
                        margin=graph_intervention_margin,
                        task_margin=graph_task_intervention_margin,
                        sample_fraction=graph_intervention_sample_fraction,
                    )
                    graph_intervention_loss = graph_losses["state"]
                    graph_task_intervention_loss = graph_losses["task"]
                weighted_graph_intervention_loss = (
                    float(graph_intervention_loss_weight) * graph_intervention_loss
                )
                weighted_graph_task_intervention_loss = (
                    float(graph_task_intervention_loss_weight) * graph_task_intervention_loss
                )
                graph_necessity_loss = activity_loss.new_zeros(())
                if (
                    float(graph_necessity_loss_weight) > 0.0
                    and int(graph_necessity_edges_per_batch) > 0
                ):
                    graph_necessity_loss = _graph_necessity_loss(
                        model,
                        outputs,
                        method,
                        horizon,
                        batch,
                        max_edges=graph_necessity_edges_per_batch,
                        margin=graph_necessity_margin,
                        sample_fraction=graph_necessity_sample_fraction,
                    )
                weighted_graph_necessity_loss = (
                    float(graph_necessity_loss_weight) * graph_necessity_loss
                )
                edge_edit_response_loss = activity_loss.new_zeros(())
                if (
                    float(edge_edit_response_loss_weight) > 0.0
                    and int(edge_edit_response_edges_per_batch) > 0
                ):
                    edge_edit_response_loss = _edge_edit_response_loss(
                        model,
                        method,
                        horizon,
                        batch,
                        max_edges=edge_edit_response_edges_per_batch,
                        margin=edge_edit_response_margin,
                        sample_fraction=edge_edit_response_sample_fraction,
                        mode=edge_edit_response_mode,
                        fixed_edges=fixed_edge_edit_edges,
                    )
                weighted_edge_edit_response_loss = (
                    float(edge_edit_response_loss_weight) * edge_edit_response_loss
                )
                activity_intervention_task_loss = activity_loss.new_zeros(())
                if float(activity_intervention_task_loss_weight) > 0.0:
                    activity_intervention_task_loss = _activity_intervention_training_loss(
                        model,
                        method,
                        horizon,
                        batch,
                        budgets=activity_intervention_budgets,
                        sample_fraction=activity_intervention_sample_fraction,
                        gain_margin=activity_intervention_margin,
                    )
                weighted_activity_intervention_task_loss = (
                    float(activity_intervention_task_loss_weight)
                    * activity_intervention_task_loss
                )
            classifier_l1_loss = _classifier_l1_penalty(model) * float(classifier_l1_weight)
            reg_loss = activity_loss.new_zeros(())
            if hasattr(model, "edge_regularization") and float(graph_edge_regularization_weight) > 0.0:
                reg_loss = model.edge_regularization() * float(graph_edge_regularization_weight)
            total_loss = (
                activity_loss
                + classifier_l1_loss
                + reg_loss
                + weighted_state_transition_loss
            )
            if not activity_only:
                total_loss = (
                    total_loss
                    + forecast_loss
                    + weighted_concept_forecast_loss
                    + weighted_concept_intervention_task_loss
                    + weighted_persistent_intervention_task_loss
                    + weighted_graph_intervention_loss
                    + weighted_graph_task_intervention_loss
                    + weighted_graph_necessity_loss
                    + weighted_edge_edit_response_loss
                    + weighted_activity_intervention_task_loss
                )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if hasattr(model, "enforce_frozen_topk_topology"):
                model.enforce_frozen_topk_topology()

            n = len(indices)
            totals["total"] += float(total_loss.item()) * n
            totals["activity"] += float(activity_loss.item()) * n
            totals["forecast"] += float(forecast_loss.item()) * n
            totals["concept_forecast"] += float(concept_forecast_loss.item()) * n
            totals["concept_forecast_weighted"] += float(weighted_concept_forecast_loss.item()) * n
            totals["gcssbm_state_transition"] += float(state_transition_loss.item()) * n
            totals["gcssbm_state_transition_weighted"] += (
                float(weighted_state_transition_loss.item()) * n
            )
            totals["concept_intervention_task"] += float(concept_intervention_task_loss.item()) * n
            totals["concept_intervention_task_weighted"] += (
                float(weighted_concept_intervention_task_loss.item()) * n
            )
            totals["persistent_intervention_task"] += float(persistent_intervention_task_loss.item()) * n
            totals["persistent_intervention_task_weighted"] += (
                float(weighted_persistent_intervention_task_loss.item()) * n
            )
            totals["graph_intervention"] += float(graph_intervention_loss.item()) * n
            totals["graph_intervention_weighted"] += float(weighted_graph_intervention_loss.item()) * n
            totals["graph_task_intervention"] += float(graph_task_intervention_loss.item()) * n
            totals["graph_task_intervention_weighted"] += (
                float(weighted_graph_task_intervention_loss.item()) * n
            )
            totals["graph_necessity"] += float(graph_necessity_loss.item()) * n
            totals["graph_necessity_weighted"] += float(weighted_graph_necessity_loss.item()) * n
            totals["edge_edit_response"] += float(edge_edit_response_loss.item()) * n
            totals["edge_edit_response_weighted"] += float(weighted_edge_edit_response_loss.item()) * n
            totals["activity_intervention_task"] += float(activity_intervention_task_loss.item()) * n
            totals["activity_intervention_task_weighted"] += float(weighted_activity_intervention_task_loss.item()) * n
            totals["graph_edge_regularization"] += float(reg_loss.item()) * n
            totals["classifier_l1"] += float(classifier_l1_loss.item()) * n
            totals["examples"] += n

        eval_topk_state = (
            model.configure_topk_for_evaluation(epoch)
            if hasattr(model, "configure_topk_for_evaluation")
            else dict(train_topk_state)
        )
        val_metrics = _evaluate_sequence(
            model,
            examples["val"],
            method,
            horizon,
            batch_size,
            num_activities,
            device,
            sil_index=sil_index,
            activity_only=activity_only,
            activity_components=activity_components,
        )
        if activity_only:
            val_score = 0.5 * (val_metrics["activity"]["accuracy"] + val_metrics["activity"]["macro_f1"])
            val_selection_loss = float(val_metrics["activity"]["loss"])
        else:
            val_score = 0.25 * (
                val_metrics["activity"]["accuracy"]
                + val_metrics["activity"]["macro_f1"]
                + val_metrics["forecast"]["accuracy"]
                + val_metrics["forecast"]["macro_f1"]
            )
            val_selection_loss = float(val_metrics["activity"]["loss"] + val_metrics["forecast"]["loss"])

        test_metrics = _evaluate_sequence(
            model,
            examples["test"],
            method,
            horizon,
            batch_size,
            num_activities,
            device,
            sil_index=sil_index,
            activity_only=activity_only,
            activity_components=activity_components,
        )
        if activity_only:
            test_score = 0.5 * (test_metrics["activity"]["accuracy"] + test_metrics["activity"]["macro_f1"])
            test_selection_loss = float(test_metrics["activity"]["loss"])
        else:
            test_score = 0.25 * (
                test_metrics["activity"]["accuracy"]
                + test_metrics["activity"]["macro_f1"]
                + test_metrics["forecast"]["accuracy"]
                + test_metrics["forecast"]["macro_f1"]
            )
            test_selection_loss = float(test_metrics["activity"]["loss"] + test_metrics["forecast"]["loss"])

        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / max(totals["examples"], 1),
            "train_activity_loss": totals["activity"] / max(totals["examples"], 1),
            "train_forecast_loss": totals["forecast"] / max(totals["examples"], 1),
            "train_concept_forecast_loss": totals["concept_forecast"] / max(totals["examples"], 1),
            "train_concept_forecast_weighted_loss": (
                totals["concept_forecast_weighted"] / max(totals["examples"], 1)
            ),
            "train_gcssbm_state_transition_loss": (
                totals["gcssbm_state_transition"] / max(totals["examples"], 1)
            ),
            "train_gcssbm_state_transition_weighted_loss": (
                totals["gcssbm_state_transition_weighted"] / max(totals["examples"], 1)
            ),
            "train_concept_intervention_task_loss": (
                totals["concept_intervention_task"] / max(totals["examples"], 1)
            ),
            "train_concept_intervention_task_weighted_loss": (
                totals["concept_intervention_task_weighted"] / max(totals["examples"], 1)
            ),
            "train_persistent_intervention_task_loss": (
                totals["persistent_intervention_task"] / max(totals["examples"], 1)
            ),
            "train_persistent_intervention_task_weighted_loss": (
                totals["persistent_intervention_task_weighted"] / max(totals["examples"], 1)
            ),
            "train_graph_intervention_loss": totals["graph_intervention"] / max(totals["examples"], 1),
            "train_graph_intervention_weighted_loss": totals["graph_intervention_weighted"] / max(totals["examples"], 1),
            "train_graph_task_intervention_loss": totals["graph_task_intervention"] / max(totals["examples"], 1),
            "train_graph_task_intervention_weighted_loss": totals["graph_task_intervention_weighted"] / max(totals["examples"], 1),
            "train_graph_necessity_loss": totals["graph_necessity"] / max(totals["examples"], 1),
            "train_graph_necessity_weighted_loss": totals["graph_necessity_weighted"] / max(totals["examples"], 1),
            "train_edge_edit_response_loss": totals["edge_edit_response"] / max(totals["examples"], 1),
            "train_edge_edit_response_weighted_loss": totals["edge_edit_response_weighted"] / max(totals["examples"], 1),
            "train_activity_intervention_task_loss": totals["activity_intervention_task"] / max(totals["examples"], 1),
            "train_activity_intervention_task_weighted_loss": totals["activity_intervention_task_weighted"] / max(totals["examples"], 1),
            "train_graph_edge_regularization_loss": totals["graph_edge_regularization"] / max(totals["examples"], 1),
            "train_classifier_l1_loss": totals["classifier_l1"] / max(totals["examples"], 1),
            "train_examples": int(totals["examples"]),
            "teacher_forcing_ratio": float(teacher_forcing_ratio),
            "activity_only": bool(activity_only),
            "topk_training_mode": topk_mode,
            "topk_train_phase": str(train_topk_state["phase"]),
            "topk_train_spatial": int(train_topk_state["spatial_top_k"]),
            "topk_train_temporal": int(train_topk_state["temporal_top_k"]),
            "topk_train_cross_temporal": int(train_topk_state["cross_temporal_top_k"]),
            "topk_eval_phase": str(eval_topk_state["phase"]),
            "topk_eval_spatial": int(eval_topk_state["spatial_top_k"]),
            "topk_eval_temporal": int(eval_topk_state["temporal_top_k"]),
            "topk_eval_cross_temporal": int(eval_topk_state["cross_temporal_top_k"]),
            "topk_checkpoint_eligible": bool(eval_topk_state["checkpoint_eligible"]),
            "topk_topology_frozen": bool(
                model.topk_topology_is_frozen() if hasattr(model, "topk_topology_is_frozen") else False
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_selection_loss": float(val_selection_loss),
            "val_score": float(val_score),
            "val_activity_accuracy": float(val_metrics["activity"]["accuracy"]),
            "val_activity_macro_f1": float(val_metrics["activity"]["macro_f1"]),
            "val_activity_top3_accuracy": float(val_metrics["activity"]["top3_accuracy"]),
            "test_selection_loss": float(test_selection_loss),
            "test_score": float(test_score),
            "test_activity_accuracy": float(test_metrics["activity"]["accuracy"]),
            "test_activity_macro_f1": float(test_metrics["activity"]["macro_f1"]),
            "test_activity_top3_accuracy": float(test_metrics["activity"]["top3_accuracy"]),
        }
        if not activity_only:
            row.update(
                {
                    "val_forecast_accuracy": float(val_metrics["forecast"]["accuracy"]),
                    "val_forecast_macro_f1": float(val_metrics["forecast"]["macro_f1"]),
                    "val_forecast_top3_accuracy": float(val_metrics["forecast"]["top3_accuracy"]),
                    "test_forecast_accuracy": float(test_metrics["forecast"]["accuracy"]),
                    "test_forecast_macro_f1": float(test_metrics["forecast"]["macro_f1"]),
                    "test_forecast_top3_accuracy": float(test_metrics["forecast"]["top3_accuracy"]),
                }
            )
            row["val_activity_forecast_accuracy"] = 0.5 * (
                row["val_activity_accuracy"] + row["val_forecast_accuracy"]
            )
            row["test_activity_forecast_accuracy"] = 0.5 * (
                row["test_activity_accuracy"] + row["test_forecast_accuracy"]
            )
            row.update(_history_forecast_diagnostic_row("val", val_metrics))
            row.update(_history_forecast_diagnostic_row("test", test_metrics))
        else:
            row["val_activity_forecast_accuracy"] = row["val_activity_accuracy"]
            row["test_activity_forecast_accuracy"] = row["test_activity_accuracy"]
        selection_value, selection_key, selection_mode = _selection_metric_value(row, early_stopping_metric)
        row["val_selection_value"] = float(selection_value)
        row["val_selection_key"] = str(selection_key)
        history.append(row)
        _log_epoch(wandb_run, row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(num_epochs):
            if activity_only:
                print(
                    f"epoch={epoch:03d} total_loss={row['train_total_loss']:.4f} "
                    f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                    f"val_act_acc={row['val_activity_accuracy']:.4f} "
                    f"test_score={test_score:.4f} test_act_acc={row['test_activity_accuracy']:.4f} "
                    f"selection={selection_key}:{selection_value:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch:03d} total_loss={row['train_total_loss']:.4f} "
                    f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                    f"val_act_acc={row['val_activity_accuracy']:.4f} "
                    f"val_fore_acc={row['val_forecast_accuracy']:.4f} "
                    f"test_score={test_score:.4f} test_act_acc={row['test_activity_accuracy']:.4f} "
                    f"test_fore_acc={row['test_forecast_accuracy']:.4f} "
                    f"selection={selection_key}:{selection_value:.4f}",
                    flush=True,
                )

        if not bool(eval_topk_state["checkpoint_eligible"]):
            continue
        if best_selection_value is None:
            improved = True
        elif selection_mode == "min":
            improved = selection_value < best_selection_value - 1e-8
        else:
            improved = selection_value > best_selection_value + 1e-8
        if improved:
            best_selection_value = float(selection_value)
            best_selection_metric = selection_key
            best_val_loss = float(val_selection_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(patience) and int(epoch) >= topk_minimum_epoch:
                break

    if best_epoch <= 0:
        raise RuntimeError("No checkpoint-eligible epoch was reached during training.")
    model.load_state_dict(best_state)
    final_topk_state = (
        model.configure_topk_for_final_evaluation()
        if hasattr(model, "configure_topk_for_final_evaluation")
        else {
            "phase": "fixed",
            "checkpoint_eligible": True,
            "spatial_top_k": int(getattr(model, "st_spatial_top_k", 0)),
            "temporal_top_k": int(getattr(model, "st_temporal_top_k", 0)),
            "cross_temporal_top_k": int(getattr(model, "st_cross_temporal_top_k", 0)),
        }
    )
    train_metrics = _evaluate_sequence(
        model,
        examples["train"],
        method,
        horizon,
        batch_size,
        num_activities,
        device,
        sil_index=sil_index,
        activity_only=activity_only,
        activity_components=activity_components,
    )
    val_metrics = _evaluate_sequence(
        model,
        examples["val"],
        method,
        horizon,
        batch_size,
        num_activities,
        device,
        sil_index=sil_index,
        activity_only=activity_only,
        activity_components=activity_components,
    )
    test_metrics = _evaluate_sequence(
        model,
        examples["test"],
        method,
        horizon,
        batch_size,
        num_activities,
        device,
        sil_index=sil_index,
        activity_only=activity_only,
        activity_components=activity_components,
    )
    test_intervention_metrics = {} if activity_only else _intervene_metrics(
        model,
        examples["test"],
        method,
        horizon,
        batch_size,
        num_activities,
        device,
    )
    test_activity_feedback_intervention_metrics = (
        {}
        if activity_only
        else _activity_feedback_intervention_metrics(
            model,
            examples["test"],
            method,
            horizon,
            batch_size,
            device,
        )
    )
    test_edge_guided_intervention_metrics = {} if activity_only else _edge_guided_intervention_metrics(
        model,
        examples["test"],
        method,
        horizon,
        batch_size,
        device,
    )
    test_graph_disabled_metrics = {}
    if bool(graph_disabled_eval) and not activity_only:
        test_graph_disabled_metrics = _evaluate_sequence_with_graph_disabled(
            model,
            examples["test"],
            method,
            horizon,
            batch_size,
            num_activities,
            device,
            sil_index=sil_index,
            activity_components=activity_components,
        )
    (
        test_graph_corrupted_metrics,
        test_graph_corruption_delta_metrics,
        graph_corruption_metadata,
    ) = _evaluate_sequence_with_corrupted_graph(
        model,
        examples["test"],
        method,
        horizon,
        batch_size,
        num_activities,
        device,
        baseline_metrics=test_metrics,
        enabled=bool(graph_corruption_eval),
        seed=int(graph_corruption_seed),
        sil_index=sil_index,
        activity_only=activity_only,
        activity_components=activity_components,
    )
    synthetic_edge_recovery_metrics = _synthetic_edge_recovery_metrics(model, metadata or {})
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_score": float(best_score),
        "best_selection_metric": best_selection_metric,
        "best_selection_value": float(best_selection_value if best_selection_value is not None else 0.0),
        "device": str(device),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_intervention_metrics": test_intervention_metrics,
        "test_activity_feedback_intervention_metrics": test_activity_feedback_intervention_metrics,
        "test_edge_guided_intervention_metrics": test_edge_guided_intervention_metrics,
        "test_graph_disabled_metrics": test_graph_disabled_metrics,
        "test_graph_corrupted_metrics": test_graph_corrupted_metrics,
        "test_graph_corruption_delta_metrics": test_graph_corruption_delta_metrics,
        "graph_corruption": graph_corruption_metadata,
        "synthetic_edge_recovery_metrics": synthetic_edge_recovery_metrics,
        "forecast_baselines": forecast_baselines,
        "forecast_metrics_used": not bool(activity_only),
        "concept_forecast_loss_weight": float(concept_forecast_loss_weight),
        "concept_forecast_loss_deadzone_std": float(concept_forecast_loss_deadzone_std),
        "gcssbm_state_transition_loss_weight": float(gcssbm_state_transition_loss_weight),
        "concept_intervention_task_loss_weight": float(concept_intervention_task_loss_weight),
        "concept_intervention_mask_ratio": float(concept_intervention_mask_ratio),
        "persistent_intervention_task_loss_weight": float(persistent_intervention_task_loss_weight),
        "persistent_intervention_sample_fraction": float(persistent_intervention_sample_fraction),
        "persistent_intervention_budgets": [int(value) for value in persistent_intervention_budgets],
        "persistent_intervention_margin": float(persistent_intervention_margin),
        "persistent_intervention_target_mode": persistent_intervention_target_mode,
        "activity_intervention_task_loss_weight": float(activity_intervention_task_loss_weight),
        "activity_intervention_sample_fraction": float(activity_intervention_sample_fraction),
        "activity_intervention_budgets": [int(value) for value in activity_intervention_budgets],
        "activity_intervention_margin": float(activity_intervention_margin),
        "forecast_horizon_loss_weights": dict(forecast_horizon_loss_weights or {}),
        "forecast_transition_tolerance_radius": int(forecast_transition_tolerance_radius),
        "forecast_transition_tolerance_weight": float(forecast_transition_tolerance_weight),
        "graph_intervention_loss_weight": float(graph_intervention_loss_weight),
        "graph_task_intervention_loss_weight": float(graph_task_intervention_loss_weight),
        "graph_task_intervention_margin": float(graph_task_intervention_margin),
        "graph_intervention_margin": float(graph_intervention_margin),
        "graph_intervention_edges_per_batch": int(graph_intervention_edges_per_batch),
        "graph_intervention_sample_fraction": float(graph_intervention_sample_fraction),
        "graph_edge_regularization_weight": float(graph_edge_regularization_weight),
        "graph_necessity_loss_weight": float(graph_necessity_loss_weight),
        "graph_necessity_margin": float(graph_necessity_margin),
        "graph_necessity_edges_per_batch": int(graph_necessity_edges_per_batch),
        "graph_necessity_sample_fraction": float(graph_necessity_sample_fraction),
        "edge_edit_response_loss_weight": float(edge_edit_response_loss_weight),
        "edge_edit_response_mode": edge_edit_response_mode,
        "edge_edit_response_margin": float(edge_edit_response_margin),
        "edge_edit_response_edges_per_batch": int(edge_edit_response_edges_per_batch),
        "edge_edit_response_sample_fraction": float(edge_edit_response_sample_fraction),
        "trained_intervention_edges": [
            dict(edge) for edge in (fixed_edge_edit_edges or [])
        ],
        "graph_disabled_eval": bool(graph_disabled_eval),
        "graph_corruption_eval": bool(graph_corruption_eval),
        "graph_corruption_seed": int(graph_corruption_seed),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "transition_prior": transition_prior_info,
        "activity_feedback_prototypes": activity_feedback_prototype_info,
        "activity_class_weighting": bool(activity_class_weighting),
        "activity_class_weight_cap": float(activity_class_weight_cap),
        "activity_sil_false_positive_penalty": float(activity_sil_false_positive_penalty),
        "train_sampling": _serializable_train_sampling_info(train_sampling_info),
        "graph_metrics": _graph_structure_metrics(model),
        "forecast_rollout_mode": str(getattr(model, "st_forecast_rollout_mode", "legacy")),
        "gcssbm_transition_mode": str(getattr(model, "gcssbm_transition_mode", "none")),
        "activity_feedback_mode": str(
            getattr(model, "st_activity_feedback_mode", "none")
        ),
        "activity_feedback_trainable_parameters": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.startswith("activity_feedback_") and parameter.requires_grad
            )
        ),
        "observed_refiner_mode": str(getattr(model, "st_observed_refiner_mode", "graph")),
        "topk_training_mode": topk_mode,
        "topk_schedule_completion_epoch": int(topk_completion_epoch),
        "topk_minimum_training_epoch": int(topk_minimum_epoch),
        "topk_final_state": dict(final_topk_state),
        "rollout_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in (
                    getattr(model, "shared_graph_layers")
                    if getattr(model, "gcssbm_transition_mode", None) == "shared"
                    else getattr(model, "forecast_graph_layers")
                    if getattr(model, "gcssbm_transition_mode", None) == "separate"
                    else getattr(model, "controlled_rollout_layers", [])
                ).parameters()
                if parameter.requires_grad
            )
            if hasattr(
                getattr(
                    model,
                    "shared_graph_layers"
                    if getattr(model, "gcssbm_transition_mode", None) == "shared"
                    else "forecast_graph_layers"
                    if getattr(model, "gcssbm_transition_mode", None) == "separate"
                    else "controlled_rollout_layers",
                    None,
                ),
                "parameters",
            )
            else 0
        ),
        "total_trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "num_sliding_window_examples": {
            name: int(value["concepts"].shape[0])
            for name, value in examples.items()
        },
    }


def _forward_outputs(
    model: nn.Module,
    method: str,
    concepts: torch.Tensor,
    key_padding_mask: torch.Tensor,
    memory_prefix_concepts: torch.Tensor | None = None,
    memory_prefix_key_padding_mask: torch.Tensor | None = None,
    previous_activity_labels: torch.Tensor | None = None,
    teacher_forcing: bool = False,
    teacher_forcing_ratio: float = 1.0,
    intervention: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if method == "motif":
        outputs = model(concepts, key_padding_mask)
    else:
        outputs = model(
            concepts,
            key_padding_mask,
            intervention=intervention,
            memory_prefix_concepts=memory_prefix_concepts,
            memory_prefix_key_padding_mask=memory_prefix_key_padding_mask,
            previous_activity_labels=previous_activity_labels,
            teacher_forcing=teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    return _apply_transition_prior_to_outputs(model, outputs)


def _forecast_logits(outputs: Dict[str, object], method: str, horizon: int) -> torch.Tensor:
    if method == "motif":
        return outputs["forecast_logits"]
    return outputs["forecast_logits_by_horizon"][horizon]


def _forecast_logits_by_step(outputs: Dict[str, object], method: str, horizon: int) -> Dict[int, torch.Tensor]:
    if "autoregressive_logits_by_step" in outputs:
        return {int(step): logits for step, logits in outputs["autoregressive_logits_by_step"].items()}
    return {int(horizon): _forecast_logits(outputs, method, horizon)}


def _sequence_forecast_loss(
    outputs: Dict[str, object],
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    class_weights_by_step: Dict[int, torch.Tensor] | None = None,
    sil_index: int | None = None,
    sil_false_positive_penalty: float = 0.0,
    horizon_loss_weights: Mapping[int, float] | None = None,
    transition_tolerance_radius: int = 0,
    transition_tolerance_weight: float = 0.0,
) -> torch.Tensor:
    class_weights_by_step = class_weights_by_step or {}
    if method == "motif" or "autoregressive_logits_by_step" not in outputs:
        logits = _forecast_logits(outputs, method, horizon)[:, -1, :]
        target = batch["forecast_labels"]
        loss = F.cross_entropy(logits, target, weight=class_weights_by_step.get(int(horizon)))
        loss = _mix_transition_tolerant_loss(
            loss,
            logits,
            target,
            batch,
            horizon=int(horizon),
            step=int(horizon),
            radius=transition_tolerance_radius,
            mix_weight=transition_tolerance_weight,
            class_weight=class_weights_by_step.get(int(horizon)),
        )
        return loss + _sil_false_positive_loss(logits, target, sil_index, sil_false_positive_penalty)

    losses = []
    step_logits_by_horizon = outputs["autoregressive_logits_by_step"]
    for step in range(1, int(horizon) + 1):
        logits = step_logits_by_horizon[step][:, -1, :]
        if step < int(horizon):
            target = batch["teacher_forcing_labels"][:, step]
        else:
            target = batch["forecast_labels"]
        loss = F.cross_entropy(logits, target, weight=class_weights_by_step.get(step))
        loss = _mix_transition_tolerant_loss(
            loss,
            logits,
            target,
            batch,
            horizon=int(horizon),
            step=step,
            radius=transition_tolerance_radius,
            mix_weight=transition_tolerance_weight,
            class_weight=class_weights_by_step.get(step),
        )
        loss = loss + _sil_false_positive_loss(logits, target, sil_index, sil_false_positive_penalty)
        losses.append(loss)
    if horizon_loss_weights is None:
        return torch.stack(losses).mean()
    return torch.stack(
        [loss * float(horizon_loss_weights[step]) for step, loss in enumerate(losses, start=1)]
    ).sum()


def _mix_transition_tolerant_loss(
    exact_loss: torch.Tensor,
    logits: torch.Tensor,
    exact_target: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    *,
    horizon: int,
    step: int,
    radius: int,
    mix_weight: float,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    radius = int(radius)
    mix_weight = float(mix_weight)
    if radius <= 0 or mix_weight <= 0.0:
        return exact_loss

    # teacher_forcing_labels contains y_t ... y_{t+H-1}; forecast_labels is y_{t+H}.
    labels_by_time = torch.cat(
        [batch["teacher_forcing_labels"][:, :horizon], batch["forecast_labels"][:, None]],
        dim=1,
    )
    # Forecast tolerance is directional: predicting a transition early may match
    # a label that occurs shortly afterward, but late predictions do not receive
    # credit for labels from earlier timesteps.
    start = int(step)
    end = min(int(horizon), int(step) + radius)
    accepted_targets = labels_by_time[:, start : end + 1]

    accepted_mask = torch.zeros_like(logits, dtype=torch.bool)
    accepted_mask.scatter_(1, accepted_targets, True)
    accepted_log_probability = torch.logsumexp(
        torch.log_softmax(logits, dim=-1).masked_fill(~accepted_mask, -torch.inf),
        dim=-1,
    )
    tolerant_per_example = -accepted_log_probability
    if class_weight is None:
        tolerant_loss = tolerant_per_example.mean()
    else:
        sample_weight = class_weight[exact_target]
        tolerant_loss = (
            (tolerant_per_example * sample_weight).sum()
            / sample_weight.sum().clamp_min(1e-12)
        )
    return (1.0 - mix_weight) * exact_loss + mix_weight * tolerant_loss


def _install_activity_feedback_prototypes(
    model: nn.Module,
    *,
    examples: Dict[str, Dict[str, np.ndarray]],
    num_activities: int,
    device: torch.device,
) -> Dict[str, object]:
    mode = str(getattr(model, "st_activity_feedback_mode", "none"))
    info: Dict[str, object] = {
        "enabled": mode == "prototype_label_to_concept",
        "installed": False,
        "smoothing": float(
            getattr(model, "st_activity_feedback_prototype_smoothing", 1.0)
        ),
        "counts": [],
    }
    if mode != "prototype_label_to_concept":
        return info

    train_examples = examples["train"]
    source = np.asarray(train_examples["concepts"][:, -1, :], dtype=np.float64)
    target = np.asarray(train_examples["future_concepts"][:, 0, :], dtype=np.float64)
    labels = np.asarray(train_examples["activity_labels"], dtype=np.int64)
    valid = (labels >= 0) & (labels < int(num_activities))
    if not np.any(valid):
        raise ValueError("Cannot estimate activity-feedback prototypes without valid labels.")

    source = source[valid]
    target = target[valid]
    labels = labels[valid]
    global_source = source.mean(axis=0)
    global_target = target.mean(axis=0)
    smoothing = float(getattr(model, "st_activity_feedback_prototype_smoothing", 1.0))
    source_prototypes = []
    target_prototypes = []
    counts = []
    for class_idx in range(int(num_activities)):
        selected = labels == class_idx
        count = int(selected.sum())
        counts.append(count)
        if count > 0:
            source_sum = source[selected].sum(axis=0)
            target_sum = target[selected].sum(axis=0)
        else:
            source_sum = np.zeros_like(global_source)
            target_sum = np.zeros_like(global_target)
        denominator = float(count) + smoothing
        if denominator > 0.0:
            source_mean = (source_sum + smoothing * global_source) / denominator
            target_mean = (target_sum + smoothing * global_target) / denominator
        else:
            source_mean = global_source
            target_mean = global_target
        source_prototypes.append(source_mean)
        target_prototypes.append(target_mean)

    model.activity_feedback_prototype_source = torch.as_tensor(
        np.stack(source_prototypes),
        dtype=torch.float32,
        device=device,
    )
    model.activity_feedback_prototype_target = torch.as_tensor(
        np.stack(target_prototypes),
        dtype=torch.float32,
        device=device,
    )
    model.activity_feedback_prototype_counts = torch.as_tensor(
        counts,
        dtype=torch.long,
        device=device,
    )
    info["installed"] = True
    info["counts"] = counts
    return info


def _install_transition_prior_from_hparams(
    model: nn.Module,
    model_hparams: Mapping[str, object],
    *,
    examples: Dict[str, Dict[str, np.ndarray]],
    horizon: int,
    num_activities: int,
    device: torch.device,
) -> Dict[str, object]:
    enabled = bool(model_hparams.get("transition_prior_enabled", False))
    weight = float(model_hparams.get("transition_prior_weight", 0.0))
    smoothing = float(model_hparams.get("transition_prior_smoothing", 1.0))
    detach_activity = bool(model_hparams.get("transition_prior_detach_activity", True))
    if weight < 0.0:
        raise ValueError("model_hparams.transition_prior_weight must be >= 0.")
    if smoothing <= 0.0:
        raise ValueError("model_hparams.transition_prior_smoothing must be > 0.")

    applied = bool(enabled and weight > 0.0)
    setattr(model, "_transition_prior_enabled", applied)
    setattr(model, "_transition_prior_weight", float(weight))
    setattr(model, "_transition_prior_detach_activity", bool(detach_activity))
    setattr(model, "_transition_prior_steps", tuple())
    info: Dict[str, object] = {
        "enabled": bool(enabled),
        "applied": applied,
        "weight": float(weight),
        "smoothing": float(smoothing),
        "detach_activity": bool(detach_activity),
        "steps": [],
    }
    if not applied:
        return info

    train_examples = examples["train"]
    train_sources = np.asarray(train_examples["activity_labels"], dtype=np.int64)
    steps: List[int] = []
    for step in range(1, int(horizon) + 1):
        train_targets = _forecast_step_targets(train_examples, step=step, horizon=horizon)
        prior = _transition_prior_probability_matrix(
            train_sources,
            train_targets,
            num_classes=num_activities,
            smoothing=smoothing,
        )
        buffer_name = f"_transition_prior_prob_h{step}"
        prior_tensor = torch.as_tensor(prior, dtype=torch.float32, device=device)
        if hasattr(model, buffer_name):
            getattr(model, buffer_name).data.copy_(prior_tensor)
        else:
            model.register_buffer(buffer_name, prior_tensor)
        steps.append(step)
    setattr(model, "_transition_prior_steps", tuple(steps))
    info["steps"] = [int(step) for step in steps]
    return info


def _transition_prior_probability_matrix(
    train_sources: np.ndarray,
    train_targets: np.ndarray,
    *,
    num_classes: int,
    smoothing: float,
) -> np.ndarray:
    counts = np.full((int(num_classes), int(num_classes)), float(smoothing), dtype=np.float64)
    sources = np.asarray(train_sources, dtype=np.int64)
    targets = np.asarray(train_targets, dtype=np.int64)
    valid = (sources >= 0) & (sources < int(num_classes)) & (targets >= 0) & (targets < int(num_classes))
    np.add.at(counts, (sources[valid], targets[valid]), 1.0)
    counts /= counts.sum(axis=1, keepdims=True)
    return counts.astype(np.float32)


def _apply_transition_prior_to_outputs(model: nn.Module, outputs: Dict[str, object]) -> Dict[str, object]:
    if not bool(getattr(model, "_transition_prior_enabled", False)):
        return outputs
    activity_logits = outputs.get("activity_logits")
    if not torch.is_tensor(activity_logits):
        return outputs

    source_probs = torch.softmax(activity_logits, dim=-1)
    if bool(getattr(model, "_transition_prior_detach_activity", True)):
        source_probs = source_probs.detach()
    weight = float(getattr(model, "_transition_prior_weight", 0.0))
    if weight <= 0.0:
        return outputs

    blended = dict(outputs)
    if isinstance(outputs.get("forecast_logits_by_horizon"), Mapping):
        blended["forecast_logits_by_horizon"] = {
            int(step): _blend_transition_prior_logits(model, logits, source_probs, int(step), weight)
            for step, logits in outputs["forecast_logits_by_horizon"].items()
        }
    if isinstance(outputs.get("autoregressive_logits_by_step"), Mapping):
        blended["autoregressive_logits_by_step"] = {
            int(step): _blend_transition_prior_logits(model, logits, source_probs, int(step), weight)
            for step, logits in outputs["autoregressive_logits_by_step"].items()
        }
    return blended


def _blend_transition_prior_logits(
    model: nn.Module,
    logits: torch.Tensor,
    source_probs: torch.Tensor,
    step: int,
    weight: float,
) -> torch.Tensor:
    prior = getattr(model, f"_transition_prior_prob_h{int(step)}", None)
    if not torch.is_tensor(prior):
        return logits
    prior = prior.to(device=logits.device, dtype=source_probs.dtype)
    prior_probs = torch.matmul(source_probs.to(device=logits.device), prior)
    prior_log = torch.log(prior_probs.clamp_min(1e-8)).to(dtype=logits.dtype)
    return logits + float(weight) * prior_log


def _future_concepts_in_model_space(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    future_concepts = batch["future_concepts"]
    calibrator = getattr(model, "calibrator", None)
    if not callable(calibrator) or future_concepts.ndim < 2:
        return future_concepts

    original_shape = future_concepts.shape
    flattened = future_concepts.reshape(-1, original_shape[-1])
    with torch.no_grad():
        calibrated = calibrator(flattened).reshape(original_shape)
    return calibrated.to(device=future_concepts.device, dtype=future_concepts.dtype)


def _concept_forecast_loss(
    model: nn.Module,
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    deadzone_std: float = 0.0,
    horizon_loss_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    predicted_by_step = outputs.get("predicted_concepts_by_step")
    if not predicted_by_step:
        return batch["concepts"].sum() * 0.0
    future_concepts = _future_concepts_in_model_space(model, batch)
    deadzone = float(deadzone_std)
    if deadzone < 0.0:
        raise ValueError("deadzone_std must be >= 0.")
    losses = []
    loss_steps = []
    for step, predicted in sorted(predicted_by_step.items()):
        step_index = int(step) - 1
        if step_index < 0 or step_index >= future_concepts.size(1):
            continue
        target = future_concepts[:, step_index, :]
        if deadzone > 0.0:
            excess = (predicted[:, -1, :] - target).abs().sub(deadzone).clamp_min(0.0)
            losses.append(F.smooth_l1_loss(excess, torch.zeros_like(excess)))
        else:
            losses.append(F.smooth_l1_loss(predicted[:, -1, :], target))
        loss_steps.append(int(step))
    if not losses:
        return batch["concepts"].sum() * 0.0
    if horizon_loss_weights is None:
        return torch.stack(losses).mean()
    return torch.stack(
        [loss * float(horizon_loss_weights[step]) for step, loss in zip(loss_steps, losses)]
    ).sum()


def _gcssbm_state_transition_loss(outputs: Dict[str, object]) -> torch.Tensor:
    priors = outputs.get("state_priors")
    targets = outputs.get("calibrated_concepts")
    valid_mask = outputs.get("state_prior_valid_mask")
    if not torch.is_tensor(priors) or not torch.is_tensor(targets) or not torch.is_tensor(valid_mask):
        raise ValueError("gcssbm outputs must include state priors, calibrated concepts, and a prior mask.")
    if priors.shape != targets.shape or valid_mask.shape != priors.shape[:2]:
        raise ValueError("Invalid gcssbm state-transition output shapes.")
    valid = valid_mask.bool()
    if not bool(valid.any()):
        return priors.sum() * 0.0
    return F.smooth_l1_loss(priors[valid], targets.detach()[valid])


def _concept_intervention_task_loss(
    model: nn.Module,
    outputs: Dict[str, object],
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    *,
    mask_ratio: float,
    class_weights_by_step: Dict[int, torch.Tensor] | None = None,
    sil_index: int | None = None,
    sil_false_positive_penalty: float = 0.0,
    horizon_loss_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    if method not in GRAPH_CBM_METHODS:
        raise ValueError("concept intervention task loss requires base_method=trace.")
    if not hasattr(model, "activity_head") or not hasattr(model, "_prediction_transform"):
        raise ValueError("concept intervention task loss requires a shared concept activity head.")
    predicted_by_step = outputs.get("predicted_concepts_by_step")
    if not predicted_by_step:
        raise ValueError("concept intervention task loss requires predicted_concepts_by_step outputs.")
    if "future_concepts" not in batch:
        raise ValueError("concept intervention task loss requires future_concepts in the batch.")

    mask_ratio = float(mask_ratio)
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("concept_intervention_mask_ratio must be in [0, 1].")

    future_concepts = _future_concepts_in_model_space(model, batch)
    class_weights_by_step = class_weights_by_step or {}
    losses = []
    loss_steps = []
    for step, predicted in sorted(predicted_by_step.items()):
        step = int(step)
        step_index = step - 1
        if step_index < 0 or step_index >= future_concepts.size(1):
            continue
        target_concepts = future_concepts[:, step_index, :]
        predicted_last = predicted[:, -1, :]
        if mask_ratio >= 1.0:
            intervened_concepts = target_concepts
        elif mask_ratio <= 0.0:
            intervened_concepts = predicted_last
        else:
            intervention_mask = torch.rand_like(target_concepts) < mask_ratio
            intervened_concepts = torch.where(intervention_mask, target_concepts, predicted_last)

        logits = model.activity_head(model._prediction_transform(intervened_concepts))
        if step < int(horizon):
            target = batch["teacher_forcing_labels"][:, step]
        else:
            target = batch["forecast_labels"]
        loss = F.cross_entropy(logits, target, weight=class_weights_by_step.get(step))
        loss = loss + _sil_false_positive_loss(
            logits,
            target,
            sil_index,
            sil_false_positive_penalty,
        )
        losses.append(loss)
        loss_steps.append(step)

    if not losses:
        raise ValueError("concept intervention task loss found no valid forecast horizons.")
    if horizon_loss_weights is None:
        return torch.stack(losses).mean()
    return torch.stack(
        [loss * float(horizon_loss_weights[step]) for step, loss in zip(loss_steps, losses)]
    ).sum()


def _class_horizon_concept_prototypes(
    model: nn.Module,
    train_examples: Dict[str, np.ndarray],
    *,
    horizon: int,
    num_activities: int,
    device: torch.device,
    smoothing: float = 1.0,
) -> torch.Tensor:
    future = torch.as_tensor(train_examples["future_concepts"], dtype=torch.float32, device=device)
    calibrator = getattr(model, "calibrator", None)
    if callable(calibrator):
        with torch.no_grad():
            future = calibrator(future.reshape(-1, future.shape[-1])).reshape_as(future)
    global_by_step = future.mean(dim=0)
    prototypes: List[torch.Tensor] = []
    for step in range(1, int(horizon) + 1):
        labels = torch.as_tensor(
            _forecast_step_targets(train_examples, step=step, horizon=horizon),
            dtype=torch.long,
            device=device,
        )
        step_values = future[:, step - 1, :]
        class_rows: List[torch.Tensor] = []
        for class_idx in range(int(num_activities)):
            selected = labels == class_idx
            count = int(selected.sum().item())
            class_sum = step_values[selected].sum(dim=0) if count else torch.zeros_like(global_by_step[step - 1])
            class_rows.append(
                (class_sum + float(smoothing) * global_by_step[step - 1])
                / (float(count) + float(smoothing))
            )
        prototypes.append(torch.stack(class_rows, dim=0))
    return torch.stack(prototypes, dim=0).detach()


def _batch_target_for_step(batch: Dict[str, torch.Tensor], step: int, horizon: int) -> torch.Tensor:
    return batch["teacher_forcing_labels"][:, int(step)] if int(step) < int(horizon) else batch["forecast_labels"]


def _subset_tensor_batch(
    batch: Dict[str, torch.Tensor], indices: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch_size = int(batch["concepts"].shape[0])
    return {
        key: value.index_select(0, indices)
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size
        else value
        for key, value in batch.items()
    }


def _true_runner_margin(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    true_logit = logits.gather(1, target[:, None]).squeeze(1)
    competitors = logits.masked_fill(F.one_hot(target, logits.shape[-1]).bool(), -torch.inf)
    runner = competitors.argmax(dim=1)
    runner_logit = logits.gather(1, runner[:, None]).squeeze(1)
    return true_logit - runner_logit, runner


def _persistent_intervention_task_loss(
    model: nn.Module,
    clean_outputs: Dict[str, object],
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    *,
    class_concept_prototypes: torch.Tensor,
    budgets: Sequence[int],
    sample_fraction: float,
    gain_margin: float,
    target_mode: str,
) -> torch.Tensor:
    concepts = batch["concepts"]
    batch_size = int(concepts.shape[0])
    sample_size = min(batch_size, max(1, int(math.ceil(batch_size * float(sample_fraction)))))
    indices = torch.randperm(batch_size, device=concepts.device)[:sample_size]
    sampled = _subset_tensor_batch(batch, indices)
    clean_by_step = _forecast_logits_by_step(clean_outputs, method, horizon)
    predicted_by_step = clean_outputs.get("predicted_concepts_by_step")
    if not isinstance(predicted_by_step, Mapping):
        raise ValueError("Persistent intervention task loss requires predicted_concepts_by_step.")

    sampled_steps = torch.randint(1, int(horizon) + 1, (sample_size,), device=concepts.device)
    sampled_budgets = torch.as_tensor(tuple(budgets), device=concepts.device)[
        torch.randint(0, len(tuple(budgets)), (sample_size,), device=concepts.device)
    ]
    head_weight = getattr(getattr(model, "activity_head", None), "weight", None)
    instance_targets = _future_concepts_in_model_space(model, sampled)
    intervention_items: List[Dict[str, object]] = []
    selected_targets = torch.empty(sample_size, dtype=torch.long, device=concepts.device)
    usable_rows = torch.zeros(sample_size, dtype=torch.bool, device=concepts.device)
    for row in range(sample_size):
        step = int(sampled_steps[row].item())
        target = _batch_target_for_step(sampled, step, horizon)[row]
        selected_targets[row] = target
        clean_logits = clean_by_step[step][indices[row], -1, :]
        _, runner = _true_runner_margin(clean_logits[None, :], target[None])
        predicted = predicted_by_step[step][indices[row], -1, :].detach()
        if target_mode == "instance_oracle":
            intervention_target = instance_targets[row, step - 1, :].detach().to(predicted)
        else:
            intervention_target = class_concept_prototypes[step - 1, target].to(predicted)
        delta = intervention_target - predicted
        score = delta.abs()
        if torch.is_tensor(head_weight) and head_weight.shape[-1] == score.numel():
            contrast = (head_weight[target] - head_weight[runner[0]]).detach()
            score = delta * contrast if target_mode == "instance_oracle" else score * contrast.abs()
        eligible = score > 0.0 if target_mode == "instance_oracle" else torch.ones_like(score, dtype=torch.bool)
        budget = min(int(sampled_budgets[row].item()), int(eligible.sum().item()))
        if budget <= 0:
            continue
        ranked_score = score.masked_fill(~eligible, -torch.inf)
        selected_concepts = torch.topk(ranked_score, k=budget, largest=True).indices
        usable_rows[row] = True
        for concept_idx in selected_concepts.tolist():
            intervention_items.append(
                {
                    "item_type": "concept",
                    "rollout_step": step,
                    "concept_idx": int(concept_idx),
                    "value": float(intervention_target[concept_idx].clamp(0.0, 1.0).item()),
                    "batch_idx": sample_size + row,
                }
            )

    if not bool(usable_rows.any()):
        return concepts.sum() * 0.0
    expanded = {
        key: value.repeat((2,) + (1,) * (value.ndim - 1))
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == sample_size
        else value
        for key, value in sampled.items()
    }
    paired_outputs = model(
        expanded["concepts"],
        expanded["key_padding_mask"],
        intervention={"mode": "persistent", "items": intervention_items},
        memory_prefix_concepts=expanded.get("memory_prefix_concepts"),
        memory_prefix_key_padding_mask=expanded.get("memory_prefix_key_padding_mask"),
    )
    paired_by_step = _forecast_logits_by_step(paired_outputs, method, horizon)
    clean_logits = torch.stack(
        [paired_by_step[int(sampled_steps[row].item())][row, -1, :] for row in range(sample_size)]
    )[usable_rows]
    selected_logits = torch.stack(
        [
            paired_by_step[int(sampled_steps[row].item())][sample_size + row, -1, :]
            for row in range(sample_size)
        ]
    )[usable_rows]
    selected_targets = selected_targets[usable_rows]
    clean_margins, _ = _true_runner_margin(clean_logits, selected_targets)
    intervened_margins, _ = _true_runner_margin(selected_logits, selected_targets)
    classification_loss = F.cross_entropy(selected_logits, selected_targets)
    gain_loss = torch.relu(
        selected_logits.new_tensor(float(gain_margin))
        - (intervened_margins - clean_margins.detach())
    )
    return classification_loss + gain_loss.mean()


def _graph_intervention_training_losses(
    model: nn.Module,
    method: str,
    batch: Dict[str, torch.Tensor],
    *,
    horizon: int,
    max_edges: int,
    margin: float,
    task_margin: float,
    sample_fraction: float = 1.0,
) -> Dict[str, torch.Tensor]:
    concepts = batch["concepts"]
    if method == "motif" or not hasattr(model, "graph_metrics") or int(max_edges) <= 0:
        zero = concepts.sum() * 0.0
        return {"state": zero, "task": zero}
    edges = _active_graph_edges(model, max_edges=int(max_edges))
    if not edges:
        zero = concepts.sum() * 0.0
        return {"state": zero, "task": zero}

    sample_fraction = float(sample_fraction)
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")
    key_padding_mask = batch["key_padding_mask"]
    memory_prefix_concepts = batch.get("memory_prefix_concepts")
    memory_prefix_key_padding_mask = batch.get("memory_prefix_key_padding_mask")
    batch_size = int(concepts.shape[0])
    sample_size = min(batch_size, max(1, int(math.ceil(batch_size * sample_fraction))))
    sample_indices = torch.arange(batch_size, device=concepts.device)
    if sample_size < batch_size:
        sample_indices = torch.randperm(batch_size, device=concepts.device)[:sample_size]
        concepts = concepts.index_select(0, sample_indices)
        key_padding_mask = key_padding_mask.index_select(0, sample_indices)
        if torch.is_tensor(memory_prefix_concepts):
            memory_prefix_concepts = memory_prefix_concepts.index_select(0, sample_indices)
        if torch.is_tensor(memory_prefix_key_padding_mask):
            memory_prefix_key_padding_mask = memory_prefix_key_padding_mask.index_select(0, sample_indices)
    sampled_batch = _subset_tensor_batch(batch, sample_indices)
    valid_mask = ~key_padding_mask
    timesteps = int(concepts.shape[1])
    margin_tensor = concepts.new_tensor(float(margin))
    edge_specs: List[Dict[str, object]] = []
    for edge in edges:
        source_time, target_time = _edge_source_target_times(edge, timesteps)
        valid_rows = valid_mask[:, source_time] & valid_mask[:, target_time]
        if not bool(valid_rows.any()):
            continue
        source = int(edge["source"])
        edge_specs.append(
            {
                "edge": edge,
                "source_time": int(source_time),
                "target_time": int(target_time),
                "valid_rows": valid_rows,
                "high": _raw_intervention_value_for_target(model, source, 0.9),
                "low": _raw_intervention_value_for_target(model, source, 0.1),
            }
        )

    if not edge_specs:
        zero = concepts.sum() * 0.0
        return {"state": zero, "task": zero}

    block_count = 2 * len(edge_specs)
    expanded_concepts = concepts.repeat((block_count, 1, 1))
    expanded_key_padding_mask = key_padding_mask.repeat((block_count, 1))
    expanded_memory_prefix_concepts = (
        memory_prefix_concepts.repeat((block_count, 1, 1))
        if torch.is_tensor(memory_prefix_concepts)
        else None
    )
    expanded_memory_prefix_key_padding_mask = (
        memory_prefix_key_padding_mask.repeat((block_count, 1))
        if torch.is_tensor(memory_prefix_key_padding_mask)
        else None
    )
    intervention_items: List[Dict[str, object]] = []
    for edge_index, spec in enumerate(edge_specs):
        edge = spec["edge"]
        source = int(edge["source"])
        source_time = int(spec["source_time"])
        for block_offset, value in enumerate((spec["high"], spec["low"])):
            block_index = (2 * edge_index) + block_offset
            intervention_items.append(
                {
                    "time_idx": source_time,
                    "concept_idx": source,
                    "value": float(value),
                    "batch_start": block_index * sample_size,
                    "batch_end": (block_index + 1) * sample_size,
                }
            )

    expanded_outputs = model(
        expanded_concepts,
        expanded_key_padding_mask,
        intervention={"mode": "persistent", "items": intervention_items},
        memory_prefix_concepts=expanded_memory_prefix_concepts,
        memory_prefix_key_padding_mask=expanded_memory_prefix_key_padding_mask,
    )

    state_losses: List[torch.Tensor] = []
    task_losses: List[torch.Tensor] = []
    expanded_logits_by_step = _forecast_logits_by_step(expanded_outputs, method, horizon)
    head_weight = getattr(getattr(model, "activity_head", None), "weight", None)
    for edge_index, spec in enumerate(edge_specs):
        edge = spec["edge"]
        state = _edge_target_state(expanded_outputs, str(edge.get("branch", "")))
        high_start = (2 * edge_index) * sample_size
        low_start = ((2 * edge_index) + 1) * sample_size
        target_time = int(spec["target_time"])
        target = int(edge["target"])
        valid_rows = spec["valid_rows"]
        if state is None:
            continue
        high_state = state[high_start : high_start + sample_size]
        low_state = state[low_start : low_start + sample_size]
        response = high_state[valid_rows, target_time, target] - low_state[valid_rows, target_time, target]
        sign = float(edge.get("sign", 1.0))
        if sign == 0.0:
            sign = 1.0
        signed_response = response * concepts.new_tensor(sign)
        state_losses.append(torch.relu(margin_tensor - signed_response).mean())

        if torch.is_tensor(head_weight) and head_weight.shape[-1] > target:
            for step, expanded_logits in expanded_logits_by_step.items():
                high_logits = expanded_logits[high_start : high_start + sample_size, -1, :]
                low_logits = expanded_logits[low_start : low_start + sample_size, -1, :]
                labels = _batch_target_for_step(sampled_batch, int(step), horizon)
                low_margin, runner = _true_runner_margin(low_logits, labels)
                high_true = high_logits.gather(1, labels[:, None]).squeeze(1)
                high_runner = high_logits.gather(1, runner[:, None]).squeeze(1)
                high_margin = high_true - high_runner
                contrast = head_weight[labels, target] - head_weight[runner, target]
                desired_direction = torch.sign(contrast.detach() * concepts.new_tensor(sign))
                usable = desired_direction != 0
                if bool(usable.any()):
                    task_losses.append(
                        torch.relu(
                            concepts.new_tensor(float(task_margin))
                            - desired_direction[usable] * (high_margin[usable] - low_margin[usable])
                        ).mean()
                    )

    zero = concepts.sum() * 0.0
    return {
        "state": torch.stack(state_losses).mean() if state_losses else zero,
        "task": torch.stack(task_losses).mean() if task_losses else zero,
    }


def _graph_necessity_loss(
    model: nn.Module,
    clean_outputs: Dict[str, object],
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    *,
    max_edges: int,
    margin: float,
    sample_fraction: float,
) -> torch.Tensor:
    concepts = batch["concepts"]
    edges = _active_graph_edges(model, max_edges=int(max_edges))
    if not edges:
        return concepts.sum() * 0.0
    batch_size = int(concepts.shape[0])
    sample_size = min(batch_size, max(1, int(math.ceil(batch_size * float(sample_fraction)))))
    indices = torch.randperm(batch_size, device=concepts.device)[:sample_size]
    sampled = _subset_tensor_batch(batch, indices)
    edge_items = [
        {
            "item_type": "edge",
            "edge_kind": str(edge["kind"]),
            "branch": str(edge["branch"]),
            "layer_index": int(edge["layer"]),
            "source_idx": int(edge["source"]),
            "target_idx": int(edge["target"]),
            "edge_scale": 0.0,
        }
        for edge in edges
    ]
    corrupted = model(
        sampled["concepts"],
        sampled["key_padding_mask"],
        intervention={"mode": "input", "items": edge_items},
        memory_prefix_concepts=sampled.get("memory_prefix_concepts"),
        memory_prefix_key_padding_mask=sampled.get("memory_prefix_key_padding_mask"),
    )
    clean_by_step = _forecast_logits_by_step(clean_outputs, method, horizon)
    corrupted_by_step = _forecast_logits_by_step(corrupted, method, horizon)
    losses: List[torch.Tensor] = []
    for step in range(1, int(horizon) + 1):
        clean_logits = clean_by_step[step].index_select(0, indices)[:, -1, :]
        corrupt_logits = corrupted_by_step[step][:, -1, :]
        labels = _batch_target_for_step(sampled, step, horizon)
        clean_margin, runner = _true_runner_margin(clean_logits, labels)
        corrupt_true = corrupt_logits.gather(1, labels[:, None]).squeeze(1)
        corrupt_runner = corrupt_logits.gather(1, runner[:, None]).squeeze(1)
        necessity_gap = clean_margin - (corrupt_true - corrupt_runner)
        losses.append(torch.relu(concepts.new_tensor(float(margin)) - necessity_gap).mean())
    return torch.stack(losses).mean()


def _edge_edit_response_loss(
    model: nn.Module,
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    *,
    max_edges: int,
    margin: float,
    sample_fraction: float,
    mode: str = "probability_l1",
    fixed_edges: Sequence[Mapping[str, object]] | None = None,
) -> torch.Tensor:
    concepts = batch["concepts"]
    edges = (
        [dict(edge) for edge in fixed_edges[: int(max_edges)]]
        if fixed_edges is not None
        else _active_graph_edges(model, max_edges=int(max_edges))
    )
    if not edges:
        return concepts.sum() * 0.0
    batch_size = int(concepts.shape[0])
    sample_size = min(batch_size, max(1, int(math.ceil(batch_size * float(sample_fraction)))))
    indices = torch.randperm(batch_size, device=concepts.device)[:sample_size]
    sampled = _subset_tensor_batch(batch, indices)
    blocks_per_edge = 3 if mode == "oracle_margin" else 2
    block_count = blocks_per_edge * len(edges)
    expanded = {
        key: value.repeat((block_count,) + (1,) * (value.ndim - 1))
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == sample_size
        else value
        for key, value in sampled.items()
    }
    edge_items: List[Dict[str, object]] = []
    for edge_index, edge in enumerate(edges):
        scales = (0.0, -1.0) if mode == "oracle_margin" else (
            -1.0 if bool(torch.rand((), device=concepts.device) < 0.5) else 0.0,
        )
        for offset, scale in enumerate(scales, start=1):
            edit_block = (blocks_per_edge * edge_index) + offset
            edge_items.append(
                {
                    "item_type": "edge",
                    "edge_kind": str(edge["kind"]),
                    "branch": str(edge["branch"]),
                    "layer_index": int(edge["layer"]),
                    "source_idx": int(edge["source"]),
                    "target_idx": int(edge["target"]),
                    "edge_scale": float(scale),
                    "batch_start": edit_block * sample_size,
                    "batch_end": (edit_block + 1) * sample_size,
                }
            )
    outputs = model(
        expanded["concepts"],
        expanded["key_padding_mask"],
        intervention={"mode": "input", "items": edge_items},
        memory_prefix_concepts=expanded.get("memory_prefix_concepts"),
        memory_prefix_key_padding_mask=expanded.get("memory_prefix_key_padding_mask"),
    )
    logits_by_step = _forecast_logits_by_step(outputs, method, horizon)
    losses: List[torch.Tensor] = []
    for edge_index in range(len(edges)):
        clean_start = (blocks_per_edge * edge_index) * sample_size
        for step, logits in logits_by_step.items():
            clean_logits = logits[clean_start : clean_start + sample_size, -1, :]
            if mode == "oracle_margin":
                delete_start = (blocks_per_edge * edge_index + 1) * sample_size
                invert_start = (blocks_per_edge * edge_index + 2) * sample_size
                delete_logits = logits[delete_start : delete_start + sample_size, -1, :]
                invert_logits = logits[invert_start : invert_start + sample_size, -1, :]
                labels = _batch_target_for_step(sampled, int(step), horizon)
                clean_margin, runner = _true_runner_margin(clean_logits, labels)
                delete_margin = (
                    delete_logits.gather(1, labels[:, None]).squeeze(1)
                    - delete_logits.gather(1, runner[:, None]).squeeze(1)
                )
                invert_margin = (
                    invert_logits.gather(1, labels[:, None]).squeeze(1)
                    - invert_logits.gather(1, runner[:, None]).squeeze(1)
                )
                best_gain = torch.maximum(delete_margin, invert_margin) - clean_margin
                losses.append(
                    torch.relu(concepts.new_tensor(float(margin)) - best_gain).mean()
                )
            else:
                edit_start = (blocks_per_edge * edge_index + 1) * sample_size
                clean_probability = torch.softmax(clean_logits, dim=-1)
                edit_probability = torch.softmax(
                    logits[edit_start : edit_start + sample_size, -1, :], dim=-1
                )
                response = (edit_probability - clean_probability).abs().mean(dim=-1)
                target = max(float(margin), torch.finfo(concepts.dtype).eps)
                losses.append(torch.relu(1.0 - response / target).mean())
    return torch.stack(losses).mean() if losses else concepts.sum() * 0.0


def _activity_intervention_training_loss(
    model: nn.Module,
    method: str,
    horizon: int,
    batch: Dict[str, torch.Tensor],
    *,
    budgets: Sequence[int],
    sample_fraction: float,
    gain_margin: float,
) -> torch.Tensor:
    concepts = batch["concepts"]
    if not bool(getattr(model, "_activity_feedback_enabled", lambda: False)()):
        return concepts.sum() * 0.0
    batch_size = int(concepts.shape[0])
    sample_size = min(batch_size, max(1, int(math.ceil(batch_size * float(sample_fraction)))))
    indices = torch.randperm(batch_size, device=concepts.device)[:sample_size]
    sampled = _subset_tensor_batch(batch, indices)
    expanded = {
        key: value.repeat((2,) + (1,) * (value.ndim - 1))
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == sample_size
        else value
        for key, value in sampled.items()
    }
    budget_values = torch.as_tensor(tuple(budgets), device=concepts.device)
    sampled_budgets = budget_values[
        torch.randint(0, len(tuple(budgets)), (sample_size,), device=concepts.device)
    ].clamp(max=int(horizon))
    items: List[Dict[str, object]] = []
    for row in range(sample_size):
        for source_step in range(int(sampled_budgets[row].item())):
            source_label = (
                sampled["activity_labels"][row]
                if source_step == 0
                else sampled["teacher_forcing_labels"][row, source_step]
            )
            items.append(
                {
                    "item_type": "activity",
                    "step": source_step,
                    "class_idx": int(source_label.item()),
                    "probability": 0.9,
                    "batch_idx": sample_size + row,
                }
            )
    outputs = model(
        expanded["concepts"],
        expanded["key_padding_mask"],
        intervention={"mode": "input", "items": items},
        memory_prefix_concepts=expanded.get("memory_prefix_concepts"),
        memory_prefix_key_padding_mask=expanded.get("memory_prefix_key_padding_mask"),
    )
    logits = _forecast_logits_by_step(outputs, method, horizon)[int(horizon)][:, -1, :]
    clean_logits = logits[:sample_size]
    intervened_logits = logits[sample_size:]
    targets = sampled["forecast_labels"]
    clean_margin, _ = _true_runner_margin(clean_logits, targets)
    intervened_margin, _ = _true_runner_margin(intervened_logits, targets)
    classification_loss = F.cross_entropy(intervened_logits, targets)
    gain_loss = torch.relu(
        intervened_logits.new_tensor(float(gain_margin))
        - (intervened_margin - clean_margin.detach())
    ).mean()
    return classification_loss + gain_loss


def _edge_target_state(outputs: Dict[str, object], branch: str) -> torch.Tensor | None:
    key_by_branch = {
        "shared": "shared_refined_concepts",
        "window": "window_refined_concepts",
        "forecast": "forecast_refined_concepts",
    }
    key = key_by_branch.get(str(branch))
    if key is not None and torch.is_tensor(outputs.get(key)):
        return outputs[key]
    for fallback in ("concept_states", "shared_refined_concepts", "forecast_refined_concepts"):
        value = outputs.get(fallback)
        if torch.is_tensor(value):
            return value
    return None


def _sil_false_positive_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sil_index: int | None,
    penalty: float,
) -> torch.Tensor:
    if sil_index is None or float(penalty) <= 0.0:
        return logits.sum() * 0.0
    non_sil = target != int(sil_index)
    if not bool(non_sil.any()):
        return logits.sum() * 0.0
    sil_prob = torch.softmax(logits[non_sil], dim=-1)[:, int(sil_index)]
    return sil_prob.mean() * float(penalty)


def _forecast_class_weights_by_step(
    examples: Dict[str, np.ndarray],
    *,
    horizon: int,
    num_classes: int,
    cap: float,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    cap = float(cap)
    weights_by_step: Dict[int, torch.Tensor] = {}
    for step in range(1, int(horizon) + 1):
        labels = examples["teacher_forcing_labels"][:, step] if step < int(horizon) else examples["forecast_labels"]
        weights = _balanced_class_weights(labels, num_classes, cap=cap)
        weights_by_step[step] = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return weights_by_step


def _balanced_class_weights(labels: np.ndarray, num_classes: int, cap: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels[labels >= 0], minlength=int(num_classes)).astype(np.float64)
    present = counts > 0.0
    weights = np.ones(int(num_classes), dtype=np.float32)
    if not np.any(present):
        return weights
    mean_count = counts[present].mean()
    weights[present] = (mean_count / np.maximum(counts[present], 1.0)).astype(np.float32)
    if cap > 0.0:
        weights = np.clip(weights, 1.0 / cap, cap)
    weights[~present] = 0.0
    present_mean = weights[present].mean()
    if present_mean > 0.0:
        weights[present] = weights[present] / present_mean
    return weights.astype(np.float32)


def _forecast_baseline_diagnostics(
    examples: Dict[str, Dict[str, np.ndarray]],
    *,
    horizon: int,
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, object]:
    baselines: Dict[str, object] = {}
    for step in range(1, int(horizon) + 1):
        train_targets = _forecast_step_targets(examples["train"], step=step, horizon=horizon)
        split_targets = {
            name: _forecast_step_targets(split_examples, step=step, horizon=horizon)
            for name, split_examples in examples.items()
        }
        baselines[f"h{step}"] = {
            "majority": _majority_baseline_metrics(train_targets, split_targets, num_classes, sil_index),
            "current_activity_transition": _transition_baseline_metrics(
                examples["train"]["activity_labels"],
                train_targets,
                {
                    name: split_examples["activity_labels"]
                    for name, split_examples in examples.items()
                },
                split_targets,
                num_classes,
                sil_index,
            ),
        }
    return baselines


def _flat_forecast_baseline_diagnostics(
    train_labels: np.ndarray,
    split_labels: Dict[str, np.ndarray],
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, object]:
    return {
        "horizon": {
            "majority": _majority_baseline_metrics(train_labels, split_labels, num_classes, sil_index),
        }
    }


def _forecast_step_targets(examples: Dict[str, np.ndarray], *, step: int, horizon: int) -> np.ndarray:
    if int(step) < int(horizon):
        return np.asarray(examples["teacher_forcing_labels"][:, int(step)], dtype=np.int64)
    return np.asarray(examples["forecast_labels"], dtype=np.int64)


def _majority_baseline_metrics(
    train_labels: np.ndarray,
    split_labels: Dict[str, np.ndarray],
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, Dict[str, float]]:
    train_labels = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(train_labels[train_labels >= 0], minlength=int(num_classes))
    majority = int(counts.argmax()) if counts.size else 0
    top3 = _topk_classes_from_counts(counts, k=3)
    return {
        name: _constant_baseline_metrics(labels, majority, top3, num_classes, sil_index)
        for name, labels in split_labels.items()
    }


def _transition_baseline_metrics(
    train_sources: np.ndarray,
    train_targets: np.ndarray,
    split_sources: Dict[str, np.ndarray],
    split_targets: Dict[str, np.ndarray],
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, Dict[str, float]]:
    counts = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    valid = (np.asarray(train_sources) >= 0) & (np.asarray(train_targets) >= 0)
    np.add.at(counts, (np.asarray(train_sources)[valid], np.asarray(train_targets)[valid]), 1)
    fallback_counts = counts.sum(axis=0)
    fallback_class = int(fallback_counts.argmax()) if fallback_counts.size else 0
    fallback_top3 = _topk_classes_from_counts(fallback_counts, k=3)
    majority_by_source = counts.argmax(axis=1)
    top3_by_source = np.stack([_topk_classes_from_counts(row, k=3) for row in counts], axis=0)

    metrics: Dict[str, Dict[str, float]] = {}
    for name, labels in split_targets.items():
        sources = np.asarray(split_sources[name], dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        known_source = (sources >= 0) & (sources < int(num_classes)) & (counts[sources.clip(0, int(num_classes) - 1)].sum(axis=1) > 0)
        preds = np.full(labels.shape, fallback_class, dtype=np.int64)
        preds[known_source] = majority_by_source[sources[known_source]]
        top3_preds = np.tile(fallback_top3[None, :], (labels.shape[0], 1))
        top3_preds[known_source] = top3_by_source[sources[known_source]]
        metrics[name] = _prediction_baseline_metrics(labels, preds, top3_preds, num_classes, sil_index)
    return metrics


def _constant_baseline_metrics(
    labels: np.ndarray,
    pred_class: int,
    top3_classes: np.ndarray,
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.full(labels.shape, int(pred_class), dtype=np.int64)
    top3_preds = np.tile(np.asarray(top3_classes, dtype=np.int64)[None, :], (labels.shape[0], 1))
    return _prediction_baseline_metrics(labels, preds, top3_preds, num_classes, sil_index)


def _prediction_baseline_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    top3_preds: np.ndarray,
    num_classes: int,
    sil_index: int | None,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    metrics = _classification_metrics(labels, preds, num_classes)
    metrics.update(_sil_metrics(labels, preds, sil_index))
    metrics["top3_accuracy"] = float(np.mean(np.any(top3_preds == labels[:, None], axis=1))) if labels.size else 0.0
    metrics["num_examples"] = int(labels.shape[0])
    return metrics


def _topk_classes_from_counts(counts: np.ndarray, k: int) -> np.ndarray:
    counts = np.asarray(counts)
    top_k = min(int(k), int(counts.shape[0]))
    ordered = np.argsort(-counts, kind="stable")[:top_k].astype(np.int64)
    if top_k < int(k):
        ordered = np.pad(ordered, (0, int(k) - top_k), constant_values=0)
    return ordered


def _scheduled_sampling_ratio(epoch: int, num_epochs: int, start_ratio: float, end_ratio: float) -> float:
    start = float(np.clip(start_ratio, 0.0, 1.0))
    end = float(np.clip(end_ratio, 0.0, 1.0))
    if int(num_epochs) <= 1:
        return start
    progress = (int(epoch) - 1) / max(int(num_epochs) - 1, 1)
    return float(start + progress * (end - start))


def _sliding_window_examples(
    split: Dict[str, np.ndarray],
    *,
    horizon: int,
    history_length: int,
    memory_prefix_length: int = 0,
    transition_boundary_radius: int = 2,
) -> Dict[str, np.ndarray]:
    concepts = np.asarray(split["concepts_std"], dtype=np.float32)
    mask = np.asarray(split["mask"], dtype=np.float32)
    example_mask = np.asarray(split.get("example_mask", mask), dtype=np.float32)
    activity_labels = np.asarray(split["activity_labels"], dtype=np.int64)
    forecast_labels = np.asarray(split[forecast_key(horizon)], dtype=np.int64)
    lengths = np.asarray(split["lengths"], dtype=np.int64)
    history_length = int(history_length)
    memory_prefix_length = max(int(memory_prefix_length), 0)
    num_concepts = int(concepts.shape[-1])

    windows: List[np.ndarray] = []
    key_padding_masks: List[np.ndarray] = []
    memory_prefix_windows: List[np.ndarray] = []
    memory_prefix_key_padding_masks: List[np.ndarray] = []
    activity_targets: List[int] = []
    forecast_targets: List[int] = []
    teacher_forcing_targets: List[np.ndarray] = []
    future_concept_targets: List[np.ndarray] = []
    transition_flags: List[bool] = []
    transition_boundary_radius = max(int(transition_boundary_radius), 0)

    for video_idx in range(concepts.shape[0]):
        length = int(lengths[video_idx])
        for timestep in range(length):
            if mask[video_idx, timestep] <= 0.0:
                continue
            if example_mask[video_idx, timestep] <= 0.0:
                continue
            activity_target = int(activity_labels[video_idx, timestep])
            forecast_target = int(forecast_labels[video_idx, timestep])
            if activity_target < 0 or forecast_target < 0:
                continue
            teacher_forcing = activity_labels[video_idx, timestep : timestep + int(horizon)]
            if teacher_forcing.shape[0] != int(horizon) or np.any(teacher_forcing < 0):
                continue
            future_concepts = concepts[video_idx, timestep + 1 : timestep + int(horizon) + 1]
            future_concept_mask = mask[video_idx, timestep + 1 : timestep + int(horizon) + 1]
            if future_concepts.shape[0] != int(horizon) or np.any(future_concept_mask <= 0.0):
                continue

            start = max(0, timestep - history_length + 1)
            history = concepts[video_idx, start : timestep + 1]
            history_mask = mask[video_idx, start : timestep + 1] > 0.0
            padded = np.zeros((history_length, num_concepts), dtype=np.float32)
            key_padding_mask = np.ones(history_length, dtype=bool)
            offset = history_length - int(history.shape[0])
            padded[offset:] = history
            key_padding_mask[offset:] = ~history_mask

            windows.append(padded)
            key_padding_masks.append(key_padding_mask)
            if memory_prefix_length > 0:
                prefix_end = start
                prefix_start = max(0, prefix_end - memory_prefix_length)
                prefix = concepts[video_idx, prefix_start:prefix_end]
                prefix_mask = mask[video_idx, prefix_start:prefix_end] > 0.0
                prefix_padded = np.zeros((memory_prefix_length, num_concepts), dtype=np.float32)
                prefix_padding_mask = np.ones(memory_prefix_length, dtype=bool)
                prefix_offset = memory_prefix_length - int(prefix.shape[0])
                if prefix.shape[0] > 0:
                    prefix_padded[prefix_offset:] = prefix
                    prefix_padding_mask[prefix_offset:] = ~prefix_mask
                memory_prefix_windows.append(prefix_padded)
                memory_prefix_key_padding_masks.append(prefix_padding_mask)
            activity_targets.append(activity_target)
            forecast_targets.append(forecast_target)
            teacher_forcing_targets.append(teacher_forcing.astype(np.int64, copy=False))
            future_concept_targets.append(future_concepts.astype(np.float32, copy=False))
            transition_flags.append(
                _has_nearby_label_transition(
                    activity_labels[video_idx],
                    length=length,
                    timestep=timestep,
                    horizon=horizon,
                    radius=transition_boundary_radius,
                )
            )

    if not windows:
        raise ValueError("No valid sliding-window examples were found for sequence training.")

    result = {
        "concepts": np.stack(windows, axis=0).astype(np.float32),
        "key_padding_mask": np.stack(key_padding_masks, axis=0).astype(bool),
        "activity_labels": np.asarray(activity_targets, dtype=np.int64),
        "forecast_labels": np.asarray(forecast_targets, dtype=np.int64),
        "teacher_forcing_labels": np.stack(teacher_forcing_targets, axis=0).astype(np.int64),
        "future_concepts": np.stack(future_concept_targets, axis=0).astype(np.float32),
        "transition_flags": np.asarray(transition_flags, dtype=bool),
    }
    if memory_prefix_length > 0:
        result["memory_prefix_concepts"] = np.stack(memory_prefix_windows, axis=0).astype(np.float32)
        result["memory_prefix_key_padding_mask"] = np.stack(memory_prefix_key_padding_masks, axis=0).astype(bool)
    return result


def _has_nearby_label_transition(
    labels: np.ndarray,
    *,
    length: int,
    timestep: int,
    horizon: int,
    radius: int,
) -> bool:
    start = max(0, int(timestep) - int(radius))
    end = min(int(length) - 1, int(timestep) + int(horizon) + int(radius))
    if end <= start:
        return False
    window = np.asarray(labels[start : end + 1], dtype=np.int64)
    window = window[window >= 0]
    return bool(window.shape[0] > 1 and np.any(window[1:] != window[:-1]))


def _activity_window_examples(
    split: Dict[str, np.ndarray],
    *,
    horizon: int,
    history_length: int,
    memory_prefix_length: int = 0,
    transition_boundary_radius: int = 2,
) -> Dict[str, np.ndarray]:
    concepts = np.asarray(split["concepts_std"], dtype=np.float32)
    mask = np.asarray(split["mask"], dtype=np.float32)
    example_mask = np.asarray(split.get("example_mask", mask), dtype=np.float32)
    activity_labels = np.asarray(split["activity_labels"], dtype=np.int64)
    lengths = np.asarray(split["lengths"], dtype=np.int64)
    history_length = int(history_length)
    memory_prefix_length = max(int(memory_prefix_length), 0)
    num_concepts = int(concepts.shape[-1])

    windows: List[np.ndarray] = []
    key_padding_masks: List[np.ndarray] = []
    memory_prefix_windows: List[np.ndarray] = []
    memory_prefix_key_padding_masks: List[np.ndarray] = []
    activity_targets: List[int] = []
    transition_flags: List[bool] = []
    transition_boundary_radius = max(int(transition_boundary_radius), 0)

    for video_idx in range(concepts.shape[0]):
        length = int(lengths[video_idx])
        for timestep in range(length):
            if mask[video_idx, timestep] <= 0.0:
                continue
            if example_mask[video_idx, timestep] <= 0.0:
                continue
            activity_target = int(activity_labels[video_idx, timestep])
            if activity_target < 0:
                continue

            start = max(0, timestep - history_length + 1)
            history = concepts[video_idx, start : timestep + 1]
            history_mask = mask[video_idx, start : timestep + 1] > 0.0
            padded = np.zeros((history_length, num_concepts), dtype=np.float32)
            key_padding_mask = np.ones(history_length, dtype=bool)
            offset = history_length - int(history.shape[0])
            padded[offset:] = history
            key_padding_mask[offset:] = ~history_mask

            windows.append(padded)
            key_padding_masks.append(key_padding_mask)
            if memory_prefix_length > 0:
                prefix_end = start
                prefix_start = max(0, prefix_end - memory_prefix_length)
                prefix = concepts[video_idx, prefix_start:prefix_end]
                prefix_mask = mask[video_idx, prefix_start:prefix_end] > 0.0
                prefix_padded = np.zeros((memory_prefix_length, num_concepts), dtype=np.float32)
                prefix_padding_mask = np.ones(memory_prefix_length, dtype=bool)
                prefix_offset = memory_prefix_length - int(prefix.shape[0])
                if prefix.shape[0] > 0:
                    prefix_padded[prefix_offset:] = prefix
                    prefix_padding_mask[prefix_offset:] = ~prefix_mask
                memory_prefix_windows.append(prefix_padded)
                memory_prefix_key_padding_masks.append(prefix_padding_mask)
            activity_targets.append(activity_target)
            transition_flags.append(
                _has_nearby_label_transition(
                    activity_labels[video_idx],
                    length=length,
                    timestep=timestep,
                    horizon=horizon,
                    radius=transition_boundary_radius,
                )
            )

    if not windows:
        raise ValueError("No valid activity-window examples were found for sequence training.")

    result = {
        "concepts": np.stack(windows, axis=0).astype(np.float32),
        "key_padding_mask": np.stack(key_padding_masks, axis=0).astype(bool),
        "activity_labels": np.asarray(activity_targets, dtype=np.int64),
        "transition_flags": np.asarray(transition_flags, dtype=bool),
    }
    if memory_prefix_length > 0:
        result["memory_prefix_concepts"] = np.stack(memory_prefix_windows, axis=0).astype(np.float32)
        result["memory_prefix_key_padding_mask"] = np.stack(memory_prefix_key_padding_masks, axis=0).astype(bool)
    return result


def _train_sampling_info(
    examples: Dict[str, np.ndarray],
    *,
    strategy: str,
    num_classes: int,
    boundary_radius: int,
    strength: float,
    rare_alpha: float,
) -> Dict[str, object]:
    strategy = str(strategy)
    if strategy not in {"uniform", "transition_aware"}:
        raise ValueError("train_sampling_strategy must be one of: uniform, transition_aware")
    labels = np.asarray(examples.get("forecast_labels", examples["activity_labels"]), dtype=np.int64)
    transition_flags = np.asarray(examples.get("transition_flags", np.zeros(labels.shape, dtype=bool)), dtype=bool)
    info: Dict[str, object] = {
        "strategy": strategy,
        "boundary_radius": int(boundary_radius),
        "strength": float(strength),
        "rare_alpha": float(rare_alpha),
        "num_examples": int(labels.shape[0]),
        "transition_example_rate": float(transition_flags.mean()) if transition_flags.size else 0.0,
    }
    if strategy == "uniform" or labels.size == 0:
        info["weights"] = None
        return info

    counts = np.bincount(labels[labels >= 0], minlength=int(num_classes)).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    alpha = max(float(rare_alpha), 0.0)
    rare = safe_counts[np.clip(labels, 0, int(num_classes) - 1)] ** (-alpha)
    rare = rare / max(float(rare.mean()), 1e-12)

    boundary_multiplier = np.where(transition_flags, max(float(strength), 1.0), 1.0)
    weights = rare * boundary_multiplier
    weights = np.where(labels >= 0, weights, 0.0).astype(np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full(labels.shape[0], 1.0 / max(labels.shape[0], 1), dtype=np.float64)
    else:
        weights = weights / total
    info["weights"] = weights
    info["max_weight"] = float(weights.max()) if weights.size else 0.0
    info["min_weight"] = float(weights.min()) if weights.size else 0.0
    return info


def _epoch_train_indices(examples: Dict[str, np.ndarray], sampling_info: Dict[str, object]) -> np.ndarray:
    num_examples = int(examples["concepts"].shape[0])
    weights = sampling_info.get("weights")
    if weights is None:
        return np.random.permutation(num_examples)
    return np.random.choice(num_examples, size=num_examples, replace=True, p=np.asarray(weights, dtype=np.float64))


def _serializable_train_sampling_info(info: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in info.items() if key != "weights"}


def _iter_example_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, int(indices.shape[0]), batch_size):
        yield indices[start : start + batch_size].astype(np.int64, copy=False)


def _example_batch_to_torch(
    examples: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch = {
        "concepts": torch.as_tensor(examples["concepts"][indices], dtype=torch.float32, device=device),
        "key_padding_mask": torch.as_tensor(examples["key_padding_mask"][indices], dtype=torch.bool, device=device),
        "activity_labels": torch.as_tensor(examples["activity_labels"][indices], dtype=torch.long, device=device),
    }
    if "forecast_labels" in examples:
        batch["forecast_labels"] = torch.as_tensor(examples["forecast_labels"][indices], dtype=torch.long, device=device)
    if "teacher_forcing_labels" in examples:
        batch["teacher_forcing_labels"] = torch.as_tensor(
            examples["teacher_forcing_labels"][indices],
            dtype=torch.long,
            device=device,
        )
    if "future_concepts" in examples:
        batch["future_concepts"] = torch.as_tensor(examples["future_concepts"][indices], dtype=torch.float32, device=device)
    if "memory_prefix_concepts" in examples:
        batch["memory_prefix_concepts"] = torch.as_tensor(
            examples["memory_prefix_concepts"][indices],
            dtype=torch.float32,
            device=device,
        )
    if "memory_prefix_key_padding_mask" in examples:
        batch["memory_prefix_key_padding_mask"] = torch.as_tensor(
            examples["memory_prefix_key_padding_mask"][indices],
            dtype=torch.bool,
            device=device,
        )
    return batch


@torch.no_grad()
def _evaluate_sequence(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    num_activities: int,
    device: torch.device,
    sil_index: int | None = None,
    activity_only: bool = False,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    activity_true, activity_pred = [], []
    forecast_true, forecast_pred = [], []
    forecast_true_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    forecast_pred_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    forecast_top3_hits_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}
    forecast_count_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}
    activity_loss_sum = 0.0
    forecast_loss_sum = 0.0
    activity_top3_hits = 0
    forecast_top3_hits = 0
    activity_count = 0
    forecast_count = 0
    concept_forecast_loss_sum = 0.0
    concept_forecast_count = 0

    for indices in _iter_example_batches(np.arange(examples["concepts"].shape[0]), batch_size):
        batch = _example_batch_to_torch(examples, indices, device)
        outputs = _forward_outputs(
            model,
            method,
            batch["concepts"],
            batch["key_padding_mask"],
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )
        activity_logits = outputs["activity_logits"][:, -1, :]

        activity_loss = F.cross_entropy(activity_logits, batch["activity_labels"])
        count = int(batch["activity_labels"].shape[0])

        activity_loss_sum += float(activity_loss.item()) * count
        activity_top3_hits += _topk_hit_count(activity_logits, batch["activity_labels"], k=3)
        activity_count += count
        activity_true.append(batch["activity_labels"].cpu().numpy())
        activity_pred.append(activity_logits.argmax(dim=-1).cpu().numpy())
        if not activity_only:
            forecast_logits = _forecast_logits(outputs, method, horizon)[:, -1, :]
            logits_by_step = _forecast_logits_by_step(outputs, method, horizon)
            forecast_loss = F.cross_entropy(forecast_logits, batch["forecast_labels"])
            forecast_loss_sum += float(forecast_loss.item()) * count
            forecast_top3_hits += _topk_hit_count(forecast_logits, batch["forecast_labels"], k=3)
            forecast_count += count
            forecast_true.append(batch["forecast_labels"].cpu().numpy())
            forecast_pred.append(forecast_logits.argmax(dim=-1).cpu().numpy())
            if outputs.get("predicted_concepts_by_step") and "future_concepts" in batch:
                concept_loss = _concept_forecast_loss(model, outputs, batch, deadzone_std=0.0)
                concept_forecast_loss_sum += float(concept_loss.item()) * count
                concept_forecast_count += count
            for step, logits in logits_by_step.items():
                step_target = batch["teacher_forcing_labels"][:, step] if step < int(horizon) else batch["forecast_labels"]
                step_logits = logits[:, -1, :]
                forecast_top3_hits_by_step[step] += _topk_hit_count(step_logits, step_target, k=3)
                forecast_count_by_step[step] += int(step_target.shape[0])
                forecast_true_by_step[step].append(step_target.cpu().numpy())
                forecast_pred_by_step[step].append(step_logits.argmax(dim=-1).cpu().numpy())

    activity_labels_np = np.concatenate(activity_true) if activity_true else np.zeros(0, dtype=np.int64)
    activity_preds_np = np.concatenate(activity_pred) if activity_pred else np.zeros(0, dtype=np.int64)
    forecast_labels_np = np.concatenate(forecast_true) if forecast_true else np.zeros(0, dtype=np.int64)
    forecast_preds_np = np.concatenate(forecast_pred) if forecast_pred else np.zeros(0, dtype=np.int64)
    activity_metrics = _classification_metrics(activity_labels_np, activity_preds_np, num_activities)
    activity_metrics.update(_component_accuracy_metrics(activity_labels_np, activity_preds_np, activity_components))
    forecast_metrics = _classification_metrics(
        forecast_labels_np,
        forecast_preds_np,
        num_activities,
    )
    forecast_metrics.update(_component_accuracy_metrics(forecast_labels_np, forecast_preds_np, activity_components))
    forecast_metrics.update(
        _sil_metrics(
            forecast_labels_np,
            forecast_preds_np,
            sil_index,
        )
    )
    forecast_by_horizon = {}
    for step in range(1, int(horizon) + 1):
        labels = np.concatenate(forecast_true_by_step[step]) if forecast_true_by_step[step] else np.zeros(0, dtype=np.int64)
        preds = np.concatenate(forecast_pred_by_step[step]) if forecast_pred_by_step[step] else np.zeros(0, dtype=np.int64)
        step_metrics = _classification_metrics(labels, preds, num_activities)
        step_metrics.update(_component_accuracy_metrics(labels, preds, activity_components))
        step_metrics.update(_sil_metrics(labels, preds, sil_index))
        step_metrics["top3_accuracy"] = forecast_top3_hits_by_step[step] / max(forecast_count_by_step[step], 1)
        step_metrics["num_examples"] = int(labels.shape[0])
        forecast_by_horizon[str(step)] = step_metrics
    activity_metrics["loss"] = activity_loss_sum / max(activity_count, 1)
    activity_metrics["top3_accuracy"] = activity_top3_hits / max(activity_count, 1)
    activity_metrics["num_examples"] = int(activity_count)
    if activity_only:
        return {"activity": activity_metrics}
    forecast_metrics["loss"] = forecast_loss_sum / max(forecast_count, 1)
    forecast_metrics["top3_accuracy"] = forecast_top3_hits / max(forecast_count, 1)
    forecast_metrics["num_examples"] = int(forecast_count)
    result = {"activity": activity_metrics, "forecast": forecast_metrics, "forecast_by_horizon": forecast_by_horizon}
    if concept_forecast_count > 0:
        result["concept_forecast"] = {
            "smooth_l1": concept_forecast_loss_sum / concept_forecast_count,
            "num_examples": int(concept_forecast_count),
        }
    return result


def _evaluate_sequence_with_graph_disabled(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    num_activities: int,
    device: torch.device,
    sil_index: int | None = None,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    saved_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    attrs = (
        "spatial_weight",
        "temporal_weight",
        "cross_temporal_weight",
        "same_weight",
        "lag_weight",
    )
    with torch.no_grad():
        for layer in _graph_modules_for_disable(model):
            for attr in attrs:
                value = getattr(layer, attr, None)
                if torch.is_tensor(value):
                    saved_tensors.append((value, value.detach().clone()))
                    value.zero_()
    try:
        metrics = _evaluate_sequence(
            model,
            examples,
            method,
            horizon,
            batch_size,
            num_activities,
            device,
            sil_index=sil_index,
            activity_only=False,
            activity_components=activity_components,
        )
    finally:
        with torch.no_grad():
            for tensor, saved in saved_tensors:
                tensor.copy_(saved)
    metrics["graph_disabled_num_zeroed_tensors"] = {"count": float(len(saved_tensors))}
    return metrics


def _graph_corruption_metadata(
    *,
    enabled: bool,
    seed: int,
    applied: bool,
    skip_reason: str | None = None,
    corrupted_tensor_pairs: int = 0,
    corrupted_entries: int = 0,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "enabled": bool(enabled),
        "applied": bool(applied),
        "mode": "shuffle_edges",
        "seed": int(seed),
        "corrupted_tensor_pairs": int(corrupted_tensor_pairs),
        "corrupted_entries": int(corrupted_entries),
    }
    if skip_reason:
        metadata["skip_reason"] = str(skip_reason)
    return metadata


@torch.no_grad()
def _evaluate_sequence_with_corrupted_graph(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    num_activities: int,
    device: torch.device,
    *,
    baseline_metrics: Dict[str, Dict[str, float]],
    enabled: bool,
    seed: int,
    sil_index: int | None = None,
    activity_only: bool = False,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    if not bool(enabled):
        return {}, {}, _graph_corruption_metadata(
            enabled=False,
            seed=seed,
            applied=False,
            skip_reason="disabled",
        )

    targets = _graph_corruption_targets(model)
    if not targets:
        return {}, {}, _graph_corruption_metadata(
            enabled=True,
            seed=seed,
            applied=False,
            skip_reason="no_graph_edges",
        )

    saved_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    seen: set[int] = set()
    for _, weight, gate, _ in targets:
        for tensor in (weight, gate):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            saved_tensors.append((tensor, tensor.detach().clone()))

    try:
        corruption_stats = _shuffle_graph_edge_targets(targets, seed=seed)
        if corruption_stats["corrupted_entries"] <= 0:
            return {}, {}, _graph_corruption_metadata(
                enabled=True,
                seed=seed,
                applied=False,
                skip_reason="insufficient_graph_edges",
            )
        corrupted_metrics = _evaluate_sequence(
            model,
            examples,
            method,
            horizon,
            batch_size,
            num_activities,
            device,
            sil_index=sil_index,
            activity_only=activity_only,
            activity_components=activity_components,
        )
        corrupted_accuracy = _accuracy_only_metrics(corrupted_metrics)
        delta_metrics = _accuracy_delta_metrics(baseline_metrics, corrupted_metrics)
        metadata = _graph_corruption_metadata(
            enabled=True,
            seed=seed,
            applied=True,
            corrupted_tensor_pairs=corruption_stats["corrupted_tensor_pairs"],
            corrupted_entries=corruption_stats["corrupted_entries"],
        )
        return corrupted_accuracy, delta_metrics, metadata
    finally:
        with torch.no_grad():
            for tensor, saved in saved_tensors:
                tensor.copy_(saved)


def _graph_corruption_targets(model: nn.Module) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    targets: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    for module_index, module in enumerate(_graph_modules_for_disable(model)):
        _append_graph_corruption_target(
            targets,
            seen_pairs,
            name=f"module_{module_index}.spatial",
            weight=getattr(module, "spatial_weight", None),
            gate=getattr(module, "spatial_gate_logits", None),
            mask=getattr(module, "spatial_mask", None),
        )
        if bool(getattr(module, "enable_same_concept_temporal", True)):
            _append_graph_corruption_target(
                targets,
                seen_pairs,
                name=f"module_{module_index}.temporal",
                weight=getattr(module, "temporal_weight", None),
                gate=getattr(module, "temporal_gate_logits", None),
                mask=None,
            )
        if bool(getattr(module, "enable_cross_temporal", False)):
            _append_graph_corruption_target(
                targets,
                seen_pairs,
                name=f"module_{module_index}.cross_temporal",
                weight=getattr(module, "cross_temporal_weight", None),
                gate=getattr(module, "cross_temporal_gate_logits", None),
                mask=getattr(module, "cross_temporal_mask", None),
            )
        same_mask = getattr(module, "same_mask", None)
        same_prune_mask = getattr(module, "same_prune_mask", None)
        if torch.is_tensor(same_mask) and torch.is_tensor(same_prune_mask):
            same_mask = same_mask * same_prune_mask
        _append_graph_corruption_target(
            targets,
            seen_pairs,
            name=f"module_{module_index}.same",
            weight=getattr(module, "same_weight", None),
            gate=getattr(module, "same_gate_logits", None),
            mask=same_mask,
        )
        lag_mask = getattr(module, "lag_mask", None)
        lag_prune_mask = getattr(module, "lag_prune_mask", None)
        if torch.is_tensor(lag_mask) and torch.is_tensor(lag_prune_mask):
            lag_mask = lag_mask * lag_prune_mask
        _append_graph_corruption_target(
            targets,
            seen_pairs,
            name=f"module_{module_index}.lag",
            weight=getattr(module, "lag_weight", None),
            gate=getattr(module, "lag_gate_logits", None),
            mask=lag_mask,
        )
    return targets


def _append_graph_corruption_target(
    targets: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    seen_pairs: set[Tuple[int, int]],
    *,
    name: str,
    weight: object,
    gate: object,
    mask: object,
) -> None:
    if not torch.is_tensor(weight) or not torch.is_tensor(gate):
        return
    if tuple(weight.shape) != tuple(gate.shape):
        return
    pair_key = (id(weight), id(gate))
    if pair_key in seen_pairs:
        return
    seen_pairs.add(pair_key)
    if torch.is_tensor(mask):
        valid_mask = mask.to(device=weight.device, dtype=torch.bool)
    else:
        valid_mask = torch.ones_like(weight, dtype=torch.bool)
    if tuple(valid_mask.shape) != tuple(weight.shape) or int(valid_mask.sum().item()) < 2:
        return
    targets.append((name, weight, gate, valid_mask))


@torch.no_grad()
def _shuffle_graph_edge_targets(
    targets: Sequence[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    seed: int,
) -> Dict[str, int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    corrupted_tensor_pairs = 0
    corrupted_entries = 0
    for _, weight, gate, valid_mask in targets:
        count = int(valid_mask.sum().item())
        if count < 2:
            continue
        perm_cpu = torch.randperm(count, generator=generator)
        if torch.equal(perm_cpu, torch.arange(count)):
            perm_cpu = perm_cpu.roll(1)
        weight_values = weight[valid_mask].detach().clone()
        gate_values = gate[valid_mask.to(device=gate.device)].detach().clone()
        weight[valid_mask] = weight_values[perm_cpu.to(device=weight_values.device)]
        gate_valid_mask = valid_mask.to(device=gate.device)
        gate[gate_valid_mask] = gate_values[perm_cpu.to(device=gate_values.device)]
        corrupted_tensor_pairs += 1
        corrupted_entries += count
    return {
        "corrupted_tensor_pairs": int(corrupted_tensor_pairs),
        "corrupted_entries": int(corrupted_entries),
    }


def _accuracy_only_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key in ("activity", "forecast"):
        value = metrics.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("accuracy"), (int, float)):
            output[key] = {"accuracy": float(value["accuracy"])}
    forecast_by_horizon = metrics.get("forecast_by_horizon")
    if isinstance(forecast_by_horizon, Mapping):
        horizon_output = {}
        for horizon, value in forecast_by_horizon.items():
            if isinstance(value, Mapping) and isinstance(value.get("accuracy"), (int, float)):
                horizon_output[str(horizon)] = {"accuracy": float(value["accuracy"])}
        if horizon_output:
            output["forecast_by_horizon"] = horizon_output
    return output


def _accuracy_delta_metrics(
    baseline_metrics: Mapping[str, object],
    corrupted_metrics: Mapping[str, object],
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key in ("activity", "forecast"):
        baseline = baseline_metrics.get(key)
        corrupted = corrupted_metrics.get(key)
        if (
            isinstance(baseline, Mapping)
            and isinstance(corrupted, Mapping)
            and isinstance(baseline.get("accuracy"), (int, float))
            and isinstance(corrupted.get("accuracy"), (int, float))
        ):
            output[key] = {
                "accuracy_delta": float(baseline["accuracy"]) - float(corrupted["accuracy"]),
            }
    baseline_by_horizon = baseline_metrics.get("forecast_by_horizon")
    corrupted_by_horizon = corrupted_metrics.get("forecast_by_horizon")
    if isinstance(baseline_by_horizon, Mapping) and isinstance(corrupted_by_horizon, Mapping):
        horizon_output = {}
        for horizon, baseline in baseline_by_horizon.items():
            corrupted = corrupted_by_horizon.get(horizon)
            if (
                isinstance(baseline, Mapping)
                and isinstance(corrupted, Mapping)
                and isinstance(baseline.get("accuracy"), (int, float))
                and isinstance(corrupted.get("accuracy"), (int, float))
            ):
                horizon_output[str(horizon)] = {
                    "accuracy_delta": float(baseline["accuracy"]) - float(corrupted["accuracy"]),
                }
        if horizon_output:
            output["forecast_by_horizon"] = horizon_output
    return output


def _graph_modules_for_disable(model: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    if hasattr(model, "_all_graph_layers"):
        modules.extend(list(model._all_graph_layers()))
    for branch_name in ("shared", "window", "forecast"):
        layers = getattr(model, f"{branch_name}_graph_layers", None)
        if layers is not None:
            modules.extend(list(layers))
    if any(hasattr(model, attr) for attr in ("same_weight", "lag_weight")):
        modules.append(model)
    unique: List[nn.Module] = []
    seen: set[int] = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        unique.append(module)
    return unique


@torch.no_grad()
def _intervene_metrics(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    num_activities: int,
    device: torch.device,
) -> Dict[str, float]:
    del num_activities
    model.eval()
    totals = {
        "num_examples": 0.0,
        "input_abs_delta_mean": 0.0,
        "concept_same_window_other_l1": 0.0,
        "concept_same_concept_other_window_l1": 0.0,
        "concept_other_window_l1": 0.0,
        "concept_past_window_l1": 0.0,
        "concept_future_window_l1": 0.0,
        "activity_logit_l1": 0.0,
        "activity_prob_l1": 0.0,
        "activity_flip_rate": 0.0,
        "activity_true_prob_delta": 0.0,
        "forecast_logit_l1": 0.0,
        "forecast_prob_l1": 0.0,
        "forecast_flip_rate": 0.0,
        "forecast_true_prob_delta": 0.0,
    }

    for indices in _iter_example_batches(np.arange(examples["concepts"].shape[0]), batch_size):
        batch = _example_batch_to_torch(examples, indices, device)
        concepts = batch["concepts"]
        key_padding_mask = batch["key_padding_mask"]
        batch_size_actual, timesteps, num_concepts = concepts.shape
        valid_mask = ~key_padding_mask
        if batch_size_actual == 0:
            continue

        baseline = _forward_outputs(
            model,
            method,
            concepts,
            key_padding_mask,
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )
        intervened_concepts = concepts.clone()
        intervention_mask = torch.zeros_like(concepts, dtype=torch.bool)
        time_indices = torch.empty(batch_size_actual, dtype=torch.long, device=device)
        concept_indices = torch.randint(num_concepts, (batch_size_actual,), device=device)
        signs = torch.where(
            torch.rand(batch_size_actual, device=device) < 0.5,
            torch.full((batch_size_actual,), -1.0, device=device),
            torch.ones(batch_size_actual, device=device),
        )

        for row_idx in range(batch_size_actual):
            valid_positions = torch.nonzero(valid_mask[row_idx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                time_idx = timesteps - 1
            else:
                selected = torch.randint(valid_positions.numel(), (1,), device=device)
                time_idx = int(valid_positions[selected].item())
            time_indices[row_idx] = time_idx

        rows = torch.arange(batch_size_actual, device=device)
        intervened_concepts[rows, time_indices, concept_indices] += signs
        intervention_mask[rows, time_indices, concept_indices] = True
        intervened = _forward_outputs(
            model,
            method,
            intervened_concepts,
            key_padding_mask,
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )

        concept_before = _concept_state_for_intervention(baseline)
        concept_after = _concept_state_for_intervention(intervened)
        concept_delta = (concept_after - concept_before).abs()
        activity_before = baseline["activity_logits"][:, -1, :]
        activity_after = intervened["activity_logits"][:, -1, :]
        forecast_before = _forecast_logits(baseline, method, horizon)[:, -1, :]
        forecast_after = _forecast_logits(intervened, method, horizon)[:, -1, :]

        valid_3d = valid_mask[:, :, None].expand_as(concept_delta)
        same_time = torch.zeros_like(concept_delta, dtype=torch.bool)
        same_time[rows, time_indices, :] = True
        same_concept = torch.zeros_like(concept_delta, dtype=torch.bool)
        same_concept[rows, :, concept_indices] = True
        past_time = torch.arange(timesteps, device=device)[None, :] < time_indices[:, None]
        future_time = torch.arange(timesteps, device=device)[None, :] > time_indices[:, None]
        past_mask = past_time[:, :, None].expand_as(concept_delta)
        future_mask = future_time[:, :, None].expand_as(concept_delta)

        activity_prob_before = torch.softmax(activity_before, dim=-1)
        activity_prob_after = torch.softmax(activity_after, dim=-1)
        forecast_prob_before = torch.softmax(forecast_before, dim=-1)
        forecast_prob_after = torch.softmax(forecast_after, dim=-1)
        activity_labels = batch["activity_labels"]
        forecast_labels = batch["forecast_labels"]

        n = float(batch_size_actual)
        totals["num_examples"] += n
        totals["input_abs_delta_mean"] += float((intervened_concepts - concepts).abs()[intervention_mask].mean().item()) * n
        totals["concept_same_window_other_l1"] += _masked_mean_abs(
            concept_delta,
            valid_3d & same_time & ~intervention_mask,
        ) * n
        totals["concept_same_concept_other_window_l1"] += _masked_mean_abs(
            concept_delta,
            valid_3d & same_concept & ~same_time,
        ) * n
        totals["concept_other_window_l1"] += _masked_mean_abs(
            concept_delta,
            valid_3d & ~same_time,
        ) * n
        totals["concept_past_window_l1"] += _masked_mean_abs(concept_delta, valid_3d & past_mask) * n
        totals["concept_future_window_l1"] += _masked_mean_abs(concept_delta, valid_3d & future_mask) * n
        totals["activity_logit_l1"] += float((activity_after - activity_before).abs().mean().item()) * n
        totals["activity_prob_l1"] += float((activity_prob_after - activity_prob_before).abs().mean().item()) * n
        totals["activity_flip_rate"] += float(
            (activity_after.argmax(dim=-1) != activity_before.argmax(dim=-1)).float().mean().item()
        ) * n
        totals["activity_true_prob_delta"] += float(
            (
                activity_prob_after[rows, activity_labels]
                - activity_prob_before[rows, activity_labels]
            ).mean().item()
        ) * n
        totals["forecast_logit_l1"] += float((forecast_after - forecast_before).abs().mean().item()) * n
        totals["forecast_prob_l1"] += float((forecast_prob_after - forecast_prob_before).abs().mean().item()) * n
        totals["forecast_flip_rate"] += float(
            (forecast_after.argmax(dim=-1) != forecast_before.argmax(dim=-1)).float().mean().item()
        ) * n
        totals["forecast_true_prob_delta"] += float(
            (
                forecast_prob_after[rows, forecast_labels]
                - forecast_prob_before[rows, forecast_labels]
            ).mean().item()
        ) * n

    count = max(totals.pop("num_examples"), 1.0)
    return {"num_examples": count, **{key: value / count for key, value in totals.items()}}


@torch.no_grad()
def _activity_feedback_intervention_metrics(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    device: torch.device,
    intervention_probability: float = 0.9,
) -> Dict[str, float]:
    enabled_fn = getattr(model, "_activity_feedback_enabled", None)
    if method not in GRAPH_CBM_METHODS or not callable(enabled_fn) or not bool(enabled_fn()):
        return {"enabled": 0.0, "num_examples": 0.0}
    if int(getattr(model, "num_activities", 0)) < 2:
        return {"enabled": 1.0, "num_examples": 0.0}

    model.eval()
    totals: Dict[str, float] = {
        "num_examples": 0.0,
        "belief_prob_l1": 0.0,
        "feedback_message_l1": 0.0,
        "concept_feedback_message_l1": 0.0,
        "logit_feedback_message_l1": 0.0,
    }
    for step in range(1, int(horizon) + 1):
        totals[f"concept_state_l1_h{step}"] = 0.0
        totals[f"forecast_prob_l1_h{step}"] = 0.0
        totals[f"forecast_flip_rate_h{step}"] = 0.0

    for indices in _iter_example_batches(np.arange(examples["concepts"].shape[0]), batch_size):
        batch = _example_batch_to_torch(examples, indices, device)
        baseline = _forward_outputs(
            model,
            method,
            batch["concepts"],
            batch["key_padding_mask"],
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )
        current_probs = baseline["effective_activity_probs_by_step"][0][:, -1, :]
        alternatives = torch.topk(current_probs, k=2, dim=-1).indices[:, 1]
        intervention = {
            "items": [
                {
                    "item_type": "activity",
                    "step": 0,
                    "batch_idx": int(row),
                    "class_idx": int(alternatives[row].item()),
                    "probability": float(intervention_probability),
                }
                for row in range(current_probs.size(0))
            ]
        }
        intervened = _forward_outputs(
            model,
            method,
            batch["concepts"],
            batch["key_padding_mask"],
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
            intervention=intervention,
        )
        after_probs = intervened["effective_activity_probs_by_step"][0][:, -1, :]
        before_message = baseline["activity_feedback_concept_messages_by_step"][0][
            :, -1, :
        ]
        after_message = intervened["activity_feedback_concept_messages_by_step"][0][
            :, -1, :
        ]
        before_logit_message = baseline["activity_feedback_logit_messages_by_step"][0][
            :, -1, :
        ]
        after_logit_message = intervened["activity_feedback_logit_messages_by_step"][0][
            :, -1, :
        ]
        count = float(current_probs.size(0))
        totals["num_examples"] += count
        totals["belief_prob_l1"] += float((after_probs - current_probs).abs().mean().item()) * count
        totals["feedback_message_l1"] += float(
            (after_message - before_message).abs().mean().item()
        ) * count
        totals["concept_feedback_message_l1"] += float(
            (after_message - before_message).abs().mean().item()
        ) * count
        totals["logit_feedback_message_l1"] += float(
            (after_logit_message - before_logit_message).abs().mean().item()
        ) * count

        baseline_logits = _forecast_logits_by_step(baseline, method, horizon)
        intervened_logits = _forecast_logits_by_step(intervened, method, horizon)
        for step in range(1, int(horizon) + 1):
            before_concepts = baseline["predicted_concepts_by_step"][step][:, -1, :]
            after_concepts = intervened["predicted_concepts_by_step"][step][:, -1, :]
            before_forecast = torch.softmax(baseline_logits[step][:, -1, :], dim=-1)
            after_forecast = torch.softmax(intervened_logits[step][:, -1, :], dim=-1)
            totals[f"concept_state_l1_h{step}"] += float(
                (after_concepts - before_concepts).abs().mean().item()
            ) * count
            totals[f"forecast_prob_l1_h{step}"] += float(
                (after_forecast - before_forecast).abs().mean().item()
            ) * count
            totals[f"forecast_flip_rate_h{step}"] += float(
                (
                    after_forecast.argmax(dim=-1)
                    != before_forecast.argmax(dim=-1)
                ).float().mean().item()
            ) * count

    count = totals.pop("num_examples")
    if count <= 0.0:
        return {"enabled": 1.0, "num_examples": 0.0}
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update(
        {
            "enabled": 1.0,
            "num_examples": count,
            "intervention_probability": float(intervention_probability),
            "forecast_prob_l1": metrics[f"forecast_prob_l1_h{int(horizon)}"],
            "forecast_flip_rate": metrics[f"forecast_flip_rate_h{int(horizon)}"],
            "concept_state_l1": metrics[f"concept_state_l1_h{int(horizon)}"],
        }
    )
    return metrics


@torch.no_grad()
def _edge_guided_intervention_metrics(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    method: str,
    horizon: int,
    batch_size: int,
    device: torch.device,
    max_edges: int = 8,
    max_examples: int = 512,
) -> Dict[str, float]:
    if method == "motif" or not hasattr(model, "graph_metrics"):
        return {}
    edges = _active_graph_edges(model, max_edges=max_edges)
    if not edges:
        return {"edge_guided_num_edges": 0.0, "edge_guided_num_events": 0.0}

    model.eval()
    example_count = min(int(max_examples), int(examples["concepts"].shape[0]))
    if example_count <= 0:
        return {"edge_guided_num_edges": float(len(edges)), "edge_guided_num_events": 0.0}

    accum: Dict[str, Dict[str, float]] = {}

    def group(name: str) -> Dict[str, float]:
        if name not in accum:
            accum[name] = {
                "events": 0.0,
                "activity_prob_l1": 0.0,
                "activity_flip_rate": 0.0,
                "forecast_prob_l1": 0.0,
                "forecast_flip_rate": 0.0,
                "shared_state_delta_mean": 0.0,
                "shared_state_delta_max": 0.0,
                "window_state_delta_mean": 0.0,
                "window_state_delta_max": 0.0,
                "forecast_state_delta_mean": 0.0,
                "forecast_state_delta_max": 0.0,
            }
        return accum[name]

    for indices in _iter_example_batches(np.arange(example_count, dtype=np.int64), batch_size):
        batch = _example_batch_to_torch(examples, indices, device)
        concepts = batch["concepts"]
        key_padding_mask = batch["key_padding_mask"]
        valid_mask = ~key_padding_mask
        baseline = _forward_outputs(
            model,
            method,
            concepts,
            key_padding_mask,
            memory_prefix_concepts=batch.get("memory_prefix_concepts"),
            memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
        )
        timesteps = int(concepts.shape[1])

        for edge in edges:
            source_time, target_time = _edge_source_target_times(edge, timesteps)
            valid_rows = valid_mask[:, source_time] & valid_mask[:, target_time]
            n = int(valid_rows.sum().item())
            if n <= 0:
                continue
            for target_z in (0.9, 0.1):
                value = _raw_intervention_value_for_target(model, int(edge["source"]), target_z)
                intervention = {
                    "mode": "persistent",
                    "items": [
                        {
                            "time_idx": source_time,
                            "concept_idx": int(edge["source"]),
                            "value": float(value),
                        }
                    ],
                }
                intervened = model(
                    concepts,
                    key_padding_mask,
                    intervention=intervention,
                    memory_prefix_concepts=batch.get("memory_prefix_concepts"),
                    memory_prefix_key_padding_mask=batch.get("memory_prefix_key_padding_mask"),
                )
                sample_metrics = _edge_guided_sample_metrics(
                    baseline,
                    intervened,
                    method,
                    horizon,
                    valid_rows,
                    target_time=target_time,
                    target_concept=int(edge["target"]),
                )
                for group_name in ("all", str(edge["branch"])):
                    stats = group(group_name)
                    stats["events"] += float(n)
                    for key, metric_value in sample_metrics.items():
                        if key.endswith("_max"):
                            stats[key] = max(stats[key], float(metric_value))
                        else:
                            stats[key] += float(metric_value) * float(n)

    result: Dict[str, float] = {
        "edge_guided_num_edges": float(len(edges)),
        "edge_guided_num_examples": float(example_count),
        "edge_guided_num_events": float(accum.get("all", {}).get("events", 0.0)),
    }
    for group_name, stats in sorted(accum.items()):
        prefix = "edge_guided" if group_name == "all" else f"edge_guided_edges_from_{group_name}"
        events = max(float(stats.get("events", 0.0)), 1.0)
        result[f"{prefix}_num_events"] = float(stats.get("events", 0.0))
        for key, value in stats.items():
            if key == "events":
                continue
            if key.endswith("_max"):
                result[f"{prefix}_{key}"] = float(value)
            else:
                result[f"{prefix}_{key}"] = float(value) / events
    return result


def _active_graph_edges(model: nn.Module, max_edges: int) -> List[Dict[str, object]]:
    edge_rows: List[Dict[str, object]] = []
    for branch_name, layer_idx, layer in _graph_branch_layers(model):
        threshold = float(getattr(layer, "edge_threshold", getattr(model, "edge_threshold", 0.2)))
        if hasattr(layer, "effective_spatial_matrix"):
            spatial = layer.effective_spatial_matrix().detach()
            spatial_gate = torch.sigmoid(layer.spatial_gate_logits).detach() * layer.spatial_mask.detach()
            spatial_active = torch.nonzero((spatial.abs() > 1e-8) & (spatial_gate > threshold), as_tuple=False)
            for source, target in spatial_active.tolist():
                edge_rows.append(
                    {
                        "branch": branch_name,
                        "layer": int(layer_idx),
                        "kind": "spatial",
                        "source": int(source),
                        "target": int(target),
                        "score": float(spatial[source, target].abs().item()),
                        "sign": float(torch.sign(spatial[source, target]).item()),
                    }
                )
        if hasattr(layer, "effective_temporal_vector"):
            temporal = layer.effective_temporal_vector().detach()
            temporal_gate = torch.sigmoid(layer.temporal_gate_logits).detach()
            temporal_active = torch.nonzero((temporal.abs() > 1e-8) & (temporal_gate > threshold), as_tuple=False)
            for concept_idx in temporal_active.flatten().tolist():
                edge_rows.append(
                    {
                        "branch": branch_name,
                        "layer": int(layer_idx),
                        "kind": "temporal",
                        "source": int(concept_idx),
                        "target": int(concept_idx),
                        "score": float(temporal[concept_idx].abs().item()),
                        "sign": float(torch.sign(temporal[concept_idx]).item()),
                    }
                )
        if hasattr(layer, "effective_cross_temporal_matrix"):
            cross = layer.effective_cross_temporal_matrix().detach()
            if cross.numel() > 0 and getattr(layer, "cross_temporal_gate_logits", None) is not None:
                cross_gate = torch.sigmoid(layer.cross_temporal_gate_logits).detach() * layer.cross_temporal_mask.detach()
                cross_active = torch.nonzero((cross.abs() > 1e-8) & (cross_gate > threshold), as_tuple=False)
                for source, target in cross_active.tolist():
                    edge_rows.append(
                        {
                            "branch": branch_name,
                            "layer": int(layer_idx),
                            "kind": "cross_temporal",
                            "source": int(source),
                            "target": int(target),
                            "score": float(cross[source, target].abs().item()),
                            "sign": float(torch.sign(cross[source, target]).item()),
                        }
                    )

    edge_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return edge_rows[: max(int(max_edges), 0)]


def _graph_branch_layers(model: nn.Module) -> List[Tuple[str, int, nn.Module]]:
    branch_layers: List[Tuple[str, int, nn.Module]] = []
    for branch_name in ("shared", "window", "forecast"):
        layers = getattr(model, f"{branch_name}_graph_layers", None)
        if layers is not None:
            branch_layers.extend((branch_name, idx, layer) for idx, layer in enumerate(layers))
    if not branch_layers and hasattr(model, "_all_graph_layers"):
        branch_layers = [("graph", idx, layer) for idx, layer in enumerate(model._all_graph_layers())]
    return branch_layers


def _cross_temporal_edge_score_rows(model: nn.Module) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for branch_name, layer_idx, layer in _graph_branch_layers(model):
        if not hasattr(layer, "effective_cross_temporal_matrix"):
            continue
        cross = layer.effective_cross_temporal_matrix().detach()
        if cross.numel() <= 0 or getattr(layer, "cross_temporal_gate_logits", None) is None:
            continue
        possible = torch.nonzero(getattr(layer, "cross_temporal_mask", torch.ones_like(cross)).detach() > 0.0, as_tuple=False)
        for source, target in possible.tolist():
            score = float(cross[source, target].abs().item())
            rows.append(
                {
                    "branch": branch_name,
                    "layer": int(layer_idx),
                    "kind": "cross_temporal",
                    "source": int(source),
                    "target": int(target),
                    "score": score,
                    "sign": float(torch.sign(cross[source, target]).item()),
                }
            )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows


def _synthetic_edge_recovery_metrics(model: nn.Module, metadata: Mapping[str, object]) -> Dict[str, float]:
    ground_truth = metadata.get("ground_truth_temporal_edges") if isinstance(metadata, Mapping) else None
    if not isinstance(ground_truth, Sequence) or not ground_truth:
        return {}
    active_edges = _active_graph_edges(model, max_edges=1_000_000)
    score_edges = _cross_temporal_edge_score_rows(model)
    metrics = _synthetic_edge_recovery_from_edges(active_edges, score_edges, ground_truth)
    for branch_name in ("shared", "window", "forecast"):
        branch_active = [edge for edge in active_edges if str(edge.get("branch")) == branch_name]
        branch_scores = [edge for edge in score_edges if str(edge.get("branch")) == branch_name]
        branch_metrics = _synthetic_edge_recovery_from_edges(
            branch_active,
            branch_scores,
            ground_truth,
            prefix=f"synthetic_edge_{branch_name}",
        )
        metrics.update(branch_metrics)
    return metrics


def _synthetic_edge_recovery_from_edges(
    active_edges: Sequence[Mapping[str, object]],
    score_edges: Sequence[Mapping[str, object]],
    ground_truth_edges: Sequence[Mapping[str, object]],
    *,
    prefix: str = "synthetic_edge",
) -> Dict[str, float]:
    ground_truth_pairs = {
        (int(edge["source"]), int(edge["target"]))
        for edge in ground_truth_edges
        if isinstance(edge, Mapping) and "source" in edge and "target" in edge
    }
    if not ground_truth_pairs:
        return {}
    gt_sources = {source for source, _ in ground_truth_pairs}
    gt_targets = {target for _, target in ground_truth_pairs}
    active_cross = [
        edge for edge in active_edges
        if str(edge.get("kind")) == "cross_temporal"
    ]
    active_pairs = {
        (int(edge["source"]), int(edge["target"]))
        for edge in active_cross
        if "source" in edge and "target" in edge
    }
    synthetic_relevant_pairs = {
        pair for pair in active_pairs
        if pair[0] in gt_sources and pair[1] in gt_targets
    }
    matched_pairs = ground_truth_pairs & active_pairs
    precision_denominator = max(len(synthetic_relevant_pairs), 1)

    rank_by_pair: Dict[Tuple[int, int], int] = {}
    for rank, edge in enumerate(score_edges, start=1):
        if str(edge.get("kind")) != "cross_temporal":
            continue
        pair = (int(edge["source"]), int(edge["target"]))
        rank_by_pair.setdefault(pair, rank)
    missing_rank = len([edge for edge in score_edges if str(edge.get("kind")) == "cross_temporal"]) + 1
    ranks = [float(rank_by_pair.get(pair, missing_rank)) for pair in sorted(ground_truth_pairs)]
    found_ranks = [float(rank_by_pair[pair]) for pair in sorted(ground_truth_pairs) if pair in rank_by_pair]

    return {
        f"{prefix}_ground_truth_edges": float(len(ground_truth_pairs)),
        f"{prefix}_active_cross_temporal_edges": float(len(active_cross)),
        f"{prefix}_synthetic_relevant_active_edges": float(len(synthetic_relevant_pairs)),
        f"{prefix}_matched_active_edges": float(len(matched_pairs)),
        f"{prefix}_recall_at_active": float(len(matched_pairs)) / float(len(ground_truth_pairs)),
        f"{prefix}_precision_active": float(len(matched_pairs)) / float(precision_denominator),
        f"{prefix}_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
        f"{prefix}_rank_found_mean": float(np.mean(found_ranks)) if found_ranks else 0.0,
        f"{prefix}_rank_found_fraction": float(len(found_ranks)) / float(len(ground_truth_pairs)),
    }


def _edge_source_target_times(edge: Dict[str, object], timesteps: int) -> Tuple[int, int]:
    if str(edge.get("kind")) == "spatial":
        source_time = max(int(timesteps) - 1, 0)
        return source_time, source_time
    source_time = max(int(timesteps) - 2, 0)
    target_time = min(source_time + 1, max(int(timesteps) - 1, 0))
    return source_time, target_time


def _raw_intervention_value_for_target(model: nn.Module, concept_idx: int, target_z: float) -> float:
    calibrator = getattr(model, "calibrator", None)
    if getattr(calibrator, "activation", None) == "learned_threshold":
        target_z = min(1.0 - 1e-4, max(1e-4, float(target_z)))
        threshold = calibrator.threshold.detach()
        sharpness = F.softplus(calibrator.log_sharpness).detach() + 1e-4
        logit = np.log(target_z / (1.0 - target_z))
        return float((threshold[int(concept_idx)] + (logit / sharpness[int(concept_idx)].clamp(min=1e-6))).item())
    return 1.0 if float(target_z) >= 0.5 else -1.0


def _edge_guided_sample_metrics(
    baseline: Dict[str, object],
    intervened: Dict[str, object],
    method: str,
    horizon: int,
    valid_rows: torch.Tensor,
    *,
    target_time: int,
    target_concept: int,
) -> Dict[str, float]:
    activity_before = baseline["activity_logits"][valid_rows, -1, :]
    activity_after = intervened["activity_logits"][valid_rows, -1, :]
    forecast_before = _forecast_logits(baseline, method, horizon)[valid_rows, -1, :]
    forecast_after = _forecast_logits(intervened, method, horizon)[valid_rows, -1, :]
    activity_prob_before = torch.softmax(activity_before, dim=-1)
    activity_prob_after = torch.softmax(activity_after, dim=-1)
    forecast_prob_before = torch.softmax(forecast_before, dim=-1)
    forecast_prob_after = torch.softmax(forecast_after, dim=-1)
    metrics = {
        "activity_prob_l1": float((activity_prob_after - activity_prob_before).abs().mean().item()),
        "activity_flip_rate": float((activity_after.argmax(dim=-1) != activity_before.argmax(dim=-1)).float().mean().item()),
        "forecast_prob_l1": float((forecast_prob_after - forecast_prob_before).abs().mean().item()),
        "forecast_flip_rate": float((forecast_after.argmax(dim=-1) != forecast_before.argmax(dim=-1)).float().mean().item()),
    }
    for branch_name, key in (
        ("shared", "shared_refined_concepts"),
        ("window", "window_refined_concepts"),
        ("forecast", "forecast_refined_concepts"),
    ):
        before = baseline.get(key)
        after = intervened.get(key)
        if not torch.is_tensor(before) or not torch.is_tensor(after):
            metrics[f"{branch_name}_state_delta_mean"] = 0.0
            metrics[f"{branch_name}_state_delta_max"] = 0.0
            continue
        delta = (after[valid_rows, int(target_time), int(target_concept)] - before[valid_rows, int(target_time), int(target_concept)]).abs()
        metrics[f"{branch_name}_state_delta_mean"] = float(delta.mean().item()) if delta.numel() else 0.0
        metrics[f"{branch_name}_state_delta_max"] = float(delta.max().item()) if delta.numel() else 0.0
    return metrics


def _concept_state_for_intervention(outputs: Dict[str, object]) -> torch.Tensor:
    for key in ("concept_states", "shared_refined_concepts", "concepts_t", "calibrated_concepts"):
        if key in outputs:
            return outputs[key]
    raise KeyError("Model outputs do not contain a concept state tensor for intervention metrics.")


def _masked_mean_abs(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.mean().item())


def _graph_structure_metrics(model: nn.Module) -> Dict[str, float]:
    if not hasattr(model, "graph_metrics"):
        return {}
    metrics = model.graph_metrics()
    active_same = float(metrics.get("active_same_time_edges", 0.0))
    active_lagged = float(metrics.get("active_lagged_edges", 0.0))
    output = {
        "same_time_density": float(metrics.get("same_time_density", 0.0)),
        "lagged_density": float(metrics.get("lagged_density", 0.0)),
        "mean_same_gate": float(metrics.get("mean_same_gate", 0.0)),
        "mean_lag_gate": float(metrics.get("mean_lag_gate", 0.0)),
        "active_edges": active_same + active_lagged,
    }
    for key, value in metrics.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    output["active_edges"] = active_same + active_lagged
    return output


def _train_linear_multitask(
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    horizon: int,
    history_length: int,
    num_activities: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    wandb_run,
    activity_class_weighting: bool,
    activity_class_weight_cap: float,
    activity_sil_false_positive_penalty: float,
    forecast_class_weighting: bool,
    forecast_class_weight_cap: float,
    forecast_sil_false_positive_penalty: float,
    classifier_l1_weight: float,
    early_stopping_metric: str,
    sil_index: int | None,
    transition_sampler_boundary_radius: int,
    activity_components: Mapping[str, np.ndarray] | None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    standardized_splits = _standardized_splits(splits)
    examples = {
        name: _sliding_window_examples(
            split,
            horizon=horizon,
            history_length=history_length,
            transition_boundary_radius=transition_sampler_boundary_radius,
        )
        for name, split in standardized_splits.items()
    }
    train_x = _flatten_linear_examples(examples["train"])
    val_x = _flatten_linear_examples(examples["val"])
    test_x = _flatten_linear_examples(examples["test"])
    standardizer = FeatureStandardizer.fit(train_x)
    train_x, val_x, test_x = standardizer.transform(train_x), standardizer.transform(val_x), standardizer.transform(test_x)

    model = MultiTaskLinearClassifier(train_x.shape[1], num_activities, horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    activity_class_weight = (
        torch.as_tensor(
            _balanced_class_weights(examples["train"]["activity_labels"], num_activities, cap=activity_class_weight_cap),
            dtype=torch.float32,
            device=device,
        )
        if activity_class_weighting
        else None
    )
    forecast_class_weights = (
        _forecast_class_weights_by_step(
            examples["train"],
            horizon=horizon,
            num_classes=num_activities,
            cap=forecast_class_weight_cap,
            device=device,
        )
        if forecast_class_weighting
        else {}
    )

    best_state = _state_to_cpu(model)
    best_selection_value: float | None = None
    best_val_loss = float("inf")
    best_score = -1.0
    best_selection_metric = str(early_stopping_metric)
    best_epoch = 0
    wait = 0
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        order = rng.permutation(train_x.shape[0])
        totals = {"total": 0.0, "activity": 0.0, "forecast": 0.0, "classifier_l1": 0.0, "examples": 0}
        for start in range(0, order.shape[0], int(batch_size)):
            idx = order[start : start + int(batch_size)]
            x = torch.as_tensor(train_x[idx], dtype=torch.float32, device=device)
            activity_target = torch.as_tensor(examples["train"]["activity_labels"][idx], dtype=torch.long, device=device)
            outputs = model(x)
            activity_logits = outputs["activity_logits"]
            activity_loss = F.cross_entropy(activity_logits, activity_target, weight=activity_class_weight)
            activity_loss = activity_loss + _sil_false_positive_loss(
                activity_logits,
                activity_target,
                sil_index,
                activity_sil_false_positive_penalty,
            )
            forecast_losses = []
            forecast_logits_by_step = outputs["forecast_logits_by_step"]
            for step in range(1, int(horizon) + 1):
                target_np = (
                    examples["train"]["teacher_forcing_labels"][idx, step]
                    if step < int(horizon)
                    else examples["train"]["forecast_labels"][idx]
                )
                target = torch.as_tensor(target_np, dtype=torch.long, device=device)
                logits = forecast_logits_by_step[step]
                step_loss = F.cross_entropy(logits, target, weight=forecast_class_weights.get(step))
                step_loss = step_loss + _sil_false_positive_loss(
                    logits,
                    target,
                    sil_index,
                    forecast_sil_false_positive_penalty,
                )
                forecast_losses.append(step_loss)
            forecast_loss = torch.stack(forecast_losses).mean()
            classifier_l1_loss = _classifier_l1_penalty(model) * float(classifier_l1_weight)
            loss = activity_loss + forecast_loss + classifier_l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = len(idx)
            totals["total"] += float(loss.item()) * n
            totals["activity"] += float(activity_loss.item()) * n
            totals["forecast"] += float(forecast_loss.item()) * n
            totals["classifier_l1"] += float(classifier_l1_loss.item()) * n
            totals["examples"] += n

        val_metrics = _evaluate_multitask_linear(
            model, val_x, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        test_metrics = _evaluate_multitask_linear(
            model, test_x, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        val_score = 0.25 * (
            val_metrics["activity"]["accuracy"]
            + val_metrics["activity"]["macro_f1"]
            + val_metrics["forecast"]["accuracy"]
            + val_metrics["forecast"]["macro_f1"]
        )
        test_score = 0.25 * (
            test_metrics["activity"]["accuracy"]
            + test_metrics["activity"]["macro_f1"]
            + test_metrics["forecast"]["accuracy"]
            + test_metrics["forecast"]["macro_f1"]
        )
        val_selection_loss = float(val_metrics["activity"]["loss"] + val_metrics["forecast"]["loss"])
        test_selection_loss = float(test_metrics["activity"]["loss"] + test_metrics["forecast"]["loss"])
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / max(totals["examples"], 1),
            "train_activity_loss": totals["activity"] / max(totals["examples"], 1),
            "train_forecast_loss": totals["forecast"] / max(totals["examples"], 1),
            "train_classifier_l1_loss": totals["classifier_l1"] / max(totals["examples"], 1),
            "train_examples": int(totals["examples"]),
            "val_selection_loss": float(val_selection_loss),
            "val_score": float(val_score),
            "val_activity_accuracy": float(val_metrics["activity"]["accuracy"]),
            "val_activity_macro_f1": float(val_metrics["activity"]["macro_f1"]),
            "val_activity_top3_accuracy": float(val_metrics["activity"]["top3_accuracy"]),
            "val_forecast_accuracy": float(val_metrics["forecast"]["accuracy"]),
            "val_forecast_macro_f1": float(val_metrics["forecast"]["macro_f1"]),
            "val_forecast_top3_accuracy": float(val_metrics["forecast"]["top3_accuracy"]),
            "test_selection_loss": float(test_selection_loss),
            "test_score": float(test_score),
            "test_activity_accuracy": float(test_metrics["activity"]["accuracy"]),
            "test_activity_macro_f1": float(test_metrics["activity"]["macro_f1"]),
            "test_activity_top3_accuracy": float(test_metrics["activity"]["top3_accuracy"]),
            "test_forecast_accuracy": float(test_metrics["forecast"]["accuracy"]),
            "test_forecast_macro_f1": float(test_metrics["forecast"]["macro_f1"]),
            "test_forecast_top3_accuracy": float(test_metrics["forecast"]["top3_accuracy"]),
        }
        row["val_activity_forecast_accuracy"] = 0.5 * (
            row["val_activity_accuracy"] + row["val_forecast_accuracy"]
        )
        row["test_activity_forecast_accuracy"] = 0.5 * (
            row["test_activity_accuracy"] + row["test_forecast_accuracy"]
        )
        row.update(_history_forecast_diagnostic_row("val", val_metrics))
        row.update(_history_forecast_diagnostic_row("test", test_metrics))
        selection_value, selection_key, selection_mode = _selection_metric_value(row, early_stopping_metric)
        row["val_selection_value"] = float(selection_value)
        row["val_selection_key"] = str(selection_key)
        history.append(row)
        _log_epoch(wandb_run, row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(num_epochs):
            print(
                f"epoch={epoch:03d} total_loss={row['train_total_loss']:.4f} "
                f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                f"val_act_acc={row['val_activity_accuracy']:.4f} "
                f"val_fore_acc={row['val_forecast_accuracy']:.4f} "
                f"test_score={test_score:.4f} test_act_acc={row['test_activity_accuracy']:.4f} "
                f"test_fore_acc={row['test_forecast_accuracy']:.4f} "
                f"selection={selection_key}:{selection_value:.4f}",
                flush=True,
            )
        if best_selection_value is None:
            improved = True
        elif selection_mode == "min":
            improved = selection_value < best_selection_value - 1e-8
        else:
            improved = selection_value > best_selection_value + 1e-8
        if improved:
            best_selection_value = float(selection_value)
            best_selection_metric = selection_key
            best_val_loss = float(val_selection_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_score": float(best_score),
        "best_selection_metric": best_selection_metric,
        "best_selection_value": float(best_selection_value if best_selection_value is not None else 0.0),
        "device": str(device),
        "standardizer": standardizer,
        "train_metrics": _evaluate_multitask_linear(
            model, train_x, examples["train"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "val_metrics": _evaluate_multitask_linear(
            model, val_x, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "test_metrics": _evaluate_multitask_linear(
            model, test_x, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "forecast_baselines": _forecast_baseline_diagnostics(
            examples,
            horizon=horizon,
            num_classes=num_activities,
            sil_index=sil_index,
        ),
        "activity_class_weighting": bool(activity_class_weighting),
        "activity_class_weight_cap": float(activity_class_weight_cap),
        "activity_sil_false_positive_penalty": float(activity_sil_false_positive_penalty),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "classifier_l1_weight": float(classifier_l1_weight),
    }
    return {"forecast_model": model.cpu(), "standardizer": standardizer}, info


def _train_linear_dynamics_shared_head(
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    horizon: int,
    history_length: int,
    num_concepts: int,
    num_activities: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    wandb_run,
    activity_class_weighting: bool,
    activity_class_weight_cap: float,
    activity_sil_false_positive_penalty: float,
    forecast_class_weighting: bool,
    forecast_class_weight_cap: float,
    forecast_sil_false_positive_penalty: float,
    concept_forecast_loss_weight: float,
    concept_forecast_loss_deadzone_std: float,
    classifier_l1_weight: float,
    early_stopping_metric: str,
    sil_index: int | None,
    transition_sampler_boundary_radius: int,
    use_concept_dynamics: bool = True,
    use_sparse_concept_dynamics: bool = False,
    sparse_top_k: int = 0,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    has_concept_dynamics = bool(use_concept_dynamics or use_sparse_concept_dynamics)
    standardized_splits = _standardized_splits(splits)
    examples = {
        name: _sliding_window_examples(
            split,
            horizon=horizon,
            history_length=history_length,
            transition_boundary_radius=transition_sampler_boundary_radius,
        )
        for name, split in standardized_splits.items()
    }
    train_x = _current_linear_dynamics_features(examples["train"])
    val_x = _current_linear_dynamics_features(examples["val"])
    test_x = _current_linear_dynamics_features(examples["test"])

    if bool(use_sparse_concept_dynamics):
        if int(sparse_top_k) < 1:
            raise ValueError("linear_sparse_dynamics_shared_head requires model_hparams.st_spatial_top_k >= 1.")
        model = LinearSparseDynamicsSharedHeadClassifier(
            num_concepts,
            num_activities,
            horizon,
            top_k=int(sparse_top_k),
        ).to(device)
    elif bool(use_concept_dynamics):
        model = LinearDynamicsSharedHeadClassifier(num_concepts, num_activities, horizon).to(device)
    else:
        model = PlainProbeSharedHeadClassifier(num_concepts, num_activities, horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    activity_class_weight = (
        torch.as_tensor(
            _balanced_class_weights(examples["train"]["activity_labels"], num_activities, cap=activity_class_weight_cap),
            dtype=torch.float32,
            device=device,
        )
        if activity_class_weighting
        else None
    )
    forecast_class_weights = (
        _forecast_class_weights_by_step(
            examples["train"],
            horizon=horizon,
            num_classes=num_activities,
            cap=forecast_class_weight_cap,
            device=device,
        )
        if forecast_class_weighting
        else {}
    )

    best_state = _state_to_cpu(model)
    best_selection_value: float | None = None
    best_val_loss = float("inf")
    best_score = -1.0
    best_selection_metric = str(early_stopping_metric)
    best_epoch = 0
    wait = 0
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        order = rng.permutation(train_x.shape[0])
        totals = {
            "total": 0.0,
            "activity": 0.0,
            "forecast": 0.0,
            "concept_forecast": 0.0,
            "classifier_l1": 0.0,
            "examples": 0,
        }
        for start in range(0, order.shape[0], int(batch_size)):
            idx = order[start : start + int(batch_size)]
            x = torch.as_tensor(train_x[idx], dtype=torch.float32, device=device)
            activity_target = torch.as_tensor(examples["train"]["activity_labels"][idx], dtype=torch.long, device=device)
            outputs = model(x)
            activity_logits = outputs["activity_logits"]
            activity_loss = F.cross_entropy(activity_logits, activity_target, weight=activity_class_weight)
            activity_loss = activity_loss + _sil_false_positive_loss(
                activity_logits,
                activity_target,
                sil_index,
                activity_sil_false_positive_penalty,
            )

            forecast_losses = []
            forecast_logits_by_step = outputs["forecast_logits_by_step"]
            for step in range(1, int(horizon) + 1):
                target_np = (
                    examples["train"]["teacher_forcing_labels"][idx, step]
                    if step < int(horizon)
                    else examples["train"]["forecast_labels"][idx]
                )
                target = torch.as_tensor(target_np, dtype=torch.long, device=device)
                logits = forecast_logits_by_step[step]
                step_loss = F.cross_entropy(logits, target, weight=forecast_class_weights.get(step))
                step_loss = step_loss + _sil_false_positive_loss(
                    logits,
                    target,
                    sil_index,
                    forecast_sil_false_positive_penalty,
                )
                forecast_losses.append(step_loss)
            forecast_loss = torch.stack(forecast_losses).mean()
            if has_concept_dynamics:
                future_concepts = torch.as_tensor(
                    examples["train"]["future_concepts"][idx],
                    dtype=torch.float32,
                    device=device,
                )
                concept_forecast_loss = _linear_dynamics_concept_forecast_loss(
                    outputs["predicted_concepts_by_step"],
                    future_concepts,
                    deadzone_std=concept_forecast_loss_deadzone_std,
                )
            else:
                concept_forecast_loss = activity_loss.new_zeros(())
            weighted_concept_forecast_loss = concept_forecast_loss * float(concept_forecast_loss_weight)
            classifier_l1_loss = _classifier_l1_penalty(model) * float(classifier_l1_weight)
            loss = activity_loss + forecast_loss + weighted_concept_forecast_loss + classifier_l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = len(idx)
            totals["total"] += float(loss.item()) * n
            totals["activity"] += float(activity_loss.item()) * n
            totals["forecast"] += float(forecast_loss.item()) * n
            totals["concept_forecast"] += float(concept_forecast_loss.item()) * n
            totals["classifier_l1"] += float(classifier_l1_loss.item()) * n
            totals["examples"] += n

        val_metrics = _evaluate_multitask_linear(
            model, val_x, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        test_metrics = _evaluate_multitask_linear(
            model, test_x, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        val_score = 0.25 * (
            val_metrics["activity"]["accuracy"]
            + val_metrics["activity"]["macro_f1"]
            + val_metrics["forecast"]["accuracy"]
            + val_metrics["forecast"]["macro_f1"]
        )
        test_score = 0.25 * (
            test_metrics["activity"]["accuracy"]
            + test_metrics["activity"]["macro_f1"]
            + test_metrics["forecast"]["accuracy"]
            + test_metrics["forecast"]["macro_f1"]
        )
        val_selection_loss = float(val_metrics["activity"]["loss"] + val_metrics["forecast"]["loss"])
        test_selection_loss = float(test_metrics["activity"]["loss"] + test_metrics["forecast"]["loss"])
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / max(totals["examples"], 1),
            "train_activity_loss": totals["activity"] / max(totals["examples"], 1),
            "train_forecast_loss": totals["forecast"] / max(totals["examples"], 1),
            "train_concept_forecast_loss": totals["concept_forecast"] / max(totals["examples"], 1),
            "train_classifier_l1_loss": totals["classifier_l1"] / max(totals["examples"], 1),
            "train_examples": int(totals["examples"]),
            "val_selection_loss": float(val_selection_loss),
            "val_score": float(val_score),
            "val_activity_accuracy": float(val_metrics["activity"]["accuracy"]),
            "val_activity_macro_f1": float(val_metrics["activity"]["macro_f1"]),
            "val_activity_top3_accuracy": float(val_metrics["activity"]["top3_accuracy"]),
            "val_forecast_accuracy": float(val_metrics["forecast"]["accuracy"]),
            "val_forecast_macro_f1": float(val_metrics["forecast"]["macro_f1"]),
            "val_forecast_top3_accuracy": float(val_metrics["forecast"]["top3_accuracy"]),
            "test_selection_loss": float(test_selection_loss),
            "test_score": float(test_score),
            "test_activity_accuracy": float(test_metrics["activity"]["accuracy"]),
            "test_activity_macro_f1": float(test_metrics["activity"]["macro_f1"]),
            "test_activity_top3_accuracy": float(test_metrics["activity"]["top3_accuracy"]),
            "test_forecast_accuracy": float(test_metrics["forecast"]["accuracy"]),
            "test_forecast_macro_f1": float(test_metrics["forecast"]["macro_f1"]),
            "test_forecast_top3_accuracy": float(test_metrics["forecast"]["top3_accuracy"]),
        }
        row["val_activity_forecast_accuracy"] = 0.5 * (
            row["val_activity_accuracy"] + row["val_forecast_accuracy"]
        )
        row["test_activity_forecast_accuracy"] = 0.5 * (
            row["test_activity_accuracy"] + row["test_forecast_accuracy"]
        )
        row.update(_history_forecast_diagnostic_row("val", val_metrics))
        row.update(_history_forecast_diagnostic_row("test", test_metrics))
        selection_value, selection_key, selection_mode = _selection_metric_value(row, early_stopping_metric)
        row["val_selection_value"] = float(selection_value)
        row["val_selection_key"] = str(selection_key)
        history.append(row)
        _log_epoch(wandb_run, row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(num_epochs):
            print(
                f"epoch={epoch:03d} total_loss={row['train_total_loss']:.4f} "
                f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                f"val_act_acc={row['val_activity_accuracy']:.4f} "
                f"val_fore_acc={row['val_forecast_accuracy']:.4f} "
                f"test_score={test_score:.4f} test_act_acc={row['test_activity_accuracy']:.4f} "
                f"test_fore_acc={row['test_forecast_accuracy']:.4f} "
                f"selection={selection_key}:{selection_value:.4f}",
                flush=True,
            )
        if best_selection_value is None:
            improved = True
        elif selection_mode == "min":
            improved = selection_value < best_selection_value - 1e-8
        else:
            improved = selection_value > best_selection_value + 1e-8
        if improved:
            best_selection_value = float(selection_value)
            best_selection_metric = selection_key
            best_val_loss = float(val_selection_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_score": float(best_score),
        "best_selection_metric": best_selection_metric,
        "best_selection_value": float(best_selection_value if best_selection_value is not None else 0.0),
        "device": str(device),
        "train_metrics": _evaluate_multitask_linear(
            model, train_x, examples["train"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "val_metrics": _evaluate_multitask_linear(
            model, val_x, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "test_metrics": _evaluate_multitask_linear(
            model, test_x, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "forecast_baselines": _forecast_baseline_diagnostics(
            examples,
            horizon=horizon,
            num_classes=num_activities,
            sil_index=sil_index,
        ),
        "activity_class_weighting": bool(activity_class_weighting),
        "activity_class_weight_cap": float(activity_class_weight_cap),
        "activity_sil_false_positive_penalty": float(activity_sil_false_positive_penalty),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "concept_forecast_loss_weight": float(concept_forecast_loss_weight),
        "concept_forecast_loss_deadzone_std": float(concept_forecast_loss_deadzone_std),
        "classifier_l1_weight": float(classifier_l1_weight),
        "shared_activity_head": True,
        "linear_concept_dynamics": bool(has_concept_dynamics),
        "linear_sparse_concept_dynamics": bool(use_sparse_concept_dynamics),
        "linear_sparse_top_k": int(sparse_top_k) if use_sparse_concept_dynamics else 0,
        "linear_dynamics_active_edges": int(model.active_edge_count()) if hasattr(model, "active_edge_count") else 0,
        "plain_current_concept_probe": not bool(has_concept_dynamics),
    }
    return {"forecast_model": model.cpu()}, info


def _current_linear_dynamics_features(examples: Dict[str, np.ndarray]) -> np.ndarray:
    concepts = np.asarray(examples["concepts"], dtype=np.float32)
    return concepts[:, -1, :].astype(np.float32)


def _linear_dynamics_concept_forecast_loss(
    predicted_by_step: Mapping[int, torch.Tensor],
    future_concepts: torch.Tensor,
    *,
    deadzone_std: float,
) -> torch.Tensor:
    losses = []
    for step, predicted in sorted(predicted_by_step.items()):
        step_index = int(step) - 1
        if step_index < 0 or step_index >= future_concepts.size(1):
            continue
        target = future_concepts[:, step_index, :]
        if float(deadzone_std) > 0.0:
            excess = (predicted - target).abs().sub(float(deadzone_std)).clamp_min(0.0)
            losses.append(F.smooth_l1_loss(excess, torch.zeros_like(excess)))
        else:
            losses.append(F.smooth_l1_loss(predicted, target))
    if not losses:
        return future_concepts.sum() * 0.0
    return torch.stack(losses).mean()


def _flatten_linear_examples(examples: Dict[str, np.ndarray]) -> np.ndarray:
    concepts = np.asarray(examples["concepts"], dtype=np.float32)
    return concepts.reshape(int(concepts.shape[0]), -1)


@torch.no_grad()
def _evaluate_multitask_linear(
    model: MultiTaskLinearClassifier,
    features: np.ndarray,
    examples: Dict[str, np.ndarray],
    horizon: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    sil_index: int | None = None,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    activity_true, activity_pred = [], []
    forecast_true, forecast_pred = [], []
    forecast_true_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    forecast_pred_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    activity_top3_hits = 0
    forecast_top3_hits = 0
    forecast_top3_hits_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}
    activity_loss_sum = 0.0
    forecast_loss_sum = 0.0
    forecast_loss_by_step: Dict[int, float] = {step: 0.0 for step in range(1, int(horizon) + 1)}
    activity_count = 0
    forecast_count = 0
    count_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}

    for start in range(0, features.shape[0], int(batch_size)):
        end = start + int(batch_size)
        x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
        outputs = model(x)
        activity_labels = torch.as_tensor(examples["activity_labels"][start:end], dtype=torch.long, device=device)
        activity_logits = outputs["activity_logits"]
        count = int(activity_labels.shape[0])
        activity_loss_sum += float(F.cross_entropy(activity_logits, activity_labels).item()) * count
        activity_top3_hits += _topk_hit_count(activity_logits, activity_labels, k=3)
        activity_count += count
        activity_true.append(activity_labels.cpu().numpy())
        activity_pred.append(activity_logits.argmax(dim=-1).cpu().numpy())

        forecast_logits_by_step = outputs["forecast_logits_by_step"]
        for step in range(1, int(horizon) + 1):
            target_np = (
                examples["teacher_forcing_labels"][start:end, step]
                if step < int(horizon)
                else examples["forecast_labels"][start:end]
            )
            target = torch.as_tensor(target_np, dtype=torch.long, device=device)
            logits = forecast_logits_by_step[step]
            step_count = int(target.shape[0])
            loss = F.cross_entropy(logits, target)
            forecast_loss_by_step[step] += float(loss.item()) * step_count
            forecast_top3_hits_by_step[step] += _topk_hit_count(logits, target, k=3)
            count_by_step[step] += step_count
            forecast_true_by_step[step].append(target.cpu().numpy())
            forecast_pred_by_step[step].append(logits.argmax(dim=-1).cpu().numpy())
            if step == int(horizon):
                forecast_loss_sum += float(loss.item()) * step_count
                forecast_top3_hits += _topk_hit_count(logits, target, k=3)
                forecast_count += step_count
                forecast_true.append(target.cpu().numpy())
                forecast_pred.append(logits.argmax(dim=-1).cpu().numpy())

    activity_labels_np = np.concatenate(activity_true) if activity_true else np.zeros(0, dtype=np.int64)
    activity_preds_np = np.concatenate(activity_pred) if activity_pred else np.zeros(0, dtype=np.int64)
    forecast_labels_np = np.concatenate(forecast_true) if forecast_true else np.zeros(0, dtype=np.int64)
    forecast_preds_np = np.concatenate(forecast_pred) if forecast_pred else np.zeros(0, dtype=np.int64)
    activity_metrics = _classification_metrics(activity_labels_np, activity_preds_np, num_classes)
    activity_metrics.update(_component_accuracy_metrics(activity_labels_np, activity_preds_np, activity_components))
    activity_metrics.update(_sil_metrics(activity_labels_np, activity_preds_np, sil_index))
    activity_metrics["loss"] = activity_loss_sum / max(activity_count, 1)
    activity_metrics["top3_accuracy"] = activity_top3_hits / max(activity_count, 1)
    activity_metrics["num_examples"] = int(activity_count)
    forecast_metrics = _classification_metrics(forecast_labels_np, forecast_preds_np, num_classes)
    forecast_metrics.update(_component_accuracy_metrics(forecast_labels_np, forecast_preds_np, activity_components))
    forecast_metrics.update(_sil_metrics(forecast_labels_np, forecast_preds_np, sil_index))
    forecast_metrics["loss"] = forecast_loss_sum / max(forecast_count, 1)
    forecast_metrics["top3_accuracy"] = forecast_top3_hits / max(forecast_count, 1)
    forecast_metrics["num_examples"] = int(forecast_count)

    forecast_by_horizon = {}
    for step in range(1, int(horizon) + 1):
        labels = np.concatenate(forecast_true_by_step[step]) if forecast_true_by_step[step] else np.zeros(0, dtype=np.int64)
        preds = np.concatenate(forecast_pred_by_step[step]) if forecast_pred_by_step[step] else np.zeros(0, dtype=np.int64)
        step_metrics = _classification_metrics(labels, preds, num_classes)
        step_metrics.update(_component_accuracy_metrics(labels, preds, activity_components))
        step_metrics.update(_sil_metrics(labels, preds, sil_index))
        step_metrics["loss"] = forecast_loss_by_step[step] / max(count_by_step[step], 1)
        step_metrics["top3_accuracy"] = forecast_top3_hits_by_step[step] / max(count_by_step[step], 1)
        step_metrics["num_examples"] = int(labels.shape[0])
        forecast_by_horizon[str(step)] = step_metrics
    return {"activity": activity_metrics, "forecast": forecast_metrics, "forecast_by_horizon": forecast_by_horizon}


def _feature_sliding_window_examples(
    split: Dict[str, np.ndarray],
    *,
    horizon: int,
    history_length: int,
    transition_boundary_radius: int = 2,
) -> Dict[str, np.ndarray]:
    features = np.asarray(split["raw_features_std"], dtype=np.float32)
    mask = np.asarray(split["mask"], dtype=np.float32)
    example_mask = np.asarray(split.get("example_mask", mask), dtype=np.float32)
    activity_labels = np.asarray(split["activity_labels"], dtype=np.int64)
    forecast_labels = np.asarray(split[forecast_key(horizon)], dtype=np.int64)
    lengths = np.asarray(split["lengths"], dtype=np.int64)
    history_length = int(history_length)
    feature_dim = int(features.shape[-1])

    windows: List[np.ndarray] = []
    key_padding_masks: List[np.ndarray] = []
    activity_targets: List[int] = []
    forecast_targets: List[int] = []
    teacher_forcing_targets: List[np.ndarray] = []
    transition_flags: List[bool] = []
    transition_boundary_radius = max(int(transition_boundary_radius), 0)

    for video_idx in range(features.shape[0]):
        length = int(lengths[video_idx])
        for timestep in range(length):
            if mask[video_idx, timestep] <= 0.0 or example_mask[video_idx, timestep] <= 0.0:
                continue
            activity_target = int(activity_labels[video_idx, timestep])
            forecast_target = int(forecast_labels[video_idx, timestep])
            if activity_target < 0 or forecast_target < 0:
                continue
            teacher_forcing = activity_labels[video_idx, timestep : timestep + int(horizon)]
            if teacher_forcing.shape[0] != int(horizon) or np.any(teacher_forcing < 0):
                continue

            start = max(0, timestep - history_length + 1)
            history = features[video_idx, start : timestep + 1]
            history_mask = mask[video_idx, start : timestep + 1] > 0.0
            padded = np.zeros((history_length, feature_dim), dtype=np.float32)
            key_padding_mask = np.ones(history_length, dtype=bool)
            offset = history_length - int(history.shape[0])
            padded[offset:] = history
            key_padding_mask[offset:] = ~history_mask

            windows.append(padded)
            key_padding_masks.append(key_padding_mask)
            activity_targets.append(activity_target)
            forecast_targets.append(forecast_target)
            teacher_forcing_targets.append(teacher_forcing.astype(np.int64, copy=False))
            transition_flags.append(
                _has_nearby_label_transition(
                    activity_labels[video_idx],
                    length=length,
                    timestep=timestep,
                    horizon=horizon,
                    radius=transition_boundary_radius,
                )
            )

    if not windows:
        raise ValueError("No valid raw-feature sliding-window examples were found.")
    return {
        "features": np.stack(windows, axis=0).astype(np.float32),
        "key_padding_mask": np.stack(key_padding_masks, axis=0).astype(bool),
        "activity_labels": np.asarray(activity_targets, dtype=np.int64),
        "forecast_labels": np.asarray(forecast_targets, dtype=np.int64),
        "teacher_forcing_labels": np.stack(teacher_forcing_targets, axis=0).astype(np.int64),
        "transition_flags": np.asarray(transition_flags, dtype=bool),
    }


def _feature_example_batch_to_torch(
    examples: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(examples["features"][indices], dtype=torch.float32, device=device),
        "key_padding_mask": torch.as_tensor(examples["key_padding_mask"][indices], dtype=torch.bool, device=device),
        "activity_labels": torch.as_tensor(examples["activity_labels"][indices], dtype=torch.long, device=device),
        "forecast_labels": torch.as_tensor(examples["forecast_labels"][indices], dtype=torch.long, device=device),
        "teacher_forcing_labels": torch.as_tensor(
            examples["teacher_forcing_labels"][indices],
            dtype=torch.long,
            device=device,
        ),
    }


def _train_feature_sequence_multitask(
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    method: str,
    horizon: int,
    history_length: int,
    input_dim: int,
    num_activities: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    wandb_run,
    activity_class_weighting: bool,
    activity_class_weight_cap: float,
    activity_sil_false_positive_penalty: float,
    forecast_class_weighting: bool,
    forecast_class_weight_cap: float,
    forecast_sil_false_positive_penalty: float,
    classifier_l1_weight: float,
    early_stopping_metric: str,
    sil_index: int | None,
    transition_sampler_boundary_radius: int,
    train_sampling_strategy: str,
    transition_sampler_strength: float,
    transition_sampler_rare_alpha: float,
    model_hparams: Dict[str, object] | None = None,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    standardized_splits = _standardized_raw_feature_splits(splits)
    examples = {
        name: _feature_sliding_window_examples(
            split,
            horizon=horizon,
            history_length=history_length,
            transition_boundary_radius=transition_sampler_boundary_radius,
        )
        for name, split in standardized_splits.items()
    }
    model = _build_feature_sequence_model(
        method=method,
        input_dim=input_dim,
        num_activities=num_activities,
        horizon=horizon,
        history_length=history_length,
        model_hparams=model_hparams,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    activity_class_weight = (
        torch.as_tensor(
            _balanced_class_weights(examples["train"]["activity_labels"], num_activities, cap=activity_class_weight_cap),
            dtype=torch.float32,
            device=device,
        )
        if activity_class_weighting
        else None
    )
    forecast_class_weights = (
        _forecast_class_weights_by_step(
            examples["train"],
            horizon=horizon,
            num_classes=num_activities,
            cap=forecast_class_weight_cap,
            device=device,
        )
        if forecast_class_weighting
        else {}
    )
    train_sampling_info = _train_sampling_info(
        examples["train"],
        strategy=train_sampling_strategy,
        num_classes=num_activities,
        boundary_radius=transition_sampler_boundary_radius,
        strength=transition_sampler_strength,
        rare_alpha=transition_sampler_rare_alpha,
    )

    best_state = _state_to_cpu(model)
    best_selection_value: float | None = None
    best_val_loss = float("inf")
    best_score = -1.0
    best_selection_metric = str(early_stopping_metric)
    best_epoch = 0
    wait = 0
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        if train_sampling_info.get("weights") is None:
            order = rng.permutation(examples["train"]["features"].shape[0])
        else:
            order = rng.choice(
                examples["train"]["features"].shape[0],
                size=examples["train"]["features"].shape[0],
                replace=True,
                p=np.asarray(train_sampling_info["weights"], dtype=np.float64),
            )
        totals = {"total": 0.0, "activity": 0.0, "forecast": 0.0, "classifier_l1": 0.0, "examples": 0}
        for indices in _iter_example_batches(order, batch_size):
            batch = _feature_example_batch_to_torch(examples["train"], indices, device)
            outputs = model(batch["features"], batch["key_padding_mask"])
            activity_logits = outputs["activity_logits"]
            activity_loss = F.cross_entropy(activity_logits, batch["activity_labels"], weight=activity_class_weight)
            activity_loss = activity_loss + _sil_false_positive_loss(
                activity_logits,
                batch["activity_labels"],
                sil_index,
                activity_sil_false_positive_penalty,
            )
            forecast_losses = []
            forecast_logits_by_step = outputs["forecast_logits_by_step"]
            for step in range(1, int(horizon) + 1):
                target = batch["teacher_forcing_labels"][:, step] if step < int(horizon) else batch["forecast_labels"]
                logits = forecast_logits_by_step[step]
                step_loss = F.cross_entropy(logits, target, weight=forecast_class_weights.get(step))
                step_loss = step_loss + _sil_false_positive_loss(
                    logits,
                    target,
                    sil_index,
                    forecast_sil_false_positive_penalty,
                )
                forecast_losses.append(step_loss)
            forecast_loss = torch.stack(forecast_losses).mean()
            classifier_l1_loss = _classifier_l1_penalty(model) * float(classifier_l1_weight)
            loss = activity_loss + forecast_loss + classifier_l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = int(indices.shape[0])
            totals["total"] += float(loss.item()) * n
            totals["activity"] += float(activity_loss.item()) * n
            totals["forecast"] += float(forecast_loss.item()) * n
            totals["classifier_l1"] += float(classifier_l1_loss.item()) * n
            totals["examples"] += n

        val_metrics = _evaluate_feature_sequence_multitask(
            model, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        test_metrics = _evaluate_feature_sequence_multitask(
            model, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        )
        val_score = 0.25 * (
            val_metrics["activity"]["accuracy"]
            + val_metrics["activity"]["macro_f1"]
            + val_metrics["forecast"]["accuracy"]
            + val_metrics["forecast"]["macro_f1"]
        )
        test_score = 0.25 * (
            test_metrics["activity"]["accuracy"]
            + test_metrics["activity"]["macro_f1"]
            + test_metrics["forecast"]["accuracy"]
            + test_metrics["forecast"]["macro_f1"]
        )
        val_selection_loss = float(val_metrics["activity"]["loss"] + val_metrics["forecast"]["loss"])
        test_selection_loss = float(test_metrics["activity"]["loss"] + test_metrics["forecast"]["loss"])
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / max(totals["examples"], 1),
            "train_activity_loss": totals["activity"] / max(totals["examples"], 1),
            "train_forecast_loss": totals["forecast"] / max(totals["examples"], 1),
            "train_classifier_l1_loss": totals["classifier_l1"] / max(totals["examples"], 1),
            "train_examples": int(totals["examples"]),
            "val_selection_loss": float(val_selection_loss),
            "val_score": float(val_score),
            "val_activity_accuracy": float(val_metrics["activity"]["accuracy"]),
            "val_activity_macro_f1": float(val_metrics["activity"]["macro_f1"]),
            "val_activity_top3_accuracy": float(val_metrics["activity"]["top3_accuracy"]),
            "val_forecast_accuracy": float(val_metrics["forecast"]["accuracy"]),
            "val_forecast_macro_f1": float(val_metrics["forecast"]["macro_f1"]),
            "val_forecast_top3_accuracy": float(val_metrics["forecast"]["top3_accuracy"]),
            "test_selection_loss": float(test_selection_loss),
            "test_score": float(test_score),
            "test_activity_accuracy": float(test_metrics["activity"]["accuracy"]),
            "test_activity_macro_f1": float(test_metrics["activity"]["macro_f1"]),
            "test_activity_top3_accuracy": float(test_metrics["activity"]["top3_accuracy"]),
            "test_forecast_accuracy": float(test_metrics["forecast"]["accuracy"]),
            "test_forecast_macro_f1": float(test_metrics["forecast"]["macro_f1"]),
            "test_forecast_top3_accuracy": float(test_metrics["forecast"]["top3_accuracy"]),
        }
        row["val_activity_forecast_accuracy"] = 0.5 * (
            row["val_activity_accuracy"] + row["val_forecast_accuracy"]
        )
        row["test_activity_forecast_accuracy"] = 0.5 * (
            row["test_activity_accuracy"] + row["test_forecast_accuracy"]
        )
        row.update(_history_forecast_diagnostic_row("val", val_metrics))
        row.update(_history_forecast_diagnostic_row("test", test_metrics))
        selection_value, selection_key, selection_mode = _selection_metric_value(row, early_stopping_metric)
        row["val_selection_value"] = float(selection_value)
        row["val_selection_key"] = str(selection_key)
        history.append(row)
        _log_epoch(wandb_run, row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(num_epochs):
            print(
                f"epoch={epoch:03d} total_loss={row['train_total_loss']:.4f} "
                f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                f"val_act_acc={row['val_activity_accuracy']:.4f} "
                f"val_fore_acc={row['val_forecast_accuracy']:.4f} "
                f"test_score={test_score:.4f} test_act_acc={row['test_activity_accuracy']:.4f} "
                f"test_fore_acc={row['test_forecast_accuracy']:.4f} "
                f"selection={selection_key}:{selection_value:.4f}",
                flush=True,
            )
        if best_selection_value is None:
            improved = True
        elif selection_mode == "min":
            improved = selection_value < best_selection_value - 1e-8
        else:
            improved = selection_value > best_selection_value + 1e-8
        if improved:
            best_selection_value = float(selection_value)
            best_selection_metric = selection_key
            best_val_loss = float(val_selection_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_score": float(best_score),
        "best_selection_metric": best_selection_metric,
        "best_selection_value": float(best_selection_value if best_selection_value is not None else 0.0),
        "device": str(device),
        "train_metrics": _evaluate_feature_sequence_multitask(
            model, examples["train"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "val_metrics": _evaluate_feature_sequence_multitask(
            model, examples["val"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "test_metrics": _evaluate_feature_sequence_multitask(
            model, examples["test"], horizon, num_activities, batch_size, device, sil_index, activity_components
        ),
        "forecast_baselines": _forecast_baseline_diagnostics(
            examples,
            horizon=horizon,
            num_classes=num_activities,
            sil_index=sil_index,
        ),
        "forecast_metrics_used": True,
        "feature_input_dim": int(input_dim),
        "feature_sequence_model": str(method),
        "activity_class_weighting": bool(activity_class_weighting),
        "activity_class_weight_cap": float(activity_class_weight_cap),
        "activity_sil_false_positive_penalty": float(activity_sil_false_positive_penalty),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "classifier_l1_weight": float(classifier_l1_weight),
        "train_sampling": _serializable_train_sampling_info(train_sampling_info),
        "total_trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "num_sliding_window_examples": {
            name: int(value["features"].shape[0])
            for name, value in examples.items()
        },
    }
    return {"forecast_model": model.cpu()}, info


@torch.no_grad()
def _evaluate_feature_sequence_multitask(
    model: nn.Module,
    examples: Dict[str, np.ndarray],
    horizon: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    sil_index: int | None = None,
    activity_components: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    activity_true, activity_pred = [], []
    forecast_true, forecast_pred = [], []
    forecast_true_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    forecast_pred_by_step: Dict[int, List[np.ndarray]] = {step: [] for step in range(1, int(horizon) + 1)}
    activity_top3_hits = 0
    forecast_top3_hits = 0
    forecast_top3_hits_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}
    activity_loss_sum = 0.0
    forecast_loss_sum = 0.0
    forecast_loss_by_step: Dict[int, float] = {step: 0.0 for step in range(1, int(horizon) + 1)}
    activity_count = 0
    forecast_count = 0
    count_by_step: Dict[int, int] = {step: 0 for step in range(1, int(horizon) + 1)}

    for start in range(0, examples["features"].shape[0], int(batch_size)):
        end = start + int(batch_size)
        features = torch.as_tensor(examples["features"][start:end], dtype=torch.float32, device=device)
        key_padding_mask = torch.as_tensor(examples["key_padding_mask"][start:end], dtype=torch.bool, device=device)
        outputs = model(features, key_padding_mask)
        activity_labels = torch.as_tensor(examples["activity_labels"][start:end], dtype=torch.long, device=device)
        activity_logits = outputs["activity_logits"]
        count = int(activity_labels.shape[0])
        activity_loss_sum += float(F.cross_entropy(activity_logits, activity_labels).item()) * count
        activity_top3_hits += _topk_hit_count(activity_logits, activity_labels, k=3)
        activity_count += count
        activity_true.append(activity_labels.cpu().numpy())
        activity_pred.append(activity_logits.argmax(dim=-1).cpu().numpy())

        forecast_logits_by_step = outputs["forecast_logits_by_step"]
        for step in range(1, int(horizon) + 1):
            target_np = (
                examples["teacher_forcing_labels"][start:end, step]
                if step < int(horizon)
                else examples["forecast_labels"][start:end]
            )
            target = torch.as_tensor(target_np, dtype=torch.long, device=device)
            logits = forecast_logits_by_step[step]
            step_count = int(target.shape[0])
            loss = F.cross_entropy(logits, target)
            forecast_loss_by_step[step] += float(loss.item()) * step_count
            forecast_top3_hits_by_step[step] += _topk_hit_count(logits, target, k=3)
            count_by_step[step] += step_count
            forecast_true_by_step[step].append(target.cpu().numpy())
            forecast_pred_by_step[step].append(logits.argmax(dim=-1).cpu().numpy())
            if step == int(horizon):
                forecast_loss_sum += float(loss.item()) * step_count
                forecast_top3_hits += _topk_hit_count(logits, target, k=3)
                forecast_count += step_count
                forecast_true.append(target.cpu().numpy())
                forecast_pred.append(logits.argmax(dim=-1).cpu().numpy())

    activity_labels_np = np.concatenate(activity_true) if activity_true else np.zeros(0, dtype=np.int64)
    activity_preds_np = np.concatenate(activity_pred) if activity_pred else np.zeros(0, dtype=np.int64)
    forecast_labels_np = np.concatenate(forecast_true) if forecast_true else np.zeros(0, dtype=np.int64)
    forecast_preds_np = np.concatenate(forecast_pred) if forecast_pred else np.zeros(0, dtype=np.int64)
    activity_metrics = _classification_metrics(activity_labels_np, activity_preds_np, num_classes)
    activity_metrics.update(_component_accuracy_metrics(activity_labels_np, activity_preds_np, activity_components))
    activity_metrics.update(_sil_metrics(activity_labels_np, activity_preds_np, sil_index))
    activity_metrics["loss"] = activity_loss_sum / max(activity_count, 1)
    activity_metrics["top3_accuracy"] = activity_top3_hits / max(activity_count, 1)
    activity_metrics["num_examples"] = int(activity_count)
    forecast_metrics = _classification_metrics(forecast_labels_np, forecast_preds_np, num_classes)
    forecast_metrics.update(_component_accuracy_metrics(forecast_labels_np, forecast_preds_np, activity_components))
    forecast_metrics.update(_sil_metrics(forecast_labels_np, forecast_preds_np, sil_index))
    forecast_metrics["loss"] = forecast_loss_sum / max(forecast_count, 1)
    forecast_metrics["top3_accuracy"] = forecast_top3_hits / max(forecast_count, 1)
    forecast_metrics["num_examples"] = int(forecast_count)

    forecast_by_horizon = {}
    for step in range(1, int(horizon) + 1):
        labels = np.concatenate(forecast_true_by_step[step]) if forecast_true_by_step[step] else np.zeros(0, dtype=np.int64)
        preds = np.concatenate(forecast_pred_by_step[step]) if forecast_pred_by_step[step] else np.zeros(0, dtype=np.int64)
        step_metrics = _classification_metrics(labels, preds, num_classes)
        step_metrics.update(_component_accuracy_metrics(labels, preds, activity_components))
        step_metrics.update(_sil_metrics(labels, preds, sil_index))
        step_metrics["loss"] = forecast_loss_by_step[step] / max(count_by_step[step], 1)
        step_metrics["top3_accuracy"] = forecast_top3_hits_by_step[step] / max(count_by_step[step], 1)
        step_metrics["num_examples"] = int(labels.shape[0])
        forecast_by_horizon[str(step)] = step_metrics
    return {"activity": activity_metrics, "forecast": forecast_metrics, "forecast_by_horizon": forecast_by_horizon}


def _train_linear(
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    horizon: int,
    history_length: int,
    num_activities: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    wandb_run,
    forecast_class_weighting: bool,
    forecast_class_weight_cap: float,
    forecast_sil_false_positive_penalty: float,
    classifier_l1_weight: float,
    early_stopping_metric: str,
    sil_index: int | None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_x, train_y = _flat_history_examples(splits["train"], forecast_key(horizon), history_length)
    val_x, val_y = _flat_history_examples(splits["val"], forecast_key(horizon), history_length)
    test_x, test_y = _flat_history_examples(splits["test"], forecast_key(horizon), history_length)
    standardizer = FeatureStandardizer.fit(train_x)
    train_x, val_x, test_x = standardizer.transform(train_x), standardizer.transform(val_x), standardizer.transform(test_x)

    model = LinearClassifier(train_x.shape[1], num_activities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    class_weight = (
        torch.as_tensor(
            _balanced_class_weights(train_y, num_activities, cap=forecast_class_weight_cap),
            dtype=torch.float32,
            device=device,
        )
        if forecast_class_weighting
        else None
    )
    best_state = _state_to_cpu(model)
    best_selection_value: float | None = None
    best_val_loss = float("inf")
    best_score = -1.0
    best_selection_metric = str(early_stopping_metric)
    best_epoch = 0
    wait = 0
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        order = rng.permutation(train_x.shape[0])
        loss_sum = 0.0
        classifier_l1_sum = 0.0
        for start in range(0, order.shape[0], int(batch_size)):
            idx = order[start : start + int(batch_size)]
            x = torch.as_tensor(train_x[idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(train_y[idx], dtype=torch.long, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            loss = loss + _sil_false_positive_loss(logits, y, sil_index, forecast_sil_false_positive_penalty)
            classifier_l1_loss = _classifier_l1_penalty(model) * float(classifier_l1_weight)
            loss = loss + classifier_l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(idx)
            classifier_l1_sum += float(classifier_l1_loss.item()) * len(idx)
        val_metrics = _evaluate_linear(model, val_x, val_y, num_activities, batch_size, device, sil_index)
        test_metrics = _evaluate_linear(model, test_x, test_y, num_activities, batch_size, device, sil_index)
        val_score = 0.5 * (val_metrics["accuracy"] + val_metrics["macro_f1"])
        test_score = 0.5 * (test_metrics["accuracy"] + test_metrics["macro_f1"])
        val_selection_loss = float(val_metrics["loss"])
        test_selection_loss = float(test_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_forecast_loss": loss_sum / max(train_x.shape[0], 1),
            "train_classifier_l1_loss": classifier_l1_sum / max(train_x.shape[0], 1),
            "val_selection_loss": float(val_selection_loss),
            "val_score": float(val_score),
            "val_forecast_accuracy": float(val_metrics["accuracy"]),
            "val_forecast_macro_f1": float(val_metrics["macro_f1"]),
            "val_forecast_top3_accuracy": float(val_metrics["top3_accuracy"]),
            "test_selection_loss": float(test_selection_loss),
            "test_score": float(test_score),
            "test_forecast_accuracy": float(test_metrics["accuracy"]),
            "test_forecast_macro_f1": float(test_metrics["macro_f1"]),
            "test_forecast_top3_accuracy": float(test_metrics["top3_accuracy"]),
        }
        for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
            for key in ("sil_true_rate", "sil_pred_rate", "sil_false_positive_rate"):
                if key in metrics:
                    row[f"{prefix}_forecast_{key}"] = float(metrics[key])
        selection_value, selection_key, selection_mode = _selection_metric_value(row, early_stopping_metric)
        row["val_selection_value"] = float(selection_value)
        row["val_selection_key"] = str(selection_key)
        history.append(row)
        _log_epoch(wandb_run, row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(num_epochs):
            print(
                f"epoch={epoch:03d} train_loss={row['train_forecast_loss']:.4f} "
                f"val_loss={val_selection_loss:.4f} val_score={val_score:.4f} "
                f"val_fore_acc={row['val_forecast_accuracy']:.4f} "
                f"test_score={test_score:.4f} test_fore_acc={row['test_forecast_accuracy']:.4f} "
                f"selection={selection_key}:{selection_value:.4f}",
                flush=True,
            )
        if best_selection_value is None:
            improved = True
        elif selection_mode == "min":
            improved = selection_value < best_selection_value - 1e-8
        else:
            improved = selection_value > best_selection_value + 1e-8
        if improved:
            best_selection_value = float(selection_value)
            best_selection_metric = selection_key
            best_val_loss = float(val_selection_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    info = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_score": float(best_score),
        "best_selection_metric": best_selection_metric,
        "best_selection_value": float(best_selection_value if best_selection_value is not None else 0.0),
        "device": str(device),
        "standardizer": standardizer,
        "train_metrics": {"forecast": _evaluate_linear(model, train_x, train_y, num_activities, batch_size, device, sil_index)},
        "val_metrics": {"forecast": _evaluate_linear(model, val_x, val_y, num_activities, batch_size, device, sil_index)},
        "test_metrics": {"forecast": _evaluate_linear(model, test_x, test_y, num_activities, batch_size, device, sil_index)},
        "forecast_baselines": _flat_forecast_baseline_diagnostics(
            train_y,
            {"train": train_y, "val": val_y, "test": test_y},
            num_activities,
            sil_index,
        ),
        "forecast_class_weighting": bool(forecast_class_weighting),
        "forecast_class_weight_cap": float(forecast_class_weight_cap),
        "forecast_sil_false_positive_penalty": float(forecast_sil_false_positive_penalty),
        "classifier_l1_weight": float(classifier_l1_weight),
    }
    return {"forecast_model": model.cpu(), "standardizer": standardizer}, info


def _flat_history_examples(
    split: Dict[str, np.ndarray],
    target_key: str,
    history_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    concepts = split["concepts"]
    targets = split[target_key]
    example_mask = np.asarray(split.get("example_mask", split["mask"]), dtype=np.float32)
    features, labels = [], []
    for video_idx in range(concepts.shape[0]):
        length = int(split["lengths"][video_idx])
        for timestep in range(length):
            if example_mask[video_idx, timestep] <= 0.0:
                continue
            target = int(targets[video_idx, timestep])
            if target < 0:
                continue
            start = max(0, timestep - int(history_length) + 1)
            history = concepts[video_idx, start : timestep + 1]
            padded = np.zeros((int(history_length), concepts.shape[-1]), dtype=np.float32)
            padded[-history.shape[0] :] = history
            features.append(padded.reshape(-1))
            labels.append(target)
    if not features:
        raise ValueError(f"No valid examples found for target {target_key!r}.")
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def _evaluate_linear(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    sil_index: int | None = None,
) -> Dict[str, float]:
    model.eval()
    logits_chunks = []
    for start in range(0, features.shape[0], int(batch_size)):
        x = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits_chunks.append(model(x).cpu())
    logits = torch.cat(logits_chunks, dim=0)
    preds = logits.argmax(dim=1).numpy()
    metrics = _classification_metrics(labels, preds, num_classes)
    metrics.update(_sil_metrics(labels, preds, sil_index))
    metrics["top3_accuracy"] = _topk_accuracy(logits, torch.as_tensor(labels, dtype=torch.long), k=3)
    metrics["loss"] = float(F.cross_entropy(logits, torch.as_tensor(labels, dtype=torch.long)).item())
    metrics["num_examples"] = int(labels.shape[0])
    return metrics


def _classification_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> Dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    preds = preds.astype(np.int64, copy=False)
    if labels.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    accuracy = float(np.mean(labels == preds))
    confusion = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)
    tp = np.diag(confusion).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    true_count = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"accuracy": accuracy, "macro_f1": float(f1.mean())}


def _topk_hit_count(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    if labels.numel() == 0:
        return 0
    top_k = min(int(k), int(logits.shape[-1]))
    top_indices = logits.topk(top_k, dim=-1).indices
    return int((top_indices == labels[:, None]).any(dim=-1).sum().item())


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    return _topk_hit_count(logits, labels, k=k) / max(int(labels.numel()), 1)


def _sil_metrics(labels: np.ndarray, preds: np.ndarray, sil_index: int | None) -> Dict[str, float]:
    if sil_index is None or labels.size == 0:
        return {
            "sil_true_rate": 0.0,
            "sil_pred_rate": 0.0,
            "sil_false_positive_rate": 0.0,
        }
    sil = int(sil_index)
    true_sil = labels == sil
    pred_sil = preds == sil
    non_sil = ~true_sil
    false_positive_rate = float(np.mean(pred_sil[non_sil])) if np.any(non_sil) else 0.0
    return {
        "sil_true_rate": float(np.mean(true_sil)),
        "sil_pred_rate": float(np.mean(pred_sil)),
        "sil_false_positive_rate": false_positive_rate,
    }


def _history_forecast_diagnostic_row(prefix: str, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    activity = metrics.get("activity", {})
    for key, value in activity.items():
        if _is_component_accuracy_key(key):
            row[f"{prefix}_activity_{key}"] = float(value)
    forecast = metrics.get("forecast", {})
    for key, value in forecast.items():
        if _is_component_accuracy_key(key):
            row[f"{prefix}_forecast_{key}"] = float(value)
    for key in ("sil_true_rate", "sil_pred_rate", "sil_false_positive_rate"):
        if key in forecast:
            row[f"{prefix}_forecast_{key}"] = float(forecast[key])
    for step, step_metrics in metrics.get("forecast_by_horizon", {}).items():
        for key, value in step_metrics.items():
            if _is_component_accuracy_key(key):
                row[f"{prefix}_forecast_h{step}_{key}"] = float(value)
        for key in ("accuracy", "macro_f1", "top3_accuracy", "sil_true_rate", "sil_pred_rate", "sil_false_positive_rate"):
            if key in step_metrics:
                row[f"{prefix}_forecast_h{step}_{key}"] = float(step_metrics[key])
    return row


def _is_component_accuracy_key(key: object) -> bool:
    key = str(key)
    return key.endswith("_accuracy") and key not in {"accuracy", "macro_f1", "top3_accuracy"}


def _state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_init_wandb(project: str | None, config: Dict[str, object]):
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("[train_model] wandb is not installed; skipping wandb logging.")
        return None
    name = config.get("logging_name")
    return wandb.init(
        project=project,
        config=config,
        name=str(name) if name else None,
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_float(mapping: Mapping[str, object] | None, *keys: str) -> float | None:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _to_float(current)


def _put_float(target: Dict[str, float], key: str, value: object) -> None:
    number = _to_float(value)
    if number is not None:
        target[key] = number


def _forecast_horizon_metric(metrics: Mapping[str, object] | None, horizon: int, metric: str) -> float | None:
    value = _nested_float(metrics, "forecast_by_horizon", str(int(horizon)), metric)
    if value is not None:
        return value
    if int(horizon) == 3:
        return _nested_float(metrics, "forecast", metric)
    return None


def _wandb_history_key(key: str) -> str:
    for prefix in ("train_", "val_", "test_", "topk_", "classifier_"):
        if key.startswith(prefix):
            return f"{prefix[:-1]}/{key[len(prefix):]}"
    return f"trainer/{key}"


def _wandb_epoch_row(
    row: Mapping[str, object],
    *,
    namespace: str | None = None,
) -> Dict[str, float | int | str]:
    clean: Dict[str, float | int | str] = {}
    if "epoch" in row:
        clean["epoch"] = int(row["epoch"])
    _put_float(clean, "train/loss", row.get("train_total_loss", row.get("train_forecast_loss")))
    _put_float(clean, "train/concept_forecast_weighted_loss", row.get("train_concept_forecast_weighted_loss"))
    _put_float(clean, "train/gcssbm_state_transition_loss", row.get("train_gcssbm_state_transition_loss"))
    _put_float(
        clean,
        "train/gcssbm_state_transition_weighted_loss",
        row.get("train_gcssbm_state_transition_weighted_loss"),
    )
    _put_float(clean, "train/concept_intervention_task_loss", row.get("train_concept_intervention_task_loss"))
    _put_float(
        clean,
        "train/concept_intervention_task_weighted_loss",
        row.get("train_concept_intervention_task_weighted_loss"),
    )
    for name in (
        "persistent_intervention_task",
        "graph_intervention",
        "graph_task_intervention",
        "graph_necessity",
    ):
        _put_float(clean, f"train/{name}_loss", row.get(f"train_{name}_loss"))
        _put_float(clean, f"train/{name}_weighted_loss", row.get(f"train_{name}_weighted_loss"))
    _put_float(clean, "val/selection", row.get("val_selection_value", row.get("val_selection_loss")))
    _put_float(clean, "val/activity_acc", row.get("val_activity_accuracy"))
    _put_float(clean, "test/activity_acc", row.get("test_activity_accuracy"))
    for split in ("val", "test"):
        for key, value in row.items():
            prefix = f"{split}_activity_"
            if not str(key).startswith(prefix) or not str(key).endswith("_accuracy"):
                continue
            component = str(key)[len(prefix) : -len("_accuracy")]
            if component in {"", "macro_f1", "top3", "forecast"}:
                continue
            _put_float(clean, f"{split}/activity_{component}_acc", value)
    for stage in ("train", "eval"):
        for family in ("spatial", "temporal", "cross_temporal"):
            _put_float(clean, f"topk/{stage}_{family}", row.get(f"topk_{stage}_{family}"))
    _put_float(clean, "topk/checkpoint_eligible", row.get("topk_checkpoint_eligible"))

    for split in ("val", "test"):
        for horizon in (1, 3):
            acc = row.get(f"{split}_forecast_h{horizon}_accuracy")
            top3 = row.get(f"{split}_forecast_h{horizon}_top3_accuracy")
            if horizon == 3:
                acc = row.get(f"{split}_forecast_accuracy", acc)
                top3 = row.get(f"{split}_forecast_top3_accuracy", top3)
            _put_float(clean, f"{split}/forecast_h{horizon}_acc", acc)
            _put_float(clean, f"{split}/forecast_h{horizon}_top3", top3)
            for key, value in row.items():
                prefix = f"{split}_forecast_h{horizon}_"
                if not str(key).startswith(prefix) or not str(key).endswith("_accuracy"):
                    continue
                component = str(key)[len(prefix) : -len("_accuracy")]
                if component in {"", "macro_f1", "top3"}:
                    continue
                _put_float(clean, f"{split}/forecast_h{horizon}_{component}_acc", value)

    # Preserve the concise dashboard aliases above, while also forwarding every
    # scalar history field so history.json and W&B carry the same information.
    for key, value in row.items():
        key = str(key)
        if key == "epoch":
            continue
        target = _wandb_history_key(key)
        number = _to_float(value)
        if number is not None:
            clean.setdefault(target, number)
        elif isinstance(value, str):
            clean.setdefault(target, value)

    if namespace:
        namespaced: Dict[str, float | int | str] = {}
        if "epoch" in clean:
            namespaced["epoch"] = clean["epoch"]
        for key, value in clean.items():
            if key != "epoch":
                namespaced[f"{namespace}/{key}"] = value
        return namespaced
    return clean


def _flatten_wandb_summary(
    target: Dict[str, float],
    value: object,
    prefix: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        child_prefix = f"{prefix}/{key}"
        if isinstance(child, Mapping):
            _flatten_wandb_summary(target, child, child_prefix)
            continue
        number = _to_float(child)
        if number is not None:
            target.setdefault(child_prefix, number)


def _wandb_summary_metrics(info: Mapping[str, object]) -> Dict[str, float]:
    metrics = {
        "test": info.get("test_metrics", {}),
        "test_graph_disabled": info.get("test_graph_disabled_metrics", {}),
        "test_graph_corrupted": info.get("test_graph_corrupted_metrics", {}),
        "test_graph_corruption_delta": info.get("test_graph_corruption_delta_metrics", {}),
    }
    summary = compact_summary_metrics(metrics)
    for source_name, source in metrics.items():
        _flatten_wandb_summary(summary, source, f"final/{source_name}")
    return {str(key): float(value) for key, value in summary.items() if _to_float(value) is not None}


def compact_summary_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    test = metrics.get("test", {})
    disabled = metrics.get("test_graph_disabled", {})
    corrupted = metrics.get("test_graph_corrupted", {})
    corruption_delta = metrics.get("test_graph_corruption_delta", {})
    summary: Dict[str, float] = {}

    _put_float(summary, "test/activity_acc", _nested_float(test, "activity", "accuracy"))
    activity_metrics = test.get("activity") if isinstance(test, Mapping) else None
    if isinstance(activity_metrics, Mapping):
        for key, value in activity_metrics.items():
            if _is_component_accuracy_key(key):
                component = str(key)[: -len("_accuracy")]
                _put_float(summary, f"test/activity_{component}_acc", value)
    _put_float(
        summary,
        "test/concept_forecast_smooth_l1",
        _nested_float(test, "concept_forecast", "smooth_l1"),
    )
    _put_float(
        summary,
        "train/concept_intervention_task_loss",
        _nested_float(metrics, "experiment", "train_concept_intervention_task_loss"),
    )
    _put_float(
        summary,
        "experiment/concept_intervention_task_loss_weight",
        _nested_float(metrics, "experiment", "concept_intervention_task_loss_weight"),
    )
    _put_float(
        summary,
        "experiment/concept_intervention_mask_ratio",
        _nested_float(metrics, "experiment", "concept_intervention_mask_ratio"),
    )
    _put_float(
        summary,
        "experiment/graph_intervention_sample_fraction",
        _nested_float(metrics, "experiment", "graph_intervention_sample_fraction"),
    )
    for name in (
        "persistent_intervention_task",
        "graph_task_intervention",
        "graph_necessity",
    ):
        _put_float(
            summary,
            f"train/{name}_loss",
            _nested_float(metrics, "experiment", f"train_{name}_loss"),
        )
        _put_float(
            summary,
            f"experiment/{name}_loss_weight",
            _nested_float(metrics, "experiment", f"{name}_loss_weight"),
        )
    _put_float(
        summary,
        "train/gcssbm_state_transition_loss",
        _nested_float(metrics, "experiment", "train_gcssbm_state_transition_loss"),
    )
    _put_float(
        summary,
        "experiment/gcssbm_state_transition_loss_weight",
        _nested_float(metrics, "experiment", "gcssbm_state_transition_loss_weight"),
    )
    for horizon in (1, 3):
        _put_float(summary, f"test/forecast_h{horizon}_acc", _forecast_horizon_metric(test, horizon, "accuracy"))
        _put_float(summary, f"test/forecast_h{horizon}_top3", _forecast_horizon_metric(test, horizon, "top3_accuracy"))
        horizon_metrics = None
        if isinstance(test, Mapping) and isinstance(test.get("forecast_by_horizon"), Mapping):
            horizon_metrics = test["forecast_by_horizon"].get(str(horizon))
        if not isinstance(horizon_metrics, Mapping) and int(horizon) == 3:
            horizon_metrics = test.get("forecast") if isinstance(test, Mapping) else None
        if isinstance(horizon_metrics, Mapping):
            for key, value in horizon_metrics.items():
                if _is_component_accuracy_key(key):
                    component = str(key)[: -len("_accuracy")]
                    _put_float(summary, f"test/forecast_h{horizon}_{component}_acc", value)
        disabled_acc = _forecast_horizon_metric(disabled, horizon, "accuracy")
        corrupted_acc = _forecast_horizon_metric(corrupted, horizon, "accuracy")
        test_acc = _forecast_horizon_metric(test, horizon, "accuracy")
        _put_float(summary, f"ablation/graph_disabled_h{horizon}_acc", disabled_acc)
        _put_float(summary, f"ablation/graph_corrupted_h{horizon}_acc", corrupted_acc)
        if test_acc is not None and disabled_acc is not None:
            summary[f"ablation/graph_disabled_delta_h{horizon}"] = float(test_acc - disabled_acc)
        _put_float(
            summary,
            f"ablation/graph_corrupted_delta_h{horizon}",
            _nested_float(corruption_delta, "forecast_by_horizon", str(horizon), "accuracy_delta"),
        )

    for key in ("sil_true_rate", "sil_pred_rate", "sil_false_positive_rate"):
        _put_float(summary, f"diagnostic/{key}", _nested_float(test, "forecast", key))
    return summary


def _log_epoch(
    wandb_run,
    row: Dict[str, object],
    *,
    namespace: str | None = None,
) -> None:
    if wandb_run is not None:
        wandb_run.log(_wandb_epoch_row(row, namespace=namespace))
