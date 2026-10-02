"""Temporal relational concept bottleneck model for activity forecasting."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .motif_activity_forecast import (
        PerChannelTemporalBlock,
        build_causal_attn_mask,
        causal_window_mean_torch,
    )
except ImportError:
    from motif_activity_forecast import (
        PerChannelTemporalBlock,
        build_causal_attn_mask,
        causal_window_mean_torch,
    )


CONCEPT_ACTIVATIONS = ("affine", "sigmoid", "learned_threshold")
CONTEXTS = ("flat", "mean", "current")
FORECAST_CONTEXTS = ("flat", "mean")
REPRESENTATION_MODES = ("combined", "z_only", "s_only")
GRAPH_ARCHITECTURES = ("spatio_temporal",)
ST_VIDEO_POOLING_MODES = ("none", "mean", "logsumexp")
ST_STATE_ACTIVATIONS = ("identity", "bounded_logit")
ST_PREDICTION_TRANSFORMS = ("identity", "centered", "logit")
ST_PAST_CONTEXT_MODES = (
    "none",
    "gated_memory",
    "summary_stats",
    "prefix_observed_memory",
    "prefix_recursive_memory",
    "prefix_graph_memory_window",
)
ST_FORECAST_ROLLOUT_MODES = ("legacy", "controlled_graph", "controlled_dense", "persistence")
ST_OBSERVED_REFINER_MODES = ("graph", "dense")
ST_TOPK_TRAINING_MODES = ("hard", "none", "gradual", "soft_train_hard_eval")
ST_ACTIVITY_FEEDBACK_MODES = (
    "none",
    "sparse_label_to_concept",
    "classifier_weight_to_concept",
    "ridge_inverse_to_concept",
    "prototype_label_to_concept",
    "mlp_label_to_concept",
    "label_autoregressive",
    "hybrid_sparse_and_label_autoregressive",
)


class ConceptCalibrator(nn.Module):
    """Per-concept calibration from standardized scores to concept evidence."""

    def __init__(self, num_concepts: int, activation: str = "affine") -> None:
        super().__init__()
        if activation not in CONCEPT_ACTIVATIONS:
            raise ValueError(f"activation must be one of {CONCEPT_ACTIVATIONS}")
        self.activation = activation
        self.scale = nn.Parameter(torch.ones(num_concepts, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_concepts, dtype=torch.float32))
        self.threshold = nn.Parameter(torch.zeros(num_concepts, dtype=torch.float32))
        self.log_sharpness = nn.Parameter(torch.zeros(num_concepts, dtype=torch.float32))

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        if self.activation == "learned_threshold":
            sharpness = F.softplus(self.log_sharpness) + 1e-4
            return torch.sigmoid((concepts - self.threshold) * sharpness)
        evidence = (concepts * self.scale) + self.bias
        if self.activation == "sigmoid":
            return torch.sigmoid(evidence)
        return evidence


def causal_window_flat_torch(states: torch.Tensor, valid_mask: torch.Tensor, window_size: int) -> torch.Tensor:
    """Flatten the last `window_size` causal states in oldest-to-current order."""

    batch_size, timesteps, channels = states.shape
    pieces = []
    for offset in range(window_size - 1, -1, -1):
        shifted = torch.zeros_like(states)
        if offset == 0:
            shifted = states
        elif offset < timesteps:
            shifted[:, offset:, :] = states[:, : timesteps - offset, :]
        pieces.append(shifted)
    flat = torch.cat(pieces, dim=-1)
    return flat * valid_mask.unsqueeze(-1)


class TemporalRelationalCBM(nn.Module):
    """Causal CBM with learned same-time and lagged concept relations.

    The model keeps every intermediate state concept-indexed. It first calibrates
    standardized CLIP concept similarities into bounded activations, then uses a
    sparse temporal concept graph to produce one state per concept and window.
    """

    def __init__(
        self,
        num_concepts: int,
        num_activities: int,
        history_length: int,
        forecast_horizons: Iterable[int],
        edge_threshold: float = 0.2,
        edge_gate_init: float = -1.0,
        forecast_context: str = "flat",
        activity_context: str = "flat",
        concept_activation: str = "affine",
        representation_mode: str = "s_only",
        future_concept_horizons: Iterable[int] | None = None,
        motif_z_attention_layers: int = 0,
        motif_z_attention_width: int = 1,
        motif_z_attention_dropout: float = 0.1,
        motif_z_attention_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_activities = int(num_activities)
        self.history_length = int(history_length)
        self.forecast_horizons = tuple(sorted({int(horizon) for horizon in forecast_horizons}))
        self.edge_threshold = float(edge_threshold)
        self.edge_gate_init = float(edge_gate_init)
        self.forecast_context = str(forecast_context)
        self.activity_context = str(activity_context)
        self.concept_activation = str(concept_activation)
        self.representation_mode = str(representation_mode)
        self.motif_z_attention_layers = int(motif_z_attention_layers)
        self.motif_z_attention_width = int(motif_z_attention_width)
        self.motif_z_attention_dropout = float(motif_z_attention_dropout)
        self.future_concept_horizons = tuple(
            sorted({int(horizon) for horizon in (future_concept_horizons or [])})
        )
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not self.forecast_horizons:
            raise ValueError("At least one forecast horizon is required.")
        if any(horizon < 1 for horizon in self.forecast_horizons):
            raise ValueError("Forecast horizons must be >= 1.")
        if any(horizon < 1 for horizon in self.future_concept_horizons):
            raise ValueError("Future concept horizons must be >= 1.")
        if self.forecast_context not in FORECAST_CONTEXTS:
            raise ValueError(f"forecast_context must be one of {FORECAST_CONTEXTS}")
        if self.activity_context not in CONTEXTS:
            raise ValueError(f"activity_context must be one of {CONTEXTS}")
        if self.concept_activation not in CONCEPT_ACTIVATIONS:
            raise ValueError(f"concept_activation must be one of {CONCEPT_ACTIVATIONS}")
        if self.representation_mode not in REPRESENTATION_MODES:
            raise ValueError(f"representation_mode must be one of {REPRESENTATION_MODES}")
        if self.motif_z_attention_layers < 0:
            raise ValueError("motif_z_attention_layers must be >= 0")
        if self.motif_z_attention_width < 1:
            raise ValueError("motif_z_attention_width must be >= 1")

        self.calibrator = ConceptCalibrator(self.num_concepts, activation=self.concept_activation)
        self.motif_z_layers = nn.ModuleList(
            [
                PerChannelTemporalBlock(
                    self.num_concepts,
                    width=self.motif_z_attention_width,
                    dropout=self.motif_z_attention_dropout,
                )
                for _ in range(self.motif_z_attention_layers)
            ]
        )
        self.motif_z_gate_logit = nn.Parameter(torch.tensor(float(motif_z_attention_gate_init)))
        self.same_weight = nn.Parameter(0.02 * torch.randn(self.num_concepts, self.num_concepts))
        self.same_gate_logits = nn.Parameter(
            torch.full((self.num_concepts, self.num_concepts), self.edge_gate_init)
        )
        num_lags = max(self.history_length - 1, 0)
        self.lag_weight = nn.Parameter(0.02 * torch.randn(num_lags, self.num_concepts, self.num_concepts))
        self.lag_gate_logits = nn.Parameter(
            torch.full((num_lags, self.num_concepts, self.num_concepts), self.edge_gate_init)
        )

        same_mask = torch.ones((self.num_concepts, self.num_concepts), dtype=torch.float32)
        same_mask.fill_diagonal_(0.0)
        lag_mask = torch.ones((num_lags, self.num_concepts, self.num_concepts), dtype=torch.float32)
        if num_lags > 0:
            diagonal = torch.eye(self.num_concepts, dtype=torch.bool).unsqueeze(0)
            lag_mask = lag_mask.masked_fill(diagonal, 0.0)
        self.register_buffer("same_mask", same_mask)
        self.register_buffer("lag_mask", lag_mask)
        self.register_buffer("same_prune_mask", same_mask.clone())
        self.register_buffer("lag_prune_mask", lag_mask.clone())

        graph_feature_dim = 3 * self.num_concepts
        self.candidate_head = nn.Linear(graph_feature_dim, self.num_concepts)
        self.gate_head = nn.Linear(graph_feature_dim, self.num_concepts)
        representation_dim = 2 * self.num_concepts if self.representation_mode == "combined" else self.num_concepts
        if self.activity_context == "flat":
            activity_input_dim = representation_dim * self.history_length
        else:
            activity_input_dim = representation_dim

        print(f"Initializing TemporalRelationalCBM with {self.num_concepts} concepts, {self.activity_input_dim} activity_input_dim, ")
        self.activity_head = nn.Linear(activity_input_dim, self.num_activities)
        if self.forecast_context == "flat":
            forecast_input_dim = representation_dim * self.history_length
        else:
            forecast_input_dim = representation_dim
        print(f"forecast_input_dim: {forecast_input_dim}")
        self.forecast_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(forecast_input_dim, self.num_activities) for horizon in self.forecast_horizons}
        )
        self.future_concept_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(forecast_input_dim, self.num_concepts) for horizon in self.future_concept_horizons}
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        same_prune_key = prefix + "same_prune_mask"
        lag_prune_key = prefix + "lag_prune_mask"
        if same_prune_key not in state_dict:
            state_dict[same_prune_key] = self.same_mask.clone()
        if lag_prune_key not in state_dict:
            state_dict[lag_prune_key] = self.lag_mask.clone()
        motif_gate_key = prefix + "motif_z_gate_logit"
        if motif_gate_key not in state_dict:
            state_dict[motif_gate_key] = self.motif_z_gate_logit.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply_motif_z_attention(
        self,
        calibrated: torch.Tensor,
        key_padding_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.motif_z_layers) == 0:
            return calibrated
        temporalized = calibrated
        causal_mask = build_causal_attn_mask(calibrated.size(1), calibrated.device)
        for layer in self.motif_z_layers:
            temporalized = layer(
                temporalized,
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask,
            )
            temporalized = temporalized * valid_mask.unsqueeze(-1)
        gate = torch.sigmoid(self.motif_z_gate_logit)
        return (calibrated + gate * (temporalized - calibrated)) * valid_mask.unsqueeze(-1)

    def _prediction_representation(self, calibrated: torch.Tensor, concept_states: torch.Tensor) -> torch.Tensor:
        if self.representation_mode == "z_only":
            return calibrated
        if self.representation_mode == "s_only":
            return concept_states
        return torch.cat([calibrated, concept_states], dim=-1)

    def effective_same_matrix(self) -> torch.Tensor:
        return self.same_weight * torch.sigmoid(self.same_gate_logits) * self.same_mask * self.same_prune_mask

    def effective_lag_matrices(self) -> torch.Tensor:
        return self.lag_weight * torch.sigmoid(self.lag_gate_logits) * self.lag_mask * self.lag_prune_mask

    def edge_regularization(self) -> torch.Tensor:
        same = self.effective_same_matrix().abs().sum()
        lag = self.effective_lag_matrices().abs().sum()
        gates = (torch.sigmoid(self.same_gate_logits) * self.same_mask * self.same_prune_mask).sum()
        if self.lag_gate_logits.numel() > 0:
            gates = gates + (torch.sigmoid(self.lag_gate_logits) * self.lag_mask * self.lag_prune_mask).sum()
        return same + lag + gates

    @torch.no_grad()
    def set_graph_priors(
        self,
        same_prior: torch.Tensor | np.ndarray,
        lag_prior: torch.Tensor | np.ndarray | None = None,
        gate_logit: float = 1.0,
    ) -> None:
        """Initialize graph weights and gates from concept-only relation priors."""

        same_prior_tensor = torch.as_tensor(same_prior, dtype=self.same_weight.dtype, device=self.same_weight.device)
        if same_prior_tensor.shape != self.same_weight.shape:
            raise ValueError(f"same_prior shape {tuple(same_prior_tensor.shape)} != {tuple(self.same_weight.shape)}")
        same_prior_tensor = same_prior_tensor * self.same_mask
        self.same_weight.zero_()
        self.same_weight.copy_(same_prior_tensor)
        self.same_gate_logits.fill_(self.edge_gate_init)
        self.same_gate_logits.masked_fill_(same_prior_tensor.abs() > 0.0, float(gate_logit))

        if self.lag_weight.numel() == 0:
            return
        if lag_prior is None:
            raise ValueError("lag_prior is required when history_length > 1")
        lag_prior_tensor = torch.as_tensor(lag_prior, dtype=self.lag_weight.dtype, device=self.lag_weight.device)
        if lag_prior_tensor.shape != self.lag_weight.shape:
            raise ValueError(f"lag_prior shape {tuple(lag_prior_tensor.shape)} != {tuple(self.lag_weight.shape)}")
        lag_prior_tensor = lag_prior_tensor * self.lag_mask
        self.lag_weight.zero_()
        self.lag_weight.copy_(lag_prior_tensor)
        self.lag_gate_logits.fill_(self.edge_gate_init)
        self.lag_gate_logits.masked_fill_(lag_prior_tensor.abs() > 0.0, float(gate_logit))

    @staticmethod
    def _topk_incoming_mask(scores: torch.Tensor, possible_mask: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep top-k source edges for each target column."""

        if top_k <= 0:
            return possible_mask.clone()
        new_mask = torch.zeros_like(possible_mask)
        num_targets = scores.shape[-1]
        flat_scores = scores.reshape(-1, num_targets)
        flat_possible = possible_mask.reshape(-1, num_targets)
        flat_new = new_mask.reshape(-1, num_targets)
        for target in range(num_targets):
            possible_indices = torch.nonzero(flat_possible[:, target] > 0.0, as_tuple=False).flatten()
            if possible_indices.numel() == 0:
                continue
            keep_count = min(int(top_k), int(possible_indices.numel()))
            values = flat_scores[possible_indices, target].abs()
            keep_local = torch.topk(values, k=keep_count, largest=True).indices
            flat_new[possible_indices[keep_local], target] = 1.0
        return new_mask

    @torch.no_grad()
    def apply_topk_pruning(self, same_top_k: int = 0, lag_top_k: int = 0) -> Dict[str, float]:
        """Freeze readable top-k incoming graph edges per target concept."""

        if same_top_k > 0:
            same_possible = self.same_mask * self.same_prune_mask
            same_scores = self.effective_same_matrix().abs()
            self.same_prune_mask.copy_(self._topk_incoming_mask(same_scores, same_possible, same_top_k))
        if lag_top_k > 0 and self.lag_weight.numel() > 0:
            lag_possible = self.lag_mask * self.lag_prune_mask
            lag_scores = self.effective_lag_matrices().abs()
            self.lag_prune_mask.copy_(self._topk_incoming_mask(lag_scores, lag_possible, lag_top_k))
        return self.graph_metrics()

    def graph_metrics(self) -> Dict[str, float]:
        same_structural = self.same_mask.detach()
        lag_structural = self.lag_mask.detach()
        same_remaining = (self.same_mask * self.same_prune_mask).detach()
        lag_remaining = (self.lag_mask * self.lag_prune_mask).detach()
        same_gate = (torch.sigmoid(self.same_gate_logits) * same_remaining).detach()
        lag_gate = (torch.sigmoid(self.lag_gate_logits) * lag_remaining).detach()
        same_weight = self.effective_same_matrix().detach()
        lag_weight = self.effective_lag_matrices().detach()
        same_active = (same_gate > self.edge_threshold) & (same_weight.abs() > 1e-8)
        lag_active = (lag_gate > self.edge_threshold) & (lag_weight.abs() > 1e-8)
        same_gate_values = same_gate[same_remaining > 0.0]
        lag_gate_values = lag_gate[lag_remaining > 0.0]
        same_possible = max(int((same_structural > 0.0).sum().item()), 1)
        lag_possible = max(int((lag_structural > 0.0).sum().item()), 1)
        same_kept = int((same_remaining > 0.0).sum().item())
        lag_kept = int((lag_remaining > 0.0).sum().item())
        return {
            "active_same_time_edges": float(same_active.sum().item()),
            "active_lagged_edges": float(lag_active.sum().item()),
            "same_time_density": float(same_active.sum().item() / same_possible),
            "lagged_density": float(lag_active.sum().item() / lag_possible),
            "mean_same_gate": float(same_gate_values.mean().item()) if same_gate_values.numel() else 0.0,
            "mean_lag_gate": float(lag_gate_values.mean().item()) if lag_gate_values.numel() else 0.0,
            "same_time_possible_edges": float(same_possible),
            "lagged_possible_edges": float(lag_possible),
            "same_time_kept_edges": float(same_kept),
            "lagged_kept_edges": float(lag_kept),
            "pruned_same_time_edges": float(same_possible - same_kept),
            "pruned_lagged_edges": float(lag_possible - lag_kept),
            "motif_z_attention_gate": float(torch.sigmoid(self.motif_z_gate_logit).detach().item()),
            "motif_z_attention_layers": float(len(self.motif_z_layers)),
        }

    def _apply_intervention(
        self,
        calibrated: torch.Tensor,
        concept_index: int,
        timestep: int,
        value: float,
    ) -> torch.Tensor:
        if concept_index < 0 or concept_index >= self.num_concepts:
            raise IndexError(f"concept_index out of range: {concept_index}")
        if timestep < 0 or timestep >= calibrated.size(1):
            raise IndexError(f"timestep out of range: {timestep}")
        intervened = calibrated.clone()
        intervened[:, timestep, concept_index] = float(value)
        return intervened

    def forward(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        batch_size, timesteps, _ = concepts.shape
        valid_mask = (~key_padding_mask).float()
        calibrated = self.calibrator(concepts) * valid_mask.unsqueeze(-1)
        if intervention is not None:
            calibrated = self._apply_intervention(
                calibrated=calibrated,
                concept_index=int(intervention["concept_index"]),
                timestep=int(intervention["timestep"]),
                value=float(intervention["value"]),
            )
            calibrated = calibrated * valid_mask.unsqueeze(-1)
        temporalized = self._apply_motif_z_attention(
            calibrated=calibrated,
            key_padding_mask=key_padding_mask,
            valid_mask=valid_mask,
        )

        same_matrix = self.effective_same_matrix()
        lag_matrices = self.effective_lag_matrices()
        concept_states = torch.zeros_like(temporalized)
        gate_values = torch.zeros_like(temporalized)
        same_messages = torch.zeros_like(temporalized)
        lag_messages = torch.zeros_like(temporalized)

        for timestep in range(timesteps):
            current = temporalized[:, timestep, :]
            same_message = current @ same_matrix
            lag_message = torch.zeros((batch_size, self.num_concepts), dtype=concepts.dtype, device=concepts.device)
            for lag in range(1, self.history_length):
                previous_timestep = timestep - lag
                if previous_timestep < 0:
                    break
                lag_message = lag_message + (temporalized[:, previous_timestep, :] @ lag_matrices[lag - 1])

            graph_features = torch.cat([current, same_message, lag_message], dim=-1)
            candidate = torch.sigmoid(self.candidate_head(graph_features))
            gate = torch.sigmoid(self.gate_head(graph_features))
            state = gate * candidate + (1.0 - gate) * current
            state = state * valid_mask[:, timestep : timestep + 1]
            gate = gate * valid_mask[:, timestep : timestep + 1]

            concept_states[:, timestep, :] = state
            gate_values[:, timestep, :] = gate
            same_messages[:, timestep, :] = same_message
            lag_messages[:, timestep, :] = lag_message

        activity_repr = self._prediction_representation(temporalized, concept_states)
        if self.activity_context == "flat":
            activity_context_repr = causal_window_flat_torch(activity_repr, valid_mask, self.history_length)
        elif self.activity_context == "mean":
            activity_context_repr = causal_window_mean_torch(activity_repr, valid_mask, self.history_length)
        else:
            activity_context_repr = activity_repr
        activity_logits = self.activity_head(activity_context_repr)
        if self.forecast_context == "flat":
            forecast_repr = causal_window_flat_torch(activity_repr, valid_mask, self.history_length)
        else:
            forecast_repr = causal_window_mean_torch(activity_repr, valid_mask, self.history_length)
        forecast_logits_by_horizon = {
            horizon: self.forecast_heads[str(horizon)](forecast_repr) for horizon in self.forecast_horizons
        }
        future_concepts_by_horizon = {
            horizon: self.future_concept_heads[str(horizon)](forecast_repr)
            for horizon in self.future_concept_horizons
        }
        return {
            "calibrated_concepts": calibrated,
            "temporalized_concepts": temporalized,
            "concept_states": concept_states,
            "gate_values": gate_values,
            "same_messages": same_messages,
            "lag_messages": lag_messages,
            "activity_repr": activity_repr,
            "prediction_repr": activity_repr,
            "representation_mode": self.representation_mode,
            "activity_context_repr": activity_context_repr,
            "forecast_repr": forecast_repr,
            "activity_logits": activity_logits,
            "forecast_logits_by_horizon": forecast_logits_by_horizon,
            "future_concepts_by_horizon": future_concepts_by_horizon,
        }

    @torch.no_grad()
    def intervene_concept(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        concept_index: int,
        timestep: int,
        value: float,
    ) -> Dict[str, object]:
        baseline = self.forward(concepts, key_padding_mask)
        intervened = self.forward(
            concepts,
            key_padding_mask,
            intervention={"concept_index": concept_index, "timestep": timestep, "value": value},
        )
        activity_delta = intervened["activity_logits"] - baseline["activity_logits"]
        forecast_delta_by_horizon = {
            horizon: intervened["forecast_logits_by_horizon"][horizon] - baseline["forecast_logits_by_horizon"][horizon]
            for horizon in self.forecast_horizons
        }
        return {
            "baseline": baseline,
            "intervened": intervened,
            "activity_delta": activity_delta,
            "forecast_delta_by_horizon": forecast_delta_by_horizon,
        }


class SpatioTemporalConceptGraphLayer(nn.Module):
    """Concept-indexed residual graph layer over concept-time nodes.

    The layer uses scalar edge weights and gates. Spatial edges are
    input-conditioned within each timestep, while temporal edges propagate
    adjacent same-concept evidence forward in time.
    """

    def __init__(
        self,
        num_concepts: int,
        edge_threshold: float = 0.2,
        edge_gate_init: float = -1.0,
        residual_gate_init: float = -2.0,
        spatial_top_k: int = 0,
        spatial_soft_threshold: float = 0.0,
        enable_spatial: bool = True,
        temporal_top_k: int = 0,
        temporal_soft_threshold: float = 0.0,
        enable_same_concept_temporal: bool = True,
        enable_cross_temporal: bool = False,
        cross_temporal_top_k: int = 0,
        cross_temporal_soft_threshold: float = 0.0,
        message_scale: float = 1.0,
        state_activation: str = "identity",
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.edge_threshold = float(edge_threshold)
        self.edge_gate_init = float(edge_gate_init)
        self.spatial_top_k = int(spatial_top_k)
        self.spatial_soft_threshold = float(spatial_soft_threshold)
        self.enable_spatial = bool(enable_spatial)
        self.temporal_top_k = int(temporal_top_k)
        self.temporal_soft_threshold = float(temporal_soft_threshold)
        self.enable_same_concept_temporal = bool(enable_same_concept_temporal)
        self.enable_cross_temporal = bool(enable_cross_temporal)
        self.cross_temporal_top_k = int(cross_temporal_top_k)
        self.cross_temporal_soft_threshold = float(cross_temporal_soft_threshold)
        self.message_scale = float(message_scale)
        self.state_activation = str(state_activation)
        if self.num_concepts < 1:
            raise ValueError("num_concepts must be >= 1")
        if self.spatial_top_k < 0:
            raise ValueError("spatial_top_k must be >= 0")
        if self.temporal_top_k < 0:
            raise ValueError("temporal_top_k must be >= 0")
        if self.cross_temporal_top_k < 0:
            raise ValueError("cross_temporal_top_k must be >= 0")
        if self.spatial_soft_threshold < 0.0 or self.spatial_soft_threshold >= 1.0:
            raise ValueError("spatial_soft_threshold must be in [0, 1)")
        if self.temporal_soft_threshold < 0.0 or self.temporal_soft_threshold >= 1.0:
            raise ValueError("temporal_soft_threshold must be in [0, 1)")
        if self.cross_temporal_soft_threshold < 0.0 or self.cross_temporal_soft_threshold >= 1.0:
            raise ValueError("cross_temporal_soft_threshold must be in [0, 1)")
        if self.message_scale <= 0.0:
            raise ValueError("message_scale must be > 0")
        if self.state_activation not in ST_STATE_ACTIVATIONS:
            raise ValueError(f"state_activation must be one of {ST_STATE_ACTIVATIONS}")

        self.spatial_weight = nn.Parameter(0.02 * torch.randn(self.num_concepts, self.num_concepts))
        self.spatial_gate_logits = nn.Parameter(
            torch.full((self.num_concepts, self.num_concepts), self.edge_gate_init)
        )
        self.spatial_source_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
        self.spatial_source_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))
        self.spatial_target_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
        self.spatial_target_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))

        self.temporal_weight = nn.Parameter(0.02 * torch.randn(self.num_concepts))
        self.temporal_gate_logits = nn.Parameter(torch.full((self.num_concepts,), self.edge_gate_init))
        self.temporal_source_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
        self.temporal_source_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))
        self.temporal_target_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
        self.temporal_target_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))
        self.residual_gate_logits = nn.Parameter(torch.full((self.num_concepts,), float(residual_gate_init)))

        if self.enable_cross_temporal:
            self.cross_temporal_weight = nn.Parameter(0.02 * torch.randn(self.num_concepts, self.num_concepts))
            self.cross_temporal_gate_logits = nn.Parameter(
                torch.full((self.num_concepts, self.num_concepts), self.edge_gate_init)
            )
            self.cross_source_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
            self.cross_source_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))
            self.cross_target_gate_scale = nn.Parameter(torch.zeros(self.num_concepts))
            self.cross_target_gate_bias = nn.Parameter(torch.full((self.num_concepts,), 2.0))
        else:
            self.register_parameter("cross_temporal_weight", None)
            self.register_parameter("cross_temporal_gate_logits", None)
            self.register_parameter("cross_source_gate_scale", None)
            self.register_parameter("cross_source_gate_bias", None)
            self.register_parameter("cross_target_gate_scale", None)
            self.register_parameter("cross_target_gate_bias", None)

        spatial_mask = torch.ones((self.num_concepts, self.num_concepts), dtype=torch.float32)
        spatial_mask.fill_diagonal_(0.0)
        self.register_buffer("spatial_mask", spatial_mask)
        self.register_buffer("cross_temporal_mask", spatial_mask.clone())
        self.register_buffer("_frozen_spatial_prune_mask", torch.empty(0), persistent=False)
        self.register_buffer("_frozen_temporal_prune_mask", torch.empty(0), persistent=False)
        self.register_buffer("_frozen_cross_temporal_prune_mask", torch.empty(0), persistent=False)

    @staticmethod
    def _topk_incoming_mask(scores: torch.Tensor, possible_mask: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            return possible_mask.clone()
        keep_count = min(int(top_k), int(scores.shape[0]))
        values = scores.abs().masked_fill(possible_mask <= 0.0, float("-inf"))
        keep_indices = torch.topk(values, k=keep_count, dim=0, largest=True).indices
        new_mask = torch.zeros_like(possible_mask)
        new_mask.scatter_(0, keep_indices, 1.0)
        return new_mask * possible_mask

    def _input_gate(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x * scale.view(1, 1, -1)) + bias.view(1, 1, -1))

    @staticmethod
    def _threshold_gate(gate: torch.Tensor, threshold: float) -> torch.Tensor:
        if threshold <= 0.0:
            return gate
        threshold = float(threshold)
        return torch.relu(gate - threshold) / max(1.0 - threshold, 1e-6)

    def spatial_prune_mask(self) -> torch.Tensor:
        if not bool(getattr(self, "enable_spatial", True)):
            return torch.zeros_like(self.spatial_mask)
        frozen = getattr(self, "_frozen_spatial_prune_mask", None)
        if frozen is not None and frozen.numel() > 0:
            return frozen
        if self.spatial_top_k <= 0:
            return self.spatial_mask
        scores = torch.sigmoid(self.spatial_gate_logits) * self.spatial_weight.abs() * self.spatial_mask
        return self._topk_incoming_mask(scores, self.spatial_mask, self.spatial_top_k)

    def temporal_prune_mask(self) -> torch.Tensor:
        if not bool(getattr(self, "enable_same_concept_temporal", True)):
            return torch.zeros_like(self.temporal_weight)
        frozen = getattr(self, "_frozen_temporal_prune_mask", None)
        if frozen is not None and frozen.numel() > 0:
            return frozen
        temporal_top_k = int(getattr(self, "temporal_top_k", 0))
        if temporal_top_k <= 0:
            return torch.ones_like(self.temporal_weight)
        keep_count = min(temporal_top_k, int(self.temporal_weight.numel()))
        scores = torch.sigmoid(self.temporal_gate_logits) * self.temporal_weight.abs()
        mask = torch.zeros_like(self.temporal_weight)
        keep = torch.topk(scores.abs(), k=keep_count, largest=True).indices
        mask[keep] = 1.0
        return mask

    def cross_temporal_prune_mask(self) -> torch.Tensor:
        if not self.enable_cross_temporal or self.cross_temporal_weight is None:
            return self.spatial_weight.new_zeros((self.num_concepts, self.num_concepts))
        frozen = getattr(self, "_frozen_cross_temporal_prune_mask", None)
        if frozen is not None and frozen.numel() > 0:
            return frozen
        cross_temporal_top_k = int(getattr(self, "cross_temporal_top_k", 0))
        if cross_temporal_top_k <= 0:
            return self.cross_temporal_mask
        scores = (
            torch.sigmoid(self.cross_temporal_gate_logits)
            * self.cross_temporal_weight.abs()
            * self.cross_temporal_mask
        )
        return self._topk_incoming_mask(scores, self.cross_temporal_mask, cross_temporal_top_k)

    @torch.no_grad()
    def freeze_topology(self) -> None:
        """Freeze current Top-K edge identities while leaving kept parameters trainable."""

        spatial = self.spatial_prune_mask().detach().clone()
        temporal = self.temporal_prune_mask().detach().clone()
        cross = self.cross_temporal_prune_mask().detach().clone()
        self._frozen_spatial_prune_mask = spatial
        self._frozen_temporal_prune_mask = temporal
        self._frozen_cross_temporal_prune_mask = cross
        # Make a serialized checkpoint recover the same topology even though
        # the frozen masks themselves are intentionally non-persistent.
        self.spatial_weight.mul_(spatial)
        self.spatial_gate_logits.masked_fill_(spatial <= 0.0, -30.0)
        self.temporal_weight.mul_(temporal)
        self.temporal_gate_logits.masked_fill_(temporal <= 0.0, -30.0)
        if self.enable_cross_temporal and self.cross_temporal_weight is not None:
            self.cross_temporal_weight.mul_(cross)
            self.cross_temporal_gate_logits.masked_fill_(cross <= 0.0, -30.0)

    @torch.no_grad()
    def enforce_frozen_topology(self) -> None:
        """Project optimizer updates back onto a previously frozen topology."""

        frozen_spatial = getattr(self, "_frozen_spatial_prune_mask", None)
        if frozen_spatial is None or frozen_spatial.numel() == 0:
            return
        self.spatial_weight.mul_(frozen_spatial)
        self.spatial_gate_logits.masked_fill_(frozen_spatial <= 0.0, -30.0)
        self.temporal_weight.mul_(self._frozen_temporal_prune_mask)
        self.temporal_gate_logits.masked_fill_(self._frozen_temporal_prune_mask <= 0.0, -30.0)
        if self.enable_cross_temporal and self.cross_temporal_weight is not None:
            self.cross_temporal_weight.mul_(self._frozen_cross_temporal_prune_mask)
            self.cross_temporal_gate_logits.masked_fill_(
                self._frozen_cross_temporal_prune_mask <= 0.0,
                -30.0,
            )

    def effective_spatial_matrix(self) -> torch.Tensor:
        if not bool(getattr(self, "enable_spatial", True)):
            return torch.zeros_like(self.spatial_weight)
        gate = self._threshold_gate(
            torch.sigmoid(self.spatial_gate_logits),
            float(getattr(self, "spatial_soft_threshold", 0.0)),
        )
        gate = gate * self.spatial_mask
        return self.spatial_weight * gate * self.spatial_prune_mask()

    def effective_temporal_vector(self) -> torch.Tensor:
        if not bool(getattr(self, "enable_same_concept_temporal", True)):
            return torch.zeros_like(self.temporal_weight)
        gate = self._threshold_gate(
            torch.sigmoid(self.temporal_gate_logits),
            float(getattr(self, "temporal_soft_threshold", 0.0)),
        )
        return self.temporal_weight * gate * self.temporal_prune_mask()

    def effective_cross_temporal_matrix(self) -> torch.Tensor:
        if not self.enable_cross_temporal or self.cross_temporal_weight is None:
            return self.spatial_weight.new_zeros((self.num_concepts, self.num_concepts))
        gate = self._threshold_gate(
            torch.sigmoid(self.cross_temporal_gate_logits),
            float(getattr(self, "cross_temporal_soft_threshold", 0.0)),
        )
        gate = gate * self.cross_temporal_mask
        return self.cross_temporal_weight * gate * self.cross_temporal_prune_mask()

    def _apply_state_update(
        self,
        current: torch.Tensor,
        message: torch.Tensor,
        update_gate: torch.Tensor,
    ) -> torch.Tensor:
        update = update_gate * message
        state_activation = getattr(self, "state_activation", "identity")
        if state_activation == "bounded_logit":
            eps = torch.finfo(current.dtype).eps
            bounded = current.clamp(min=eps, max=1.0 - eps)
            logits = torch.log(bounded) - torch.log1p(-bounded)
            return torch.sigmoid(logits + update)
        return current + update

    def edge_regularization(self) -> torch.Tensor:
        spatial_gate = self._threshold_gate(
            torch.sigmoid(self.spatial_gate_logits),
            float(getattr(self, "spatial_soft_threshold", 0.0)),
        ) * self.spatial_mask
        spatial_prune = self.spatial_prune_mask()
        temporal_gate = self._threshold_gate(
            torch.sigmoid(self.temporal_gate_logits),
            float(getattr(self, "temporal_soft_threshold", 0.0)),
        )
        temporal_prune = self.temporal_prune_mask()
        reg = self.effective_spatial_matrix().abs().sum()
        reg = reg + self.effective_temporal_vector().abs().sum()
        reg = reg + (spatial_gate * spatial_prune).sum()
        reg = reg + (temporal_gate * temporal_prune).sum()
        if self.enable_cross_temporal and self.cross_temporal_gate_logits is not None:
            cross_gate = self._threshold_gate(
                torch.sigmoid(self.cross_temporal_gate_logits),
                float(getattr(self, "cross_temporal_soft_threshold", 0.0)),
            )
            cross_prune = self.cross_temporal_prune_mask()
            cross_gate = cross_gate * self.cross_temporal_mask * cross_prune
            reg = reg + self.effective_cross_temporal_matrix().abs().sum() + cross_gate.sum()
        return reg

    @torch.no_grad()
    def set_spatial_prior(
        self,
        same_prior: torch.Tensor | np.ndarray,
        gate_logit: float,
    ) -> None:
        prior = torch.as_tensor(same_prior, dtype=self.spatial_weight.dtype, device=self.spatial_weight.device)
        if prior.shape != self.spatial_weight.shape:
            raise ValueError(f"same_prior shape {tuple(prior.shape)} != {tuple(self.spatial_weight.shape)}")
        prior = prior * self.spatial_mask
        self.spatial_weight.zero_()
        self.spatial_weight.copy_(prior)
        self.spatial_gate_logits.fill_(self.edge_gate_init)
        self.spatial_gate_logits.masked_fill_(prior.abs() > 0.0, float(gate_logit))

    @torch.no_grad()
    def set_cross_temporal_prior(
        self,
        lag_prior: torch.Tensor | np.ndarray,
        gate_logit: float,
    ) -> None:
        if not self.enable_cross_temporal or self.cross_temporal_weight is None:
            return
        prior = torch.as_tensor(
            lag_prior,
            dtype=self.cross_temporal_weight.dtype,
            device=self.cross_temporal_weight.device,
        )
        if prior.shape != self.cross_temporal_weight.shape:
            raise ValueError(f"lag_prior shape {tuple(prior.shape)} != {tuple(self.cross_temporal_weight.shape)}")
        prior = prior * self.cross_temporal_mask
        self.cross_temporal_weight.zero_()
        self.cross_temporal_weight.copy_(prior)
        self.cross_temporal_gate_logits.fill_(self.edge_gate_init)
        self.cross_temporal_gate_logits.masked_fill_(prior.abs() > 0.0, float(gate_logit))

    def graph_metrics(self) -> Dict[str, float]:
        spatial_possible = int((self.spatial_mask > 0.0).sum().item())
        spatial_kept_mask = self.spatial_prune_mask()
        spatial_kept = int((spatial_kept_mask > 0.0).sum().item())
        spatial_gate = self._threshold_gate(
            torch.sigmoid(self.spatial_gate_logits),
            float(getattr(self, "spatial_soft_threshold", 0.0)),
        )
        spatial_gate = spatial_gate * self.spatial_mask * spatial_kept_mask
        spatial_weight = self.effective_spatial_matrix().detach()
        spatial_gate_detached = spatial_gate.detach()
        spatial_active = (spatial_gate_detached > self.edge_threshold) & (spatial_weight.abs() > 1e-8)
        spatial_gate_values = spatial_gate_detached[spatial_kept_mask > 0.0]

        temporal_prune = self.temporal_prune_mask().detach()
        temporal_gate = self._threshold_gate(
            torch.sigmoid(self.temporal_gate_logits),
            float(getattr(self, "temporal_soft_threshold", 0.0)),
        ).detach() * temporal_prune
        temporal_weight = self.effective_temporal_vector().detach()
        temporal_active = (temporal_gate > self.edge_threshold) & (temporal_weight.abs() > 1e-8)
        lag_possible = int(self.temporal_gate_logits.numel())
        temporal_kept = int((temporal_prune > 0.0).sum().item())
        lag_kept = temporal_kept
        lag_active_count = int(temporal_active.sum().item())
        temporal_gate_values = temporal_gate[temporal_prune > 0.0]
        lag_gate_sum = float(temporal_gate_values.sum().item()) if temporal_gate_values.numel() else 0.0
        lag_gate_count = int(temporal_gate_values.numel())
        cross_possible = 0
        cross_kept = 0
        cross_active_count = 0

        if self.enable_cross_temporal and self.cross_temporal_gate_logits is not None:
            cross_prune = self.cross_temporal_prune_mask().detach()
            cross_gate = self._threshold_gate(
                torch.sigmoid(self.cross_temporal_gate_logits),
                float(getattr(self, "cross_temporal_soft_threshold", 0.0)),
            ).detach() * self.cross_temporal_mask * cross_prune
            cross_weight = self.effective_cross_temporal_matrix().detach()
            cross_active = (cross_gate > self.edge_threshold) & (cross_weight.abs() > 1e-8)
            cross_values = cross_gate[cross_prune > 0.0]
            cross_possible = int((self.cross_temporal_mask > 0.0).sum().item())
            cross_kept = int((cross_prune > 0.0).sum().item())
            cross_active_count = int(cross_active.sum().item())
            lag_possible += cross_possible
            lag_kept += cross_kept
            lag_active_count += cross_active_count
            lag_gate_sum += float(cross_values.sum().item()) if cross_values.numel() else 0.0
            lag_gate_count += int(cross_values.numel())

        return {
            "active_same_time_edges": float(spatial_active.sum().item()),
            "active_lagged_edges": float(lag_active_count),
            "active_same_concept_temporal_edges": float(temporal_active.sum().item()),
            "active_cross_temporal_edges": float(cross_active_count),
            "same_time_possible_edges": float(spatial_possible),
            "lagged_possible_edges": float(lag_possible),
            "same_concept_temporal_possible_edges": float(self.temporal_gate_logits.numel()),
            "cross_temporal_possible_edges": float(cross_possible),
            "same_time_kept_edges": float(spatial_kept),
            "lagged_kept_edges": float(lag_kept),
            "same_concept_temporal_kept_edges": float(temporal_kept),
            "cross_temporal_kept_edges": float(cross_kept),
            "same_gate_sum": float(spatial_gate_values.sum().item()) if spatial_gate_values.numel() else 0.0,
            "same_gate_count": float(spatial_gate_values.numel()),
            "lag_gate_sum": float(lag_gate_sum),
            "lag_gate_count": float(lag_gate_count),
            "same_concept_temporal_gate_sum": float(temporal_gate_values.sum().item())
            if temporal_gate_values.numel()
            else 0.0,
            "same_concept_temporal_gate_count": float(temporal_gate_values.numel()),
        }

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_interventions: list[Dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        spatial_base = self._apply_edge_interventions(
            self.effective_spatial_matrix(),
            edge_interventions,
            "spatial",
            batch_size=x.size(0),
        )
        source_gate = self._input_gate(x, self.spatial_source_gate_scale, self.spatial_source_gate_bias)
        target_gate = self._input_gate(x, self.spatial_target_gate_scale, self.spatial_target_gate_bias)
        # Equivalent to applying A_sp[b,t,i,j] = base[i,j] * source_gate[b,t,i] * target_gate[b,t,j],
        # without materializing the full [B,T,K,K] tensor for long Breakfast sequences.
        spatial_message = torch.matmul(x * source_gate, spatial_base) * target_gate

        temporal_message = torch.zeros_like(x)
        if x.size(1) > 1:
            pair_valid = valid_mask[:, :-1] * valid_mask[:, 1:]
            temporal_source_gate = self._input_gate(
                x[:, :-1, :],
                self.temporal_source_gate_scale,
                self.temporal_source_gate_bias,
            )
            temporal_target_gate = self._input_gate(
                x[:, 1:, :],
                self.temporal_target_gate_scale,
                self.temporal_target_gate_bias,
            )
            temporal_edge = self._apply_edge_interventions(
                self.effective_temporal_vector(),
                edge_interventions,
                "temporal",
                batch_size=x.size(0),
            )
            if temporal_edge.ndim == 1:
                temporal_edge = temporal_edge.view(1, 1, self.num_concepts)
            else:
                temporal_edge = temporal_edge.unsqueeze(1)
            temporal_edge = temporal_edge * temporal_source_gate * temporal_target_gate
            temporal_edge = temporal_edge * pair_valid.unsqueeze(-1)
            temporal_message[:, 1:, :] = temporal_message[:, 1:, :] + (x[:, :-1, :] * temporal_edge)

            if self.enable_cross_temporal:
                cross_base = self._apply_edge_interventions(
                    self.effective_cross_temporal_matrix(),
                    edge_interventions,
                    "cross_temporal",
                    batch_size=x.size(0),
                )
                cross_source_gate = self._input_gate(
                    x[:, :-1, :],
                    self.cross_source_gate_scale,
                    self.cross_source_gate_bias,
                )
                cross_target_gate = self._input_gate(
                    x[:, 1:, :],
                    self.cross_target_gate_scale,
                    self.cross_target_gate_bias,
                )
                cross_message = torch.matmul(x[:, :-1, :] * cross_source_gate, cross_base) * cross_target_gate
                cross_message = cross_message * pair_valid.unsqueeze(-1)
                temporal_message[:, 1:, :] = temporal_message[:, 1:, :] + cross_message

        update_gate = torch.sigmoid(self.residual_gate_logits).view(1, 1, self.num_concepts)
        refined = self._apply_state_update(
            x,
            (spatial_message + temporal_message) * float(getattr(self, "message_scale", 1.0)),
            update_gate,
        ) * valid_mask.unsqueeze(-1)
        return refined, {
            "spatial_messages": spatial_message * valid_mask.unsqueeze(-1),
            "temporal_messages": temporal_message * valid_mask.unsqueeze(-1),
            "update_gates": update_gate.expand_as(x) * valid_mask.unsqueeze(-1),
        }

    def forward_next(
        self,
        previous: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_interventions: list[Dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply this graph layer as a single +1 concept transition."""

        valid_mask = valid_mask.float()
        target = previous
        spatial_base = self._apply_edge_interventions(
            self.effective_spatial_matrix(),
            edge_interventions,
            "spatial",
            batch_size=previous.size(0),
        )
        source_gate = self._input_gate(target, self.spatial_source_gate_scale, self.spatial_source_gate_bias)
        target_gate = self._input_gate(target, self.spatial_target_gate_scale, self.spatial_target_gate_bias)
        spatial_message = torch.matmul(target * source_gate, spatial_base) * target_gate

        temporal_source_gate = self._input_gate(
            previous,
            self.temporal_source_gate_scale,
            self.temporal_source_gate_bias,
        )
        temporal_target_gate = self._input_gate(
            target,
            self.temporal_target_gate_scale,
            self.temporal_target_gate_bias,
        )
        temporal_edge = self._apply_edge_interventions(
            self.effective_temporal_vector(),
            edge_interventions,
            "temporal",
            batch_size=previous.size(0),
        )
        if temporal_edge.ndim == 1:
            temporal_edge = temporal_edge.view(1, 1, self.num_concepts)
        else:
            temporal_edge = temporal_edge.unsqueeze(1)
        temporal_edge = temporal_edge * temporal_source_gate * temporal_target_gate
        temporal_message = previous * temporal_edge * valid_mask.unsqueeze(-1)

        if self.enable_cross_temporal:
            cross_base = self._apply_edge_interventions(
                self.effective_cross_temporal_matrix(),
                edge_interventions,
                "cross_temporal",
                batch_size=previous.size(0),
            )
            cross_source_gate = self._input_gate(
                previous,
                self.cross_source_gate_scale,
                self.cross_source_gate_bias,
            )
            cross_target_gate = self._input_gate(
                target,
                self.cross_target_gate_scale,
                self.cross_target_gate_bias,
            )
            cross_message = torch.matmul(previous * cross_source_gate, cross_base) * cross_target_gate
            temporal_message = temporal_message + (cross_message * valid_mask.unsqueeze(-1))

        update_gate = torch.sigmoid(self.residual_gate_logits).view(1, 1, self.num_concepts)
        refined = self._apply_state_update(
            target,
            (spatial_message + temporal_message) * float(getattr(self, "message_scale", 1.0)),
            update_gate,
        ) * valid_mask.unsqueeze(-1)
        return refined, {
            "spatial_messages": spatial_message * valid_mask.unsqueeze(-1),
            "temporal_messages": temporal_message * valid_mask.unsqueeze(-1),
            "update_gates": update_gate.expand_as(previous) * valid_mask.unsqueeze(-1),
        }

    def _apply_edge_interventions(
        self,
        values: torch.Tensor,
        edge_interventions: list[Dict[str, object]] | None,
        edge_kind: str,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if not edge_interventions:
            return values
        matching_items = []
        for item in edge_interventions:
            kind = str(item.get("edge_kind", item.get("kind", ""))).lower().replace(" ", "_")
            if kind not in {str(edge_kind), f"edge_{edge_kind}"}:
                continue
            matching_items.append(item)
        if not matching_items:
            return values

        has_batch_selectors = any(
            "batch_idx" in item or "batch_start" in item or "batch_end" in item
            for item in matching_items
        )
        adjusted = (
            values.unsqueeze(0).expand(int(batch_size), *values.shape).clone()
            if has_batch_selectors
            else values.clone()
        )
        for item in matching_items:
            has_value = item.get("edge_value") is not None
            has_delta = item.get("edge_delta") is not None
            has_scale = item.get("edge_scale") is not None
            if sum(bool(flag) for flag in (has_value, has_delta, has_scale)) != 1:
                continue
            if has_batch_selectors:
                if "batch_start" in item or "batch_end" in item:
                    start = 0 if item.get("batch_start") is None else int(item["batch_start"])
                    end = None if item.get("batch_end") is None else int(item["batch_end"])
                    batch_selector: slice | int = slice(start, end)
                else:
                    batch_idx = item.get("batch_idx")
                    batch_selector = slice(None) if batch_idx is None else int(batch_idx)
            if edge_kind == "temporal":
                concept_idx = int(item.get("concept_idx", item.get("target_idx", -1)))
                if concept_idx < 0 or concept_idx >= values.numel():
                    continue
                index = (batch_selector, concept_idx) if has_batch_selectors else concept_idx
                if has_value:
                    adjusted[index] = float(item["edge_value"])
                elif has_delta:
                    adjusted[index] = adjusted[index] + float(item["edge_delta"])
                else:
                    adjusted[index] = adjusted[index] * float(item["edge_scale"])
                continue

            source_idx = int(item.get("source_idx", -1))
            target_idx = int(item.get("target_idx", -1))
            if (
                source_idx < 0
                or target_idx < 0
                or source_idx >= values.shape[0]
                or target_idx >= values.shape[1]
            ):
                continue
            index = (
                (batch_selector, source_idx, target_idx)
                if has_batch_selectors
                else (source_idx, target_idx)
            )
            if has_value:
                adjusted[index] = float(item["edge_value"])
            elif has_delta:
                adjusted[index] = adjusted[index] + float(item["edge_delta"])
            else:
                adjusted[index] = adjusted[index] * float(item["edge_scale"])
        return adjusted


class DenseTemporalConceptRolloutLayer(nn.Module):
    """Graph-free dense residual transition with a graph-layer-matched parameter count."""

    def __init__(
        self,
        num_concepts: int,
        *,
        residual_gate_init: float = -2.0,
        enable_cross_temporal: bool = True,
        state_activation: str = "identity",
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.state_activation = str(state_activation)
        self.num_branches = 4 if bool(enable_cross_temporal) else 2
        if self.state_activation not in ST_STATE_ACTIVATIONS:
            raise ValueError(f"state_activation must be one of {ST_STATE_ACTIVATIONS}")

        k = self.num_concepts
        self.branch_weights = nn.ParameterList(
            [nn.Parameter(0.02 * torch.randn(k, k)) for _ in range(self.num_branches)]
        )
        self.branch_bias = nn.Parameter(torch.zeros(self.num_branches, k))
        self.gate_scale = nn.Parameter(torch.zeros(self.num_branches, k))
        self.gate_bias = nn.Parameter(torch.full((self.num_branches, k), 2.0))
        self.residual_gate_logits = nn.Parameter(torch.full((k,), float(residual_gate_init)))
        self.norm_scale = nn.Parameter(torch.ones(k))
        self.norm_bias = nn.Parameter(torch.zeros(k))
        # A graph layer without cross-temporal edges has two fewer K-vectors in
        # its matrix branches; these featurewise affine terms close that exact
        # capacity gap while remaining active in the dense transition.
        if self.num_branches == 2:
            self.extra_scale = nn.Parameter(torch.ones(k))
            self.extra_bias = nn.Parameter(torch.zeros(k))
        else:
            self.register_parameter("extra_scale", None)
            self.register_parameter("extra_bias", None)

    def _apply_state_update(self, current: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if self.state_activation == "bounded_logit":
            eps = torch.finfo(current.dtype).eps
            bounded = current.clamp(min=eps, max=1.0 - eps)
            logits = torch.log(bounded) - torch.log1p(-bounded)
            return torch.sigmoid(logits + update)
        return current + update

    def _transition(self, current: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sources = [current, current, previous, previous][: self.num_branches]
        messages = []
        for index, (source, weight) in enumerate(zip(sources, self.branch_weights)):
            gate = torch.sigmoid(
                source * self.gate_scale[index].view(1, 1, -1)
                + self.gate_bias[index].view(1, 1, -1)
            )
            messages.append(gate * (torch.matmul(source, weight) + self.branch_bias[index]))
        message = torch.stack(messages, dim=0).sum(dim=0)
        if self.extra_scale is not None and self.extra_bias is not None:
            message = message * self.extra_scale.view(1, 1, -1) + self.extra_bias.view(1, 1, -1)
        message = F.layer_norm(
            message,
            (self.num_concepts,),
            self.norm_scale,
            self.norm_bias,
        )
        update_gate = torch.sigmoid(self.residual_gate_logits).view(1, 1, -1)
        return self._apply_state_update(current, update_gate * message), update_gate

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_interventions: list[Dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del edge_interventions
        previous = torch.zeros_like(x)
        if x.size(1) > 1:
            previous[:, 1:, :] = x[:, :-1, :]
        refined, update_gate = self._transition(x, previous)
        valid = valid_mask.float().unsqueeze(-1)
        refined = refined * valid
        return refined, {
            "spatial_messages": torch.zeros_like(x),
            "temporal_messages": (refined - x) * valid,
            "update_gates": update_gate.expand_as(x) * valid,
        }

    def forward_next(
        self,
        previous: torch.Tensor,
        valid_mask: torch.Tensor,
        edge_interventions: list[Dict[str, object]] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del edge_interventions
        refined, update_gate = self._transition(previous, previous)
        valid = valid_mask.float().unsqueeze(-1)
        refined = refined * valid
        return refined, {
            "spatial_messages": torch.zeros_like(previous),
            "temporal_messages": (refined - previous) * valid,
            "update_gates": update_gate.expand_as(previous) * valid,
        }


class SpatioTemporalConceptGraphBase(nn.Module):
    """CBM variant with concept-preserving spatio-temporal graph refinement."""

    graph_architecture = "spatio_temporal"

    def __init__(
        self,
        num_concepts: int,
        num_activities: int,
        history_length: int,
        forecast_horizons: Iterable[int],
        edge_threshold: float = 0.2,
        edge_gate_init: float = -1.0,
        forecast_context: str = "flat",
        activity_context: str = "flat",
        concept_activation: str = "affine",
        representation_mode: str = "s_only",
        future_concept_horizons: Iterable[int] | None = None,
        motif_z_attention_layers: int = 0,
        motif_z_attention_width: int = 1,
        motif_z_attention_dropout: float = 0.1,
        motif_z_attention_gate_init: float = -2.0,
        st_graph_layers: int = 1,
        st_task_graph_layers: int = 1,
        st_spatial_top_k: int = 0,
        st_spatial_soft_threshold: float = 0.0,
        st_enable_spatial: bool = True,
        st_temporal_top_k: int = 0,
        st_temporal_soft_threshold: float = 0.0,
        st_enable_same_concept_temporal: bool = True,
        st_residual_gate_init: float = -2.0,
        st_enable_cross_temporal: bool = False,
        st_cross_temporal_top_k: int = 0,
        st_cross_temporal_soft_threshold: float = 0.0,
        st_message_scale: float = 1.0,
        st_video_pooling: str = "none",
        st_state_activation: str = "identity",
        st_prediction_transform: str = "identity",
        st_past_context_mode: str = "none",
        st_past_context_gate_init: float = -2.0,
        st_memory_prefix_length: int = 0,
        st_summary_short_alpha: float = 0.5,
        st_summary_long_alpha: float = 0.1,
        st_summary_seen_threshold: float = 0.5,
        st_forecast_rollout_mode: str = "legacy",
        st_controlled_dense_reuse_forecast_layer: bool = False,
        st_observed_refiner_mode: str = "graph",
        st_topk_training_mode: str = "hard",
        st_topk_warmup_epochs: int = 20,
        st_topk_ramp_epochs: int = 30,
        use_concept_calibrator: bool = True,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_activities = int(num_activities)
        self.history_length = int(history_length)
        self.forecast_horizons = tuple(sorted({int(horizon) for horizon in forecast_horizons}))
        self.edge_threshold = float(edge_threshold)
        self.edge_gate_init = float(edge_gate_init)
        self.forecast_context = str(forecast_context)
        self.activity_context = str(activity_context)
        self.concept_activation = str(concept_activation)
        self.representation_mode = str(representation_mode)
        self.motif_z_attention_layers = int(motif_z_attention_layers)
        self.motif_z_attention_width = int(motif_z_attention_width)
        self.motif_z_attention_dropout = float(motif_z_attention_dropout)
        self.future_concept_horizons = tuple(
            sorted({int(horizon) for horizon in (future_concept_horizons or [])})
        )
        self.st_graph_layers = int(st_graph_layers)
        self.st_task_graph_layers = int(st_task_graph_layers)
        self.st_spatial_top_k = int(st_spatial_top_k)
        self.st_spatial_soft_threshold = float(st_spatial_soft_threshold)
        self.st_enable_spatial = bool(st_enable_spatial)
        self.st_temporal_top_k = int(st_temporal_top_k)
        self.st_temporal_soft_threshold = float(st_temporal_soft_threshold)
        self.st_enable_same_concept_temporal = bool(st_enable_same_concept_temporal)
        self.st_residual_gate_init = float(st_residual_gate_init)
        self.st_enable_cross_temporal = bool(st_enable_cross_temporal)
        self.st_cross_temporal_top_k = int(st_cross_temporal_top_k)
        self.st_cross_temporal_soft_threshold = float(st_cross_temporal_soft_threshold)
        self.st_message_scale = float(st_message_scale)
        self.st_video_pooling = str(st_video_pooling)
        self.st_state_activation = str(st_state_activation)
        self.st_prediction_transform = str(st_prediction_transform)
        self.st_past_context_mode = str(st_past_context_mode)
        self.st_past_context_gate_init = float(st_past_context_gate_init)
        self.st_memory_prefix_length = int(st_memory_prefix_length)
        self.st_summary_short_alpha = float(st_summary_short_alpha)
        self.st_summary_long_alpha = float(st_summary_long_alpha)
        self.st_summary_seen_threshold = float(st_summary_seen_threshold)
        self.st_forecast_rollout_mode = str(st_forecast_rollout_mode)
        self.st_controlled_dense_reuse_forecast_layer = bool(
            st_controlled_dense_reuse_forecast_layer
        )
        self.st_observed_refiner_mode = str(st_observed_refiner_mode)
        self.st_topk_training_mode = str(st_topk_training_mode)
        self.st_topk_warmup_epochs = int(st_topk_warmup_epochs)
        self.st_topk_ramp_epochs = int(st_topk_ramp_epochs)
        self.st_target_spatial_top_k = int(st_spatial_top_k)
        self.st_target_temporal_top_k = int(st_temporal_top_k)
        self.st_target_cross_temporal_top_k = int(st_cross_temporal_top_k)
        self.use_concept_calibrator = bool(use_concept_calibrator)

        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not self.forecast_horizons:
            raise ValueError("At least one forecast horizon is required.")
        if any(horizon < 1 for horizon in self.forecast_horizons):
            raise ValueError("Forecast horizons must be >= 1.")
        if any(horizon < 1 for horizon in self.future_concept_horizons):
            raise ValueError("Future concept horizons must be >= 1.")
        if self.forecast_context not in FORECAST_CONTEXTS:
            raise ValueError(f"forecast_context must be one of {FORECAST_CONTEXTS}")
        if self.activity_context not in CONTEXTS:
            raise ValueError(f"activity_context must be one of {CONTEXTS}")
        if self.concept_activation not in CONCEPT_ACTIVATIONS:
            raise ValueError(f"concept_activation must be one of {CONCEPT_ACTIVATIONS}")
        if self.representation_mode not in REPRESENTATION_MODES:
            raise ValueError(f"representation_mode must be one of {REPRESENTATION_MODES}")
        if self.motif_z_attention_layers < 0:
            raise ValueError("motif_z_attention_layers must be >= 0")
        if self.motif_z_attention_width < 1:
            raise ValueError("motif_z_attention_width must be >= 1")
        if self.st_graph_layers < 0:
            raise ValueError("st_graph_layers must be >= 0")
        if self.st_task_graph_layers < 0:
            raise ValueError("st_task_graph_layers must be >= 0")
        if self.st_temporal_top_k < 0:
            raise ValueError("st_temporal_top_k must be >= 0")
        if self.st_cross_temporal_top_k < 0:
            raise ValueError("st_cross_temporal_top_k must be >= 0")
        if self.st_temporal_soft_threshold < 0.0 or self.st_temporal_soft_threshold >= 1.0:
            raise ValueError("st_temporal_soft_threshold must be in [0, 1)")
        if self.st_cross_temporal_soft_threshold < 0.0 or self.st_cross_temporal_soft_threshold >= 1.0:
            raise ValueError("st_cross_temporal_soft_threshold must be in [0, 1)")
        if self.st_message_scale <= 0.0:
            raise ValueError("st_message_scale must be > 0")
        if self.st_video_pooling not in ST_VIDEO_POOLING_MODES:
            raise ValueError(f"st_video_pooling must be one of {ST_VIDEO_POOLING_MODES}")
        if self.st_state_activation not in ST_STATE_ACTIVATIONS:
            raise ValueError(f"st_state_activation must be one of {ST_STATE_ACTIVATIONS}")
        if self.st_prediction_transform not in ST_PREDICTION_TRANSFORMS:
            raise ValueError(f"st_prediction_transform must be one of {ST_PREDICTION_TRANSFORMS}")
        if self.st_past_context_mode not in ST_PAST_CONTEXT_MODES:
            raise ValueError(f"st_past_context_mode must be one of {ST_PAST_CONTEXT_MODES}")
        if self.st_forecast_rollout_mode not in ST_FORECAST_ROLLOUT_MODES:
            raise ValueError(f"st_forecast_rollout_mode must be one of {ST_FORECAST_ROLLOUT_MODES}")
        if self.st_observed_refiner_mode not in ST_OBSERVED_REFINER_MODES:
            raise ValueError(f"st_observed_refiner_mode must be one of {ST_OBSERVED_REFINER_MODES}")
        if self.st_topk_training_mode not in ST_TOPK_TRAINING_MODES:
            raise ValueError(f"st_topk_training_mode must be one of {ST_TOPK_TRAINING_MODES}")
        if self.st_topk_warmup_epochs < 0:
            raise ValueError("st_topk_warmup_epochs must be >= 0")
        if self.st_topk_ramp_epochs < 1:
            raise ValueError("st_topk_ramp_epochs must be >= 1")
        if self.st_memory_prefix_length < 0:
            raise ValueError("st_memory_prefix_length must be >= 0")
        if not 0.0 < self.st_summary_short_alpha <= 1.0:
            raise ValueError("st_summary_short_alpha must be in (0, 1]")
        if not 0.0 < self.st_summary_long_alpha <= 1.0:
            raise ValueError("st_summary_long_alpha must be in (0, 1]")

        if self.use_concept_calibrator:
            self.calibrator = ConceptCalibrator(self.num_concepts, activation=self.concept_activation) 
        else:
            self.calibrator = nn.Identity()
        self.motif_z_layers = nn.ModuleList(
            [
                PerChannelTemporalBlock(
                    self.num_concepts,
                    width=self.motif_z_attention_width,
                    dropout=self.motif_z_attention_dropout,
                )
                for _ in range(self.motif_z_attention_layers)
            ]
        )
        self.motif_z_gate_logit = nn.Parameter(torch.tensor(float(motif_z_attention_gate_init)))

        self.past_context_read_gate_logits = nn.Parameter(
            torch.full((self.num_concepts,), self.st_past_context_gate_init)
        )
        self.past_memory_update_logits = nn.Parameter(
            torch.full((self.num_concepts,), self.st_past_context_gate_init)
        )
        self.past_summary_weights = nn.Parameter(torch.zeros(4, self.num_concepts))
        self.shared_graph_layers = self._make_observed_refiner_layers(self.st_graph_layers)
        self.window_graph_layers = self._make_observed_refiner_layers(self.st_task_graph_layers)
        self.forecast_graph_layers = self._make_observed_refiner_layers(self.st_task_graph_layers)
        self.activity_head = nn.Linear(self.num_concepts, self.num_activities)
        self.forecast_concept_projection = nn.Linear(self.num_concepts, self.num_concepts)
        self.shared_forecast_head = nn.Linear(self.num_concepts + self.num_activities, self.num_activities)
        self.forecast_heads = nn.ModuleDict(
            {str(horizon): self.shared_forecast_head for horizon in self.forecast_horizons}
        )
        self.future_concept_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(self.num_concepts, self.num_concepts) for horizon in self.future_concept_horizons}
        )
        self.video_head = (
            nn.Linear(self.num_concepts, self.num_activities)
            if self.st_video_pooling != "none"
            else None
        )

    def _make_graph_layers(self, count: int) -> nn.ModuleList:
        return nn.ModuleList(
            [
                SpatioTemporalConceptGraphLayer(
                    num_concepts=self.num_concepts,
                    edge_threshold=self.edge_threshold,
                    edge_gate_init=self.edge_gate_init,
                    residual_gate_init=self.st_residual_gate_init,
                    spatial_top_k=self.st_spatial_top_k,
                    spatial_soft_threshold=self.st_spatial_soft_threshold,
                    enable_spatial=self.st_enable_spatial,
                    temporal_top_k=self.st_temporal_top_k,
                    temporal_soft_threshold=self.st_temporal_soft_threshold,
                    enable_same_concept_temporal=self.st_enable_same_concept_temporal,
                    enable_cross_temporal=self.st_enable_cross_temporal,
                    cross_temporal_top_k=self.st_cross_temporal_top_k,
                    cross_temporal_soft_threshold=self.st_cross_temporal_soft_threshold,
                    message_scale=self.st_message_scale,
                    state_activation=self.st_state_activation,
                )
                for _ in range(int(count))
            ]
        )

    def _make_observed_refiner_layers(self, count: int) -> nn.ModuleList:
        if self.st_observed_refiner_mode == "graph":
            return self._make_graph_layers(count)
        return nn.ModuleList(
            [
                DenseTemporalConceptRolloutLayer(
                    self.num_concepts,
                    residual_gate_init=self.st_residual_gate_init,
                    enable_cross_temporal=self.st_enable_cross_temporal,
                    state_activation=self.st_state_activation,
                )
                for _ in range(int(count))
            ]
        )

    def _all_graph_layers(self) -> list[SpatioTemporalConceptGraphLayer]:
        layers = list(self.shared_graph_layers) + list(self.window_graph_layers) + list(self.forecast_graph_layers)
        controlled = getattr(self, "controlled_rollout_layers", [])
        layers.extend(controlled)
        return [layer for layer in layers if isinstance(layer, SpatioTemporalConceptGraphLayer)]

    def _set_runtime_topk(self, spatial: int, temporal: int, cross_temporal: int) -> None:
        self.st_spatial_top_k = int(spatial)
        self.st_temporal_top_k = int(temporal)
        self.st_cross_temporal_top_k = int(cross_temporal)
        for layer in self._all_graph_layers():
            layer.spatial_top_k = int(spatial)
            layer.temporal_top_k = int(temporal)
            layer.cross_temporal_top_k = int(cross_temporal)

    def _topk_state(self, phase: str, checkpoint_eligible: bool) -> Dict[str, object]:
        return {
            "phase": str(phase),
            "checkpoint_eligible": bool(checkpoint_eligible),
            "spatial_top_k": int(self.st_spatial_top_k),
            "temporal_top_k": int(self.st_temporal_top_k),
            "cross_temporal_top_k": int(self.st_cross_temporal_top_k),
        }

    def topk_schedule_completion_epoch(self) -> int:
        if self.st_topk_training_mode != "gradual":
            return 1
        return int(self.st_topk_warmup_epochs + self.st_topk_ramp_epochs)

    def topk_minimum_training_epoch(self) -> int:
        return self.topk_schedule_completion_epoch()

    def freeze_topk_topology(self) -> None:
        for layer in self._all_graph_layers():
            layer.freeze_topology()

    def enforce_frozen_topk_topology(self) -> None:
        for layer in self._all_graph_layers():
            layer.enforce_frozen_topology()

    def topk_topology_is_frozen(self) -> bool:
        layers = self._all_graph_layers()
        return bool(layers) and all(
            getattr(layer, "_frozen_spatial_prune_mask", None) is not None
            and layer._frozen_spatial_prune_mask.numel() > 0
            for layer in layers
        )

    def _gradual_topk_values(self, epoch: int) -> tuple[int, int, int, str, bool]:
        epoch = int(epoch)
        completion = self.topk_schedule_completion_epoch()
        if epoch <= self.st_topk_warmup_epochs:
            return 0, 0, 0, "warmup", False
        if epoch >= completion:
            return (
                self.st_target_spatial_top_k,
                self.st_target_temporal_top_k,
                self.st_target_cross_temporal_top_k,
                "target",
                True,
            )
        progress = (epoch - self.st_topk_warmup_epochs) / float(self.st_topk_ramp_epochs)

        def interpolate(maximum: int, target: int) -> int:
            if target <= 0:
                return 0
            return max(int(target), int(round(maximum + progress * (target - maximum))))

        return (
            interpolate(max(self.num_concepts - 1, 1), self.st_target_spatial_top_k),
            interpolate(self.num_concepts, self.st_target_temporal_top_k),
            interpolate(max(self.num_concepts - 1, 1), self.st_target_cross_temporal_top_k),
            "ramp",
            False,
        )

    def configure_topk_for_training(self, epoch: int) -> Dict[str, object]:
        mode = self.st_topk_training_mode
        if mode == "hard":
            values = (
                self.st_target_spatial_top_k,
                self.st_target_temporal_top_k,
                self.st_target_cross_temporal_top_k,
            )
            phase, eligible = "hard", True
        elif mode in {"none", "soft_train_hard_eval"}:
            values = (0, 0, 0)
            phase, eligible = ("all_edges", True)
        else:
            *values, phase, eligible = self._gradual_topk_values(epoch)
        self._set_runtime_topk(*values)
        return self._topk_state(phase, eligible)

    def configure_topk_for_evaluation(self, epoch: int) -> Dict[str, object]:
        mode = self.st_topk_training_mode
        if mode == "none":
            values, phase, eligible = (0, 0, 0), "all_edges", True
        elif mode == "gradual":
            *values, phase, eligible = self._gradual_topk_values(epoch)
        else:
            values = (
                self.st_target_spatial_top_k,
                self.st_target_temporal_top_k,
                self.st_target_cross_temporal_top_k,
            )
            phase = "hard_eval" if mode == "soft_train_hard_eval" else "hard"
            eligible = True
        self._set_runtime_topk(*values)
        return self._topk_state(phase, eligible)

    def configure_topk_for_final_evaluation(self) -> Dict[str, object]:
        if self.st_topk_training_mode == "none":
            self._set_runtime_topk(0, 0, 0)
            return self._topk_state("all_edges", True)
        self._set_runtime_topk(
            self.st_target_spatial_top_k,
            self.st_target_temporal_top_k,
            self.st_target_cross_temporal_top_k,
        )
        return self._topk_state("target", True)

    def _apply_motif_z_attention(
        self,
        calibrated: torch.Tensor,
        key_padding_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.motif_z_layers) == 0:
            return calibrated
        temporalized = calibrated
        causal_mask = build_causal_attn_mask(calibrated.size(1), calibrated.device)
        for layer in self.motif_z_layers:
            temporalized = layer(
                temporalized,
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask,
            )
            temporalized = temporalized * valid_mask.unsqueeze(-1)
        gate = torch.sigmoid(self.motif_z_gate_logit)
        return (calibrated + gate * (temporalized - calibrated)) * valid_mask.unsqueeze(-1)

    def _apply_past_context(
        self,
        calibrated: torch.Tensor,
        valid_mask: torch.Tensor,
        memory_prefix_concepts: torch.Tensor | None = None,
        memory_prefix_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        mode = getattr(self, "st_past_context_mode", "none")
        self._last_prefix_memory_context = None
        if mode == "none":
            self._last_past_context_delta_mean = 0.0
            return calibrated, {
                "past_context_mode": "none",
                "past_context_state": torch.zeros_like(calibrated),
                "past_context_delta": torch.zeros_like(calibrated),
            }
        if mode == "gated_memory":
            enriched, debug = self._apply_gated_concept_memory(calibrated, valid_mask)
        elif mode == "summary_stats":
            enriched, debug = self._apply_summary_stat_context(calibrated, valid_mask)
        elif mode in {"prefix_observed_memory", "prefix_recursive_memory"}:
            enriched, debug = self._apply_prefix_observed_memory(
                calibrated,
                valid_mask,
                memory_prefix_concepts=memory_prefix_concepts,
                memory_prefix_key_padding_mask=memory_prefix_key_padding_mask,
            )
        elif mode == "prefix_graph_memory_window":
            enriched, debug = self._apply_prefix_graph_memory_window(
                calibrated,
                valid_mask,
                memory_prefix_concepts=memory_prefix_concepts,
                memory_prefix_key_padding_mask=memory_prefix_key_padding_mask,
            )
        else:
            raise ValueError(f"Unknown st_past_context_mode: {mode}")
        delta = (enriched - calibrated) * valid_mask.unsqueeze(-1)
        self._last_past_context_delta_mean = float(delta.detach().abs().mean().item())
        debug["past_context_mode"] = mode
        debug["past_context_delta"] = delta
        return enriched, debug

    def _apply_gated_concept_memory(
        self,
        calibrated: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        batch_size, _, _ = calibrated.shape
        memory_state = torch.zeros((batch_size, self.num_concepts), dtype=calibrated.dtype, device=calibrated.device)
        memory = torch.zeros_like(calibrated)
        update_gate = torch.sigmoid(self.past_memory_update_logits).view(1, self.num_concepts)
        read_gate = torch.sigmoid(self.past_context_read_gate_logits).view(1, 1, self.num_concepts)
        for timestep in range(calibrated.size(1)):
            valid = valid_mask[:, timestep : timestep + 1]
            current = calibrated[:, timestep, :]
            updated = (update_gate * current) + ((1.0 - update_gate) * memory_state)
            memory_state = torch.where(valid > 0.0, updated, memory_state)
            memory[:, timestep, :] = memory_state * valid
        enriched = calibrated + read_gate * (memory - calibrated)
        enriched = enriched * valid_mask.unsqueeze(-1)
        return enriched, {
            "past_context_state": memory,
            "past_memory_update_gate": update_gate.expand_as(memory_state),
            "past_context_read_gate": read_gate.expand_as(calibrated),
        }

    def _memory_update_from_sequence(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(values.size(0))
        memory_state = torch.zeros((batch_size, self.num_concepts), dtype=values.dtype, device=values.device)
        memory = torch.zeros_like(values)
        update_gate = torch.sigmoid(self.past_memory_update_logits).view(1, self.num_concepts)
        for timestep in range(values.size(1)):
            valid = valid_mask[:, timestep : timestep + 1]
            current = values[:, timestep, :]
            updated = (update_gate * current) + ((1.0 - update_gate) * memory_state)
            memory_state = torch.where(valid > 0.0, updated, memory_state)
            memory[:, timestep, :] = memory_state * valid
        return memory_state, memory

    def _prefix_memory_from_inputs(
        self,
        reference: torch.Tensor,
        memory_prefix_concepts: torch.Tensor | None,
        memory_prefix_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory_prefix_concepts is None or int(memory_prefix_concepts.size(1)) == 0:
            prefix_calibrated = reference.new_zeros((reference.size(0), 0, self.num_concepts))
            prefix_valid_mask = reference.new_zeros((reference.size(0), 0))
        else:
            prefix_calibrated = self.calibrator(memory_prefix_concepts)
            if memory_prefix_key_padding_mask is None:
                prefix_valid_mask = prefix_calibrated.new_ones(prefix_calibrated.shape[:2])
            else:
                prefix_valid_mask = (~memory_prefix_key_padding_mask).float()
            prefix_calibrated = prefix_calibrated * prefix_valid_mask.unsqueeze(-1)
        prefix_state, prefix_memory = self._memory_update_from_sequence(prefix_calibrated, prefix_valid_mask)
        has_prefix = (prefix_valid_mask.sum(dim=1, keepdim=True) > 0.0).float()
        return prefix_state, prefix_memory, has_prefix

    def _apply_prefix_observed_memory(
        self,
        calibrated: torch.Tensor,
        valid_mask: torch.Tensor,
        memory_prefix_concepts: torch.Tensor | None,
        memory_prefix_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        prefix_state, prefix_memory, has_prefix = self._prefix_memory_from_inputs(
            calibrated,
            memory_prefix_concepts,
            memory_prefix_key_padding_mask,
        )
        read_gate = torch.sigmoid(self.past_context_read_gate_logits).view(1, 1, self.num_concepts)
        prefix_context = prefix_state[:, None, :].expand_as(calibrated)
        enriched = calibrated + read_gate * (prefix_context - calibrated)
        enriched = torch.where(has_prefix[:, None, :] > 0.0, enriched, calibrated)
        enriched = enriched * valid_mask.unsqueeze(-1)
        recursive_context = torch.where(
            has_prefix[:, None, :] > 0.0,
            prefix_context,
            calibrated,
        )
        self._last_prefix_memory_context = recursive_context
        return enriched, {
            "past_context_state": prefix_context * valid_mask.unsqueeze(-1),
            "past_prefix_memory": prefix_memory,
            "past_prefix_memory_state": prefix_state,
            "past_prefix_has_observed": has_prefix.squeeze(-1),
            "past_memory_update_gate": torch.sigmoid(self.past_memory_update_logits).view(1, self.num_concepts).expand_as(prefix_state),
            "past_context_read_gate": read_gate.expand_as(calibrated),
        }

    def _apply_prefix_graph_memory_window(
        self,
        calibrated: torch.Tensor,
        valid_mask: torch.Tensor,
        memory_prefix_concepts: torch.Tensor | None,
        memory_prefix_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        prefix_state, prefix_memory, has_prefix = self._prefix_memory_from_inputs(
            calibrated,
            memory_prefix_concepts,
            memory_prefix_key_padding_mask,
        )
        graph_memory_window = prefix_state[:, None, :]
        graph_memory_valid = has_prefix
        self._last_prefix_memory_context = None
        return calibrated, {
            "past_context_state": graph_memory_window.expand_as(calibrated) * valid_mask.unsqueeze(-1),
            "past_prefix_memory": prefix_memory,
            "past_prefix_memory_state": prefix_state,
            "past_prefix_has_observed": has_prefix.squeeze(-1),
            "past_memory_update_gate": torch.sigmoid(self.past_memory_update_logits).view(1, self.num_concepts).expand_as(prefix_state),
            "past_context_read_gate": torch.sigmoid(self.past_context_read_gate_logits).view(1, 1, self.num_concepts).expand_as(calibrated),
            "graph_memory_window": graph_memory_window,
            "graph_memory_valid": graph_memory_valid,
        }

    def _apply_recursive_forecast_memory(
        self,
        predicted_concepts: torch.Tensor,
        memory_context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update_gate = torch.sigmoid(self.past_memory_update_logits).view(1, 1, self.num_concepts)
        read_gate = torch.sigmoid(self.past_context_read_gate_logits).view(1, 1, self.num_concepts)
        updated_memory = (update_gate * predicted_concepts) + ((1.0 - update_gate) * memory_context)
        updated_memory = torch.where(valid_mask.unsqueeze(-1) > 0.0, updated_memory, memory_context)
        enriched = predicted_concepts + read_gate * (updated_memory - predicted_concepts)
        enriched = enriched * valid_mask.unsqueeze(-1)
        return enriched, updated_memory

    @staticmethod
    def _causal_ema(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        alpha = float(alpha)
        batch_size, _, channels = values.shape
        state = torch.zeros((batch_size, channels), dtype=values.dtype, device=values.device)
        output = torch.zeros_like(values)
        for timestep in range(values.size(1)):
            valid = valid_mask[:, timestep : timestep + 1]
            current = values[:, timestep, :]
            updated = (alpha * current) + ((1.0 - alpha) * state)
            state = torch.where(valid > 0.0, updated, state)
            output[:, timestep, :] = state * valid
        return output

    def _causal_time_since_seen(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        age = torch.zeros((values.size(0), values.size(2)), dtype=values.dtype, device=values.device)
        output = torch.zeros_like(values)
        threshold = float(self.st_summary_seen_threshold)
        scale = float(max(int(getattr(self, "history_length", 1)), 1))
        for timestep in range(values.size(1)):
            valid = valid_mask[:, timestep : timestep + 1]
            seen = (values[:, timestep, :] > threshold) & (valid > 0.0)
            aged = age + valid
            age = torch.where(seen, torch.zeros_like(age), aged)
            output[:, timestep, :] = torch.clamp(age / scale, min=0.0, max=1.0) * valid
        return output

    def _apply_summary_stat_context(
        self,
        calibrated: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        short_ema = self._causal_ema(calibrated, valid_mask, self.st_summary_short_alpha)
        long_ema = self._causal_ema(calibrated, valid_mask, self.st_summary_long_alpha)
        trend = short_ema - long_ema
        time_since_seen = self._causal_time_since_seen(calibrated, valid_mask)
        summary_features = torch.stack(
            [
                short_ema - calibrated,
                long_ema - calibrated,
                trend,
                time_since_seen,
            ],
            dim=2,
        )
        summary_delta = (summary_features * self.past_summary_weights.view(1, 1, 4, self.num_concepts)).sum(dim=2)
        read_gate = torch.sigmoid(self.past_context_read_gate_logits).view(1, 1, self.num_concepts)
        enriched = (calibrated + read_gate * summary_delta) * valid_mask.unsqueeze(-1)
        return enriched, {
            "past_context_state": summary_delta * valid_mask.unsqueeze(-1),
            "summary_short_ema": short_ema,
            "summary_long_ema": long_ema,
            "summary_trend": trend * valid_mask.unsqueeze(-1),
            "summary_time_since_seen": time_since_seen,
            "past_context_read_gate": read_gate.expand_as(calibrated),
        }

    def _prediction_transform(self, states: torch.Tensor) -> torch.Tensor:
        transform = getattr(self, "st_prediction_transform", "identity")
        if transform == "logit":
            eps = torch.finfo(states.dtype).eps
            bounded = states.clamp(min=eps, max=1.0 - eps)
            return torch.log(bounded) - torch.log1p(-bounded)
        if transform == "centered":
            return (2.0 * states) - 1.0
        return states

    @staticmethod
    def _zero_layer_info(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "spatial_messages": torch.zeros_like(x),
            "temporal_messages": torch.zeros_like(x),
            "update_gates": torch.zeros_like(x),
        }

    def _run_graph_layers(
        self,
        layers: nn.ModuleList,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
        branch: str = "shared",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        layer_info = self._zero_layer_info(x)
        refined = x
        for layer_index, layer in enumerate(layers):
            refined, layer_info = layer(
                refined,
                valid_mask,
                edge_interventions=self._edge_intervention_items(intervention, branch, layer_index),
            )
            refined = self._apply_persistent_intervention(
                refined,
                intervention=intervention,
                clamp_source=clamp_source,
            )
        return refined, layer_info

    @staticmethod
    def _intervention_mode(intervention: Dict[str, object] | None) -> str:
        if not isinstance(intervention, dict):
            return "input"
        return str(intervention.get("mode") or intervention.get("intervention_mode") or "input").lower()

    @staticmethod
    def _intervention_items(intervention: Dict[str, object]) -> list[Dict[str, object]]:
        items = intervention.get("items")
        if items is None:
            items = [intervention]
        return [dict(item) for item in items if isinstance(item, dict)]

    def _concept_intervention_items(self, intervention: Dict[str, object]) -> list[Dict[str, object]]:
        return [
            item for item in self._intervention_items(intervention)
            if str(item.get("item_type", item.get("type", "concept"))).lower() in {"concept", "node"}
            and ("concept_idx" in item or "concept_index" in item)
            and ("time_idx" in item or "timestep" in item or "rollout_step" in item)
        ]

    @staticmethod
    def _intervention_batch_selector(
        item: Dict[str, object],
        intervention: Dict[str, object],
    ) -> slice | int:
        if "batch_start" in item or "batch_end" in item:
            start = 0 if item.get("batch_start") is None else int(item["batch_start"])
            end = None if item.get("batch_end") is None else int(item["batch_end"])
            return slice(start, end)
        batch_idx = item.get("batch_idx", intervention.get("batch_idx"))
        return slice(None) if batch_idx is None else int(batch_idx)

    def _edge_intervention_items(
        self,
        intervention: Dict[str, object] | None,
        branch: str,
        layer_index: int,
    ) -> list[Dict[str, object]]:
        if not isinstance(intervention, dict):
            return []
        rows: list[Dict[str, object]] = []
        for item in self._intervention_items(intervention):
            if str(item.get("item_type", item.get("type", ""))).lower() != "edge":
                continue
            item_branch = str(item.get("branch", "all")).lower()
            if item_branch not in {"all", str(branch).lower()}:
                continue
            selected_layer = item.get("layer_index")
            if selected_layer is not None and int(selected_layer) != int(layer_index):
                continue
            rows.append(item)
        return rows

    def _shift_concept_intervention_times(
        self,
        intervention: Dict[str, object] | None,
        offset: int,
    ) -> Dict[str, object] | None:
        if not isinstance(intervention, dict) or int(offset) == 0:
            return intervention

        def shift_item(item: Dict[str, object]) -> Dict[str, object]:
            shifted = dict(item)
            if str(shifted.get("item_type", shifted.get("type", "concept"))).lower() in {"concept", "node"}:
                if "time_idx" in shifted:
                    shifted["time_idx"] = int(shifted["time_idx"]) + int(offset)
                if "timestep" in shifted:
                    shifted["timestep"] = int(shifted["timestep"]) + int(offset)
            return shifted

        shifted_intervention = dict(intervention)
        items = intervention.get("items")
        if isinstance(items, list):
            shifted_intervention["items"] = [
                shift_item(item) if isinstance(item, dict) else item
                for item in items
            ]
        else:
            shifted_intervention = shift_item(shifted_intervention)
        return shifted_intervention

    def _apply_persistent_intervention(
        self,
        states: torch.Tensor,
        intervention: Dict[str, object] | None,
        clamp_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._intervention_mode(intervention) not in {"persistent", "clamp", "state"}:
            return states
        if not isinstance(intervention, dict):
            return states
        intervened = states.clone()
        source = clamp_source if clamp_source is not None else states
        for item in self._concept_intervention_items(intervention):
            if "rollout_step" in item:
                continue
            concept_idx = int(item.get("concept_idx", item.get("concept_index")))
            time_idx = int(item.get("time_idx", item.get("timestep")))
            has_value = "value" in item and item["value"] is not None
            has_delta = "delta" in item and item["delta"] is not None
            if has_value == has_delta:
                raise ValueError("Exactly one of intervention value or delta must be provided.")
            if concept_idx < 0 or concept_idx >= self.num_concepts:
                raise IndexError(f"concept_idx out of range: {concept_idx}")
            if time_idx < 0 or time_idx >= states.size(1):
                raise IndexError(f"time_idx out of range: {time_idx}")
            batch_selector = self._intervention_batch_selector(item, intervention)
            if has_value:
                intervened[batch_selector, time_idx, concept_idx] = source[batch_selector, time_idx, concept_idx]
            else:
                intervened[batch_selector, time_idx, concept_idx] = (
                    intervened[batch_selector, time_idx, concept_idx] + float(item["delta"])
                )
        return intervened

    def _apply_input_intervention(
        self,
        concepts: torch.Tensor,
        intervention: Dict[str, object],
    ) -> torch.Tensor:
        intervened = concepts.clone()
        for item in self._concept_intervention_items(intervention):
            if "rollout_step" in item:
                continue
            concept_idx = int(item.get("concept_idx", item.get("concept_index")))
            time_idx = int(item.get("time_idx", item.get("timestep")))
            has_value = "value" in item and item["value"] is not None
            has_delta = "delta" in item and item["delta"] is not None
            if has_value == has_delta:
                raise ValueError("Exactly one of intervention value or delta must be provided.")
            if concept_idx < 0 or concept_idx >= self.num_concepts:
                raise IndexError(f"concept_idx out of range: {concept_idx}")
            if time_idx < 0 or time_idx >= concepts.size(1):
                raise IndexError(f"time_idx out of range: {time_idx}")
            batch_selector = self._intervention_batch_selector(item, intervention)
            if has_value:
                intervened[batch_selector, time_idx, concept_idx] = float(item["value"])
            else:
                intervened[batch_selector, time_idx, concept_idx] = (
                    intervened[batch_selector, time_idx, concept_idx] + float(item["delta"])
                )
        return intervened

    def _apply_rollout_concept_intervention(
        self,
        states: torch.Tensor,
        intervention: Dict[str, object] | None,
        rollout_step: int,
    ) -> torch.Tensor:
        if not isinstance(intervention, dict):
            return states
        updated = states
        for item in self._concept_intervention_items(intervention):
            if item.get("rollout_step") is None or int(item["rollout_step"]) != int(rollout_step):
                continue
            concept_idx = int(item.get("concept_idx", item.get("concept_index")))
            if concept_idx < 0 or concept_idx >= self.num_concepts:
                raise IndexError(f"concept_idx out of range: {concept_idx}")
            has_value = item.get("value") is not None
            has_delta = item.get("delta") is not None
            if has_value == has_delta:
                raise ValueError("Exactly one of intervention value or delta must be provided.")
            batch_selector = self._intervention_batch_selector(item, intervention)
            if updated is states:
                updated = states.clone()
            if has_value:
                value = float(item["value"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError("Future rollout concept values must be in [0, 1].")
                updated[batch_selector, -1, concept_idx] = value
            else:
                updated[batch_selector, -1, concept_idx] = torch.clamp(
                    updated[batch_selector, -1, concept_idx] + float(item["delta"]),
                    min=0.0,
                    max=1.0,
                )
        return updated

    def _pool_video(self, states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor | None:
        if self.video_head is None:
            return None
        if self.st_video_pooling == "mean":
            summed = (states * valid_mask.unsqueeze(-1)).sum(dim=1)
            count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return summed / count
        masked = states.masked_fill(valid_mask.unsqueeze(-1) <= 0.0, -1e9)
        pooled = torch.logsumexp(masked, dim=1)
        count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return pooled - torch.log(count)

    def edge_regularization(self) -> torch.Tensor:
        reg = self.calibrator.scale.sum() * 0.0
        for layer in self._all_graph_layers():
            reg = reg + layer.edge_regularization()
        return reg

    @torch.no_grad()
    def set_graph_priors(
        self,
        same_prior: torch.Tensor | np.ndarray,
        lag_prior: torch.Tensor | np.ndarray | None = None,
        gate_logit: float = 1.0,
    ) -> None:
        for layer in self._all_graph_layers():
            layer.set_spatial_prior(same_prior, gate_logit=gate_logit)
        if self.st_enable_cross_temporal and lag_prior is not None:
            lag_prior_tensor = torch.as_tensor(lag_prior)
            if lag_prior_tensor.ndim == 3 and lag_prior_tensor.shape[0] > 0:
                adjacent_prior = lag_prior_tensor[0]
            else:
                adjacent_prior = lag_prior_tensor
            for layer in self._all_graph_layers():
                layer.set_cross_temporal_prior(adjacent_prior, gate_logit=gate_logit)

    @torch.no_grad()
    def apply_topk_pruning(self, same_top_k: int = 0, lag_top_k: int = 0) -> Dict[str, float]:
        if same_top_k > 0:
            for layer in self._all_graph_layers():
                layer.spatial_top_k = int(same_top_k)
        if lag_top_k > 0:
            for layer in self._all_graph_layers():
                layer.cross_temporal_top_k = int(lag_top_k)
        return self.graph_metrics()

    def graph_metrics(self) -> Dict[str, float]:
        aggregate = {
            "active_same_time_edges": 0.0,
            "active_lagged_edges": 0.0,
            "active_same_concept_temporal_edges": 0.0,
            "active_cross_temporal_edges": 0.0,
            "same_time_possible_edges": 0.0,
            "lagged_possible_edges": 0.0,
            "same_concept_temporal_possible_edges": 0.0,
            "cross_temporal_possible_edges": 0.0,
            "same_time_kept_edges": 0.0,
            "lagged_kept_edges": 0.0,
            "same_concept_temporal_kept_edges": 0.0,
            "cross_temporal_kept_edges": 0.0,
            "same_gate_sum": 0.0,
            "same_gate_count": 0.0,
            "lag_gate_sum": 0.0,
            "lag_gate_count": 0.0,
            "same_concept_temporal_gate_sum": 0.0,
            "same_concept_temporal_gate_count": 0.0,
        }
        branch_groups = {
            "shared": self.shared_graph_layers,
            "window": self.window_graph_layers,
            "forecast": self.forecast_graph_layers,
        }
        branch_counts: Dict[str, Dict[str, float]] = {}
        for branch_name, layers in branch_groups.items():
            branch_counts[branch_name] = {"active_same_time_edges": 0.0, "active_lagged_edges": 0.0}
            for layer in layers:
                if not isinstance(layer, SpatioTemporalConceptGraphLayer):
                    continue
                metrics = layer.graph_metrics()
                for key in aggregate:
                    aggregate[key] += float(metrics[key])
                branch_counts[branch_name]["active_same_time_edges"] += float(metrics["active_same_time_edges"])
                branch_counts[branch_name]["active_lagged_edges"] += float(metrics["active_lagged_edges"])

        same_possible = max(aggregate["same_time_possible_edges"], 1.0)
        lag_possible = max(aggregate["lagged_possible_edges"], 1.0)
        same_gate_count = max(aggregate["same_gate_count"], 1.0)
        lag_gate_count = max(aggregate["lag_gate_count"], 1.0)
        output = {
            "active_same_time_edges": aggregate["active_same_time_edges"],
            "active_lagged_edges": aggregate["active_lagged_edges"],
            "same_time_density": aggregate["active_same_time_edges"] / same_possible,
            "lagged_density": aggregate["active_lagged_edges"] / lag_possible,
            "mean_same_gate": aggregate["same_gate_sum"] / same_gate_count,
            "mean_lag_gate": aggregate["lag_gate_sum"] / lag_gate_count,
            "same_time_possible_edges": aggregate["same_time_possible_edges"],
            "lagged_possible_edges": aggregate["lagged_possible_edges"],
            "same_time_kept_edges": aggregate["same_time_kept_edges"],
            "lagged_kept_edges": aggregate["lagged_kept_edges"],
            "same_concept_temporal_kept_edges": aggregate["same_concept_temporal_kept_edges"],
            "cross_temporal_kept_edges": aggregate["cross_temporal_kept_edges"],
            "active_same_concept_temporal_edges": aggregate["active_same_concept_temporal_edges"],
            "active_cross_temporal_edges": aggregate["active_cross_temporal_edges"],
            "pruned_same_time_edges": aggregate["same_time_possible_edges"] - aggregate["same_time_kept_edges"],
            "pruned_lagged_edges": aggregate["lagged_possible_edges"] - aggregate["lagged_kept_edges"],
            "pruned_same_concept_temporal_edges": aggregate["same_concept_temporal_possible_edges"]
            - aggregate["same_concept_temporal_kept_edges"],
            "pruned_cross_temporal_edges": aggregate["cross_temporal_possible_edges"]
            - aggregate["cross_temporal_kept_edges"],
            "motif_z_attention_gate": float(torch.sigmoid(self.motif_z_gate_logit).detach().item()),
            "motif_z_attention_layers": float(len(self.motif_z_layers)),
            "st_graph_layers": float(self.st_graph_layers),
            "st_task_graph_layers": float(self.st_task_graph_layers),
            "st_enable_cross_temporal": float(self.st_enable_cross_temporal),
            "st_enable_same_concept_temporal": float(getattr(self, "st_enable_same_concept_temporal", True)),
            "st_spatial_top_k": float(getattr(self, "st_spatial_top_k", 0)),
            "st_temporal_top_k": float(getattr(self, "st_temporal_top_k", 0)),
            "st_cross_temporal_top_k": float(getattr(self, "st_cross_temporal_top_k", 0)),
            "st_target_spatial_top_k": float(getattr(self, "st_target_spatial_top_k", 0)),
            "st_target_temporal_top_k": float(getattr(self, "st_target_temporal_top_k", 0)),
            "st_target_cross_temporal_top_k": float(
                getattr(self, "st_target_cross_temporal_top_k", 0)
            ),
            "st_message_scale": float(getattr(self, "st_message_scale", 1.0)),
            "st_state_activation_bounded_logit": float(getattr(self, "st_state_activation", "identity") == "bounded_logit"),
            "st_prediction_transform_logit": float(getattr(self, "st_prediction_transform", "identity") == "logit"),
            "st_past_context_enabled": float(getattr(self, "st_past_context_mode", "none") != "none"),
            "st_past_context_mode_gated_memory": float(
                getattr(self, "st_past_context_mode", "none") == "gated_memory"
            ),
            "st_past_context_mode_summary_stats": float(
                getattr(self, "st_past_context_mode", "none") == "summary_stats"
            ),
            "st_past_context_mode_prefix_observed_memory": float(
                getattr(self, "st_past_context_mode", "none") == "prefix_observed_memory"
            ),
            "st_past_context_mode_prefix_recursive_memory": float(
                getattr(self, "st_past_context_mode", "none") == "prefix_recursive_memory"
            ),
            "st_past_context_mode_prefix_graph_memory_window": float(
                getattr(self, "st_past_context_mode", "none") == "prefix_graph_memory_window"
            ),
            "st_memory_prefix_length": float(getattr(self, "st_memory_prefix_length", 0)),
            "st_past_context_delta_mean_abs": float(getattr(self, "_last_past_context_delta_mean", 0.0)),
            "st_past_context_read_gate_mean": float(
                torch.sigmoid(self.past_context_read_gate_logits).detach().mean().item()
            )
            if hasattr(self, "past_context_read_gate_logits")
            else 0.0,
        }
        for branch_name, counts in branch_counts.items():
            output[f"st_{branch_name}_active_same_time_edges"] = counts["active_same_time_edges"]
            output[f"st_{branch_name}_active_lagged_edges"] = counts["active_lagged_edges"]
        return output

    def forward(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        memory_prefix_concepts: torch.Tensor | None = None,
        memory_prefix_key_padding_mask: torch.Tensor | None = None,
        previous_activity_labels: torch.Tensor | None = None,
        teacher_forcing: bool = False,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, object]:
        valid_mask = (~key_padding_mask).float()
        if intervention is not None:
            concepts = self._apply_input_intervention(concepts, intervention)
        calibrated = self.calibrator(concepts) * valid_mask.unsqueeze(-1)
        calibrated = self._apply_persistent_intervention(
            calibrated,
            intervention=intervention,
            clamp_source=calibrated,
        )
        past_context_concepts, past_context_debug = self._apply_past_context(
            calibrated,
            valid_mask,
            memory_prefix_concepts=memory_prefix_concepts,
            memory_prefix_key_padding_mask=memory_prefix_key_padding_mask,
        )
        past_context_concepts = self._apply_persistent_intervention(
            past_context_concepts,
            intervention=intervention,
            clamp_source=calibrated,
        )
        graph_memory_mode = getattr(self, "st_past_context_mode", "none") == "prefix_graph_memory_window"
        graph_input = past_context_concepts
        graph_valid_mask = valid_mask
        graph_key_padding_mask = key_padding_mask
        graph_clamp_source = calibrated
        graph_intervention = intervention
        if graph_memory_mode:
            memory_window = past_context_debug["graph_memory_window"]
            memory_valid = past_context_debug["graph_memory_valid"]
            graph_input = torch.cat([memory_window, graph_input], dim=1)
            graph_valid_mask = torch.cat([memory_valid, valid_mask], dim=1)
            graph_key_padding_mask = graph_valid_mask <= 0.0
            graph_clamp_source = torch.cat([memory_window, calibrated], dim=1)
            graph_intervention = self._shift_concept_intervention_times(intervention, 1)

        temporalized_full = self._apply_motif_z_attention(
            calibrated=graph_input,
            key_padding_mask=graph_key_padding_mask,
            valid_mask=graph_valid_mask,
        )
        temporalized_full = self._apply_persistent_intervention(
            temporalized_full,
            intervention=graph_intervention,
            clamp_source=graph_clamp_source,
        )

        shared_refined_full, shared_info_full = self._run_graph_layers(
            self.shared_graph_layers,
            temporalized_full,
            graph_valid_mask,
            intervention=graph_intervention,
            clamp_source=graph_clamp_source,
            branch="shared",
        )
        window_refined_full, window_info_full = self._run_graph_layers(
            self.window_graph_layers,
            shared_refined_full,
            graph_valid_mask,
            intervention=graph_intervention,
            clamp_source=graph_clamp_source,
            branch="window",
        )
        forecast_refined_full, forecast_info_full = self._run_graph_layers(
            self.forecast_graph_layers,
            shared_refined_full,
            graph_valid_mask,
            intervention=graph_intervention,
            clamp_source=graph_clamp_source,
            branch="forecast",
        )
        graph_memory_debug: Dict[str, object] = {}
        if graph_memory_mode:
            graph_memory_debug = {
                "graph_memory_temporalized": temporalized_full[:, :1, :],
                "graph_memory_refined": shared_refined_full[:, :1, :],
                "graph_memory_window_refined": window_refined_full[:, :1, :],
                "graph_memory_forecast_refined": forecast_refined_full[:, :1, :],
            }
            temporalized = temporalized_full[:, 1:, :]
            shared_refined = shared_refined_full[:, 1:, :]
            window_refined = window_refined_full[:, 1:, :]
            forecast_refined = forecast_refined_full[:, 1:, :]
            shared_info = {key: value[:, 1:, :] for key, value in shared_info_full.items()}
            window_info = {key: value[:, 1:, :] for key, value in window_info_full.items()}
            forecast_info = {key: value[:, 1:, :] for key, value in forecast_info_full.items()}
        else:
            temporalized = temporalized_full
            shared_refined = shared_refined_full
            window_refined = window_refined_full
            forecast_refined = forecast_refined_full
            shared_info = shared_info_full
            window_info = window_info_full
            forecast_info = forecast_info_full
        (
            window_refined,
            forecast_refined,
            observed_activity_logit_residual,
            observed_activity_feedback_debug,
        ) = (
            self._apply_observed_activity_feedback(
                window_refined,
                forecast_refined,
                valid_mask,
                intervention=intervention,
                clamp_source=calibrated,
            )
        )
        window_prediction = self._prediction_transform(window_refined)
        forecast_prediction = self._prediction_transform(forecast_refined)

        activity_logits = self.activity_head(window_prediction) + observed_activity_logit_residual
        forecast_logits_by_horizon, autoregressive_logits_by_step = self._autoregressive_forecast_logits(
            forecast_refined=forecast_refined,
            activity_logits=activity_logits,
            previous_activity_labels=previous_activity_labels,
            teacher_forcing=teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            intervention=intervention,
            clamp_source=calibrated,
        )
        future_concepts_by_horizon = {
            horizon: self.future_concept_heads[str(horizon)](forecast_prediction)
            for horizon in self.future_concept_horizons
        }
        video_repr = self._pool_video(shared_refined, valid_mask)
        video_logits = self.video_head(video_repr) if self.video_head is not None and video_repr is not None else None

        return {
            "calibrated_concepts": calibrated,
            "past_context_concepts": past_context_concepts,
            **past_context_debug,
            **graph_memory_debug,
            **observed_activity_feedback_debug,
            "temporalized_concepts": temporalized,
            "shared_refined_concepts": shared_refined,
            "window_refined_concepts": window_refined,
            "forecast_refined_concepts": forecast_refined,
            "concept_states": shared_refined,
            "gate_values": shared_info["update_gates"],
            "same_messages": window_info["spatial_messages"],
            "lag_messages": forecast_info["temporal_messages"],
            "activity_repr": window_prediction,
            "prediction_repr": window_prediction,
            "representation_mode": self.representation_mode,
            "st_state_activation": getattr(self, "st_state_activation", "identity"),
            "st_prediction_transform": getattr(self, "st_prediction_transform", "identity"),
            "activity_context_repr": window_prediction,
            "forecast_repr": forecast_prediction,
            "activity_logits": activity_logits,
            "forecast_logits_by_horizon": forecast_logits_by_horizon,
            "autoregressive_logits_by_step": autoregressive_logits_by_step,
            "future_concepts_by_horizon": future_concepts_by_horizon,
            "video_repr": video_repr,
            "video_logits": video_logits,
        }

    def _apply_observed_activity_feedback(
        self,
        window_refined: torch.Tensor,
        forecast_refined: torch.Tensor,
        valid_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        del valid_mask, intervention, clamp_source
        logit_residual = window_refined.new_zeros(
            (*window_refined.shape[:-1], self.num_activities)
        )
        return window_refined, forecast_refined, logit_residual, {}

    def _autoregressive_forecast_logits(
        self,
        forecast_refined: torch.Tensor,
        activity_logits: torch.Tensor,
        previous_activity_labels: torch.Tensor | None,
        teacher_forcing: bool,
        teacher_forcing_ratio: float,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        del intervention, clamp_source
        max_horizon = max(self.forecast_horizons)
        if previous_activity_labels is not None:
            if previous_activity_labels.ndim == 1:
                previous_activity_labels = previous_activity_labels[:, None, None].expand(
                    -1,
                    forecast_refined.size(1),
                    1,
                )
            elif previous_activity_labels.ndim == 2:
                previous_activity_labels = previous_activity_labels[:, None, :].expand(
                    -1,
                    forecast_refined.size(1),
                    -1,
                )
        ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        prev_activity_context = torch.softmax(activity_logits.detach(), dim=-1)
        if teacher_forcing and previous_activity_labels is not None:
            ground_truth_activity = previous_activity_labels[..., 0].long()
            ground_truth_context = F.one_hot(ground_truth_activity, num_classes=self.num_activities).float()
            if ratio >= 1.0:
                prev_activity_context = ground_truth_context
            elif ratio > 0.0:
                use_ground_truth = torch.rand(
                    ground_truth_activity.shape,
                    device=ground_truth_activity.device,
                ) < ratio
                prev_activity_context = torch.where(
                    use_ground_truth.unsqueeze(-1),
                    ground_truth_context,
                    prev_activity_context,
                )

        logits_by_step: Dict[int, torch.Tensor] = {}
        forecast_head_input_dim = int(self.shared_forecast_head.in_features)
        use_projected_concepts = forecast_head_input_dim == self.num_concepts + self.num_activities
        forecast_prediction = self._prediction_transform(forecast_refined)
        concept_features = (
            self.forecast_concept_projection(forecast_prediction)
            if use_projected_concepts
            else forecast_prediction
        )
        for step in range(1, max_horizon + 1):
            if use_projected_concepts:
                forecast_input = torch.cat([concept_features, prev_activity_context], dim=-1)
            elif hasattr(self, "previous_activity_projection"):
                previous_activity_features = self.previous_activity_projection(prev_activity_context)
                forecast_input = torch.cat([forecast_prediction, previous_activity_features], dim=-1)
            else:
                raise RuntimeError(
                    "Forecast head input dimension does not match the new K + activity context layout "
                    "and no legacy previous_activity_projection is available."
                )
            step_logits = self.shared_forecast_head(forecast_input)
            logits_by_step[step] = step_logits
            if teacher_forcing and previous_activity_labels is not None:
                if previous_activity_labels.ndim == 3:
                    prev_index = min(step, previous_activity_labels.size(-1) - 1)
                    ground_truth_activity = previous_activity_labels[..., prev_index].long()
                else:
                    ground_truth_activity = previous_activity_labels.long()
                predicted_context = torch.softmax(step_logits.detach(), dim=-1)
                ground_truth_context = F.one_hot(ground_truth_activity, num_classes=self.num_activities).float()
                if ratio >= 1.0:
                    prev_activity_context = ground_truth_context
                elif ratio <= 0.0:
                    prev_activity_context = predicted_context
                else:
                    use_ground_truth = torch.rand(
                        ground_truth_activity.shape,
                        device=ground_truth_activity.device,
                    ) < ratio
                    prev_activity_context = torch.where(
                        use_ground_truth.unsqueeze(-1),
                        ground_truth_context,
                        predicted_context,
                    )
            else:
                prev_activity_context = torch.softmax(step_logits.detach(), dim=-1)
        forecast_logits_by_horizon = {
            horizon: logits_by_step[horizon]
            for horizon in self.forecast_horizons
        }
        return forecast_logits_by_horizon, logits_by_step

    @torch.no_grad()
    def intervene(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        time_idx: int,
        concept_idx: int,
        batch_idx: int | None = None,
        value: float | None = None,
        delta: float | None = None,
    ) -> Dict[str, object]:
        intervention = {
            "time_idx": int(time_idx),
            "concept_idx": int(concept_idx),
            "batch_idx": batch_idx,
            "value": value,
            "delta": delta,
        }
        baseline = self.forward(concepts, key_padding_mask)
        intervened = self.forward(concepts, key_padding_mask, intervention=intervention)
        forecast_delta_by_horizon = {
            horizon: intervened["forecast_logits_by_horizon"][horizon] - baseline["forecast_logits_by_horizon"][horizon]
            for horizon in self.forecast_horizons
        }
        return {
            "baseline": baseline,
            "intervened": intervened,
            "refined_before": baseline["concept_states"],
            "refined_after": intervened["concept_states"],
            "window_refined_before": baseline["window_refined_concepts"],
            "window_refined_after": intervened["window_refined_concepts"],
            "forecast_refined_before": baseline["forecast_refined_concepts"],
            "forecast_refined_after": intervened["forecast_refined_concepts"],
            "activity_logits_before": baseline["activity_logits"],
            "activity_logits_after": intervened["activity_logits"],
            "activity_delta": intervened["activity_logits"] - baseline["activity_logits"],
            "forecast_logits_before_by_horizon": baseline["forecast_logits_by_horizon"],
            "forecast_logits_after_by_horizon": intervened["forecast_logits_by_horizon"],
            "forecast_delta_by_horizon": forecast_delta_by_horizon,
        }

    @torch.no_grad()
    def intervene_concept(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        concept_index: int,
        timestep: int,
        value: float,
    ) -> Dict[str, object]:
        return self.intervene(
            concepts=concepts,
            key_padding_mask=key_padding_mask,
            time_idx=timestep,
            concept_idx=concept_index,
            value=value,
        )


class ActivityAutoregressiveGraphCBM(SpatioTemporalConceptGraphBase):
    """Graph-refined concepts with autoregressive activity-label forecasting."""


class GraphCBM(SpatioTemporalConceptGraphBase):
    """Graph-CBM: calibrate concepts, refine observed windows, then roll out H1--H3.

    A shared sparse spatio-temporal graph refines the observed concept states.
    Task-specific graph layers recursively predict future concept states, and one
    shared activity head maps both observed and forecast concept states to labels.
    """

    def __init__(
        self,
        *args,
        st_activity_feedback_mode: str = "none",
        st_activity_feedback_top_k: int = 10,
        st_activity_feedback_gate_init: float = -2.0,
        st_activity_feedback_history_steps: int = 0,
        st_activity_feedback_history_gate_init: float = 0.0,
        st_activity_feedback_hidden_dim: int = 64,
        st_activity_feedback_ridge_lambda: float = 0.1,
        st_activity_feedback_probability_eps: float = 1e-4,
        st_activity_feedback_prototype_smoothing: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.st_activity_feedback_mode = str(st_activity_feedback_mode)
        self.st_activity_feedback_top_k = int(st_activity_feedback_top_k)
        self.st_target_activity_feedback_top_k = int(st_activity_feedback_top_k)
        self.st_activity_feedback_gate_init = float(st_activity_feedback_gate_init)
        self.st_activity_feedback_history_steps = int(st_activity_feedback_history_steps)
        self.st_activity_feedback_history_gate_init = float(st_activity_feedback_history_gate_init)
        self.st_activity_feedback_hidden_dim = int(st_activity_feedback_hidden_dim)
        self.st_activity_feedback_ridge_lambda = float(st_activity_feedback_ridge_lambda)
        self.st_activity_feedback_probability_eps = float(st_activity_feedback_probability_eps)
        self.st_activity_feedback_prototype_smoothing = float(
            st_activity_feedback_prototype_smoothing
        )
        if self.st_activity_feedback_mode not in ST_ACTIVITY_FEEDBACK_MODES:
            raise ValueError(
                f"st_activity_feedback_mode must be one of {ST_ACTIVITY_FEEDBACK_MODES}"
            )
        if self.st_activity_feedback_top_k < 0:
            raise ValueError("st_activity_feedback_top_k must be >= 0")
        if self.st_activity_feedback_history_steps < 0:
            raise ValueError("st_activity_feedback_history_steps must be >= 0")
        if self.st_activity_feedback_hidden_dim <= 0:
            raise ValueError("st_activity_feedback_hidden_dim must be > 0")
        if self.st_activity_feedback_ridge_lambda <= 0.0:
            raise ValueError("st_activity_feedback_ridge_lambda must be > 0")
        if not 0.0 < self.st_activity_feedback_probability_eps < 0.5:
            raise ValueError("st_activity_feedback_probability_eps must be in (0, 0.5)")
        if self.st_activity_feedback_prototype_smoothing < 0.0:
            raise ValueError("st_activity_feedback_prototype_smoothing must be >= 0")

        if self._activity_feedback_uses_sparse_matrix():
            
            self.activity_feedback_weight = nn.Parameter(
                torch.empty(self.num_activities, self.num_concepts)
            )
            nn.init.normal_(self.activity_feedback_weight, mean=0.0, std=0.01)
            self.activity_feedback_gate_logits = nn.Parameter(
                torch.full(
                    (self.num_activities, self.num_concepts),
                    self.st_activity_feedback_gate_init,
                )
            )
        if self.st_activity_feedback_mode == "mlp_label_to_concept":
            self.activity_feedback_decoder = nn.Sequential(
                nn.Linear(
                    self.num_activities,
                    self.st_activity_feedback_hidden_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Linear(
                    self.st_activity_feedback_hidden_dim,
                    self.num_concepts,
                    bias=False,
                ),
            )
        if self._activity_feedback_updates_logits():
            self.activity_feedback_logit_weight = nn.Parameter(
                torch.empty(self.num_activities, self.num_activities)
            )
            nn.init.normal_(self.activity_feedback_logit_weight, mean=0.0, std=0.01)
        if self.st_activity_feedback_mode == "prototype_label_to_concept":
            self.register_buffer(
                "activity_feedback_prototype_source",
                torch.empty(0, self.num_concepts),
            )
            self.register_buffer(
                "activity_feedback_prototype_target",
                torch.empty(0, self.num_concepts),
            )
            self.register_buffer(
                "activity_feedback_prototype_counts",
                torch.empty(0),
            )
        if self._activity_feedback_enabled() and not self._activity_feedback_uses_sparse_matrix():
            self.activity_feedback_strength_logit = nn.Parameter(
                torch.tensor(self.st_activity_feedback_gate_init, dtype=torch.float32)
            )
        elif self.st_activity_feedback_mode == "hybrid_sparse_and_label_autoregressive":
            self.activity_feedback_strength_logit = nn.Parameter(
                torch.tensor(self.st_activity_feedback_gate_init, dtype=torch.float32)
            )
        if self._activity_feedback_enabled() and self.st_activity_feedback_history_steps > 0:
            self.activity_feedback_history_gate_logits = nn.Parameter(
                torch.full(
                    (self.st_activity_feedback_history_steps,),
                    self.st_activity_feedback_history_gate_init,
                )
            )
        del self.forecast_concept_projection
        del self.shared_forecast_head
        self.forecast_heads = nn.ModuleDict(
            {str(horizon): self.activity_head for horizon in self.forecast_horizons}
        )
        if self.st_forecast_rollout_mode == "controlled_graph":
            self.controlled_rollout_layers = self._make_graph_layers(self.st_task_graph_layers)
        elif self.st_forecast_rollout_mode == "controlled_dense":
            if self.st_controlled_dense_reuse_forecast_layer:
                if self.st_observed_refiner_mode != "dense":
                    raise ValueError(
                        "st_controlled_dense_reuse_forecast_layer requires "
                        "st_observed_refiner_mode='dense'."
                    )
                # TRACE reuses its forecast graph transition at every horizon.
                # Reusing the matched dense forecast transition provides the
                # same sharing pattern without adding a second transition bank.
                self.controlled_rollout_layers = self.forecast_graph_layers
            else:
                self.controlled_rollout_layers = nn.ModuleList(
                    [
                        DenseTemporalConceptRolloutLayer(
                            self.num_concepts,
                            residual_gate_init=self.st_residual_gate_init,
                            enable_cross_temporal=self.st_enable_cross_temporal,
                            state_activation=self.st_state_activation,
                        )
                        for _ in range(self.st_task_graph_layers)
                    ]
                )
        else:
            self.controlled_rollout_layers = nn.ModuleList()

    def _activity_feedback_enabled(self) -> bool:
        return getattr(self, "st_activity_feedback_mode", "none") != "none"

    def _activity_feedback_uses_sparse_matrix(self) -> bool:
        return getattr(self, "st_activity_feedback_mode", "none") in {
            "sparse_label_to_concept",
            "hybrid_sparse_and_label_autoregressive",
        }

    def _activity_feedback_updates_concepts(self) -> bool:
        return getattr(self, "st_activity_feedback_mode", "none") in {
            "sparse_label_to_concept",
            "classifier_weight_to_concept",
            "ridge_inverse_to_concept",
            "prototype_label_to_concept",
            "mlp_label_to_concept",
            "hybrid_sparse_and_label_autoregressive",
        }

    def _activity_feedback_updates_logits(self) -> bool:
        return getattr(self, "st_activity_feedback_mode", "none") in {
            "label_autoregressive",
            "hybrid_sparse_and_label_autoregressive",
        }

    def _activity_feedback_history_steps(self) -> int:
        if not self._activity_feedback_enabled():
            return 0
        return int(getattr(self, "st_activity_feedback_history_steps", 0))

    def _activity_feedback_prune_mask(self) -> torch.Tensor:
        if not self._activity_feedback_uses_sparse_matrix():
            return self.activity_head.weight.new_zeros(
                (self.num_activities, self.num_concepts)
            )
        frozen = getattr(self, "_frozen_activity_feedback_prune_mask", None)
        if torch.is_tensor(frozen) and frozen.numel() > 0:
            return frozen
        top_k = int(getattr(self, "st_activity_feedback_top_k", 0))
        if top_k <= 0 or top_k >= self.num_concepts:
            return torch.ones_like(self.activity_feedback_weight)
        scores = self.activity_feedback_weight.abs() * torch.sigmoid(
            self.activity_feedback_gate_logits
        )
        keep = torch.topk(scores, k=top_k, dim=1, largest=True).indices
        mask = torch.zeros_like(scores)
        return mask.scatter(1, keep, 1.0)

    def effective_activity_feedback_matrix(self) -> torch.Tensor:
        if not self._activity_feedback_uses_sparse_matrix():
            return self.activity_head.weight.new_zeros(
                (self.num_activities, self.num_concepts)
            )
        return (
            self.activity_feedback_weight
            * torch.sigmoid(self.activity_feedback_gate_logits)
            * self._activity_feedback_prune_mask()
        )

    def _activity_feedback_topk_for_spatial(self, spatial_top_k: int) -> int:
        target = int(getattr(self, "st_target_activity_feedback_top_k", 0))
        if target <= 0 or target >= self.num_concepts:
            return 0
        if int(spatial_top_k) <= 0:
            return 0
        spatial_max = max(self.num_concepts - 1, 1)
        spatial_target = int(getattr(self, "st_target_spatial_top_k", 0))
        if spatial_target <= 0 or spatial_target >= spatial_max:
            return target
        progress = (spatial_max - int(spatial_top_k)) / float(spatial_max - spatial_target)
        progress = max(0.0, min(1.0, progress))
        return max(target, int(round(self.num_concepts + progress * (target - self.num_concepts))))

    def _gradual_activity_feedback_topk(self, epoch: int) -> int:
        target = int(getattr(self, "st_target_activity_feedback_top_k", 0))
        if target <= 0 or target >= self.num_concepts:
            return 0
        if int(epoch) <= self.st_topk_warmup_epochs:
            return 0
        completion = self.topk_schedule_completion_epoch()
        if int(epoch) >= completion:
            return target
        progress = (int(epoch) - self.st_topk_warmup_epochs) / float(
            self.st_topk_ramp_epochs
        )
        return max(
            target,
            int(round(self.num_concepts + progress * (target - self.num_concepts))),
        )

    def _set_runtime_topk(self, spatial: int, temporal: int, cross_temporal: int) -> None:
        super()._set_runtime_topk(spatial, temporal, cross_temporal)
        if self._activity_feedback_uses_sparse_matrix():
            self.st_activity_feedback_top_k = self._activity_feedback_topk_for_spatial(spatial)

    def _topk_state(self, phase: str, checkpoint_eligible: bool) -> Dict[str, object]:
        state = super()._topk_state(phase, checkpoint_eligible)
        state["activity_feedback_top_k"] = (
            int(getattr(self, "st_activity_feedback_top_k", 0))
            if self._activity_feedback_uses_sparse_matrix()
            else 0
        )
        return state

    def configure_topk_for_training(self, epoch: int) -> Dict[str, object]:
        state = super().configure_topk_for_training(epoch)
        if self._activity_feedback_uses_sparse_matrix():
            mode = self.st_topk_training_mode
            if mode in {"none", "soft_train_hard_eval"}:
                feedback_top_k = 0
            elif mode == "gradual":
                feedback_top_k = self._gradual_activity_feedback_topk(epoch)
            else:
                feedback_top_k = self.st_target_activity_feedback_top_k
            self.st_activity_feedback_top_k = int(feedback_top_k)
            state["activity_feedback_top_k"] = int(feedback_top_k)
        return state

    def configure_topk_for_evaluation(self, epoch: int) -> Dict[str, object]:
        state = super().configure_topk_for_evaluation(epoch)
        if self._activity_feedback_uses_sparse_matrix():
            if self.st_topk_training_mode == "none":
                feedback_top_k = 0
            elif self.st_topk_training_mode == "gradual":
                feedback_top_k = self._gradual_activity_feedback_topk(epoch)
            else:
                feedback_top_k = self.st_target_activity_feedback_top_k
            self.st_activity_feedback_top_k = int(feedback_top_k)
            state["activity_feedback_top_k"] = int(feedback_top_k)
        return state

    def configure_topk_for_final_evaluation(self) -> Dict[str, object]:
        state = super().configure_topk_for_final_evaluation()
        if self._activity_feedback_uses_sparse_matrix():
            feedback_top_k = (
                0
                if self.st_topk_training_mode == "none"
                else self.st_target_activity_feedback_top_k
            )
            self.st_activity_feedback_top_k = int(feedback_top_k)
            state["activity_feedback_top_k"] = int(feedback_top_k)
        return state

    @torch.no_grad()
    def freeze_topk_topology(self) -> None:
        super().freeze_topk_topology()
        if self._activity_feedback_uses_sparse_matrix():
            mask = self._activity_feedback_prune_mask().detach().clone()
            self._frozen_activity_feedback_prune_mask = mask
            self.activity_feedback_weight.mul_(mask)
            self.activity_feedback_gate_logits.masked_fill_(mask <= 0.0, -30.0)

    @torch.no_grad()
    def enforce_frozen_topk_topology(self) -> None:
        super().enforce_frozen_topk_topology()
        frozen = getattr(self, "_frozen_activity_feedback_prune_mask", None)
        if self._activity_feedback_uses_sparse_matrix() and torch.is_tensor(frozen) and frozen.numel() > 0:
            self.activity_feedback_weight.mul_(frozen)
            self.activity_feedback_gate_logits.masked_fill_(frozen <= 0.0, -30.0)

    def topk_topology_is_frozen(self) -> bool:
        graph_frozen = super().topk_topology_is_frozen()
        if not self._activity_feedback_uses_sparse_matrix():
            return graph_frozen
        feedback_frozen = getattr(self, "_frozen_activity_feedback_prune_mask", None)
        return graph_frozen and torch.is_tensor(feedback_frozen) and feedback_frozen.numel() > 0

    def edge_regularization(self) -> torch.Tensor:
        regularization = super().edge_regularization()
        if self._activity_feedback_uses_sparse_matrix():
            mask = self._activity_feedback_prune_mask()
            regularization = regularization + self.effective_activity_feedback_matrix().abs().sum()
            regularization = regularization + (
                torch.sigmoid(self.activity_feedback_gate_logits) * mask
            ).sum()
        if self._activity_feedback_enabled() and hasattr(
            self, "activity_feedback_history_gate_logits"
        ):
            regularization = regularization + torch.sigmoid(
                self.activity_feedback_history_gate_logits
            ).sum()
        return regularization

    def graph_metrics(self) -> Dict[str, float]:
        metrics = super().graph_metrics()
        enabled = self._activity_feedback_enabled()
        metrics["st_activity_feedback_enabled"] = float(enabled)
        metrics["st_activity_feedback_top_k"] = float(
            getattr(self, "st_activity_feedback_top_k", 0)
            if self._activity_feedback_uses_sparse_matrix()
            else 0
        )
        metrics["st_activity_feedback_history_steps"] = float(
            self._activity_feedback_history_steps()
        )
        metrics["st_activity_feedback_updates_concepts"] = float(
            self._activity_feedback_updates_concepts()
        )
        metrics["st_activity_feedback_updates_logits"] = float(
            self._activity_feedback_updates_logits()
        )
        metrics["activity_feedback_strength"] = float(
            torch.sigmoid(self.activity_feedback_strength_logit).detach().item()
        ) if hasattr(self, "activity_feedback_strength_logit") else 0.0
        metrics["activity_feedback_history_gate_mean"] = float(
            torch.sigmoid(self.activity_feedback_history_gate_logits).detach().mean().item()
        ) if hasattr(self, "activity_feedback_history_gate_logits") else 0.0
        if not self._activity_feedback_uses_sparse_matrix():
            metrics.update(
                {
                    "activity_feedback_possible_edges": 0.0,
                    "activity_feedback_kept_edges": 0.0,
                    "activity_feedback_active_edges": 0.0,
                    "activity_feedback_density": 0.0,
                    "activity_feedback_mean_abs_weight": 0.0,
                    "activity_feedback_mean_gate": 0.0,
                }
            )
            return metrics
        matrix = self.effective_activity_feedback_matrix().detach()
        mask = self._activity_feedback_prune_mask().detach()
        gate = torch.sigmoid(self.activity_feedback_gate_logits).detach() * mask
        possible = float(matrix.numel())
        kept = float((mask > 0.0).sum().item())
        active = float(((gate > self.edge_threshold) & (matrix.abs() > 1e-8)).sum().item())
        kept_values = matrix[mask > 0.0]
        kept_gates = gate[mask > 0.0]
        metrics.update(
            {
                "activity_feedback_possible_edges": possible,
                "activity_feedback_kept_edges": kept,
                "activity_feedback_active_edges": active,
                "activity_feedback_density": active / max(possible, 1.0),
                "activity_feedback_mean_abs_weight": float(kept_values.abs().mean().item())
                if kept_values.numel()
                else 0.0,
                "activity_feedback_mean_gate": float(kept_gates.mean().item())
                if kept_gates.numel()
                else 0.0,
            }
        )
        return metrics

    def _run_graph_next_layers(
        self,
        layers: nn.ModuleList,
        previous: torch.Tensor,
        valid_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
        branch: str = "forecast",
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        layer_info = self._zero_layer_info(previous)
        refined = previous
        for layer_index, layer in enumerate(layers):
            refined, layer_info = layer.forward_next(
                refined,
                valid_mask,
                edge_interventions=self._edge_intervention_items(intervention, branch, layer_index),
            )
            refined = self._apply_persistent_intervention(
                refined,
                intervention=intervention,
                clamp_source=clamp_source,
            )
        return refined, layer_info

    def _activity_interventions_by_step(
        self,
        intervention: Dict[str, object] | None,
        max_horizon: int,
    ) -> Dict[int, list[Dict[str, object]]]:
        if not isinstance(intervention, dict):
            return {}
        by_key: Dict[tuple[int, int | None], Dict[str, object]] = {}
        for item in self._intervention_items(intervention):
            if str(item.get("item_type", item.get("type", ""))).lower() != "activity":
                continue
            if not self._activity_feedback_enabled():
                raise ValueError(
                    "Activity interventions require an enabled st_activity_feedback_mode."
                )
            if "step" not in item:
                raise ValueError("Activity intervention requires step.")
            step = int(item["step"])
            min_step = -self._activity_feedback_history_steps()
            if step < min_step or step >= int(max_horizon):
                raise IndexError(
                    "Activity intervention step must be in "
                    f"[{min_step}, {int(max_horizon) - 1}]: {step}"
                )
            class_idx = item.get("class_idx", item.get("class_index"))
            if class_idx is None:
                raise ValueError("Activity intervention requires class_idx.")
            class_idx = int(class_idx)
            if class_idx < 0 or class_idx >= self.num_activities:
                raise IndexError(f"Activity class_idx out of range: {class_idx}")
            if "probability" not in item:
                raise ValueError("Activity intervention requires probability.")
            probability = float(item["probability"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Activity intervention probability must be in [0, 1].")
            normalized = dict(item)
            normalized["step"] = step
            normalized["class_idx"] = class_idx
            normalized["probability"] = probability
            batch_idx = normalized.get("batch_idx")
            batch_key = None if batch_idx is None else int(batch_idx)
            by_key[(step, batch_key)] = normalized
        by_step: Dict[int, list[Dict[str, object]]] = {}
        for (step, _), item in by_key.items():
            by_step.setdefault(step, []).append(item)
        return by_step

    def _apply_activity_probability_intervention(
        self,
        probabilities: torch.Tensor,
        items: list[Dict[str, object]],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not items:
            return probabilities
        intervened = probabilities.clone()
        for item in items:
            batch_idx = item.get("batch_idx")
            if batch_idx is None:
                rows = torch.arange(probabilities.size(0), device=probabilities.device)
            else:
                batch_idx = int(batch_idx)
                if batch_idx < 0 or batch_idx >= probabilities.size(0):
                    raise IndexError(f"Activity intervention batch_idx out of range: {batch_idx}")
                rows = torch.tensor([batch_idx], device=probabilities.device)
            time_positions = torch.arange(valid_mask.size(1), device=valid_mask.device).view(1, -1)
            last_valid = torch.where(
                valid_mask[rows] > 0.0,
                time_positions.expand(rows.numel(), -1),
                torch.full_like(time_positions.expand(rows.numel(), -1), -1),
            ).max(dim=1).values
            if torch.any(last_valid < 0):
                raise ValueError("Activity intervention requires at least one valid timestep per row.")

            class_idx = int(item["class_idx"])
            target = float(item["probability"])
            selected = intervened[rows, last_valid, :]
            if self.num_activities == 1:
                selected.fill_(1.0)
            else:
                old_selected = selected[:, class_idx]
                old_remaining = 1.0 - old_selected
                scale = (1.0 - target) / old_remaining.clamp_min(1e-12)
                selected = selected * scale.unsqueeze(-1)
                degenerate = old_remaining <= 1e-12
                if torch.any(degenerate):
                    selected[degenerate] = (1.0 - target) / float(self.num_activities - 1)
                selected[:, class_idx] = target
            intervened[rows, last_valid, :] = selected
        return intervened

    def _feedback_activity_probabilities(
        self,
        predicted: torch.Tensor,
        previous_activity_labels: torch.Tensor | None,
        source_step: int,
        teacher_forcing: bool,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        if not teacher_forcing or previous_activity_labels is None:
            return predicted
        labels = previous_activity_labels
        if labels.ndim == 1:
            labels = labels[:, None, None].expand(-1, predicted.size(1), 1)
        elif labels.ndim == 2:
            labels = labels[:, None, :].expand(-1, predicted.size(1), -1)
        label_index = min(int(source_step), labels.size(-1) - 1)
        ground_truth = F.one_hot(
            labels[..., label_index].long(),
            num_classes=self.num_activities,
        ).to(dtype=predicted.dtype)
        ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        if ratio >= 1.0:
            return ground_truth
        if ratio <= 0.0:
            return predicted
        use_ground_truth = torch.rand(
            ground_truth.shape[:-1],
            device=ground_truth.device,
        ) < ratio
        return torch.where(use_ground_truth.unsqueeze(-1), ground_truth, predicted)

    def _apply_activity_feedback(
        self,
        concepts: torch.Tensor,
        probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message = self._activity_concept_feedback_message(probabilities)
        message = message * valid_mask.unsqueeze(-1)
        if getattr(self, "st_state_activation", "identity") == "bounded_logit":
            eps = torch.finfo(concepts.dtype).eps
            bounded = concepts.clamp(min=eps, max=1.0 - eps)
            logits = torch.log(bounded) - torch.log1p(-bounded)
            updated = torch.sigmoid(logits + message)
        else:
            updated = concepts + message
        return updated * valid_mask.unsqueeze(-1), message

    def _activity_feedback_strength(self) -> torch.Tensor:
        parameter = getattr(self, "activity_feedback_strength_logit", None)
        if torch.is_tensor(parameter):
            return torch.sigmoid(parameter)
        return self.activity_head.weight.new_ones(())

    def _activity_concept_feedback_message(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        mode = getattr(self, "st_activity_feedback_mode", "none")
        if mode in {
            "sparse_label_to_concept",
            "hybrid_sparse_and_label_autoregressive",
        }:
            return torch.matmul(probabilities, self.effective_activity_feedback_matrix())
        if mode == "classifier_weight_to_concept":
            return (
                torch.matmul(probabilities, self.activity_head.weight)
                * self._activity_feedback_strength()
            )
        if mode == "ridge_inverse_to_concept":
            eps = float(getattr(self, "st_activity_feedback_probability_eps", 1e-4))
            log_probabilities = torch.log(probabilities.clamp_min(eps))
            centered_log_probabilities = log_probabilities - log_probabilities.mean(
                dim=-1,
                keepdim=True,
            )
            classifier = self.activity_head.weight.detach()
            classifier = classifier - classifier.mean(dim=0, keepdim=True)
            gram = classifier @ classifier.transpose(0, 1)
            ridge = float(getattr(self, "st_activity_feedback_ridge_lambda", 0.1))
            gram = gram + torch.eye(
                self.num_activities,
                device=gram.device,
                dtype=gram.dtype,
            ) * ridge
            original_shape = centered_log_probabilities.shape
            flattened = centered_log_probabilities.reshape(-1, self.num_activities)
            coefficients = torch.linalg.solve(gram, flattened.transpose(0, 1)).transpose(0, 1)
            projected = (coefficients @ classifier).reshape(
                *original_shape[:-1],
                self.num_concepts,
            )
            return projected * self._activity_feedback_strength()
        if mode == "prototype_label_to_concept":
            source = getattr(self, "activity_feedback_prototype_source", None)
            target = getattr(self, "activity_feedback_prototype_target", None)
            if not torch.is_tensor(source) or not torch.is_tensor(target) or source.numel() == 0:
                raise RuntimeError(
                    "prototype_label_to_concept requires training-split prototypes to be installed."
                )
            source = source.to(device=probabilities.device, dtype=probabilities.dtype)
            target = target.to(device=probabilities.device, dtype=probabilities.dtype)
            source = self.calibrator(source)
            target = self.calibrator(target)
            if getattr(self, "st_state_activation", "identity") == "bounded_logit":
                eps = float(getattr(self, "st_activity_feedback_probability_eps", 1e-4))
                source = torch.logit(source.clamp(min=eps, max=1.0 - eps))
                target = torch.logit(target.clamp(min=eps, max=1.0 - eps))
            transition = target - source
            return (
                torch.matmul(probabilities, transition)
                * self._activity_feedback_strength()
            )
        if mode == "mlp_label_to_concept":
            return self.activity_feedback_decoder(probabilities) * self._activity_feedback_strength()
        return probabilities.new_zeros((*probabilities.shape[:-1], self.num_concepts))

    def _activity_logit_feedback_message(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if not self._activity_feedback_updates_logits():
            return probabilities.new_zeros((*probabilities.shape[:-1], self.num_activities))
        message = torch.matmul(probabilities, self.activity_feedback_logit_weight)
        message = message - message.mean(dim=-1, keepdim=True)
        return message * self._activity_feedback_strength()

    def _replace_activity_distribution(
        self,
        probabilities: torch.Tensor,
        items: list[Dict[str, object]],
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Apply proportional probability replacement to selected batch rows."""

        if not items:
            return probabilities
        intervened = probabilities.clone()
        for item in items:
            batch_idx = item.get("batch_idx")
            if batch_idx is None:
                rows = torch.nonzero(eligible, as_tuple=False).flatten()
            else:
                batch_idx = int(batch_idx)
                if batch_idx < 0 or batch_idx >= probabilities.size(0):
                    raise IndexError(f"Activity intervention batch_idx out of range: {batch_idx}")
                if not bool(eligible[batch_idx]):
                    raise ValueError("Activity intervention source timestep is not available.")
                rows = torch.tensor([batch_idx], device=probabilities.device)
            if rows.numel() == 0:
                continue
            class_idx = int(item["class_idx"])
            target = float(item["probability"])
            selected = intervened[rows, :]
            if self.num_activities == 1:
                selected.fill_(1.0)
            else:
                old_selected = selected[:, class_idx]
                old_remaining = 1.0 - old_selected
                scale = (1.0 - target) / old_remaining.clamp_min(1e-12)
                selected = selected * scale.unsqueeze(-1)
                degenerate = old_remaining <= 1e-12
                if torch.any(degenerate):
                    selected[degenerate] = (1.0 - target) / float(self.num_activities - 1)
                selected[:, class_idx] = target
            intervened[rows, :] = selected
        return intervened

    def _apply_observed_activity_feedback(
        self,
        window_refined: torch.Tensor,
        forecast_refined: torch.Tensor,
        valid_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        history_steps = self._activity_feedback_history_steps()
        if history_steps <= 0:
            logit_residual = window_refined.new_zeros(
                (*window_refined.shape[:-1], self.num_activities)
            )
            return window_refined, forecast_refined, logit_residual, {}

        max_horizon = max(self.forecast_horizons)
        activity_interventions = self._activity_interventions_by_step(intervention, max_horizon)
        preliminary_logits = self.activity_head(self._prediction_transform(window_refined))
        preliminary_probabilities = torch.softmax(preliminary_logits, dim=-1)
        batch_rows = torch.arange(window_refined.size(0), device=window_refined.device)
        time_positions = torch.arange(valid_mask.size(1), device=valid_mask.device).view(1, -1)
        last_valid = torch.where(
            valid_mask > 0.0,
            time_positions.expand(valid_mask.size(0), -1),
            torch.full_like(time_positions.expand(valid_mask.size(0), -1), -1),
        ).max(dim=1).values
        if torch.any(last_valid < 0):
            raise ValueError("Historical activity feedback requires at least one valid timestep.")

        total_concept_message = window_refined.new_zeros(
            (window_refined.size(0), window_refined.size(-1))
        )
        total_logit_message = window_refined.new_zeros(
            (window_refined.size(0), self.num_activities)
        )
        probabilities_by_step: Dict[int, torch.Tensor] = {}
        concept_messages_by_step: Dict[int, torch.Tensor] = {}
        logit_messages_by_step: Dict[int, torch.Tensor] = {}
        for lag in range(1, history_steps + 1):
            source_positions = last_valid - lag
            eligible = source_positions >= 0
            safe_positions = source_positions.clamp_min(0)
            eligible = eligible & (valid_mask[batch_rows, safe_positions] > 0.0)
            probabilities = preliminary_probabilities[batch_rows, safe_positions, :]
            probabilities = self._replace_activity_distribution(
                probabilities,
                activity_interventions.get(-lag, []),
                eligible,
            )
            probabilities = probabilities * eligible.unsqueeze(-1)
            lag_gate = torch.sigmoid(self.activity_feedback_history_gate_logits[lag - 1])
            concept_message = self._activity_concept_feedback_message(probabilities) * lag_gate
            concept_message = concept_message * eligible.unsqueeze(-1)
            logit_message = self._activity_logit_feedback_message(probabilities) * lag_gate
            logit_message = logit_message * eligible.unsqueeze(-1)
            total_concept_message = total_concept_message + concept_message
            total_logit_message = total_logit_message + logit_message
            probabilities_by_step[-lag] = probabilities
            concept_messages_by_step[-lag] = concept_message
            logit_messages_by_step[-lag] = logit_message

        def update_current(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            before = states[batch_rows, last_valid, :]
            if getattr(self, "st_state_activation", "identity") == "bounded_logit":
                eps = torch.finfo(states.dtype).eps
                bounded = before.clamp(min=eps, max=1.0 - eps)
                logits = torch.log(bounded) - torch.log1p(-bounded)
                after = torch.sigmoid(logits + total_concept_message)
            else:
                after = before + total_concept_message
            updated = states.clone()
            updated[batch_rows, last_valid, :] = after
            updated = self._apply_persistent_intervention(
                updated,
                intervention=intervention,
                clamp_source=clamp_source,
            )
            return updated, before

        if self._activity_feedback_updates_concepts():
            updated_window, pre_feedback_current = update_current(window_refined)
            updated_forecast, _ = update_current(forecast_refined)
        else:
            updated_window = window_refined
            updated_forecast = forecast_refined
            pre_feedback_current = window_refined[batch_rows, last_valid, :]
        post_feedback_current = updated_window[batch_rows, last_valid, :]
        observed_logit_residual = preliminary_logits.new_zeros(preliminary_logits.shape)
        observed_logit_residual[batch_rows, last_valid, :] = total_logit_message
        return updated_window, updated_forecast, observed_logit_residual, {
            "effective_activity_probs_by_history_step": probabilities_by_step,
            "activity_feedback_messages_by_history_step": concept_messages_by_step,
            "activity_feedback_concept_messages_by_history_step": concept_messages_by_step,
            "activity_feedback_logit_messages_by_history_step": logit_messages_by_step,
            "activity_feedback_history_total_message": total_concept_message,
            "activity_feedback_history_total_concept_message": total_concept_message,
            "activity_feedback_history_total_logit_message": total_logit_message,
            "pre_history_feedback_current_concepts": pre_feedback_current,
            "post_history_feedback_current_concepts": post_feedback_current,
            "preliminary_history_activity_logits": preliminary_logits,
        }

    def _autoregressive_forecast_logits(
        self,
        forecast_refined: torch.Tensor,
        activity_logits: torch.Tensor,
        previous_activity_labels: torch.Tensor | None,
        teacher_forcing: bool,
        teacher_forcing_ratio: float,
        intervention: Dict[str, object] | None = None,
        clamp_source: torch.Tensor | None = None,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        max_horizon = max(self.forecast_horizons)
        valid_mask = getattr(
            self,
            "_forecast_rollout_valid_mask",
            forecast_refined.new_ones(forecast_refined.shape[:2]),
        )
        predicted_concepts_by_step: Dict[int, torch.Tensor] = {}
        pre_feedback_concepts_by_step: Dict[int, torch.Tensor] = {}
        forecast_graph_info_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
        logits_by_step: Dict[int, torch.Tensor] = {}
        effective_activity_probs_by_step: Dict[int, torch.Tensor] = {}
        activity_feedback_concept_messages_by_step: Dict[int, torch.Tensor] = {}
        activity_feedback_logit_messages_by_step: Dict[int, torch.Tensor] = {}
        current_concepts = forecast_refined
        memory_context = getattr(self, "_last_prefix_memory_context", None)
        if self.st_forecast_rollout_mode == "legacy":
            rollout_layers = self.forecast_graph_layers if len(self.forecast_graph_layers) > 0 else self.shared_graph_layers
        elif self.st_forecast_rollout_mode == "persistence":
            rollout_layers = nn.ModuleList()
        else:
            rollout_layers = self.controlled_rollout_layers
        if not self._activity_feedback_enabled():
            activity_interventions = self._activity_interventions_by_step(intervention, max_horizon)
            if activity_interventions:
                raise ValueError("Activity feedback is disabled.")
        else:
            activity_interventions = self._activity_interventions_by_step(intervention, max_horizon)

        source_probabilities = torch.softmax(activity_logits, dim=-1)
        for step in range(1, max_horizon + 1):
            source_step = step - 1
            if self._activity_feedback_enabled():
                effective_probabilities = self._feedback_activity_probabilities(
                    source_probabilities,
                    previous_activity_labels,
                    source_step,
                    teacher_forcing,
                    teacher_forcing_ratio,
                )
                effective_probabilities = self._apply_activity_probability_intervention(
                    effective_probabilities,
                    activity_interventions.get(source_step, []),
                    valid_mask,
                )
            else:
                effective_probabilities = source_probabilities
            effective_activity_probs_by_step[source_step] = effective_probabilities

            if self.st_forecast_rollout_mode == "persistence":
                # Component ablation: each H1--H3 prediction reads the same
                # current concept state. No graph, dense transition, memory,
                # or activity-feedback path can pass information forward.
                graph_info = self._zero_layer_info(current_concepts)
            else:
                current_concepts, graph_info = self._run_graph_next_layers(
                    rollout_layers,
                    current_concepts,
                    valid_mask,
                    intervention=intervention,
                    clamp_source=clamp_source,
                    branch="forecast" if len(self.forecast_graph_layers) > 0 else "shared",
                )
            if (
                getattr(self, "st_past_context_mode", "none") == "prefix_recursive_memory"
                and memory_context is not None
            ):
                current_concepts, memory_context = self._apply_recursive_forecast_memory(
                    current_concepts,
                    memory_context,
                    valid_mask,
                )
            pre_feedback_concepts_by_step[step] = current_concepts
            if self._activity_feedback_updates_concepts():
                current_concepts, feedback_message = self._apply_activity_feedback(
                    current_concepts,
                    effective_probabilities,
                    valid_mask,
                )
                current_concepts = self._apply_persistent_intervention(
                    current_concepts,
                    intervention=intervention,
                    clamp_source=clamp_source,
                )
            else:
                feedback_message = torch.zeros_like(current_concepts)
            current_concepts = self._apply_rollout_concept_intervention(
                current_concepts,
                intervention,
                step,
            )
            activity_feedback_concept_messages_by_step[source_step] = feedback_message
            predicted_concepts_by_step[step] = current_concepts
            forecast_graph_info_by_step[step] = graph_info
            base_logits = self.activity_head(self._prediction_transform(current_concepts))
            logit_feedback_message = self._activity_logit_feedback_message(
                effective_probabilities
            ) * valid_mask.unsqueeze(-1)
            activity_feedback_logit_messages_by_step[source_step] = logit_feedback_message
            logits_by_step[step] = base_logits + logit_feedback_message
            source_probabilities = torch.softmax(logits_by_step[step], dim=-1)
        effective_activity_probs_by_step[max_horizon] = source_probabilities
        self._last_predicted_concepts_by_step = predicted_concepts_by_step
        self._last_pre_feedback_concepts_by_step = pre_feedback_concepts_by_step
        self._last_forecast_graph_info_by_step = forecast_graph_info_by_step
        self._last_effective_activity_probs_by_step = effective_activity_probs_by_step
        self._last_activity_feedback_messages_by_step = (
            activity_feedback_concept_messages_by_step
        )
        self._last_activity_feedback_logit_messages_by_step = (
            activity_feedback_logit_messages_by_step
        )
        forecast_logits_by_horizon = {
            horizon: logits_by_step[horizon]
            for horizon in self.forecast_horizons
        }
        return forecast_logits_by_horizon, logits_by_step

    def forward(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        memory_prefix_concepts: torch.Tensor | None = None,
        memory_prefix_key_padding_mask: torch.Tensor | None = None,
        previous_activity_labels: torch.Tensor | None = None,
        teacher_forcing: bool = False,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, object]:
        # The base class calibrates concepts and applies the shared graph to
        # observed windows. This subclass adds the recursive forecast rollout
        # and exposes the intermediate concept states for intervention analysis.
        self._forecast_rollout_valid_mask = (~key_padding_mask).float()
        outputs = super().forward(
            concepts,
            key_padding_mask,
            intervention=intervention,
            memory_prefix_concepts=memory_prefix_concepts,
            memory_prefix_key_padding_mask=memory_prefix_key_padding_mask,
            previous_activity_labels=previous_activity_labels,
            teacher_forcing=teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        predicted_concepts_by_step = getattr(self, "_last_predicted_concepts_by_step", {})
        pre_feedback_concepts_by_step = getattr(self, "_last_pre_feedback_concepts_by_step", {})
        forecast_graph_info_by_step = getattr(self, "_last_forecast_graph_info_by_step", {})
        effective_activity_probs_by_step = getattr(
            self,
            "_last_effective_activity_probs_by_step",
            {},
        )
        activity_feedback_messages_by_step = getattr(
            self,
            "_last_activity_feedback_messages_by_step",
            {},
        )
        activity_feedback_logit_messages_by_step = getattr(
            self,
            "_last_activity_feedback_logit_messages_by_step",
            {},
        )
        if hasattr(self, "_last_predicted_concepts_by_step"):
            delattr(self, "_last_predicted_concepts_by_step")
        if hasattr(self, "_last_pre_feedback_concepts_by_step"):
            delattr(self, "_last_pre_feedback_concepts_by_step")
        if hasattr(self, "_last_forecast_graph_info_by_step"):
            delattr(self, "_last_forecast_graph_info_by_step")
        if hasattr(self, "_last_effective_activity_probs_by_step"):
            delattr(self, "_last_effective_activity_probs_by_step")
        if hasattr(self, "_last_activity_feedback_messages_by_step"):
            delattr(self, "_last_activity_feedback_messages_by_step")
        if hasattr(self, "_last_activity_feedback_logit_messages_by_step"):
            delattr(self, "_last_activity_feedback_logit_messages_by_step")
        if hasattr(self, "_forecast_rollout_valid_mask"):
            delattr(self, "_forecast_rollout_valid_mask")
        outputs["predicted_concepts_by_step"] = dict(predicted_concepts_by_step)
        outputs["pre_feedback_concepts_by_step"] = dict(pre_feedback_concepts_by_step)
        outputs["forecast_graph_info_by_step"] = dict(forecast_graph_info_by_step)
        outputs["effective_activity_probs_by_step"] = dict(effective_activity_probs_by_step)
        outputs["activity_feedback_messages_by_step"] = dict(
            activity_feedback_messages_by_step
        )
        outputs["activity_feedback_concept_messages_by_step"] = dict(
            activity_feedback_messages_by_step
        )
        outputs["activity_feedback_logit_messages_by_step"] = dict(
            activity_feedback_logit_messages_by_step
        )
        outputs["future_concepts_by_horizon"] = {
            horizon: predicted_concepts_by_step[horizon]
            for horizon in self.future_concept_horizons
            if horizon in predicted_concepts_by_step
        }
        return outputs


class GraphConceptStateBottleneckModel(GraphCBM):
    """Graph-structured state-space model whose state coordinates are concepts."""

    def __init__(
        self,
        *args,
        gcssbm_transition_mode: str = "shared",
        gcssbm_transition_layers: int = 1,
        gcssbm_update_gate_init: float = 0.0,
        **kwargs,
    ) -> None:
        transition_mode = str(gcssbm_transition_mode)
        transition_layers = int(gcssbm_transition_layers)
        if transition_mode not in {"shared", "separate"}:
            raise ValueError("gcssbm_transition_mode must be 'shared' or 'separate'.")
        if transition_layers < 1:
            raise ValueError("gcssbm_transition_layers must be >= 1.")
        if str(kwargs.get("st_past_context_mode", "none")) != "none":
            raise ValueError("gcssbm does not support st_past_context_mode != 'none'.")
        if str(kwargs.get("st_activity_feedback_mode", "none")) != "none":
            raise ValueError("gcssbm does not support activity-label feedback.")
        if int(kwargs.get("motif_z_attention_layers", 0)) != 0:
            raise ValueError("gcssbm requires motif_z_attention_layers=0.")
        if str(kwargs.get("st_observed_refiner_mode", "graph")) != "graph":
            raise ValueError("gcssbm requires st_observed_refiner_mode='graph'.")
        if str(kwargs.get("st_forecast_rollout_mode", "legacy")) != "legacy":
            raise ValueError("gcssbm requires st_forecast_rollout_mode='legacy'.")
        if str(kwargs.get("st_state_activation", "bounded_logit")) != "bounded_logit":
            raise ValueError("gcssbm requires st_state_activation='bounded_logit'.")
        if str(kwargs.get("st_prediction_transform", "logit")) != "logit":
            raise ValueError("gcssbm requires st_prediction_transform='logit'.")

        # The shared branch is the observed-state transition. Task-specific
        # stacks from ConceptForecastCBM are disabled and, when requested, a
        # separately initialized forecast transition is installed below.
        kwargs["st_graph_layers"] = transition_layers
        kwargs["st_task_graph_layers"] = 0
        kwargs["st_past_context_mode"] = "none"
        kwargs["st_state_activation"] = "bounded_logit"
        kwargs["st_prediction_transform"] = "logit"
        kwargs["st_forecast_rollout_mode"] = "legacy"
        kwargs["st_activity_feedback_mode"] = "none"
        super().__init__(*args, **kwargs)

        self.gcssbm_transition_mode = transition_mode
        self.gcssbm_transition_layers = transition_layers
        self.gcssbm_update_gate_init = float(gcssbm_update_gate_init)
        self.state_update_gate_bias = nn.Parameter(
            torch.full((self.num_concepts,), self.gcssbm_update_gate_init)
        )
        self.state_update_gate_observation_scale = nn.Parameter(
            torch.zeros(self.num_concepts)
        )
        self.state_update_gate_prior_scale = nn.Parameter(
            torch.zeros(self.num_concepts)
        )

        if self.gcssbm_transition_mode == "separate":
            self.forecast_graph_layers = self._make_graph_layers(transition_layers)
            self.forecast_graph_layers.load_state_dict(self.shared_graph_layers.state_dict())
        else:
            self.forecast_graph_layers = nn.ModuleList()
        # GCSSBM directly supervises and exports its recurrent states; separate
        # per-horizon projection heads would create unused, non-semantic state.
        self.future_concept_heads = nn.ModuleDict()

    def _state_update_gate(
        self,
        observation: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            self.state_update_gate_bias.view(1, 1, -1)
            + self.state_update_gate_observation_scale.view(1, 1, -1) * (observation - 0.5)
            + self.state_update_gate_prior_scale.view(1, 1, -1) * (prior - 0.5)
        )
        return torch.sigmoid(logits)

    def _graph_only_intervention(
        self,
        intervention: Dict[str, object] | None,
        *,
        shared_edges_apply_to_forecast: bool = False,
    ) -> Dict[str, object] | None:
        if not isinstance(intervention, dict):
            return None
        edge_items = []
        for raw_item in self._intervention_items(intervention):
            if str(raw_item.get("item_type", raw_item.get("type", ""))).lower() != "edge":
                continue
            item = dict(raw_item)
            if shared_edges_apply_to_forecast and str(item.get("branch", "all")).lower() == "shared":
                item["branch"] = "all"
            edge_items.append(item)
        if not edge_items:
            return None
        return {"mode": "input", "items": edge_items}

    def _apply_state_intervention_at_time(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        timestep: int,
        intervention: Dict[str, object] | None,
    ) -> torch.Tensor:
        if not isinstance(intervention, dict):
            return state
        mode = self._intervention_mode(intervention)
        if mode not in {"state", "pulse", "persistent", "clamp"}:
            return state
        updated = state.clone()
        for item in self._concept_intervention_items(intervention):
            if "rollout_step" in item:
                continue
            item_time = int(item.get("time_idx", item.get("timestep")))
            if item_time != int(timestep):
                continue
            concept_idx = int(item.get("concept_idx", item.get("concept_index")))
            if concept_idx < 0 or concept_idx >= self.num_concepts:
                raise IndexError(f"concept_idx out of range: {concept_idx}")
            has_value = item.get("value") is not None
            has_delta = item.get("delta") is not None
            if has_value == has_delta:
                raise ValueError("Exactly one of intervention value or delta must be provided.")
            batch_selector = self._intervention_batch_selector(item, intervention)
            if has_value:
                if mode in {"state", "pulse"}:
                    value = float(item["value"])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError("State intervention values must be in [0, 1].")
                    updated[batch_selector, 0, concept_idx] = value
                else:
                    updated[batch_selector, 0, concept_idx] = observation[
                        batch_selector, 0, concept_idx
                    ]
            else:
                updated[batch_selector, 0, concept_idx] = torch.clamp(
                    updated[batch_selector, 0, concept_idx] + float(item["delta"]),
                    min=0.0,
                    max=1.0,
                )
        return updated

    def forward(
        self,
        concepts: torch.Tensor,
        key_padding_mask: torch.Tensor,
        intervention: Dict[str, object] | None = None,
        memory_prefix_concepts: torch.Tensor | None = None,
        memory_prefix_key_padding_mask: torch.Tensor | None = None,
        previous_activity_labels: torch.Tensor | None = None,
        teacher_forcing: bool = False,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, object]:
        del (
            memory_prefix_concepts,
            memory_prefix_key_padding_mask,
            previous_activity_labels,
            teacher_forcing,
            teacher_forcing_ratio,
        )
        if concepts.ndim != 3 or key_padding_mask.shape != concepts.shape[:2]:
            raise ValueError("Expected concepts [B,T,C] and key_padding_mask [B,T].")

        mode = self._intervention_mode(intervention)
        input_concepts = concepts
        if isinstance(intervention, dict) and mode not in {"state", "pulse"}:
            input_concepts = self._apply_input_intervention(input_concepts, intervention)
        valid_mask = (~key_padding_mask).float()
        calibrated = self.calibrator(input_concepts) * valid_mask.unsqueeze(-1)
        observed_graph_intervention = self._graph_only_intervention(intervention)
        forecast_graph_intervention = self._graph_only_intervention(
            intervention,
            shared_edges_apply_to_forecast=self.gcssbm_transition_mode == "shared",
        )

        batch_size, timesteps, _ = calibrated.shape
        previous_state = calibrated.new_zeros((batch_size, 1, self.num_concepts))
        has_state = torch.zeros((batch_size, 1), dtype=torch.bool, device=calibrated.device)
        filtered_steps: list[torch.Tensor] = []
        prior_steps: list[torch.Tensor] = []
        gate_steps: list[torch.Tensor] = []
        prior_valid_steps: list[torch.Tensor] = []
        observed_graph_info: list[Dict[str, torch.Tensor]] = []

        for timestep in range(timesteps):
            observation = calibrated[:, timestep : timestep + 1, :]
            current_valid = valid_mask[:, timestep : timestep + 1] > 0.0
            predicted_prior, graph_info = self._run_graph_next_layers(
                self.shared_graph_layers,
                previous_state,
                has_state.float(),
                intervention=observed_graph_intervention,
                branch="shared",
            )
            prior = torch.where(has_state.unsqueeze(-1), predicted_prior, observation)
            gate = self._state_update_gate(observation, prior)
            filtered = prior + gate * (observation - prior)
            filtered = self._apply_state_intervention_at_time(
                filtered,
                observation,
                timestep,
                intervention,
            )
            output_state = torch.where(
                current_valid.unsqueeze(-1),
                filtered,
                torch.zeros_like(filtered),
            )
            previous_state = torch.where(
                current_valid.unsqueeze(-1),
                filtered,
                previous_state,
            )
            prior_valid = current_valid & has_state
            has_state = has_state | current_valid

            filtered_steps.append(output_state)
            prior_steps.append(prior * prior_valid.unsqueeze(-1))
            gate_steps.append(gate * current_valid.unsqueeze(-1))
            prior_valid_steps.append(prior_valid)
            observed_graph_info.append(graph_info)

        filtered_states = torch.cat(filtered_steps, dim=1)
        state_priors = torch.cat(prior_steps, dim=1)
        state_update_gates = torch.cat(gate_steps, dim=1)
        state_prior_valid_mask = torch.cat(prior_valid_steps, dim=1)
        prediction_states = self._prediction_transform(filtered_states)
        activity_logits = self.activity_head(prediction_states)

        rollout_layers = (
            self.shared_graph_layers
            if self.gcssbm_transition_mode == "shared"
            else self.forecast_graph_layers
        )
        predicted_concepts_by_step: Dict[int, torch.Tensor] = {}
        forecast_graph_info_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
        autoregressive_logits_by_step: Dict[int, torch.Tensor] = {}
        current_states = filtered_states
        max_horizon = max(self.forecast_horizons)
        for step in range(1, max_horizon + 1):
            current_states, graph_info = self._run_graph_next_layers(
                rollout_layers,
                current_states,
                valid_mask,
                intervention=forecast_graph_intervention,
                branch="forecast",
            )
            current_states = self._apply_rollout_concept_intervention(
                current_states,
                intervention,
                step,
            )
            predicted_concepts_by_step[step] = current_states
            forecast_graph_info_by_step[step] = graph_info
            autoregressive_logits_by_step[step] = self.activity_head(
                self._prediction_transform(current_states)
            )

        forecast_logits_by_horizon = {
            horizon: autoregressive_logits_by_step[horizon]
            for horizon in self.forecast_horizons
        }
        forecast_refined = predicted_concepts_by_step.get(1, filtered_states)
        observed_spatial_messages = torch.cat(
            [info["spatial_messages"] for info in observed_graph_info],
            dim=1,
        )
        observed_temporal_messages = torch.cat(
            [info["temporal_messages"] for info in observed_graph_info],
            dim=1,
        )
        return {
            "calibrated_concepts": calibrated,
            "filtered_concept_states": filtered_states,
            "state_priors": state_priors,
            "state_prior_valid_mask": state_prior_valid_mask,
            "state_update_gates": state_update_gates,
            "concept_states": filtered_states,
            "shared_refined_concepts": filtered_states,
            "window_refined_concepts": filtered_states,
            "forecast_refined_concepts": forecast_refined,
            "temporalized_concepts": calibrated,
            "gate_values": state_update_gates,
            "same_messages": observed_spatial_messages,
            "lag_messages": observed_temporal_messages,
            "activity_repr": prediction_states,
            "prediction_repr": prediction_states,
            "activity_context_repr": prediction_states,
            "forecast_repr": self._prediction_transform(forecast_refined),
            "representation_mode": self.representation_mode,
            "st_state_activation": self.st_state_activation,
            "st_prediction_transform": self.st_prediction_transform,
            "activity_logits": activity_logits,
            "forecast_logits_by_horizon": forecast_logits_by_horizon,
            "autoregressive_logits_by_step": autoregressive_logits_by_step,
            "predicted_concepts_by_step": predicted_concepts_by_step,
            "pre_feedback_concepts_by_step": dict(predicted_concepts_by_step),
            "forecast_graph_info_by_step": forecast_graph_info_by_step,
            "future_concepts_by_horizon": {
                horizon: predicted_concepts_by_step[horizon]
                for horizon in self.future_concept_horizons
                if horizon in predicted_concepts_by_step
            },
            "video_repr": None,
            "video_logits": None,
            "gcssbm_transition_mode": self.gcssbm_transition_mode,
        }


# Compatibility alias for already-evaluated checkpoints. New code constructs
# GraphCBM directly and records base_method=trace.
ConceptForecastCBM = GraphCBM
