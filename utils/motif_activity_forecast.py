"""Lightweight MoTIF-style temporal model for activity classification and forecasting."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_valid_mean_std(concepts: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-concept statistics from valid train windows only."""

    valid = mask.astype(bool)
    flattened = concepts[valid]
    mean = flattened.mean(axis=0).astype(np.float32)
    std = flattened.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def standardize_concepts(
    concepts: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Standardize concepts and zero padded windows."""

    standardized = (concepts - mean[None, None, :]) / std[None, None, :]
    return (standardized * mask[:, :, None]).astype(np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            div_term_cos = torch.exp(
                torch.arange(0, d_model - 1, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 1::2] = torch.cos(position * div_term_cos)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class DiagQKVd(nn.Module):
    def __init__(self, channels: int, width: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.q = nn.Conv1d(channels, channels * width, 1, groups=channels, bias=bias)
        self.k = nn.Conv1d(channels, channels * width, 1, groups=channels, bias=bias)
        self.v = nn.Conv1d(channels, channels * width, 1, groups=channels, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, timesteps, channels = x.shape
        x_channel_first = x.transpose(1, 2)
        q = self.q(x_channel_first).transpose(1, 2).view(batch, timesteps, channels, self.width)
        k = self.k(x_channel_first).transpose(1, 2).view(batch, timesteps, channels, self.width)
        v = self.v(x_channel_first).transpose(1, 2).view(batch, timesteps, channels, self.width)
        return q, k, v


class ChannelTimeNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class PerChannelFFN(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channel_first = x.transpose(1, 2)
        y = self.fc2(self.dropout(self.activation(self.fc1(x_channel_first))))
        return y.transpose(1, 2)


class PerChannelTemporalBlock(nn.Module):
    """Per-concept temporal attention with strict causal masking support."""

    def __init__(self, channels: int, width: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.qkv = DiagQKVd(channels, width)
        self.scale = width ** -0.5
        self.norm1 = ChannelTimeNorm(channels)
        self.norm2 = ChannelTimeNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PerChannelFFN(channels, dropout=dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, timesteps, _ = x.shape
        y = self.norm1(x)
        q, k, v = self.qkv(y)
        scores = torch.einsum("btcd,bucd->bctu", q, k) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.view(1, 1, timesteps, timesteps), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(batch, 1, 1, timesteps), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.attn_weights = weights.detach()
        out = torch.einsum("bctu,bucd->btcd", weights, v).mean(dim=-1)
        x = x + self.dropout(out)
        z = self.ffn(self.norm2(x))
        return x + self.dropout(z)


class PerConceptAffine(nn.Module):
    def __init__(self, num_concepts: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_concepts))
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.softplus(x * self.scale + self.bias) - math.log(2.0)
        return y.clamp(min=0.0)


def build_causal_attn_mask(timesteps: int, device: torch.device) -> torch.Tensor:
    """Return a mask with True entries where future attention should be blocked."""

    return torch.triu(torch.ones((timesteps, timesteps), dtype=torch.bool, device=device), diagonal=1)


def causal_window_mean_torch(states: torch.Tensor, valid_mask: torch.Tensor, window_size: int) -> torch.Tensor:
    """Compute a causal running mean over the last `window_size` valid windows."""

    masked = states * valid_mask.unsqueeze(-1)
    cumsum_states = torch.cumsum(masked, dim=1)
    cumsum_mask = torch.cumsum(valid_mask.unsqueeze(-1), dim=1)
    padded_states = torch.cat([torch.zeros_like(cumsum_states[:, :1]), cumsum_states], dim=1)
    padded_mask = torch.cat([torch.zeros_like(cumsum_mask[:, :1]), cumsum_mask], dim=1)
    timesteps = states.size(1)
    device = states.device
    end_idx = torch.arange(1, timesteps + 1, device=device)
    start_idx = torch.clamp(end_idx - window_size, min=0)
    window_sum = padded_states[:, end_idx, :] - padded_states[:, start_idx, :]
    window_count = padded_mask[:, end_idx, :] - padded_mask[:, start_idx, :]
    return window_sum / torch.clamp(window_count, min=1.0)


class MotifActivityForecastModel(nn.Module):
    """Causal MoTIF variant for per-window activity classification and forecasting."""

    def __init__(
        self,
        num_concepts: int,
        num_activities: int,
        history_length: int,
        transformer_layers: int = 1,
        dropout: float = 0.1,
        dimension: int = 1,
        max_sequence_length: int = 2000,
        shared_activity_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.num_activities = num_activities
        self.history_length = history_length
        self.max_sequence_length = max_sequence_length
        self.shared_activity_head = bool(shared_activity_head)
        self.posenc = PositionalEncoding(num_concepts, dropout=dropout, max_len=max_sequence_length)
        self.layers = nn.ModuleList(
            [PerChannelTemporalBlock(num_concepts, width=dimension, dropout=dropout) for _ in range(transformer_layers)]
        )
        self.norm = nn.LayerNorm(num_concepts)
        self.concept_predictor = PerConceptAffine(num_concepts)
        self.activity_head = nn.Linear(num_concepts * 2, num_activities)
        self.forecast_head = (
            self.activity_head
            if self.shared_activity_head
            else nn.Linear(num_concepts * 2, num_activities)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        causal_mask = build_causal_attn_mask(x.size(1), x.device)
        valid_mask = (~key_padding_mask).float()
        x = self.posenc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        backbone_t = self.norm(x)
        concepts_t = self.concept_predictor(backbone_t)

        activity_repr = torch.cat([backbone_t, concepts_t], dim=-1)
        history_backbone = causal_window_mean_torch(backbone_t, valid_mask, self.history_length)
        history_concepts = causal_window_mean_torch(concepts_t, valid_mask, self.history_length)
        forecast_repr = torch.cat([history_backbone, history_concepts], dim=-1)

        activity_logits = self.activity_head(activity_repr)
        forecast_logits = self.forecast_head(forecast_repr)
        return {
            "backbone_t": backbone_t,
            "concepts_t": concepts_t,
            "activity_repr": activity_repr,
            "forecast_repr": forecast_repr,
            "activity_logits": activity_logits,
            "forecast_logits": forecast_logits,
        }
