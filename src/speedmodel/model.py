"""
Small sequence regressor: per-frame invariant features -> single speed value.

Architecture (mirrors the paper this design was benchmarked against, with
attention properly masked so padded time steps don't influence the result):

    (B, T, N_FEATURES)
        -> Conv1d embed (temporal, kernel=3)
        -> GRU or LSTM
        -> masked additive attention over time
        -> Linear -> (B,) speed
"""

from __future__ import annotations

import torch
import torch.nn as nn

from speedmodel.features import N_FEATURES


class MaskedAttention(nn.Module):
    """Additive attention over the time dimension, ignoring padded frames."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, H), mask: (B, T) with 1.0 = real frame, 0.0 = padding
        scores = self.score(seq).squeeze(-1)              # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)             # (B, T)
        weights = torch.nan_to_num(weights, nan=0.0)        # guards fully-empty tracks
        context = torch.sum(seq * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context


class SpeedRegressor(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.1,
                 rnn_type: str = "gru"):
        super().__init__()

        self.embed = nn.Conv1d(N_FEATURES, hidden_size, kernel_size=3, padding=1)
        self.embed_act = nn.ReLU()

        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = MaskedAttention(rnn_out_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_out_size, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (B, T, N_FEATURES), mask: (B, T)
        x = features.transpose(1, 2)          # (B, N_FEATURES, T) for Conv1d
        x = self.embed_act(self.embed(x))
        x = x.transpose(1, 2)                  # back to (B, T, H)

        rnn_out, _ = self.rnn(x)               # (B, T, rnn_out_size)
        context = self.attention(rnn_out, mask)
        context = self.dropout(context)

        speed = self.head(context).squeeze(-1)  # (B,)
        return speed
