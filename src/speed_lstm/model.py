"""
Two-layer unidirectional LSTM encoder (hidden size 64, dropout 0.1 between
layers) with temporal attention pooling and a small regression head
(Linear 64->32, ReLU, Linear 32->1). Hidden/cell state starts at zero for
every window; nothing carries over between windows.

`Predictor` wraps a saved checkpoint for inference: it restores the
network, the saved feature/normalization metadata, computes v2 features
from raw timestamps + boxes, and converts the standardized prediction back
to physical speed (m/s), clamped at zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from speed_lstm import features_v2 as fv2

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_seq: (batch, T, hidden). mask: (batch, T), 1=valid, 0=padding.
        scores = self.score(hidden_seq).squeeze(-1)  # (batch, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * hidden_seq, dim=1)  # (batch, hidden)
        return context


class SpeedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_seq, _ = self.lstm(features)      # (batch, T, hidden)
        context = self.attention(hidden_seq, mask)
        return self.head(context).squeeze(-1)     # (batch,) standardized prediction


class Predictor:
    """Loads a checkpoint saved by speed_lstm.train and predicts physical speed (m/s) from raw inputs."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.mode = ckpt["mode"]
        self.feature_version = ckpt.get("feature_version", fv2.__name__)
        self.n_observations = ckpt["n_observations"]
        self.max_timestamp_gap = ckpt["max_timestamp_gap"]
        self.output_units = ckpt.get("output_units", "m/s")
        self.split_name = ckpt.get("split_name")
        self.seed = ckpt.get("seed")

        self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float64)
        self.feature_std = np.asarray(ckpt["feature_std"], dtype=np.float64)
        self.target_mean = float(ckpt["target_mean"])
        self.target_std = float(ckpt["target_std"])

        self.model = SpeedLSTM(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict(self, timestamps, boxes2d=None, boxes3d=None) -> float:
        """
        timestamps: (n_observations,) increasing seconds for ONE consistently tracked vehicle.
        boxes2d: (n_observations, 4) [xmin, ymin, xmax, ymax], required for mode in {2d, combined}.
        boxes3d: (n_observations, 6) [center_x, center_y, center_z, length, width, height] meters,
                 required for mode in {3d, combined}.

        The caller is responsible for supplying a single, consecutively tracked vehicle — this
        does not (and cannot, given only geometry + timestamps) verify vehicle identity.
        """
        t = np.asarray(timestamps, dtype=np.float64)
        if len(t) != self.n_observations:
            raise ValueError(f"Expected {self.n_observations} observations, got {len(t)}")

        box2d = np.asarray(boxes2d, dtype=np.float64) if boxes2d is not None else None
        center3d = dims3d = None
        if boxes3d is not None:
            boxes3d = np.asarray(boxes3d, dtype=np.float64)
            center3d, dims3d = boxes3d[:, :3], boxes3d[:, 3:6]

        feats = fv2.compute_features(self.mode, box2d, center3d, dims3d, t)  # (n_observations-1, F)
        feats = (feats - self.feature_mean) / self.feature_std

        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, T, F)
        with torch.no_grad():
            pred_std = self.model(x).item()

        pred = pred_std * self.target_std + self.target_mean
        return max(pred, 0.0)
