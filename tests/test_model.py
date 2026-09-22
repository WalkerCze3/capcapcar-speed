import numpy as np
import torch

from speed_lstm import features_v2 as fv2
from speed_lstm.model import Predictor, SpeedLSTM


def test_forward_shape():
    model = SpeedLSTM(input_size=41)
    x = torch.randn(4, 15, 41)
    out = model(x)
    assert out.shape == (4,)


def test_checkpoint_roundtrip_predictor(tmp_path):
    mode = "3d"
    input_size = fv2.n_features(mode)
    model = SpeedLSTM(input_size=input_size)

    feature_mean = np.zeros(input_size)
    feature_std = np.ones(input_size)
    ckpt = {
        "model_state": model.state_dict(),
        "mode": mode,
        "input_size": input_size,
        "hidden_size": model.hidden_size,
        "feature_version": "v2",
        "n_observations": 16,
        "max_timestamp_gap": 0.2,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": 20.0,
        "target_std": 5.0,
        "output_units": "m/s",
        "seed": 42,
        "split_name": "speed_balanced",
    }
    ckpt_path = tmp_path / "best.pt"
    torch.save(ckpt, ckpt_path)

    predictor = Predictor(ckpt_path)
    t = np.cumsum(np.full(16, 0.033))
    center3d = np.zeros((16, 3))
    center3d[:, 0] = np.arange(16) * 0.6
    center3d[:, 1] = 3.0
    center3d[:, 2] = 0.75
    dims3d = np.tile([15.0, 5.7, 4.5], (16, 1))
    boxes3d = np.concatenate([center3d, dims3d], axis=1)

    speed = predictor.predict(timestamps=t, boxes3d=boxes3d)
    assert isinstance(speed, float)
    assert speed >= 0.0  # clamped


def test_predict_clamps_negative_to_zero(tmp_path):
    mode = "3d"
    input_size = fv2.n_features(mode)
    model = SpeedLSTM(input_size=input_size)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head[-1].bias.fill_(-1.0)  # force a very negative standardized prediction

    ckpt = {
        "model_state": model.state_dict(), "mode": mode, "input_size": input_size,
        "hidden_size": model.hidden_size, "feature_version": "v2", "n_observations": 16,
        "max_timestamp_gap": 0.2, "feature_mean": np.zeros(input_size), "feature_std": np.ones(input_size),
        "target_mean": 0.0, "target_std": 1.0, "output_units": "m/s", "seed": 42, "split_name": "x",
    }
    ckpt_path = tmp_path / "best.pt"
    torch.save(ckpt, ckpt_path)
    predictor = Predictor(ckpt_path)

    t = np.cumsum(np.full(16, 0.033))
    center3d = np.tile([0.0, 3.0, 0.75], (16, 1))
    dims3d = np.tile([15.0, 5.7, 4.5], (16, 1))
    boxes3d = np.concatenate([center3d, dims3d], axis=1)
    speed = predictor.predict(timestamps=t, boxes3d=boxes3d)
    assert speed == 0.0
