"""Vanishing-point CNN: crop geometry, labels, loss, and checkpoint round trip."""

import numpy as np
import pytest
import torch

from speed_lstm import vp_cnn
from speed_lstm.vp_cnn import CropGeom


def test_vp_crop_direction_round_trip():
    g = CropGeom.from_box((100, 200, 300, 260))
    for vp in (np.array([5000.0, -300.0, 1.0]), np.array([1.0, 0.1, 0.0]), np.array([150.0, 210.0, 1.0])):
        back = vp_cnn.crop_dir_to_vp(vp_cnn.vp_to_crop_dir(vp, g), g)
        assert np.allclose(np.cross(back / np.linalg.norm(back), vp / np.linalg.norm(vp)), 0.0, atol=1e-9)


def test_crop_coordinates_match_image_pixels():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(frame, (330, 210), 3, (255, 255, 255), -1)
    g = CropGeom.from_box((300, 180, 380, 240))           # side = 80 * 1.3 = 104, center (340, 210)
    crop = vp_cnn.crop_image(frame, g, size=104)          # 1 px per crop pixel
    ys, xs = np.nonzero(crop[..., 0] > 128)
    # Pixel (330, 210) sits 10 px left of the crop center at row center.
    assert xs.mean() + 0.5 == pytest.approx(104 / 2 - 10, abs=0.6)
    assert ys.mean() + 0.5 == pytest.approx(104 / 2, abs=0.6)


def test_flip_label_matches_mirrored_point():
    g = CropGeom(cx=0.0, cy=0.0, side=2.0)               # crop coords == image coords
    vp = np.array([0.4, -0.2, 1.0])
    mirrored = np.array([-0.4, -0.2, 1.0])
    assert np.allclose(vp_cnn.flip_dir(vp_cnn.vp_to_crop_dir(vp, g)), vp_cnn.vp_to_crop_dir(mirrored, g))


def test_loss_is_sign_invariant_and_zero_at_target():
    t = torch.nn.functional.normalize(torch.randn(8, 2, 3), dim=-1)
    assert vp_cnn.direction_loss(t, t).item() == pytest.approx(0.0, abs=1e-6)
    assert vp_cnn.direction_loss(-t, t).item() == pytest.approx(0.0, abs=1e-6)
    assert vp_cnn.direction_loss(torch.roll(t, 1, dims=0), t).item() > 0.01


def test_model_outputs_unit_directions_and_learns():
    torch.manual_seed(0)
    model = vp_cnn.VPNet(width=8)
    x = torch.rand(16, 3, 64, 64)
    target = torch.nn.functional.normalize(torch.tensor([[[0.3, -0.9, 0.2], [0.95, 0.1, 0.05]]]), dim=-1).repeat(16, 1, 1)
    out = model(x)
    assert out.shape == (16, 2, 3)
    assert torch.allclose(out.norm(dim=-1), torch.ones(16, 2), atol=1e-5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = vp_cnn.direction_loss(model(x), target).item()
    for _ in range(60):
        loss = vp_cnn.direction_loss(model(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_predictor_round_trip(tmp_path):
    model = vp_cnn.VPNet(width=8)
    path = tmp_path / "vp.pt"
    torch.save({"model": model.state_dict(), "width": 8, "crop_size": 64, "crop_pad": 0.15}, path)
    pred = vp_cnn.VPPredictor(str(path), device="cpu")
    crops = np.random.default_rng(0).integers(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    geoms = [CropGeom.from_box((10 * i, 20, 10 * i + 50, 60)) for i in range(5)]
    vp1, vp2 = pred.predict_vps(crops, geoms)
    assert vp1.shape == (5, 3) and vp2.shape == (5, 3)
    # With flip TTA each direction is the average of the crop and its mirror, mapped back consistently.
    d = pred.predict_dirs(crops)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)
