import numpy as np

from speed_lstm import features_v2 as fv2


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.concatenate([[0], rng.uniform(0.02, 0.04, 15)]))

    box2d = np.zeros((16, 4))
    box2d[:, 0] = 100 + np.cumsum(rng.normal(2.0, 0.1, 16))
    box2d[:, 2] = box2d[:, 0] + 20 + rng.normal(0, 0.2, 16)
    box2d[:, 1] = 50 + np.cumsum(rng.normal(0.3, 0.1, 16))
    box2d[:, 3] = box2d[:, 1] + 10 + rng.normal(0, 0.2, 16)

    center3d = np.zeros((16, 3))
    center3d[:, 0] = np.cumsum(rng.normal(0.6, 0.05, 16))
    center3d[:, 1] = 3.0 + rng.normal(0, 0.05, 16)
    center3d[:, 2] = 0.75
    dims3d = np.tile([15.0, 5.7, 4.5], (16, 1)) + rng.normal(0, 0.01, (16, 3))

    return box2d, center3d, dims3d, t


def test_shapes():
    box2d, center3d, dims3d, t = _synthetic()
    assert fv2.compute_features_2d(box2d, t).shape == (15, 53)
    assert fv2.compute_features_3d(center3d, dims3d, t).shape == (15, 41)
    assert fv2.compute_features_combined(box2d, center3d, dims3d, t).shape == (15, 93)


def test_causal_no_future_leakage():
    box2d, center3d, dims3d, t = _synthetic()
    f2d = fv2.compute_features_2d(box2d, t)
    f3d = fv2.compute_features_3d(center3d, dims3d, t)

    box2d_mod = box2d.copy()
    box2d_mod[15, [0, 2]] += 500  # translate (not distort) only the LAST raw observation's box
    center3d_mod = center3d.copy()
    center3d_mod[15, 0] += 500
    dims3d_mod = dims3d.copy()
    dims3d_mod[15, 0] += 5

    f2d_mod = fv2.compute_features_2d(box2d_mod, t)
    f3d_mod = fv2.compute_features_3d(center3d_mod, dims3d_mod, t)

    # Output rows 0..13 correspond to raw observations 1..14 and must be
    # unaffected by a change to raw observation 15; only output row 14 may differ.
    assert np.allclose(f2d[:14], f2d_mod[:14])
    assert np.allclose(f3d[:14], f3d_mod[:14])
    assert not np.allclose(f2d[14], f2d_mod[14])
    assert not np.allclose(f3d[14], f3d_mod[14])


def test_translation_invariance_of_motion_features():
    box2d, center3d, dims3d, t = _synthetic()
    f3d = fv2.compute_features_3d(center3d, dims3d, t)

    shifted = center3d.copy()
    shifted[:, 0] += 1000.0  # shift the whole track by a constant in x
    f3d_shifted = fv2.compute_features_3d(shifted, dims3d, t)

    # Column layout: raw(7) rate(7) log(3) rel(3) vel(12) mask(4) accel(3) accel_mask(1) elapsed(1)
    # Everything except the raw-geometry block (first 7 cols, which include absolute position)
    # should be invariant to a constant translation.
    assert np.allclose(f3d[:, 7:], f3d_shifted[:, 7:])
    assert not np.allclose(f3d[:, 0], f3d_shifted[:, 0])  # raw center_x DID shift


def test_lag_velocity_masks_match_history_availability():
    box2d, center3d, dims3d, t = _synthetic()
    f3d = fv2.compute_features_3d(center3d, dims3d, t)
    # 3D column layout: raw(7)=0:7 rate(7)=7:14 log(3)=14:17 rel(3)=17:20
    # vel(12)=20:32 mask(4)=32:36 accel(3)=36:39 accel_mask(1)=39
    mask = f3d[:, 32:36]  # lags (1,3,5,10)
    assert np.all(mask[:, 0] == 1)                       # lag 1 always available
    assert np.all(mask[:2, 1] == 0) and np.all(mask[2:, 1] == 1)   # lag 3 needs j>=2
    assert np.all(mask[:4, 2] == 0) and np.all(mask[4:, 2] == 1)   # lag 5 needs j>=4
    assert np.all(mask[:9, 3] == 0) and np.all(mask[9:, 3] == 1)   # lag 10 needs j>=9


def test_acceleration_mask_only_invalid_at_first_step():
    box2d, center3d, dims3d, t = _synthetic()
    f3d = fv2.compute_features_3d(center3d, dims3d, t)
    accel_mask = f3d[:, 39]
    assert accel_mask[0] == 0
    assert np.all(accel_mask[1:] == 1)


def test_combined_drops_duplicate_elapsed_time():
    box2d, center3d, dims3d, t = _synthetic()
    f2d = fv2.compute_features_2d(box2d, t)
    f3d = fv2.compute_features_3d(center3d, dims3d, t)
    fc = fv2.compute_features_combined(box2d, center3d, dims3d, t)
    assert fc.shape[1] == f2d.shape[1] + f3d.shape[1] - 1
    assert np.allclose(fc[:, :53], f2d)               # 2D block (incl. its elapsed-time col) kept whole
    assert np.allclose(fc[:, 53:], f3d[:, :-1])         # 3D block minus its own (duplicate) elapsed-time col


def test_one_step_rate_matches_hand_calculation():
    t = np.array([0.0, 0.1, 0.25, 0.35] + [0.35 + 0.1 * i for i in range(1, 13)])
    center3d = np.zeros((16, 3))
    center3d[:, 0] = np.arange(16) * 2.0  # 2 m per raw step (not per second)
    dims3d = np.tile([15.0, 5.7, 4.5], (16, 1))
    f3d = fv2.compute_features_3d(center3d, dims3d, t)
    rate_x = f3d[:, 7]  # one-step rate of raw col 0 (center_x)
    expected = (center3d[1:, 0] - center3d[:-1, 0]) / (t[1:] - t[:-1])
    assert np.allclose(rate_x, expected)
