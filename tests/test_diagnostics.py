import numpy as np

from speed_lstm.diagnostics import errors_by_speed_range, full_report, mae, rmse, speed_coverage


def test_mae_rmse_basic():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 5.0])
    assert mae(pred, target) == np.mean([0, 0, 2])
    assert rmse(pred, target) == np.sqrt(np.mean([0, 0, 4]))


def test_speed_coverage_sums_to_one():
    targets = np.array([1.0, 6.0, 11.0, 22.0, 40.0])
    coverage = speed_coverage(targets)
    assert abs(sum(v["fraction"] for v in coverage.values()) - 1.0) < 1e-9


def test_errors_by_speed_range_empty_bucket_is_none():
    pred = np.array([1.0, 2.0])
    target = np.array([1.0, 2.0])
    report = errors_by_speed_range(pred, target, ranges=[(0, 5), (25, 30)])
    assert report["[25,30)"]["count"] == 0
    assert report["[25,30)"]["mae"] is None


def test_full_report_shape():
    pred = np.array([10.0, 12.0, 21.0])
    target = np.array([11.0, 12.0, 20.0])
    report = full_report(pred, target)
    assert report["n"] == 3
    assert "speed_coverage" in report and "errors_by_speed_range" in report
