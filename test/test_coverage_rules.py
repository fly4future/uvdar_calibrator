"""Assert-based checks for sample-selection and readiness logic.

Run directly: python3 test/test_coverage_rules.py
No test framework on purpose -- these guard silent-wrong-answer logic (the
cached per-sample metric, and the goodenough rule) that a GUI cannot reveal
by eye.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from uvdar_calibrator.engine import coverage  # noqa: E402
from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.calibrator import Calibrator, CalibratorConfig  # noqa: E402


def _synthetic_corners(board, image_size, cx, cy, scale):
    """Build a flat x-major/y-minor [row, col] point grid centered at (cx, cy)."""
    width, height = image_size
    pts = []
    for x in range(board.n_cols):
        for y in range(board.n_rows):
            col = cx * width + (x - (board.n_cols - 1) / 2.0) * scale
            row = cy * height + (y - (board.n_rows - 1) / 2.0) * scale
            pts.append([row, col])
    return np.asarray(pts, dtype=float)


def test_cached_metric_matches_fresh_computation():
    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    cal = Calibrator(board, CalibratorConfig())
    cal.image_size = (960, 600)

    corners = _synthetic_corners(board, cal.image_size, 0.5, 0.5, 40.0)
    fresh = coverage.sample_metric(
        corners, board, cal.image_size, label="synthetic",
        valid_region=cal.valid_region_px(),
    )
    cached = cal._build_metric(corners, "synthetic")

    assert fresh is not None, "sample_metric returned None for a valid grid"
    for key in ("x", "y", "size", "skew"):
        assert abs(fresh[key] - cached[key]) < 1e-12, (
            f"cached metric[{key}]={cached[key]} != fresh {fresh[key]}"
        )
    print("ok: cached metric matches fresh sample_metric")


def test_db_metrics_tracks_accepted_samples():
    """db_metrics() must stay aligned with db, skipping unclassifiable views."""
    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    cal = Calibrator(board, CalibratorConfig())
    cal.image_size = (960, 600)

    assert cal.db_metrics() == [], "empty db should yield no metrics"

    from uvdar_calibrator.engine.calibrator import Sample
    for cx, cy in ((0.3, 0.3), (0.7, 0.6)):
        corners = _synthetic_corners(board, cal.image_size, cx, cy, 40.0)
        cal.db.append(Sample(
            params=[cx, cy, 0.2, 0.0],
            image=np.zeros((600, 960), dtype=np.uint8),
            corners=corners,
            image_path="synthetic",
            metric=cal._build_metric(corners, "synthetic"),
        ))

    metrics = cal.db_metrics()
    assert len(metrics) == 2, f"expected 2 metrics, got {len(metrics)}"
    assert metrics[0]["x"] < metrics[1]["x"], "metrics out of order"
    print("ok: db_metrics tracks accepted samples")


def test_goodenough_requires_both_count_and_range():
    ranges = (0.6, 0.6, 0.3, 0.45)

    # Full range coverage, but only 3 samples -> not ready.
    few = [[0.0, 0.0, 0.0, 0.0], [0.9, 0.9, 0.5, 0.6], [0.4, 0.4, 0.25, 0.3]]
    good, progress = coverage.compute_goodenough(few, ranges, min_db_size=20)
    assert all(row[3] >= 1.0 for row in progress), "expected full range"
    assert not good, "3 samples with full range must not be ready"

    # Many samples, but all identical -> no range -> not ready.
    many = [[0.5, 0.5, 0.2, 0.1]] * 40
    good, progress = coverage.compute_goodenough(many, ranges, min_db_size=20)
    assert not good, "40 identical samples must not be ready"

    # Both satisfied -> ready.
    both = few * 7  # 21 samples spanning the same full range
    good, _ = coverage.compute_goodenough(both, ranges, min_db_size=20)
    assert good, "21 samples with full range must be ready"
    print("ok: goodenough requires both sample count and range coverage")


def test_bins_do_not_gate_readiness():
    ranges = (0.6, 0.6, 0.3, 0.45)
    db = [[0.0, 0.0, 0.0, 0.0], [0.9, 0.9, 0.5, 0.6], [0.4, 0.4, 0.25, 0.3]] * 7

    plain, _ = coverage.compute_goodenough(db, ranges, min_db_size=20)
    # An empty metrics list means every spatial bin is missing. If bins gated,
    # this would flip the answer.
    binned, _, report = coverage.compute_goodenough_with_bins(
        db, [], ranges, min_db_size=20,
    )
    assert plain == binned, (
        f"bin report changed readiness: {plain} -> {binned}; bins must be hints only"
    )
    assert report["missing"]["quadrants"], "expected missing bins in this fixture"
    print("ok: bin report does not gate readiness")


if __name__ == "__main__":
    test_cached_metric_matches_fresh_computation()
    test_db_metrics_tracks_accepted_samples()
    test_goodenough_requires_both_count_and_range()
    test_bins_do_not_gate_readiness()
    print("all checks passed")
