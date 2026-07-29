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


if __name__ == "__main__":
    test_cached_metric_matches_fresh_computation()
    test_db_metrics_tracks_accepted_samples()
    print("all checks passed")
