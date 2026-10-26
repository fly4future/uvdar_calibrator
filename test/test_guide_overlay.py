"""Assert-based checks for the on-image "put the grid here" suggestion box.

Run directly: python3 test/test_guide_overlay.py
No test framework on purpose -- this guards a box that is only visible in the
live preview, so a wrong position would otherwise go unnoticed until someone
followed the advice on the drone.

The box is drawn from normalized coordinates that live in the same frame of
reference as coverage.get_parameters: the full image rectangle, or the
camera's usable (fisheye) circle when one is configured. These checks pin
the mapping so the box cannot silently drift away from the cell it names.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from uvdar_calibrator.engine import coverage  # noqa: E402
from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.apps.gui import guide_box_rect  # noqa: E402

IMAGE_SIZE = (960, 600)
GUIDE = {"x": 0.835, "y": 0.165, "size": 0.34, "text": "put the grid in the top-right"}


def _synthetic_corners(board, cx_px, cy_px, scale=40.0):
    """A flat x-major/y-minor [row, col] grid centered at (cx_px, cy_px)."""
    pts = []
    for x in range(board.n_cols):
        for y in range(board.n_rows):
            col = cx_px + (x - (board.n_cols - 1) / 2.0) * scale
            row = cy_px + (y - (board.n_rows - 1) / 2.0) * scale
            pts.append([row, col])
    return np.asarray(pts, dtype=float)


def test_normalized_to_pixel_inverts_sample_metric():
    """A sample's normalized position must map back to its pixel position."""
    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    corners = _synthetic_corners(board, cx_px=700.0, cy_px=150.0)

    for valid_region in (None, (480.0, 300.0, 270.0)):
        metric = coverage.sample_metric(
            corners, board, IMAGE_SIZE, label="synthetic",
            valid_region=valid_region,
        )
        px, py = coverage.normalized_to_pixel(
            metric["x"], metric["y"], IMAGE_SIZE, valid_region,
        )
        assert abs(px - 700.0) < 1e-6, f"x: {px} != 700"
        assert abs(py - 150.0) < 1e-6, f"y: {py} != 150"
    print("ok: normalized_to_pixel inverts sample_metric's normalization")


def test_full_frame_box_is_unchanged():
    """Without a valid region the box must land exactly where it always did."""
    left, top, right, bottom, cx, cy = guide_box_rect(GUIDE, IMAGE_SIZE, None)
    width, height = IMAGE_SIZE

    # The legacy geometry, verbatim.
    want_cx = int(round(GUIDE["x"] * width))
    want_cy = int(round(GUIDE["y"] * height))
    box = int(round(GUIDE["size"] * min(width, height) * 2.2))
    box = max(60, min(box, int(0.85 * min(width, height))))
    want = (
        max(0, want_cx - box // 2),
        max(0, want_cy - box // 2),
        min(width - 1, want_cx + box // 2),
        min(height - 1, want_cy + box // 2),
    )
    assert (left, top, right, bottom) == want, (
        (left, top, right, bottom), want
    )
    assert (cx, cy) == (want_cx, want_cy), (cx, cy)
    print("ok: full-frame box geometry is unchanged")


def test_circle_region_box_follows_the_cells():
    """With a valid circle the box must land where the coverage cell means."""
    width, height = IMAGE_SIZE
    valid_region = (480.0, 300.0, 270.0)

    left, top, right, bottom, cx, cy = guide_box_rect(GUIDE, IMAGE_SIZE, valid_region)

    expect_x = 480.0 - 270.0 + GUIDE["x"] * 2 * 270.0
    expect_y = 300.0 - 270.0 + GUIDE["y"] * 2 * 270.0
    assert abs(cx - expect_x) <= 1, f"cx {cx} != {expect_x}"
    assert abs(cy - expect_y) <= 1, f"cy {cy} != {expect_y}"

    # The box is sized against the usable circle, not the whole sensor.
    scale = coverage.normalized_length_scale(IMAGE_SIZE, valid_region)
    assert scale == 2 * 270.0, scale
    box_cap = int(0.85 * min(width, height))
    box = max(60, min(int(round(GUIDE["size"] * scale * 2.2)), box_cap))
    assert (right - left) <= box and (bottom - top) <= box, (right - left, box)

    # And that is where a sample classified into that cell actually sits.
    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    corners = _synthetic_corners(board, cx_px=expect_x, cy_px=expect_y)
    metric = coverage.sample_metric(
        corners, board, IMAGE_SIZE, label="synthetic", valid_region=valid_region,
    )
    assert metric["cell"] == "top-right", metric["cell"]
    print("ok: circle-normalized box lands in the cell it names")


def test_batch_app_draws_the_guide():
    """The folder app must show the box too, not just the live app."""
    try:
        import tkinter as tk
    except Exception:
        print("ok: batch app draws the guide (skipped: no tkinter)")
        return

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ok: batch app draws the guide (skipped: no display -- {exc})")
        return

    from uvdar_calibrator.apps.gui import BatchCalibrationApp
    from uvdar_calibrator.engine.calibrator import Calibrator, CalibratorConfig, Sample

    root.withdraw()
    app = BatchCalibrationApp(root, image_dir=".")
    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    cal = Calibrator(board, CalibratorConfig())
    cal.image_size = IMAGE_SIZE
    corners = _synthetic_corners(board, cx_px=300.0, cy_px=200.0)
    cal.db.append(Sample(
        params=[0.3, 0.3, 0.2, 0.0],
        image=np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8),
        corners=corners,
        image_path="synthetic",
        metric=cal._build_metric(corners, "synthetic"),
    ))
    app.calibrator = cal
    app._guide_db_len = -1

    calls = []
    app._render_frame = lambda *a, **kw: calls.append(kw)
    app._show_current_sample()
    root.destroy()

    assert len(calls) == 1, f"expected one render, got {len(calls)}"
    assert calls[0].get("show_guidance") is True, (
        "the folder app must draw the guide box"
    )
    print("ok: batch app draws the guide")


if __name__ == "__main__":
    test_normalized_to_pixel_inverts_sample_metric()
    test_full_frame_box_is_unchanged()
    test_circle_region_box_follows_the_cells()
    test_batch_app_draws_the_guide()
    print("all checks passed")
