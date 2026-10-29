"""
The calibration target is selectable: UV LED grid (default) or checkerboard.

Each detector must find its own target and nothing else, in the same point
order, and switching targets in a live session must drop the old samples.
"""

import queue
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.detection import (  # noqa: E402
    find_image_files,
    get_corners,
    PATTERN_CHESSBOARD,
    PATTERN_LED_GRID,
    read_image_gray,
)

IMAGE_DIR = Path(__file__).resolve().parents[1] / "example_images"
BOARD = LedGridBoard(6, 4, 50.0)
SQUARE_PX = 50


def _checkerboard_image():
    """Render a board whose inner corners form BOARD's 7 x 5 point grid."""
    squares_x, squares_y = BOARD.n_cols + 1, BOARD.n_rows + 1
    board = np.zeros((squares_y * SQUARE_PX, squares_x * SQUARE_PX), np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                board[y * SQUARE_PX:(y + 1) * SQUARE_PX,
                      x * SQUARE_PX:(x + 1) * SQUARE_PX] = 255
    image = np.full((600, 960), 128, np.uint8)
    top, left = 100, 200
    image[top:top + board.shape[0], left:left + board.shape[1]] = board
    return image, (top, left)


def test_checkerboard_is_found_only_in_checkerboard_mode():
    image, (top, left) = _checkerboard_image()

    ok, corners, _ = get_corners(image, BOARD, PATTERN_CHESSBOARD)
    assert ok and corners.shape == (BOARD.n_points, 2), corners

    # Every inner corner sits on the square lattice, [row, col] order.
    rows = (corners[:, 0] - top) / SQUARE_PX
    cols = (corners[:, 1] - left) / SQUARE_PX
    assert np.allclose(rows, np.round(rows), atol=0.05), rows
    assert np.allclose(cols, np.round(cols), atol=0.05), cols
    # x-major/y-minor: consecutive points walk the short (y) side first.
    lattice = np.column_stack([np.round(rows), np.round(cols)])
    steps = np.abs(np.diff(lattice[:BOARD.n_rows], axis=0)).sum(axis=1)
    assert np.all(steps == 1), lattice[:BOARD.n_rows]

    ok, _, _ = get_corners(image, BOARD, PATTERN_LED_GRID)
    assert not ok, "the LED-grid detector accepted a checkerboard"
    print("ok: a checkerboard is found only in checkerboard mode, x-major")


def test_led_images_are_not_checkerboards():
    files = find_image_files(str(IMAGE_DIR))
    assert files, f"no example images in {IMAGE_DIR}"
    for path in files:
        ok, _, _ = get_corners(read_image_gray(path), BOARD, PATTERN_CHESSBOARD)
        assert not ok, f"{Path(path).name} detected as a checkerboard"
    print(f"ok: none of the {len(files)} LED images reads as a checkerboard")


def test_switching_the_target_drops_samples():
    import tkinter as tk

    from uvdar_calibrator.apps.gui import LiveCalibrationApp
    from uvdar_calibrator.engine.calibrator import Calibrator, CalibratorConfig, Sample

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ok: target switch (skipped: no display -- {exc})")
        return

    class _StubConsumer:
        def __init__(self):
            self.capturing = threading.Event()
            self.capturing.set()
            self.handle_lock = threading.Lock()

        def wait_until_idle(self):
            pass

    root.withdraw()
    calibrator = Calibrator(BOARD, CalibratorConfig())
    calibrator.image_size = (960, 600)
    try:
        app = LiveCalibrationApp(
            root,
            calibrator=calibrator,
            result_queue=queue.Queue(),
            preview_queue=queue.Queue(),
            consumer=_StubConsumer(),
            subscribe_fn=lambda topic: topic,
        )
        calibrator.db.append(Sample(
            params=[0.3, 0.3, 0.2, 0.0],
            image=np.zeros((600, 960), dtype=np.uint8),
            corners=np.zeros((BOARD.n_points, 2)),
            image_path="synthetic",
        ))

        app.pattern.set(PATTERN_CHESSBOARD)
        app._on_pattern_changed()
        assert calibrator.pattern == PATTERN_CHESSBOARD
        assert not calibrator.db, "samples of the old target survived"
        assert app.consumer.capturing.is_set(), "capture was not resumed"
    finally:
        root.destroy()
    print("ok: switching the target drops the old samples")


if __name__ == "__main__":
    test_checkerboard_is_found_only_in_checkerboard_mode()
    test_led_images_are_not_checkerboards()
    test_switching_the_target_drops_samples()
    print("all checks passed")
