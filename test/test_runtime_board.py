"""
The LED-grid geometry can be changed while a live session is running.

A new square count changes the point grid the detector must find and the
solver indexes points by, so samples collected for the old grid must go. A
pitch change alone leaves the points where they are; only the world scale
changes, so the samples are kept and the user re-solves.
"""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uvdar_calibrator.apps.gui import LiveCalibrationApp  # noqa: E402
from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.calibrator import (  # noqa: E402
    Calibrator,
    CalibratorConfig,
    Sample,
)

BOARD = LedGridBoard(6, 4, 50.0)


class _StubConsumer:
    def __init__(self):
        self.capturing = threading.Event()
        self.capturing.set()
        self.raw_queue = queue.Queue()
        self.handle_lock = threading.Lock()
        self.paused_when_idle = None

    def wait_until_idle(self):
        self.paused_when_idle = self.capturing.is_set()


def _build_app(board=BOARD):
    root = tk.Tk()
    root.withdraw()
    calibrator = Calibrator(board, CalibratorConfig())
    calibrator.image_size = (960, 600)
    consumer = _StubConsumer()
    app = LiveCalibrationApp(
        root,
        calibrator=calibrator,
        result_queue=queue.Queue(),
        preview_queue=queue.Queue(),
        consumer=consumer,
        subscribe_fn=lambda topic: topic,
    )
    return root, app, calibrator, consumer


def _add_sample(calibrator, board):
    corners = np.zeros((board.n_cols * board.n_rows, 2), dtype=float)
    calibrator.db.append(Sample(
        params=[0.3, 0.3, 0.2, 0.0],
        image=np.zeros((600, 960), dtype=np.uint8),
        corners=corners,
        image_path="synthetic",
        metric=calibrator._build_metric(corners, "synthetic"),
    ))
    calibrator.calibrated = True


def test_fields_are_visible_and_editable_in_live():
    root, app, calibrator, _consumer = _build_app(LedGridBoard(5, 3, 25.0))
    try:
        for widget in app.board_option_widgets:
            assert widget.winfo_manager() == "pack", "the LED-grid fields are hidden"
            assert str(widget.cget("state")) != "disabled", (
                "live board fields must be editable"
            )
        assert app.n_sq_x.get() == 5 and app.n_sq_y.get() == 3
        assert abs(app.spacing_mm.get() - 25.0) < 1e-9
    finally:
        root.destroy()
    print("ok: the LED-grid fields are visible and editable in live mode")


def test_count_change_swaps_the_board_and_drops_old_samples():
    root, app, calibrator, consumer = _build_app()
    try:
        _add_sample(calibrator, BOARD)
        _add_sample(calibrator, BOARD)
        app._update_progress_panel()

        app.n_sq_x.set(4)
        app.n_sq_y.set(3)
        app.spacing_mm.set(40.0)
        app._on_board_fields_committed()

        assert calibrator.board == LedGridBoard(4, 3, 40.0), calibrator.board
        assert len(calibrator.db) == 0, "samples from the old grid survived"
        assert calibrator.calibrated is False
        assert calibrator.Xt.shape == (calibrator.board.n_points, 1), (
            "the world grid still describes the old board"
        )
        assert calibrator.Yt.shape == (calibrator.board.n_points, 1)
        assert calibrator.Xt[-1, 0] == 4 * 40.0
        assert calibrator.Yt[-1, 0] == 3 * 40.0
        assert consumer.paused_when_idle is False, (
            "capture was not paused before the board swap"
        )
        assert consumer.capturing.is_set(), "capture was not resumed"
        assert "capture a new set" in app.bottom_status.cget("text")
    finally:
        root.destroy()
    print("ok: a new square count swaps the board and drops the old samples")


def test_spacing_change_keeps_samples_but_invalidates_the_model():
    root, app, calibrator, _consumer = _build_app()
    try:
        _add_sample(calibrator, BOARD)
        app._update_progress_panel()

        app.spacing_mm.set(60.0)
        app._on_board_fields_committed()

        assert calibrator.board == LedGridBoard(6, 4, 60.0), calibrator.board
        assert len(calibrator.db) == 1, "the sample was dropped for a pitch change"
        assert calibrator.calibrated is False, "the old-scale model stayed valid"
        assert calibrator.Xt[-1, 0] == 6 * 60.0, "the world scale did not follow the pitch"
        assert calibrator.Yt[-1, 0] == 4 * 60.0
        assert str(app.export_button.cget("state")) == "disabled"
        assert str(app.plots_button.cget("state")) == "disabled"
        assert "samples kept" in app.bottom_status.cget("text")
    finally:
        root.destroy()
    print("ok: a pitch change keeps the samples and asks for a re-solve")


def test_board_swap_rebuilds_the_world_points_for_the_new_grid():
    cal = Calibrator(LedGridBoard(6, 4, 50.0), CalibratorConfig())
    cal.image_size = (960, 600)
    assert cal.Xt.shape == (35, 1)

    cal.board = LedGridBoard(7, 5, 50.0)
    assert cal.Xt.shape == (48, 1), "world points still describe the 6x4 grid"
    assert cal.Yt.shape == (48, 1)
    assert cal.calibrated is False

    cal.db.clear()
    _add_sample(cal, cal.board)
    Xp_abs, Yp_abs, ima_proc = cal.assemble()
    assert Xp_abs.shape == (48, 1, 1)
    assert Yp_abs.shape == (48, 1, 1)
    assert ima_proc == [1]
    print("ok: swapping the board rebuilds Xt/Yt for the new point grid")


def test_assemble_refuses_samples_from_another_grid():
    cal = Calibrator(LedGridBoard(7, 5, 50.0), CalibratorConfig())
    cal.image_size = (960, 600)
    _add_sample(cal, LedGridBoard(6, 4, 50.0))

    try:
        cal.assemble()
    except RuntimeError as exc:
        assert "geometry disagree" in str(exc), exc
    else:
        raise AssertionError("assemble accepted a sample from another grid")
    print("ok: assemble explains a sample/board grid mismatch")


def test_invalid_values_revert_to_the_running_board():
    root, app, calibrator, _consumer = _build_app()
    try:
        app.n_sq_x.set(0)
        app._on_board_fields_committed()
        assert calibrator.board == BOARD
        assert app.n_sq_x.get() == BOARD.n_sq_x

        app.spacing_mm.set(0.0)
        app._on_board_fields_committed()
        assert calibrator.board == BOARD
        assert abs(app.spacing_mm.get() - BOARD.spacing_mm) < 1e-9
    finally:
        root.destroy()
    print("ok: invalid grid values are rejected and the fields revert")


if __name__ == "__main__":
    try:
        tk.Tk().destroy()
    except tk.TclError as exc:
        print(f"ok: runtime board changes (skipped: no display -- {exc})")
        sys.exit(0)

    test_fields_are_visible_and_editable_in_live()
    test_count_change_swaps_the_board_and_drops_old_samples()
    test_spacing_change_keeps_samples_but_invalidates_the_model()
    test_board_swap_rebuilds_the_world_points_for_the_new_grid()
    test_assemble_refuses_samples_from_another_grid()
    test_invalid_values_revert_to_the_running_board()
    print("all checks passed")
