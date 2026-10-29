"""Checks for mirroring the incoming live frames.

Some camera drivers deliver a left-right flipped image. The live path can
flip such a frame back before it reaches the queues, so the preview, the
detection, the coverage advice and the saved model all describe the same
view.

The interesting part is not the flip itself but what it does to the board's
reading: the detector walks the grid in image order, so a mirrored image
reverses the board's x axis. The model that comes out is therefore the
mirror of the unmirrored one -- still a valid model of the (mirrored) stream,
which is what these checks pin down.

Run: python3 test/test_mirror.py
"""

from pathlib import Path
import contextlib
import io
import queue
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.calibrator import (  # noqa: E402
    Calibrator,
    CalibratorConfig,
    Sample,
)
from uvdar_calibrator.engine.detection import (  # noqa: E402
    find_image_files,
    get_corners,
    mirror_horizontal,
    read_image_gray,
)
from uvdar_calibrator.engine.ocam_model import (  # noqa: E402
    cam2world,
    DEFAULT_FORWARD_VIEW_HFOV_DEG,
)

IMAGE_DIR = ROOT / "example_images"
BOARD = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)


def test_mirror_horizontal_flips_and_copies():
    """The flip must be left-right, and must not touch the caller's array."""
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)

    mirrored = mirror_horizontal(img)

    assert np.array_equal(mirrored, img[:, ::-1]), mirrored
    # The live path hands the same array to the preview queue, the detection
    # queue and Sample.image. A flip that aliased the input would rewrite the
    # pixels of already-accepted samples while their recorded corners still
    # described the old orientation.
    assert not np.shares_memory(mirrored, img), "the mirror aliases its input"
    assert mirrored.flags["C_CONTIGUOUS"], "OpenCV needs a contiguous array"
    assert np.array_equal(img, np.arange(12, dtype=np.uint8).reshape(3, 4)), (
        "the input was modified in place"
    )
    print("ok: mirror_horizontal flips left-right into a fresh contiguous array")


def test_mirrored_frame_keeps_the_board_reading():
    """
    A mirrored frame must read as the same board, mirrored.

    The detector returns the points in image order, so a mirrored frame
    labels the board differently: the axis that runs horizontally in the
    photo is reversed. That is one of the four symmetries of the point grid
    (the board is photographed at different angles, so it is not always the
    same axis that runs horizontally), and the solver absorbs such a
    symmetry -- it is a rigid turn of the board in its own plane, which the
    per-image rotation R takes care of.

    What must not happen is a *transposed* reading: the board's 6x4 squares
    are not a 4x6 board, and a transposed labelling would not fit at all.
    """
    files = find_image_files(str(IMAGE_DIR))
    assert files, f"no example images in {IMAGE_DIR}"

    checked = 0
    for path in files:
        img = read_image_gray(path)
        ok, corners, _ = get_corners(img, BOARD)
        if not ok or corners is None or len(corners) != BOARD.n_cols * BOARD.n_rows:
            continue

        mirrored_img = mirror_horizontal(img)
        ok, mirrored_corners, _ = get_corners(mirrored_img, BOARD)
        assert ok and mirrored_corners is not None, f"{path}: no detection when mirrored"
        assert len(mirrored_corners) == len(corners)

        corners = np.asarray(corners, dtype=float)
        mirrored_corners = np.asarray(mirrored_corners, dtype=float)
        width = img.shape[1]

        # Every mirrored point must be the mirror of one of the original
        # points -- the same dots, seen from the other side.
        labelling = {}
        for k in range(len(mirrored_corners)):
            target = np.array([mirrored_corners[k][0], width - 1 - mirrored_corners[k][1]])
            distances = np.linalg.norm(corners - target, axis=1)
            source = int(np.argmin(distances))
            assert distances[source] < 0.05, (
                f"{path}: mirrored point {k} matches no original point"
            )
            labelling[divmod(k, BOARD.n_rows)] = divmod(source, BOARD.n_rows)

        symmetry = _grid_symmetry(labelling)
        assert symmetry is not None, (
            f"{path}: the mirrored reading is not a symmetry of the point grid"
        )
        checked += 1

    assert checked >= 5, f"only {checked} usable detections"
    print(f"ok: {checked} mirrored frames read as the same board, mirrored")


def _grid_symmetry(labelling):
    """
    Name the symmetry of the point grid that ``labelling`` follows.

    ``labelling`` maps a (x, y) board index of the mirrored reading to the
    (x, y) board index of the original reading. Returns None when the
    mapping is not one of the grid's four symmetries -- a transposed
    reading, which would be a different physical board.
    """
    n_cols, n_rows = BOARD.n_cols, BOARD.n_rows
    for name, flip_x, flip_y in (
        ("identity", False, False),
        ("x-flip", True, False),
        ("y-flip", False, True),
        ("x-flip and y-flip", True, True),
    ):
        if all(
            labelling[(x, y)] == (
                n_cols - 1 - x if flip_x else x,
                n_rows - 1 - y if flip_y else y,
            )
            for x in range(n_cols)
            for y in range(n_rows)
        ):
            return name
    return None


def test_mirrored_calibration_is_still_good():
    """
    Calibrating a mirrored stream must still produce a good model.

    The model of a mirrored stream is the mirror of the unmirrored one, so
    the reprojection error must stay small and the forward view must still
    straighten the grid (the same criterion test_forward_view.py uses).
    """
    files = find_image_files(str(IMAGE_DIR))
    assert files, f"no example images in {IMAGE_DIR}"

    def solve(mirrored):
        cal = Calibrator(BOARD, CalibratorConfig())
        for path in files:
            img = read_image_gray(path)
            cal.handle_frame(mirror_horizontal(img) if mirrored else img, path)
        with contextlib.redirect_stdout(io.StringIO()):
            cal.cal_fromcorners()
        return cal

    plain = solve(False).last_ocam_model
    calibrator = solve(True)
    model = calibrator.last_ocam_model
    assert model is not None, "the mirrored calibration produced no model"
    avg_error = float(np.nanmean(calibrator.reprojection_err))
    assert avg_error < 1.0, f"average reprojection error {avg_error:.3f} px"

    # The column centre moves with the mirror; the row centre does not.
    assert abs(model.xc - plain.xc) < 1.0, (model.xc, plain.xc)
    assert abs(model.yc - (960 - 1 - plain.yc)) < 1.0, (model.yc, plain.yc)

    rms = _forward_view_collinearity(calibrator, model)
    assert rms is not None, "no full detections to measure"
    assert rms < 0.5, f"forward view barely straightens the grid: {rms:.3f} px"
    print(
        f"ok: mirrored calibration: {len(calibrator.db)} samples, reprojection "
        f"{avg_error:.3f} px, forward view {rms:.3f} px"
    )


def _forward_view_collinearity(calibrator, model):
    """Mean per-line collinearity of the grid in the forward view, in px."""
    eps = 1.0
    probes = cam2world(
        np.array(
            [[model.xc, model.xc, model.xc + eps],
             [model.yc, model.yc + eps, model.yc]],
            dtype=float,
        ),
        model,
    )
    fwd = probes[:, 0] / np.linalg.norm(probes[:, 0])
    right = probes[:, 1] - fwd * (fwd @ probes[:, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    focal = 0.5 * model.width / np.tan(np.radians(DEFAULT_FORWARD_VIEW_HFOV_DEG) / 2.0)

    residuals = []
    for sample in calibrator.db:
        rows, cols = sample.corners[:, 0], sample.corners[:, 1]
        rays = cam2world(np.array([rows, cols]), model)
        along = fwd @ rays
        if not (along > 0).all():
            continue
        points = np.column_stack([(right @ rays) / along, (down @ rays) / along]) * focal
        grid = points.reshape(BOARD.n_cols, BOARD.n_rows, 2)
        for lines in (grid, grid.transpose(1, 0, 2)):
            for line in lines:
                centred = line - line.mean(axis=0)
                residuals.append(
                    np.linalg.svd(centred, compute_uv=False)[-1] / np.sqrt(len(line))
                )
    if not residuals:
        return None
    return float(np.mean(residuals))


class _StubConsumer:
    """The bits of FrameConsumerThread the GUI actually touches."""

    def __init__(self):
        self.capturing = threading.Event()
        self.capturing.set()
        self.raw_queue = queue.Queue()

    def wait_until_idle(self):
        pass


def test_live_toggle_switches_orientation_and_drops_old_samples():
    """Flipping mid-session must not leave two orientations in one solve."""
    try:
        import tkinter as tk
    except Exception:
        print("ok: live mirror toggle (skipped: no tkinter)")
        return

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ok: live mirror toggle (skipped: no display -- {exc})")
        return

    from uvdar_calibrator.apps.gui import LiveCalibrationApp

    root.withdraw()
    calibrator = Calibrator(BOARD, CalibratorConfig())
    calls = []
    app = LiveCalibrationApp(
        root,
        calibrator=calibrator,
        result_queue=queue.Queue(),
        preview_queue=queue.Queue(),
        consumer=_StubConsumer(),
        subscribe_fn=lambda topic: topic,
        set_mirror_fn=calls.append,
    )

    corners = np.zeros((BOARD.n_cols * BOARD.n_rows, 2), dtype=float)
    calibrator.image_size = (960, 600)
    calibrator.db.append(Sample(
        params=[0.3, 0.3, 0.2, 0.0],
        image=np.zeros((600, 960), dtype=np.uint8),
        corners=corners,
        image_path="synthetic",
        metric=calibrator._build_metric(corners, "synthetic"),
    ))
    calibrator.calibrated = True

    app.mirror.set(True)
    app._on_mirror_toggled()
    root.destroy()

    assert calls == [True], calls
    assert len(calibrator.db) == 0, "the old samples were kept"
    assert calibrator.calibrated is False
    print("ok: live mirror toggle flips the stream and drops the old samples")


if __name__ == "__main__":
    test_mirror_horizontal_flips_and_copies()
    test_mirrored_frame_keeps_the_board_reading()
    test_mirrored_calibration_is_still_good()
    test_live_toggle_switches_orientation_and_drops_old_samples()
    print("all checks passed")
