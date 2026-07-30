"""Check the forward view really undistorts, and stays inside the image.

The forward view exists to eyeball whether a calibration is any good: the LED
grid's rows and columns are straight in the world, the fisheye bows them, and
a correct rectilinear remap must straighten them again. That collinearity
residual is the test -- it needs no ground truth beyond "the board is planar
and regular", and it is independent of how wide the view is framed, since FOV
only scales and crops the projection plane.

Uses the real calibrated rig and the real detected points, so the thresholds
mean something.

Run: python3 test/test_forward_view.py
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uvdar_calibrator.engine.ocam_model import (  # noqa: E402
    build_forward_view_maps,
    cam2world,
    DEFAULT_FORWARD_VIEW_HFOV_DEG,
    OCamModel,
)

# The real UVDAR rig, from calibrating example_images/ (calib_results.txt).
MODEL = OCamModel(
    xc=272.7,
    yc=515.448,
    width=960,
    height=600,
    ss=np.array([-4.707577e02, 0.0, 1.726291e-03, -3.575386e-06, 7.742914e-09]),
)

POINT_DUMPS = ROOT / "example_images" / "detected_marker_previews"
N_COLS, N_ROWS = 7, 5  # the 6x4-square board's point grid


def _view_basis(model):
    """The (fwd, right, down) frame build_forward_view_maps projects onto."""
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
    return fwd, right, np.cross(fwd, right)


def _collinearity_rms(points):
    """Mean RMS perpendicular deviation of each board row/column from a line."""
    grid = points.reshape(N_COLS, N_ROWS, 2)  # x-major, y-minor as generated
    residuals = []
    for lines in (grid, grid.transpose(1, 0, 2)):
        for line in lines:
            centred = line - line.mean(axis=0)
            # Smallest singular value is the spread about the best-fit line.
            residuals.append(
                np.linalg.svd(centred, compute_uv=False)[-1] / np.sqrt(len(line))
            )
    return float(np.mean(residuals))


def test_forward_view_straightens_the_grid():
    dumps = sorted(POINT_DUMPS.glob("*_points.txt"))
    assert dumps, f"no detected-point dumps in {POINT_DUMPS}"

    fwd, right, down = _view_basis(MODEL)
    # Pixels per unit of the projection plane, so both numbers are in px.
    focal = 0.5 * MODEL.width / np.tan(np.radians(DEFAULT_FORWARD_VIEW_HFOV_DEG) / 2.0)

    raw_rms, rect_rms = [], []
    for dump in dumps:
        data = np.loadtxt(dump, delimiter=",", skiprows=1)
        if len(data) != N_COLS * N_ROWS:
            continue  # a partial detection tells us nothing about straightness
        rows, cols = data[:, 1], data[:, 2]

        rays = cam2world(np.array([rows, cols]), MODEL)
        along = fwd @ rays
        assert (along > 0).all(), "board points must be in front of the camera"

        raw_rms.append(_collinearity_rms(np.column_stack([cols, rows])))
        rect_rms.append(
            _collinearity_rms(
                np.column_stack([(right @ rays) / along, (down @ rays) / along]) * focal
            )
        )

    assert len(raw_rms) >= 20, f"only {len(raw_rms)} full detections"
    raw, rect = float(np.mean(raw_rms)), float(np.mean(rect_rms))
    assert rect < raw / 3.0, (
        f"forward view barely straightens the grid: {raw:.3f} px raw "
        f"vs {rect:.3f} px undistorted"
    )

    print(
        f"ok: {len(raw_rms)} images, per-line collinearity "
        f"{raw:.3f} px raw -> {rect:.3f} px undistorted ({raw / rect:.1f}x straighter)"
    )


def test_maps_stay_inside_the_source_image():
    """Every non-sentinel coordinate must be a real pixel."""
    map_x, map_y = build_forward_view_maps(MODEL)
    ok = (map_x >= 0) & (map_y >= 0)
    assert ok.any(), "the LUT is entirely invalid"

    # Out-of-image coords would make remap sample the border colour, which
    # looks like real image content instead of the blank it should be.
    assert map_x[ok].max() < MODEL.width, map_x[ok].max()
    assert map_y[ok].max() < MODEL.height, map_y[ok].max()

    print(f"ok: LUT covers {ok.mean():.0%} of the view, all coords in bounds")


if __name__ == "__main__":
    test_forward_view_straightens_the_grid()
    test_maps_stay_inside_the_source_image()
    print("all checks passed")
