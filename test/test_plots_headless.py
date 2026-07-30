"""Check the diagnostics build as four bare figures, no windows.

The plot functions were converted off pyplot's global figure manager onto
explicit Axes so they could share one window. If any of them slips back to
`plt.*`, it reopens its own window at calibration time -- which is invisible
in a headless check unless the check looks for it, so this one does.

Run: python3 test/test_plots_headless.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uvdar_calibrator.diagnostics import plots  # noqa: E402
from uvdar_calibrator.engine.ocam_model import OCamModel  # noqa: E402

EXPECTED_TABS = ["Reprojection", "Error analysis", "Projection function", "Extrinsics"]

N_SQ_X, N_SQ_Y = 6, 4
N_POINTS = (N_SQ_X + 1) * (N_SQ_Y + 1)
N_IMAGES = 3


def _synthetic_inputs():
    """A tiny stand-in for a solved calibration: 3 views of a 7x5 grid."""
    model = OCamModel(
        xc=240.0,
        yc=320.0,
        width=640,
        height=480,
        ss=np.array([-8.457e01, 0.0, 6.181e-04, -1.079e-06, 3.907e-09]),
    )

    # Board points, x-major / y-minor as the engine generates them.
    xs, ys = np.meshgrid(np.arange(N_SQ_X + 1), np.arange(N_SQ_Y + 1), indexing="ij")
    Xt = (xs.ravel() * 50.0).reshape(-1, 1)
    Yt = (ys.ravel() * 50.0).reshape(-1, 1)

    # RRfin[:, :, k] is a 3x3 whose third column is the translation.
    RRfin = np.zeros((3, 3, N_IMAGES))
    for k in range(N_IMAGES):
        RRfin[:, :, k] = np.array(
            [
                [1.0, 0.0, 20.0 * k],
                [0.0, 1.0, 10.0 * k],
                [0.0, 0.0, 800.0 + 50.0 * k],
            ]
        )

    rng = np.random.default_rng(0)
    Xp_abs = rng.uniform(50, 430, size=(N_POINTS, 1, N_IMAGES))
    Yp_abs = rng.uniform(50, 590, size=(N_POINTS, 1, N_IMAGES))

    ima_proc = list(range(1, N_IMAGES + 1))  # 1-based, MATLAB convention
    images = [np.zeros((480, 640), dtype=np.uint8) for _ in range(N_IMAGES)]

    return model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs, images


def test_build_diagnostic_figures():
    import matplotlib.pyplot as plt

    model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs, images = _synthetic_inputs()

    figures = plots.build_diagnostic_figures(
        model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs,
        images=images, n_sq_y=N_SQ_Y,
    )

    assert [title for title, _ in figures] == EXPECTED_TABS, [t for t, _ in figures]

    for title, fig in figures:
        assert fig.axes, f"{title!r} figure has no axes"
        for ax in fig.axes:
            artists = ax.lines + ax.collections + ax.images + ax.texts
            assert artists, f"{title!r} has an empty axes"

    # The reprojection tab must hold one panel per sample in one figure -- the
    # whole point of the refactor.
    reproj = dict(figures)["Reprojection"]
    assert len(reproj.axes) == N_IMAGES, len(reproj.axes)

    # No pyplot figure anywhere: a stray plt.figure() is exactly the
    # regression that would put windows back on screen.
    assert plt.get_fignums() == [], plt.get_fignums()

    print(f"ok: {len(figures)} figures ({', '.join(t for t, _ in figures)}), 0 pyplot windows")


if __name__ == "__main__":
    test_build_diagnostic_figures()
    print("all checks passed")
