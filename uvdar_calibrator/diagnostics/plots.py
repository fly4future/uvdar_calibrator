"""
Matplotlib diagnostics for a completed calibration.

Every function here draws into an ``Axes``/``Figure`` handed to it and shows
nothing: :func:`build_diagnostic_figures` assembles the four diagnostics as
plain :class:`matplotlib.figure.Figure` objects, and
:mod:`uvdar_calibrator.diagnostics.plot_window` puts them on screen as tabs of
one window. Pyplot is deliberately not imported -- it is the global figure
manager, and going through it is what used to open one blocking window per
accepted sample (N + 3 windows to close by hand).

They operate on explicit arrays plus an
:class:`~uvdar_calibrator.engine.ocam_model.OCamModel` (no engine/GUI state).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from matplotlib.figure import Figure
import numpy as np

from ..engine.ocam_model import (
    idx,
    OCamModel,
    omni3d2pixel,
    reprojectpoints_adv,
    world2cam,
)

COLORS = "brgkcm"


def draw_axes(
    ax,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    n_sq_y: int,
) -> None:
    Xp_abs = np.asarray(Xp_abs).reshape(-1)
    Yp_abs = np.asarray(Yp_abs).reshape(-1)

    xo_X = Xp_abs[0:: n_sq_y + 1]
    yo_X = Yp_abs[0:: n_sq_y + 1]

    xo_Y = Xp_abs[: n_sq_y + 1]
    yo_Y = Yp_abs[: n_sq_y + 1]

    ax.plot(yo_X, xo_X, "g-", linewidth=2)
    ax.plot(yo_Y, xo_Y, "g-", linewidth=2)

    if len(xo_X) < 2 or len(xo_Y) < 2:
        return

    delta = 40.0

    uX = np.array(
        [
            xo_X[1] - xo_X[0],
            yo_X[1] - yo_X[0],
            0.0,
        ]
    )

    uY = np.array(
        [
            xo_Y[1] - xo_Y[0],
            yo_Y[1] - yo_Y[0],
            0.0,
        ]
    )

    origin = np.array(
        [
            xo_X[0],
            yo_X[0],
            0.0,
        ]
    )

    def normed(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    Xloc = normed(np.cross(uX, np.cross(uX, uY))) + normed(uX)
    Xloc = normed(Xloc) * delta + origin

    Yloc = normed(np.cross(np.cross(uX, uY), uY)) + normed(uY)
    Yloc = normed(Yloc) * delta + origin

    Oloc = normed(np.cross(np.cross(uX, uY), uY)) + normed(
        np.cross(uX, np.cross(uX, uY))
    )
    Oloc = normed(Oloc) * delta + origin

    ax.text(
        Xloc[1],
        Xloc[0],
        "X",
        color="g",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        Yloc[1],
        Yloc[0],
        "Y",
        color="g",
        fontsize=14,
        fontweight="bold",
        ha="center",
    )

    ax.text(
        Oloc[1],
        Oloc[0],
        "O",
        color="g",
        fontsize=14,
        fontweight="bold",
    )


def draw_reprojection(
    fig: Figure,
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    images: Optional[Sequence[np.ndarray]] = None,
    n_sq_y: int = 4,
) -> None:
    """Draw one panel per accepted sample into a single figure's subplot grid."""
    if ocam_model.ss is None:
        print("Need to calibrate before showing image reprojection.")
        return

    ima_proc = list(ima_proc)
    n = len(ima_proc)
    if n == 0:
        return

    # Squarish grid: with 23 samples the panels are small, which is what the
    # toolbar's zoom is for -- better than 23 windows.
    n_cols = int(math.ceil(math.sqrt(n)))
    n_rows = int(math.ceil(n / n_cols))

    for panel, kk in enumerate(ima_proc, start=1):
        k = idx(kk)
        ax = fig.add_subplot(n_rows, n_cols, panel)

        if images is not None and k < len(images):
            I = images[k]  # noqa: E741 -- MATLAB port keeps upstream's name
        else:
            I = 255 * np.ones(  # noqa: E741
                (
                    ocam_model.height,
                    ocam_model.width,
                )
            )

        xx = RRfin[:, :, k] @ np.vstack(
            [
                np.asarray(Xt).ravel(),
                np.asarray(Yt).ravel(),
                np.ones(np.asarray(Xt).size),
            ]
        )

        m = world2cam(xx, ocam_model)

        xp = m[0, :]
        yp = m[1, :]

        ax.imshow(I, cmap="gray")

        ax.set_title(f"Image {kk}", fontsize=9)

        ax.plot(
            Yp_abs[:, 0, k],
            Xp_abs[:, 0, k],
            "r+",
        )

        ax.plot(
            yp,
            xp,
            COLORS[(kk - 1) % 6] + "o",
            fillstyle="none",
        )

        ax.plot(
            ocam_model.yc,
            ocam_model.xc,
            "ro",
            fillstyle="none",
        )

        ax.axis(
            [
                1,
                ocam_model.width,
                ocam_model.height,
                1,
            ]
        )
        ax.set_xticks([])
        ax.set_yticks([])

        draw_axes(
            ax,
            Xp_abs[:, 0, k],
            Yp_abs[:, 0, k],
            n_sq_y,
        )

    fig.suptitle("Image points (+) and reprojected grid points (o)")


def draw_error_analysis(
    ax,
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
) -> None:
    err = []
    stderr = []
    MSE = 0.0

    if (
        ocam_model.c is None
        and ocam_model.d is None
        and ocam_model.e is None
    ):
        ocam_model.c = 1.0
        ocam_model.d = 0.0
        ocam_model.e = 0.0

    for i in ima_proc:
        k = idx(i)

        xx = RRfin[:, :, k] @ np.vstack(
            [
                np.asarray(Xt).ravel(),
                np.asarray(Yt).ravel(),
                np.ones(np.asarray(Xt).size),
            ]
        )

        xp1, yp1 = omni3d2pixel(
            ocam_model.ss,
            xx,
            ocam_model.width,
            ocam_model.height,
        )

        xp = (
            xp1 * ocam_model.c
            + yp1 * ocam_model.d
            + ocam_model.xc
        )

        yp = (
            xp1 * ocam_model.e
            + yp1
            + ocam_model.yc
        )

        sqerr = (Xp_abs[:, 0, k] - xp) ** 2 + (
            Yp_abs[:, 0, k] - yp
        ) ** 2

        err.append(float(np.nanmean(np.sqrt(sqerr))))
        stderr.append(float(np.nanstd(np.sqrt(sqerr))))

        MSE += float(np.nansum(sqerr))

        ax.plot(
            Xp_abs[:, 0, k] - xp,
            Yp_abs[:, 0, k] - yp,
            COLORS[(i - 1) % 6] + "+",
        )

    ax.grid(True)
    ax.set_title("Analyse error")
    ax.set_xlabel("X residual [pixels]")
    ax.set_ylabel("Y residual [pixels]")

    print("\nAverage reprojection error computed for each chessboard [pixels]:\n")

    for e, s in zip(err, stderr):
        print(f" {e:3.2f} ± {s:3.2f}")

    print(f"\nAverage error [pixels]\n\n {np.nanmean(err):f}")
    print(f"\nSum of squared errors\n\n {MSE:f}")

    print("ss =")
    print(ocam_model.ss)


def draw_projection_function(fig: Figure, ocam_model: OCamModel) -> None:
    ss = ocam_model.ss

    if ss is None:
        raise ValueError("Cannot plot calibration results because ocam_model.ss is empty.")

    rho = np.arange(
        0,
        int(np.floor(ocam_model.width / 2)) + 1,
        dtype=float,
    )

    f_rho = np.polyval(np.asarray(ss)[::-1], rho)

    angle_deg = np.degrees(np.arctan2(rho, -f_rho)) - 90.0

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(rho, f_rho)
    ax1.grid(True)
    ax1.axis("equal")
    ax1.set_xlabel("Distance 'rho' from the image center in pixels")
    ax1.set_ylabel("f(rho)")
    ax1.set_title("Forward projection function")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(rho, angle_deg)
    ax2.grid(True)
    ax2.set_xlabel("Distance 'rho' from the image center in pixels")
    ax2.set_ylabel("Degrees")
    ax2.set_title("Angle of optical ray as a function of distance from circle center (pixels)")


def print_calib_summary(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
) -> None:
    """Print the per-image reprojection table and the fitted model, no plots."""
    M = np.column_stack(
        [
            np.asarray(Xt).ravel(),
            np.asarray(Yt).ravel(),
            np.zeros(np.asarray(Xt).size),
        ]
    )

    reprojectpoints_adv(
        ocam_model,
        RRfin,
        ima_proc,
        Xp_abs,
        Yp_abs,
        M,
    )

    print("ss =")
    print(ocam_model.ss)

    print("xc =")
    print(ocam_model.xc)

    print("yc =")
    print(ocam_model.yc)


def draw_extrinsics(
    ax,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
) -> None:
    """Draw board points and per-sample camera translations; ax must be 3d."""
    ax.scatter(
        np.asarray(Xt).ravel(),
        np.asarray(Yt).ravel(),
        np.zeros(np.asarray(Xt).size),
        marker="o",
    )

    for i in ima_proc:
        k = idx(i)
        T = RRfin[:, 2, k]

        ax.scatter(T[0], T[1], T[2], marker="^")
        ax.text(T[0], T[1], T[2], str(i))

    ax.set_title("Extrinsic approximation from RRfin translations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def build_diagnostic_figures(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    images: Optional[Sequence[np.ndarray]] = None,
    n_sq_y: int = 4,
) -> List[Tuple[str, Figure]]:
    """
    Build the four diagnostics as ``(tab title, Figure)`` pairs.

    Nothing is shown; pass the result to
    :func:`uvdar_calibrator.diagnostics.plot_window.show_diagnostics`. The
    layout engine is set per figure rather than calling ``tight_layout`` here,
    because that needs a renderer these figures do not have until a canvas is
    attached.
    """
    figures: List[Tuple[str, Figure]] = []

    fig_reproj = Figure(figsize=(11, 8))
    draw_reprojection(
        fig_reproj, ocam_model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs,
        images=images, n_sq_y=n_sq_y,
    )
    figures.append(("Reprojection", fig_reproj))

    fig_err = Figure(figsize=(11, 8), layout="constrained")
    draw_error_analysis(
        fig_err.add_subplot(1, 1, 1),
        ocam_model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs,
    )
    figures.append(("Error analysis", fig_err))

    fig_proj = Figure(figsize=(11, 8), layout="constrained")
    draw_projection_function(fig_proj, ocam_model)
    figures.append(("Projection function", fig_proj))

    fig_ext = Figure(figsize=(11, 8), layout="constrained")
    draw_extrinsics(
        fig_ext.add_subplot(1, 1, 1, projection="3d"),
        RRfin, ima_proc, Xt, Yt,
    )
    figures.append(("Extrinsics", fig_ext))

    return figures
