"""
Matplotlib plotting/diagnostics for a completed calibration.

All functions here are plotting-only and block on window close, as in the
original tool. They operate on explicit arrays plus an
:class:`~uvdar_calibrator.engine.ocam_model.OCamModel` (no engine/GUI state).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..engine.ocam_model import (
    _idx,
    OCamModel,
    omni3d2pixel,
    reprojectpoints_adv,
    world2cam,
)


def draw_axes(
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

    plt.plot(yo_X, xo_X, "g-", linewidth=2)
    plt.plot(yo_Y, xo_Y, "g-", linewidth=2)

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

    plt.text(
        Xloc[1],
        Xloc[0],
        "X",
        color="g",
        fontsize=14,
        fontweight="bold",
    )

    plt.text(
        Yloc[1],
        Yloc[0],
        "Y",
        color="g",
        fontsize=14,
        fontweight="bold",
        ha="center",
    )

    plt.text(
        Oloc[1],
        Oloc[0],
        "O",
        color="g",
        fontsize=14,
        fontweight="bold",
    )


def reproject_calib(
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
    colors = "brgkcm"

    if ocam_model.ss is None:
        print("Need to calibrate before showing image reprojection.")
        return

    for kk in ima_proc:
        k = _idx(kk)

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

        plt.figure(5 + kk)
        plt.clf()

        plt.imshow(I, cmap="gray")

        plt.title(f"Image {kk} - Image points (+) and reprojected grid points (o)")

        plt.plot(
            Yp_abs[:, 0, k],
            Xp_abs[:, 0, k],
            "r+",
        )

        plt.plot(
            yp,
            xp,
            colors[(kk - 1) % 6] + "o",
            fillstyle="none",
        )

        plt.plot(
            ocam_model.yc,
            ocam_model.xc,
            "ro",
            fillstyle="none",
        )

        plt.axis(
            [
                1,
                ocam_model.width,
                ocam_model.height,
                1,
            ]
        )

        draw_axes(
            Xp_abs[:, 0, k],
            Yp_abs[:, 0, k],
            n_sq_y,
        )

    plt.show()


def analyse_error(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
) -> None:
    plt.figure(5)
    plt.clf()

    colors = "brgkcm"

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
        k = _idx(i)

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

        plt.plot(
            Xp_abs[:, 0, k] - xp,
            Yp_abs[:, 0, k] - yp,
            colors[(i - 1) % 6] + "+",
        )

    plt.grid(True)
    plt.title("Analyse error")
    plt.xlabel("X residual [pixels]")
    plt.ylabel("Y residual [pixels]")

    print("\nAverage reprojection error computed for each chessboard [pixels]:\n")

    for e, s in zip(err, stderr):
        print(f" {e:3.2f} ± {s:3.2f}")

    print(f"\nAverage error [pixels]\n\n {np.nanmean(err):f}")
    print(f"\nSum of squared errors\n\n {MSE:f}")

    print("ss =")
    print(ocam_model.ss)

    plt.show()


def plot_calibration_results(ocam_model: OCamModel) -> None:
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

    plt.figure(3)
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(rho, f_rho)
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Distance 'rho' from the image center in pixels")
    plt.ylabel("f(rho)")
    plt.title("Forward projection function")

    plt.subplot(2, 1, 2)
    plt.plot(rho, angle_deg)
    plt.grid(True)
    plt.xlabel("Distance 'rho' from the image center in pixels")
    plt.ylabel("Degrees")
    plt.title("Angle of optical ray as a function of distance from circle center (pixels)")

    plt.tight_layout()
    plt.show()


def show_calib_results(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
) -> None:
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

    plot_calibration_results(ocam_model)


def show_extrinsic(
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        np.asarray(Xt).ravel(),
        np.asarray(Yt).ravel(),
        np.zeros(np.asarray(Xt).size),
        marker="o",
    )

    for i in ima_proc:
        k = _idx(i)
        T = RRfin[:, 2, k]

        ax.scatter(T[0], T[1], T[2], marker="^")
        ax.text(T[0], T[1], T[2], str(i))

    ax.set_title("Extrinsic approximation from RRfin translations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
