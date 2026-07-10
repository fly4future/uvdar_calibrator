"""
OCamCalib (Scaramuzza) omnidirectional camera model and solver math.

Faithful, minimally refactored port of the MATLAB toolbox
ctu-mrs/OCamCalib_UVDAR (itself a UV-LED-grid fork of Scaramuzza's
OCamCalib). This module is pure math over ``(Xt, Yt, Xp_abs, Yp_abs,
xc, yc, ...)`` arrays -- it has no knowledge of "samples", "goodness",
or GUI state, exactly like ``cv2.calibrateCamera`` is pure math with no
knowledge of why those particular corners were chosen.

Conventions (load-bearing -- do not "fix"):

- ``ima_proc`` holds MATLAB-style **1-based** image numbers; ``_idx()``
  converts to a 0-based Python/numpy index wherever an array is touched.
- Detected marker points are stored ``[row, col]`` (``Xp_abs`` = row,
  ``Yp_abs`` = col) -- the reverse of OpenCV's usual ``[x, y]``.
- Board points ``Xt``/``Yt`` are generated x-major, y-minor (outer loop
  over x, inner loop over y).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------

@dataclass
class OCamModel:
    xc: float
    yc: float
    width: int
    height: int
    c: float = 1.0
    d: float = 0.0
    e: float = 0.0
    ss: Optional[np.ndarray] = None
    invpol: Optional[np.ndarray] = None


def _idx(i: int) -> int:
    """Convert MATLAB-style 1-based image number to Python index."""
    return int(i) - 1


def _as_col(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


# -----------------------------------------------------------------------------
# Projection functions
# -----------------------------------------------------------------------------

def omni3d2pixel(
    ss: Sequence[float],
    xx: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(xx, dtype=float).copy()
    ss = np.asarray(ss, dtype=float).ravel()

    ind0 = np.where((xx[0, :] == 0) & (xx[1, :] == 0))[0]

    if ind0.size:
        xx[0, ind0] = np.finfo(float).eps
        xx[1, ind0] = np.finfo(float).eps

    denom = np.sqrt(xx[0, :] ** 2 + xx[1, :] ** 2)
    m = xx[2, :] / denom

    poly_coef = ss[::-1].copy()

    rho = np.full_like(m, np.nan, dtype=float)

    for j, mj in enumerate(m):
        poly_tmp = poly_coef.copy()
        poly_tmp[-2] = poly_coef[-2] - mj

        roots = np.roots(poly_tmp)

        real_pos = roots[
            (np.abs(np.imag(roots)) < 1e-9)
            & (np.real(roots) > 0)
        ]
        real_pos = np.real(real_pos)

        if real_pos.size == 1:
            rho[j] = real_pos[0]
        elif real_pos.size > 1:
            rho[j] = np.min(real_pos)

    x = xx[0, :] / denom * rho
    y = xx[1, :] / denom * rho

    return x, y


def world2cam(M: np.ndarray, ocam_model: OCamModel) -> np.ndarray:
    x, y = omni3d2pixel(
        ocam_model.ss,
        M,
        ocam_model.width,
        ocam_model.height,
    )

    m = np.empty((2, len(x)), dtype=float)

    m[0, :] = x * ocam_model.c + y * ocam_model.d + ocam_model.xc
    m[1, :] = x * ocam_model.e + y + ocam_model.yc

    return m


def getpoint(ss: Sequence[float], m: np.ndarray) -> np.ndarray:
    ss = np.asarray(ss, dtype=float).ravel()
    rho = np.sqrt(m[0, :] ** 2 + m[1, :] ** 2)

    return np.vstack(
        [
            m[0, :],
            m[1, :],
            np.polyval(ss[::-1], rho),
        ]
    )


def cam2world(m: np.ndarray, ocam_model: OCamModel) -> np.ndarray:
    m = np.asarray(m, dtype=float)

    if m.ndim == 1:
        m = m.reshape(2, 1)

    n_points = m.shape[1]

    A = np.array(
        [
            [ocam_model.c, ocam_model.d],
            [ocam_model.e, 1.0],
        ],
        dtype=float,
    )

    T = np.array(
        [
            [ocam_model.xc],
            [ocam_model.yc],
        ],
        dtype=float,
    ) @ np.ones((1, n_points))

    mp = np.linalg.inv(A) @ (m - T)

    M = getpoint(ocam_model.ss, mp)

    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0

    return M / norms


# -----------------------------------------------------------------------------
# Calibration solver
# -----------------------------------------------------------------------------

def plot_RR(
    RR: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xpt: np.ndarray,
    Ypt: np.ndarray,
    figure_number: int = 0,
) -> int:
    selected = 0

    for i in range(RR.shape[2]):
        RRdef = RR[:, :, i]

        R11, R21, R31 = RRdef[0, 0], RRdef[1, 0], RRdef[2, 0]
        R12, R22, R32 = RRdef[0, 1], RRdef[1, 1], RRdef[2, 1]
        T1, T2 = RRdef[0, 2], RRdef[1, 2]

        MA = R21 * Xt + R22 * Yt + T2
        MB = Ypt * (R31 * Xt + R32 * Yt)

        MC = R11 * Xt + R12 * Yt + T1
        MD = Xpt * (R31 * Xt + R32 * Yt)

        rho = np.sqrt(Xpt ** 2 + Ypt ** 2)
        rho2 = Xpt ** 2 + Ypt ** 2

        PP1 = np.vstack(
            [
                np.hstack([MA, MA * rho, MA * rho2]),
                np.hstack([MC, MC * rho, MC * rho2]),
            ]
        )

        PP = np.hstack(
            [
                PP1,
                np.vstack([-Ypt, -Xpt]),
            ]
        )

        QQ = np.vstack([MB, MD])

        s = np.linalg.pinv(PP) @ QQ
        ss = s[:3].ravel()

        if figure_number > 0:
            x = np.arange(0, 621)
            plt.figure(figure_number)
            plt.subplot(1, RR.shape[2], i + 1)
            plt.plot(x, np.polyval(ss[::-1], x))
            plt.grid(True)
            plt.axis("equal")

        if ss[-1] >= 0:
            selected = i

    return selected


def calibrate(
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    xc: float,
    yc: float,
    taylor_order_default: int,
    ima_proc: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    Xt = _as_col(Xt)
    Yt = _as_col(Yt)

    Xp = np.asarray(Xp_abs, dtype=float) - xc
    Yp = np.asarray(Yp_abs, dtype=float) - yc

    n_ima = Xp.shape[2]

    RRfin = np.zeros((3, 3, n_ima), dtype=float)

    for kk in ima_proc:
        k = _idx(kk)

        Ypt = _as_col(Yp[:, 0, k])
        Xpt = _as_col(Xp[:, 0, k])

        A = np.hstack(
            [
                Xt * Ypt,
                Yt * Ypt,
                -Xt * Xpt,
                -Yt * Xpt,
                Ypt,
                -Xpt,
            ]
        )

        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T

        R11, R12, R21, R22, T1, T2 = V[:, -1]

        AA = ((R11 * R12) + (R21 * R22)) ** 2
        BB = R11 ** 2 + R21 ** 2
        CC = R12 ** 2 + R22 ** 2

        roots = np.roots([1.0, CC - BB, -AA])

        R32_2 = np.real(
            roots[
                (np.abs(np.imag(roots)) < 1e-9)
                & (np.real(roots) >= 0)
            ]
        )

        R31_list = []
        R32_list = []

        for val in R32_2:
            for sg in (1.0, -1.0):
                sqrt_val = sg * math.sqrt(max(val, 0.0))

                R32_list.append(sqrt_val)

                if abs(val) < 1e-12:
                    tmp = math.sqrt(max(CC - BB, 0.0))

                    R31_list.append(tmp)
                    R31_list.append(-tmp)

                    R32_list.append(sqrt_val)
                else:
                    R31_list.append(
                        -(R11 * R12 + R21 * R22) / sqrt_val
                    )

        if not R31_list:
            return np.array(0), np.array(0)

        candidates = []

        for r31, r32 in zip(R31_list, R32_list):
            for sg in (1.0, -1.0):
                Lb = 1.0 / math.sqrt(R11 ** 2 + R21 ** 2 + r31 ** 2)

                candidates.append(
                    sg
                    * Lb
                    * np.array(
                        [
                            [R11, R12, T1],
                            [R21, R22, T2],
                            [r31, r32, 0.0],
                        ]
                    )
                )

        RR = np.stack(candidates, axis=2)

        first = np.array([Xpt[0, 0], Ypt[0, 0]])

        distances = [
            np.linalg.norm(RR[:2, 2, c] - first)
            for c in range(RR.shape[2])
        ]

        min_ind = int(np.argmin(distances))

        RR1 = []

        for c in range(RR.shape[2]):
            if (
                np.sign(RR[0, 2, c]) == np.sign(RR[0, 2, min_ind])
                and np.sign(RR[1, 2, c]) == np.sign(RR[1, 2, min_ind])
            ):
                RR1.append(RR[:, :, c])

        if not RR1:
            return np.array(0), np.array(0)

        RR1 = np.stack(RR1, axis=2)

        nm = plot_RR(RR1, Xt, Yt, Xpt, Ypt, 0)

        RRfin[:, :, k] = RR1[:, :, nm]

    RRfin, ss = omni_find_parameters_fun(
        Xt,
        Yt,
        Xp_abs,
        Yp_abs,
        xc,
        yc,
        RRfin,
        taylor_order_default,
        ima_proc,
    )

    return RRfin, ss


def omni_find_parameters_fun(
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    xc: float,
    yc: float,
    RRfin: np.ndarray,
    taylor_order: int,
    ima_proc: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    Xp = np.asarray(Xp_abs, dtype=float) - xc
    Yp = np.asarray(Yp_abs, dtype=float) - yc

    Xt = _as_col(Xt)
    Yt = _as_col(Yt)

    min_order = 4
    max_order = int(taylor_order)

    orders = [max_order] if max_order <= min_order else list(range(min_order, max_order + 1))

    last_s = None
    last_ss = None

    for order in orders:
        PP = np.empty((0, 0))
        QQ_parts = []
        count = 0

        for i in ima_proc:
            count += 1
            k = _idx(i)

            RRdef = RRfin[:, :, k]

            R11, R21, R31 = RRdef[0, 0], RRdef[1, 0], RRdef[2, 0]
            R12, R22, R32 = RRdef[0, 1], RRdef[1, 1], RRdef[2, 1]
            T1, T2 = RRdef[0, 2], RRdef[1, 2]

            Xpt = _as_col(Xp[:, 0, k])
            Ypt = _as_col(Yp[:, 0, k])

            MA = R21 * Xt + R22 * Yt + T2
            MB = Ypt * (R31 * Xt + R32 * Yt)

            MC = R11 * Xt + R12 * Yt + T1
            MD = Xpt * (R31 * Xt + R32 * Yt)

            PP1 = np.vstack([MA, MC])

            rho_base = np.sqrt(Xpt ** 2 + Ypt ** 2)

            for j in range(2, order + 1):
                rho = rho_base ** j
                PP1 = np.hstack(
                    [
                        PP1,
                        np.vstack([MA * rho, MC * rho]),
                    ]
                )

            if PP.size == 0:
                PP = np.hstack(
                    [
                        PP1,
                        np.vstack([-Ypt, -Xpt]),
                    ]
                )
            else:
                top = np.hstack(
                    [
                        PP,
                        np.zeros((PP.shape[0], 1)),
                    ]
                )

                bottom = np.hstack(
                    [
                        PP1,
                        np.zeros((PP1.shape[0], count - 1)),
                        np.vstack([-Ypt, -Xpt]),
                    ]
                )

                PP = np.vstack([top, bottom])

            QQ_parts.append(np.vstack([MB, MD]))

        QQ = np.vstack(QQ_parts)

        s = np.linalg.pinv(PP) @ QQ

        last_s = s.ravel()
        last_ss = last_s[:order]

    s = last_s
    ss_raw = last_ss

    count = 0

    for j in ima_proc:
        count += 1
        k = _idx(j)

        RRfin[2, 2, k] = s[len(ss_raw) + count - 1]

    ss = np.concatenate([[ss_raw[0]], [0.0], ss_raw[1:]])

    return RRfin, ss


# -----------------------------------------------------------------------------
# Reprojection, center search, refinement
# -----------------------------------------------------------------------------

def reprojectpoints(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    err = []
    stderr = []
    MSE = 0.0

    for i in ima_proc:
        k = _idx(i)

        xx = RRfin[:, :, k] @ np.vstack(
            [
                np.asarray(Xt).ravel(),
                np.asarray(Yt).ravel(),
                np.ones(np.asarray(Xt).size),
            ]
        )

        xp, yp = omni3d2pixel(
            ocam_model.ss,
            xx,
            ocam_model.width,
            ocam_model.height,
        )

        stt = np.sqrt(
            (Xp_abs[:, 0, k] - ocam_model.xc - xp) ** 2
            + (Yp_abs[:, 0, k] - ocam_model.yc - yp) ** 2
        )

        err.append(float(np.nanmean(stt)))
        stderr.append(float(np.nanstd(stt)))

        MSE += float(
            np.nansum(
                (Xp_abs[:, 0, k] - ocam_model.xc - xp) ** 2
                + (Yp_abs[:, 0, k] - ocam_model.yc - yp) ** 2
            )
        )

    err = np.asarray(err)
    stderr = np.asarray(stderr)

    print("\nAverage reprojection error computed for each chessboard [pixels]:\n")

    for e, s in zip(err, stderr):
        print(f" {e:3.2f} ± {s:3.2f}")

    avg = float(np.nanmean(err))

    print(f"\nAverage error [pixels]\n\n {avg:f}")
    print(f"\nSum of squared errors\n\n {MSE:f}")

    if avg < 1.0:
        print("\nAverage error is below 1 pixel: OK.\n")
    else:
        print("\nWARNING: average error is above 1 pixel. Check marker extraction/calibration.\n")

    return err, stderr, MSE


def reprojectPoints_fun(
    Xt,
    Yt,
    Xp_abs,
    Yp_abs,
    xc,
    yc,
    RRfin,
    ss,
    ima_proc,
    width,
    height,
) -> float:
    MSE = 0.0

    Xt = _as_col(Xt)
    Yt = _as_col(Yt)

    for i in ima_proc:
        k = _idx(i)

        xx = RRfin[:, :, k] @ np.vstack(
            [
                Xt.ravel(),
                Yt.ravel(),
                np.ones(Xt.size),
            ]
        )

        xp, yp = omni3d2pixel(ss, xx, width, height)

        if np.isnan(xp).any() or np.isnan(yp).any():
            return float("nan")

        MSE += float(
            np.sum(
                (Xp_abs[:, 0, k] - xc - xp) ** 2
                + (Yp_abs[:, 0, k] - yc - yp) ** 2
            )
        )

    return MSE


# lowercase alias used in earlier code
reprojectpoints_fun = reprojectPoints_fun


def reprojectpoints_adv(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    M: np.ndarray,
):
    M = np.asarray(M, dtype=float).copy()

    M[:, 2] = 1.0

    err = []
    stderr = []
    MSE = 0.0

    for i in ima_proc:
        k = _idx(i)

        Mc = RRfin[:, :, k] @ M.T

        m = world2cam(Mc, ocam_model)

        xp = m[0, :]
        yp = m[1, :]

        sqerr = (Xp_abs[:, 0, k] - xp) ** 2 + (Yp_abs[:, 0, k] - yp) ** 2

        err.append(float(np.nanmean(np.sqrt(sqerr))))
        stderr.append(float(np.nanstd(np.sqrt(sqerr))))

        MSE += float(np.nansum(sqerr))

    print("\nAverage reprojection error computed for each chessboard [pixels]:\n")

    for e, s in zip(err, stderr):
        print(f" {e:3.2f} ± {s:3.2f}")

    print(f"\nAverage error [pixels]\n\n {np.nanmean(err):f}")
    print(f"\nSum of squared errors\n\n {MSE:f}")

    return np.asarray(err), np.asarray(stderr), MSE


def findcenter(
    ocam_model: OCamModel,
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    taylor_order: int,
    ima_proc: Sequence[int],
) -> np.ndarray:
    """
    Original MATLAB-style exhaustive center search.

    Mutates ``ocam_model`` (xc, yc, ss) and returns the recalibrated RRfin.
    """
    print("\nComputing center coordinates.\n")

    pxc = ocam_model.xc
    pyc = ocam_model.yc

    width = ocam_model.width
    height = ocam_model.height

    regwidth = width / 2.0
    regheight = height / 2.0

    yceil = 5
    xceil = 5

    xregstart = pxc - (regheight / 2.0)
    xregstop = pxc + (regheight / 2.0)

    yregstart = pyc - (regwidth / 2.0)
    yregstop = pyc + (regwidth / 2.0)

    print("Iteration ", end="", flush=True)

    for glc in range(1, 10):
        xs = np.linspace(xregstart, xregstop + 1.0 / xceil, xceil + 2)
        ys = np.linspace(yregstart, yregstop + 1.0 / yceil, yceil + 2)

        xreg, yreg = np.meshgrid(xs, ys, indexing="ij")

        MSEA = np.full(xreg.shape, np.inf)

        for ic in range(xreg.shape[0]):
            for jc in range(xreg.shape[1]):
                xc = float(xreg[ic, jc])
                yc = float(yreg[ic, jc])

                RRfin, ss = calibrate(
                    Xt,
                    Yt,
                    Xp_abs,
                    Yp_abs,
                    xc,
                    yc,
                    taylor_order,
                    ima_proc,
                )

                if np.isscalar(RRfin) or np.size(RRfin) == 1:
                    continue

                MSE = reprojectpoints_fun(
                    Xt,
                    Yt,
                    Xp_abs,
                    Yp_abs,
                    xc,
                    yc,
                    RRfin,
                    ss,
                    ima_proc,
                    width,
                    height,
                )

                if not np.isnan(MSE):
                    MSEA[ic, jc] = MSE

        ind = np.unravel_index(np.argmin(MSEA), MSEA.shape)

        ocam_model.xc = float(xreg[ind])
        ocam_model.yc = float(yreg[ind])

        dx_reg = abs((xregstop - xregstart) / xceil)
        dy_reg = abs((yregstop - yregstart) / yceil)

        xregstart = ocam_model.xc - dx_reg
        xregstop = ocam_model.xc + dx_reg

        yregstart = ocam_model.yc - dy_reg
        yregstop = ocam_model.yc + dy_reg

        print(f"{glc}...", end="", flush=True)

    print("\n")

    RRfin, ss = calibrate(
        Xt,
        Yt,
        Xp_abs,
        Yp_abs,
        ocam_model.xc,
        ocam_model.yc,
        taylor_order,
        ima_proc,
    )

    ocam_model.ss = np.asarray(ss, dtype=float)

    reprojectpoints(ocam_model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs)

    print("xc =", ocam_model.xc)
    print("yc =", ocam_model.yc)

    return RRfin


def findcenter_fast(
    ocam_model: OCamModel,
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    taylor_order: int,
    ima_proc: Sequence[int],
    iterations: int = 4,
    grid_size: int = 3,
    start_radius_fraction: float = 0.20,
) -> Optional[np.ndarray]:
    """
    Faster center search.

    The original MATLAB-style findcenter search is accurate but slow because it
    recalibrates many times. This fast version searches a smaller grid and
    shrinks the search radius after each pass.

    Mutates ``ocam_model`` (xc, yc, ss) and returns the recalibrated RRfin,
    or None when the search fails (the caller should keep the previous
    center/extrinsics in that case).
    """
    print("\nComputing center coordinates using FAST search.\n")

    width = ocam_model.width
    height = ocam_model.height

    best_xc = float(ocam_model.xc)
    best_yc = float(ocam_model.yc)

    radius_x = height * start_radius_fraction
    radius_y = width * start_radius_fraction

    best_mse = float("inf")
    best_RRfin = None
    best_ss = None

    total_tests = iterations * grid_size * grid_size
    test_count = 0

    for it in range(1, iterations + 1):
        xs = np.linspace(best_xc - radius_x, best_xc + radius_x, grid_size)
        ys = np.linspace(best_yc - radius_y, best_yc + radius_y, grid_size)

        print(f"Fast center iteration {it}/{iterations}...", flush=True)

        for xc in xs:
            for yc in ys:
                test_count += 1

                RRfin, ss = calibrate(
                    Xt,
                    Yt,
                    Xp_abs,
                    Yp_abs,
                    float(xc),
                    float(yc),
                    taylor_order,
                    ima_proc,
                )

                if np.isscalar(RRfin) or np.size(RRfin) == 1:
                    continue

                mse = reprojectpoints_fun(
                    Xt,
                    Yt,
                    Xp_abs,
                    Yp_abs,
                    float(xc),
                    float(yc),
                    RRfin,
                    ss,
                    ima_proc,
                    width,
                    height,
                )

                if not np.isnan(mse) and mse < best_mse:
                    best_mse = mse
                    best_xc = float(xc)
                    best_yc = float(yc)
                    best_RRfin = RRfin
                    best_ss = ss

        radius_x *= 0.35
        radius_y *= 0.35

        print(
            f"  best center so far: xc={best_xc:.3f}, "
            f"yc={best_yc:.3f}, MSE={best_mse:.3f} "
            f"({test_count}/{total_tests} center tests)"
        )

    if best_RRfin is None or best_ss is None:
        print("Fast center search failed. Keeping previous center.")
        return None

    ocam_model.xc = best_xc
    ocam_model.yc = best_yc
    ocam_model.ss = np.asarray(best_ss, dtype=float)

    print("\nFast center search complete.")
    print("xc =", ocam_model.xc)
    print("yc =", ocam_model.yc)

    reprojectpoints(ocam_model, best_RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs)

    return best_RRfin


def recomp_corner_calib(
    ocam_model: OCamModel,
    images: Sequence[np.ndarray],
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    ima_proc: Sequence[int],
    wintx: Optional[int] = None,
    winty: Optional[int] = None,
    min_movement_px: float = 0.05,
    max_movement_px: float = 8.0,
) -> None:
    """
    Refine detected image points near their current measured locations.

    Mutates ``Xp_abs``/``Yp_abs`` in place.

    Important safety rule:
        Never replace measured points with the model reprojection itself.

    The older Python version used the current model projection as the initial
    point and, if refinement failed, wrote the projected points into Xp_abs/Yp_abs.
    That can make reprojection error become exactly 0.000000, because the code is
    then comparing the model against points generated by the model. This function
    avoids that by starting from the existing measured points and keeping the
    original measured points whenever refinement is suspicious or unavailable.
    """
    if cv2 is None:
        print("OpenCV unavailable; keeping original detected points.")
        return

    if wintx is None:
        wintx = max(
            round(ocam_model.width / 128),
            round(ocam_model.height / 96),
        )

    if winty is None:
        winty = wintx

    wintx = int(round(wintx))
    winty = int(round(winty))

    print(f"Window size = {2 * wintx + 1}x{2 * winty + 1}")
    print("Safe corner refinement: refining from measured UV points, not model projections.")

    changed_total = 0
    kept_total = 0

    for kk in ima_proc:
        k = _idx(kk)
        I = images[k].astype(np.uint8)  # noqa: E741 -- MATLAB port keeps upstream's name

        # OpenCV cornerSubPix expects points as [col, row]. Existing data stores
        # Xp_abs=row and Yp_abs=col.
        original_cv = np.vstack(
            [
                Yp_abs[:, 0, k],
                Xp_abs[:, 0, k],
            ]
        ).T.astype(np.float32)

        if wintx == 0 or winty == 0:
            print(f"  image {kk}: refinement window is zero; keeping original detected points")
            kept_total += original_cv.shape[0]
            continue

        try:
            term = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                1e-3,
            )

            refined = cv2.cornerSubPix(
                I,
                original_cv.reshape(-1, 1, 2),
                (winty, wintx),
                (-1, -1),
                term,
            ).reshape(-1, 2)

            movement = np.linalg.norm(refined - original_cv, axis=1)

            # If refinement barely moves every point, it is harmless but not useful.
            # If it moves points too far, it probably locked onto the wrong bright dot
            # or a bad gradient. In both cases, keep measured points.
            suspicious = (movement > max_movement_px) | ~np.isfinite(movement)
            usable = ~suspicious

            if not np.any(usable):
                print(f"  image {kk}: refinement rejected; keeping original detected points")
                kept_total += original_cv.shape[0]
                continue

            updated = original_cv.copy()
            updated[usable] = refined[usable]

            Yp_abs[:, 0, k] = updated[:, 0]
            Xp_abs[:, 0, k] = updated[:, 1]

            changed = int(np.sum((movement >= min_movement_px) & usable))
            kept = int(np.sum(~usable))
            changed_total += changed
            kept_total += kept

            print(
                f"  image {kk}: refined {changed}/{len(movement)} points "
                f"(median move={np.median(movement):.3f}px, max move={np.max(movement):.3f}px)"
            )

        except Exception as exc:
            print(
                f"  image {kk}: corner refinement failed ({exc}); "
                "keeping original detected points"
            )
            kept_total += original_cv.shape[0]

    print(
        f"Corners recomputed safely. Changed points: {changed_total}, "
        f"kept/rejected: {kept_total}\nDone"
    )


# -----------------------------------------------------------------------------
# Inverse polynomial
# -----------------------------------------------------------------------------

def invFUN(
    ss: Sequence[float],
    theta: np.ndarray,
    radius: float,
) -> np.ndarray:
    m = np.tan(theta)

    ss = np.asarray(ss, dtype=float).ravel()
    poly_coef = ss[::-1].copy()

    r = np.full_like(m, np.inf, dtype=float)

    for j, mj in enumerate(m):
        poly_tmp = poly_coef.copy()
        poly_tmp[-2] = poly_coef[-2] - mj

        roots = np.roots(poly_tmp)

        res = np.real(
            roots[
                (np.abs(np.imag(roots)) < 1e-9)
                & (np.real(roots) > 0)
                & (np.real(roots) < radius)
            ]
        )

        if res.size == 1:
            r[j] = res[0]

    return r


def findinvpoly2(
    ss: Sequence[float],
    radius: float,
    N: int,
):
    theta = np.arange(-np.pi / 2, 1.20 + 1e-12, 0.01)

    r = invFUN(ss, theta, radius)

    ind = np.isfinite(r)

    theta = theta[ind]
    r = r[ind]

    if len(theta) <= N:
        raise RuntimeError("Not enough valid points to compute inverse polynomial.")

    pol = np.polyfit(theta, r, N)
    err = np.abs(r - np.polyval(pol, theta))

    return pol, err, N


def findinvpoly(
    ss: Sequence[float],
    radius: float,
    N: Optional[int] = None,
):
    if N is None:
        maxerr = np.inf
        N = 1
        pol = None
        err = None

        while maxerr > 0.01:
            N += 1
            pol, err, _ = findinvpoly2(ss, radius, N)
            maxerr = np.max(err)

            if N > 30:
                break

        return pol, err, N

    return findinvpoly2(ss, radius, N)


# -----------------------------------------------------------------------------
# Save/export
# -----------------------------------------------------------------------------

def _serializable_calib_dict(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    taylor_order: Optional[int],
) -> dict:
    return {
        "Xt": Xt,
        "Yt": Yt,
        "Xp_abs": Xp_abs,
        "Yp_abs": Yp_abs,
        "RRfin": RRfin,
        "ima_proc": np.asarray(ima_proc),
        "taylor_order": taylor_order,
        "xc": ocam_model.xc,
        "yc": ocam_model.yc,
        "width": ocam_model.width,
        "height": ocam_model.height,
        "c": ocam_model.c,
        "d": ocam_model.d,
        "e": ocam_model.e,
        "ss": ocam_model.ss,
        "invpol": ocam_model.invpol,
    }


def saving_calib(
    ocam_model: OCamModel,
    RRfin: np.ndarray,
    ima_proc: Sequence[int],
    Xt: np.ndarray,
    Yt: np.ndarray,
    Xp_abs: np.ndarray,
    Yp_abs: np.ndarray,
    taylor_order: Optional[int],
    output_dir: str = ".",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / "Omni_Calib_Results.npz"

    if npz_path.exists():
        pfn = 0

        while (out / f"Omni_Calib_Results_old{pfn}.npz").exists():
            pfn += 1

        shutil.copyfile(npz_path, out / f"Omni_Calib_Results_old{pfn}.npz")

        print(
            "Copying the current Omni_Calib_Results.npz file to "
            f"Omni_Calib_Results_old{pfn}.npz"
        )

    calib_dict = _serializable_calib_dict(
        ocam_model, RRfin, ima_proc, Xt, Yt, Xp_abs, Yp_abs, taylor_order
    )

    np.savez(npz_path, **calib_dict)

    try:
        # SciPy is optional. Import dynamically so editors do not warn if it
        # is not installed; the .npz file is always saved above.
        scipy_io = __import__("scipy.io", fromlist=["savemat"])
        scipy_io.savemat(
            out / "Omni_Calib_Results.mat",
            {"calib_data": calib_dict},
        )
    except Exception:
        pass

    print("done")


def export_data(
    ocam_model: OCamModel,
    output_dir: str = ".",
) -> None:
    if ocam_model.invpol is None:
        radius = math.sqrt(
            (ocam_model.width / 2.0) ** 2
            + (ocam_model.height / 2.0) ** 2
        )

        ocam_model.invpol, _, _ = findinvpoly(ocam_model.ss, radius)

    path = Path(output_dir) / "calib_results.txt"

    with open(path, "w", encoding="utf-8") as fid:
        fid.write(
            "#polynomial coefficients for the DIRECT mapping function "
            "(ocam_model.ss in MATLAB). These are used by cam2world\n\n"
        )

        fid.write(f"{len(ocam_model.ss)} ")
        fid.write(" ".join(f"{v:e}" for v in ocam_model.ss))
        fid.write("\n\n")

        fid.write(
            "#polynomial coefficients for the inverse mapping function "
            "(ocam_model.invpol in MATLAB). These are used by world2cam\n\n"
        )

        fid.write(f"{len(ocam_model.invpol)} ")
        fid.write(" ".join(f"{v:f}" for v in ocam_model.invpol[::-1]))
        fid.write("\n\n")

        fid.write('#center: "row" and "column", starting from 0 (C convention)\n\n')
        fid.write(f"{ocam_model.xc:f} {ocam_model.yc:f}\n\n")

        fid.write('#affine parameters "c", "d", "e"\n\n')
        fid.write(f"{ocam_model.c:f} {ocam_model.d:f} {ocam_model.e:f}\n\n")

        fid.write('#image size: "height" and "width"\n\n')
        fid.write(f"{ocam_model.height:d} {ocam_model.width:d}\n\n")

    print(f"Exported {path}")
