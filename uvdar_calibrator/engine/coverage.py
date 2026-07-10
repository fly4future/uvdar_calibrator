"""
Sample selection and calibration-readiness scoring.

The core of this module (``get_parameters`` / ``is_good_sample`` /
``compute_goodenough``) is ported from ROS image_pipeline's
``camera_calibration.calibrator.Calibrator``, adapted to this repo's point
storage: corners are flat ``(n_points, 2)`` arrays in ``[row, col]`` order,
laid out x-major/y-minor (outer loop over board x, inner loop over board y)
rather than ROS's row-major ``(rows, cols)`` grids.

Every detected view is reduced to four normalized numbers
``[p_x, p_y, p_size, skew]``. A new sample is only accepted if its L1
distance from every already-accepted sample exceeds a threshold; readiness
is the min/max range covered by accepted samples on each axis.

The bin-based classification at the bottom of this module (COVERAGE_*_BINS
etc.) predates the ROS-style selection and is kept only as a supplementary
"next images to capture" report. It must not gate calibration:
``compute_goodenough`` is the single source of truth for readiness.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .board import LedGridBoard

# NOTE: ROS camera_calibration defaults, carried over unchanged. They were
# tuned for a hand-held checkerboard filling much of a pinhole camera's
# frame; the UV LED grid here is smaller in frame and captured from farther
# away, so these likely need retuning later (do not change casually --
# retune against real capture sets and observe acceptance behavior).
DEFAULT_SAMPLE_THRESHOLD = 0.2               # is_good_sample min L1 param distance
DEFAULT_PARAM_RANGES = (0.7, 0.7, 0.4, 0.5)  # compute_goodenough targets (X, Y, Size, Skew)
DEFAULT_MIN_DB_SIZE = 40                     # db size forcing goodenough regardless of ranges

PARAM_NAMES = ("X", "Y", "Size", "Skew")


# -----------------------------------------------------------------------------
# ROS-style sample parameters
# -----------------------------------------------------------------------------

def outside_corner_indices(n_cols: int, n_rows: int) -> Dict[str, int]:
    """
    Flat indices of the four outside corners of the point grid.

    The flat layout is x-major/y-minor: ``idx(x, y) = x * n_rows + y``.
    This traces the same consistent loop around the quadrilateral that
    ROS's ``_get_outside_corners`` does
    (up_left -> up_right -> down_right -> down_left), just re-derived for
    x-major storage instead of row-major.
    """
    return {
        "up_left": 0,
        "up_right": (n_cols - 1) * n_rows,
        "down_right": n_cols * n_rows - 1,
        "down_left": n_rows - 1,
    }


def _get_outside_corners_xy(
    corners_xy: np.ndarray,
    board: LedGridBoard,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four outside corners as (x, y) points."""
    idx = outside_corner_indices(board.n_cols, board.n_rows)
    return (
        corners_xy[idx["up_left"]],
        corners_xy[idx["up_right"]],
        corners_xy[idx["down_right"]],
        corners_xy[idx["down_left"]],
    )


def _calculate_area(corners) -> float:
    """
    Compute the area of the quadrilateral spanned by the four outside corners.

    Shoelace via the diagonal cross product, as in ROS.
    """
    (up_left, up_right, down_right, down_left) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0] * q[1] - p[1] * q[0]) / 2.0


def _calculate_skew(corners) -> float:
    """
    Skew: angle deviation from 90 degrees at the up_right outside corner.

    0 = no skew, 1 = high skew. As in ROS.
    """
    up_left, up_right, down_right, _ = corners

    def angle(a, b, c):
        """Angle between lines ab, bc."""
        ab = a - b
        cb = c - b
        return math.acos(
            np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        )

    return min(1.0, 2.0 * abs((math.pi / 2.0) - angle(up_left, up_right, down_right)))


def get_parameters(
    corners_row_col: np.ndarray,
    board: LedGridBoard,
    image_size: Tuple[int, int],
) -> List[float]:
    """
    Reduce a detected view to ``[p_x, p_y, p_size, skew]``.

    ``corners_row_col`` is ``(n_points, 2)`` in ``[row, col]`` order,
    x-major/y-minor. ``image_size`` is ``(width, height)``.
    """
    width, height = image_size
    corners_xy = np.asarray(corners_row_col, dtype=float)[:, ::-1]  # (row,col) -> (x,y)

    oc = _get_outside_corners_xy(corners_xy, board)
    area = _calculate_area(oc)
    skew = _calculate_skew(oc)

    border = math.sqrt(area)
    # For X and Y, we "shrink" the image all around by approx. half the board
    # size. Otherwise large boards are penalized because you can't get much
    # X/Y variation. (Comment and formula as in ROS.)
    p_x = min(1.0, max(0.0, (float(np.mean(corners_xy[:, 0])) - border / 2.0) / (width - border)))
    p_y = min(1.0, max(0.0, (float(np.mean(corners_xy[:, 1])) - border / 2.0) / (height - border)))
    p_size = math.sqrt(area / (width * height))

    return [p_x, p_y, p_size, skew]


def param_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """L1 distance between two parameter vectors."""
    return sum(abs(a - b) for (a, b) in zip(p1, p2))


def is_good_sample(
    params: Sequence[float],
    db_params: Sequence[Sequence[float]],
    threshold: float = DEFAULT_SAMPLE_THRESHOLD,
) -> bool:
    """
    Return True if the sample is sufficiently different from every accepted one.

    An empty database always accepts (as in ROS).
    """
    if not db_params:
        return True

    d = min(param_distance(params, p) for p in db_params)
    return d > threshold


def compute_goodenough(
    db_params: Sequence[Sequence[float]],
    param_ranges: Sequence[float] = DEFAULT_PARAM_RANGES,
    min_db_size: int = DEFAULT_MIN_DB_SIZE,
) -> Tuple[bool, List[Tuple[str, float, float, float]]]:
    """
    Judge readiness from the min/max range covered by accepted samples.

    Returns ``(goodenough, [(name, lo, hi, progress), ...])`` where progress
    per axis is ``min((hi - lo) / target_range, 1.0)``. "Good enough" means
    progress is 1.0 on every axis, or the database has grown large
    regardless (``>= min_db_size`` samples). Returns ``(False, [])`` for an
    empty database.
    """
    if not db_params:
        return False, []

    all_params = [list(p) for p in db_params]
    min_params = all_params[0]
    max_params = all_params[0]
    for params in all_params[1:]:
        min_params = [min(a, b) for (a, b) in zip(min_params, params)]
        max_params = [max(a, b) for (a, b) in zip(max_params, params)]

    # Don't reward small size or skew
    min_params = [min_params[0], min_params[1], 0.0, 0.0]

    progress = [
        min((hi - lo) / r, 1.0)
        for (lo, hi, r) in zip(min_params, max_params, param_ranges)
    ]

    goodenough = (len(all_params) >= min_db_size) or all(p == 1.0 for p in progress)

    return goodenough, list(zip(PARAM_NAMES, min_params, max_params, progress))


def format_progress(
    progress: List[Tuple[str, float, float, float]],
    goodenough: bool,
    n_samples: int,
    bar_width: int = 24,
) -> str:
    """Human-readable text rendering of compute_goodenough output."""
    lines = [f"Accepted samples: {n_samples}"]
    if not progress:
        lines.append("No samples yet.")
        return "\n".join(lines)

    for name, lo, hi, p in progress:
        n = round(bar_width * p)
        lines.append(
            f"{name:5s} [{'#' * n}{'.' * (bar_width - n)}] "
            f"lo={lo:.2f} hi={hi:.2f} progress={100.0 * p:.0f}%"
        )

    lines.append(
        "Status: ready to calibrate."
        if goodenough
        else "Status: NOT ready -- capture more varied views (see hints below)."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Supplementary bin-based coverage report
# -----------------------------------------------------------------------------
# Kept only to drive the "next images to capture" suggestion text (a nice
# addition ROS doesn't have). This must NOT gate calibration.

COVERAGE_X_BINS = (
    (0.00, 0.33, "left"),
    (0.33, 0.67, "center"),
    (0.67, 1.01, "right"),
)
COVERAGE_Y_BINS = (
    (0.00, 0.33, "top"),
    (0.33, 0.67, "middle"),
    (0.67, 1.01, "bottom"),
)
COVERAGE_SIZE_BINS = (
    (0.00, 0.15, "far/small"),
    (0.15, 0.32, "medium"),
    (0.32, 2.00, "close/large"),
)
COVERAGE_SKEW_BINS = (
    (0.00, 0.15, "front-on"),
    (0.15, 0.35, "moderately tilted"),
    (0.35, 2.00, "strongly tilted"),
)


def _coverage_bin(value: float, bins) -> str:
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return bins[-1][2]


def _coverage_bar(label: str, filled: int, total: int, width: int = 24) -> str:
    total = max(int(total), 1)
    filled = max(0, min(int(filled), total))
    n = round(width * filled / total)
    return f"{label:11s} [{'#' * n}{'.' * (width - n)}] {filled}/{total}"


def sample_metric(
    corners_row_col: np.ndarray,
    board: LedGridBoard,
    image_size: Tuple[int, int],
    label: str = "",
) -> Optional[dict]:
    """Classify one detected view into x/y/size/skew/quadrant bins."""
    corners = np.asarray(corners_row_col, dtype=float)
    rows = corners[:, 0]
    cols = corners[:, 1]
    ok = np.isfinite(rows) & np.isfinite(cols)
    if np.sum(ok) < 4:
        return None

    rows = rows[ok]
    cols = cols[ok]
    width = float(image_size[0])
    height = float(image_size[1])

    x_center = float(np.mean(cols) / max(width, 1.0))
    y_center = float(np.mean(rows) / max(height, 1.0))
    bbox_w = float(np.max(cols) - np.min(cols))
    bbox_h = float(np.max(rows) - np.min(rows))
    size = float(np.sqrt(max(bbox_w * bbox_h, 0.0) / max(width * height, 1.0)))

    # Estimate perspective tilt/skew from the two grid axes. The data is stored
    # x-major/y-minor, so reshape as [x_index, y_index, row_col].
    skew = 0.0
    try:
        nx = board.n_cols
        ny = board.n_rows
        rc = corners.reshape(nx, ny, 2)
        axis_x = np.nanmean(rc[-1, :, :], axis=0) - np.nanmean(rc[0, :, :], axis=0)
        axis_y = np.nanmean(rc[:, -1, :], axis=0) - np.nanmean(rc[:, 0, :], axis=0)
        denom = np.linalg.norm(axis_x) * np.linalg.norm(axis_y)
        if denom > 1e-9:
            cosang = abs(float(np.dot(axis_x, axis_y) / denom))
            cosang = max(-1.0, min(1.0, cosang))
            angle_deg = float(np.degrees(np.arccos(cosang)))
            skew = min(abs(90.0 - angle_deg) / 45.0, 1.0)
    except Exception:
        skew = 0.0

    quadrant = (
        "L" if x_center < 0.5 else "R",
        "T" if y_center < 0.5 else "B",
    )

    return {
        "label": label,
        "x": max(0.0, min(1.0, x_center)),
        "y": max(0.0, min(1.0, y_center)),
        "size": max(0.0, size),
        "skew": max(0.0, min(1.0, skew)),
        "x_bin": _coverage_bin(x_center, COVERAGE_X_BINS),
        "y_bin": _coverage_bin(y_center, COVERAGE_Y_BINS),
        "size_bin": _coverage_bin(size, COVERAGE_SIZE_BINS),
        "skew_bin": _coverage_bin(skew, COVERAGE_SKEW_BINS),
        "quadrant": "".join(quadrant),
    }


def compute_bin_coverage(metrics: List[dict]) -> dict:
    """Aggregate per-sample bin metrics into filled/missing bin sets."""
    x_bins = {m["x_bin"] for m in metrics}
    y_bins = {m["y_bin"] for m in metrics}
    size_bins = {m["size_bin"] for m in metrics}
    skew_bins = {m["skew_bin"] for m in metrics}
    quadrants = {m["quadrant"] for m in metrics}

    required_x = {b[2] for b in COVERAGE_X_BINS}
    required_y = {b[2] for b in COVERAGE_Y_BINS}
    required_size = {b[2] for b in COVERAGE_SIZE_BINS}
    # Strong tilt is useful, but forcing it can make UV extraction fragile.
    required_skew = {"front-on", "moderately tilted"}
    required_quadrants = {"LT", "RT", "LB", "RB"}

    missing = {
        "x": sorted(required_x - x_bins),
        "y": sorted(required_y - y_bins),
        "size": sorted(required_size - size_bins),
        "tilt": sorted(required_skew - skew_bins),
        "quadrants": sorted(required_quadrants - quadrants),
    }

    return {
        "metrics": metrics,
        "accepted_images": len(metrics),
        "missing": missing,
        "sets": {
            "x": x_bins,
            "y": y_bins,
            "size": size_bins,
            "tilt": skew_bins,
            "quadrants": quadrants,
        },
    }


def coverage_suggestions(report: dict) -> List[str]:
    missing = report["missing"]
    suggestions = []
    if missing["x"]:
        suggestions.append("move the board/dot grid toward: " + ", ".join(missing["x"]))
    if missing["y"]:
        suggestions.append("move the board/dot grid toward: " + ", ".join(missing["y"]))
    if missing["size"]:
        suggestions.append("capture size(s): " + ", ".join(missing["size"]))
    if missing["tilt"]:
        suggestions.append("capture tilt(s): " + ", ".join(missing["tilt"]))
    if missing["quadrants"]:
        suggestions.append("include image quadrants: " + ", ".join(missing["quadrants"]))
    return suggestions


def format_bin_coverage(report: dict) -> str:
    """Supplementary, non-gating text report of the bin classification."""
    sets = report["sets"]
    lines = []
    lines.append("Supplementary coverage hints (bin classification of ACCEPTED samples).")
    lines.append("Note: near-duplicate images are rejected before reaching this report;")
    lines.append("readiness is judged by the X/Y/Size/Skew range progress, not these bins.")
    lines.append(_coverage_bar("x position", len(sets["x"]), len(COVERAGE_X_BINS)))
    lines.append(_coverage_bar("y position", len(sets["y"]), len(COVERAGE_Y_BINS)))
    lines.append(_coverage_bar("size", len(sets["size"]), len(COVERAGE_SIZE_BINS)))
    lines.append(_coverage_bar("tilt", len(sets["tilt"] & {"front-on", "moderately tilted"}), 2))
    lines.append(_coverage_bar("quadrants", len(sets["quadrants"]), 4))

    suggestions = coverage_suggestions(report)
    if suggestions:
        lines.append("Next images to capture:")
        for s in suggestions:
            lines.append(f"  - {s}")
    else:
        lines.append("All coverage bins are represented.")

    lines.append("")
    lines.append("Per-sample coverage metrics:")
    lines.append("sample,x_bin,y_bin,size_bin,tilt_bin,x_norm,y_norm,size,tilt")
    for m in report["metrics"]:
        lines.append(
            f"{m['label']},{m['x_bin']},{m['y_bin']},{m['size_bin']},{m['skew_bin']},"
            f"{m['x']:.3f},{m['y']:.3f},{m['size']:.3f},{m['skew']:.3f}"
        )
    return "\n".join(lines)


def plot_bin_coverage(metrics: List[dict]) -> None:
    """Scatter plot of accepted board locations in the image (supplementary)."""
    if not metrics:
        return

    import matplotlib.pyplot as plt

    xs = [m["x"] for m in metrics]
    ys = [m["y"] for m in metrics]
    sizes = [m["size"] * 450.0 + 20.0 for m in metrics]

    plt.figure("UV-DAR calibration coverage")
    plt.clf()
    plt.scatter(xs, ys, s=sizes, alpha=0.65)
    for m in metrics:
        plt.text(m["x"], m["y"], str(m["label"]), fontsize=8)
    plt.axvline(0.33, linestyle="--", linewidth=1)
    plt.axvline(0.67, linestyle="--", linewidth=1)
    plt.axhline(0.33, linestyle="--", linewidth=1)
    plt.axhline(0.67, linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.xlabel("horizontal board location in image")
    plt.ylabel("vertical board location in image")
    plt.title(f"Accepted sample coverage ({len(metrics)} samples)")
    plt.grid(True)
    plt.show()
