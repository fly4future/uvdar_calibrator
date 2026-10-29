"""
Pattern detection: UV LED grid (default) or a printed checkerboard.

The main entry point is :func:`get_corners`, which mirrors the shape of ROS
camera_calibration's ``get_corners``: ``(ok, corners, board)``. ``corners``
keep the ``[row, col]``, x-major/y-minor ordering used throughout this
codebase (this is load-bearing for ``Xt``/``Yt`` alignment -- do not change).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .board import LedGridBoard

try:
    import cv2
except Exception:
    cv2 = None


#: Folder (inside the output dir) the calibrator writes per-image detection
#: previews into.
PREVIEW_DIR_NAME = "detected_marker_previews"

IMAGE_EXTENSIONS = ("jpg", "jpeg", "bmp", "png", "tif", "tiff")


def find_image_files(image_dir: str) -> List[str]:
    """Sorted image files directly inside ``image_dir``."""
    return sorted(
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )


def read_image_gray(path: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for image reading/extraction. Run: pip install opencv-python"
        )

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def mirror_horizontal(img: np.ndarray) -> np.ndarray:
    """
    Flip an image left-right and return a new, contiguous array.

    ``cv2.flip`` rather than ``img[:, ::-1]`` on purpose: the numpy view has
    negative strides, which OpenCV refuses and which still aliases the
    caller's buffer. The live path hands the *same* array to the preview
    queue, the detection queue and ``Sample.image``, so an in-place flip
    would rewrite the pixels of already-accepted samples while their
    recorded corner coordinates still describe the old orientation.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for mirroring. Run: pip install opencv-python"
        )

    return cv2.flip(img, 1)


def save_detected_marker_preview(img, pts_row_col, output_path):
    """
    Save a preview image showing detected marker points.

    pts_row_col is in [row, col] order.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")

    if img.ndim == 2:
        preview = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        preview = img.copy()

    for idx, (row, col) in enumerate(pts_row_col, start=1):
        cv2.circle(preview, (int(round(col)), int(round(row))), 8, (0, 0, 255), 2)
        cv2.putText(
            preview,
            str(idx),
            (int(round(col)) + 8, int(round(row)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), preview)


def _reorder_row_major_to_x_major(
    points_col_row: np.ndarray,
    n_cols: int,
    n_rows: int,
    transposed: bool = False,
) -> np.ndarray:
    """
    Reorder a flat, gridded array of (col, row) points from OpenCV's
    row-major layout (index = row * n_cols + col) into this codebase's
    x-major, y-minor [row, col] order (outer loop over board x, inner loop
    over board y), matching Xt/Yt generation.

    Used by both the chessboard and circle-grid detectors -- they receive
    the same shape of OpenCV output and need the identical reindex.
    ``transposed=True`` handles the circle-grid detector's case where
    OpenCV found the pattern rotated (n_rows columns x n_cols rows).
    """
    pts = []

    for x in range(n_cols):
        for y in range(n_rows):
            cv_index = (x * n_rows + y) if transposed else (y * n_cols + x)
            col, row = points_col_row[cv_index]
            pts.append([row, col])

    return np.asarray(pts, dtype=float)


def _try_chessboard_detection(
    img: np.ndarray,
    expected_cols: int,
    expected_rows: int,
) -> Optional[np.ndarray]:
    """
    Find a printed checkerboard's inner corners; returns [row, col] points or None.

    ``CALIB_CB_FAST_CHECK`` rejects a frame without a board (most live
    frames) quickly instead of running the full search.
    """
    pattern_size = (expected_cols, expected_rows)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(img, pattern_size, flags=flags)
    if not ok or corners is None:
        return None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term).reshape(-1, 2)

    # OpenCV gives row-major image order. Convert to the x-major order
    # used by Xt/Yt: x outer, y inner.
    return _reorder_row_major_to_x_major(corners, expected_cols, expected_rows)


def _make_led_blob_detector():
    """Blob detector tuned for small bright UV LED dots."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 700
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    # OpenCV's default of 10 px merges neighbouring LEDs on a small or
    # distant grid.
    params.minDistBetweenBlobs = 3
    return cv2.SimpleBlobDetector_create(params)


def _try_circle_grid_detection(
    img: np.ndarray,
    expected_cols: int,
    expected_rows: int,
) -> Optional[np.ndarray]:
    """
    Find the UV LED grid with OpenCV's symmetric circle-grid detector.

    Tries the board in both orientations. Returns [row, col] points or None.
    """
    blob_detector = _make_led_blob_detector()
    for pcols, prows in ((expected_cols, expected_rows), (expected_rows, expected_cols)):
        ok, centers = cv2.findCirclesGrid(
            img,
            (pcols, prows),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=blob_detector,
        )
        if not ok or centers is None:
            continue

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        # Convert OpenCV row-major output to the same x-major, y-minor
        # order used by Xt/Yt: for x in 0..n_sq_x, for y in 0..n_sq_y.
        # If OpenCV found the transposed layout, transpose back.
        transposed = (pcols, prows) != (expected_cols, expected_rows)
        return _reorder_row_major_to_x_major(
            centers, expected_cols, expected_rows, transposed=transposed
        )

    return None


#: Calibration targets :func:`get_corners` can look for.
PATTERN_LED_GRID = "led"
PATTERN_CHESSBOARD = "chessboard"
PATTERNS = (PATTERN_LED_GRID, PATTERN_CHESSBOARD)


def get_corners(
    img: np.ndarray,
    board: LedGridBoard,
    pattern: str = PATTERN_LED_GRID,
) -> Tuple[bool, Optional[np.ndarray], LedGridBoard]:
    """
    Detect the calibration pattern in one image.

    Mirrors ROS camera_calibration's ``get_corners`` return contract:
    ``(ok, corners, board)``. ``corners`` is ``(n_points, 2)`` in
    ``[row, col]`` order, x-major/y-minor -- aligned with the board's
    ``Xt``/``Yt`` generation order. The LED grid's LEDs and a checkerboard's
    inner corners play the same role, so both use the same point grid.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")
    if pattern == PATTERN_LED_GRID:
        pts = _try_circle_grid_detection(img, board.n_cols, board.n_rows)
    elif pattern == PATTERN_CHESSBOARD:
        pts = _try_chessboard_detection(img, board.n_cols, board.n_rows)
    else:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")
    return (pts is not None), pts, board
