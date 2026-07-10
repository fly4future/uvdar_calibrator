"""
Pattern detection: chessboard -> circle grid -> UV bright-dot fallback.

The main entry point is :func:`get_corners`, which mirrors the shape of ROS
camera_calibration's ``get_corners``: ``(ok, corners, board)``. ``corners``
keep the ``[row, col]``, x-major/y-minor ordering used throughout this
codebase (this is load-bearing for ``Xt``/``Yt`` alignment -- do not change).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .board import LedGridBoard

try:
    import cv2
except Exception:
    cv2 = None


def _normalize_extension(extension: str) -> List[str]:
    ext = extension.lower().strip().lstrip(".")

    if ext in {"all", "*", ""}:
        return ["jpg", "jpeg", "bmp", "png", "tif", "tiff"]

    if ext == "j":
        return ["jpg", "jpeg"]

    if ext in {"jpg", "jpeg"}:
        return ["jpg", "jpeg"]

    if ext in {"bmp", "png", "tif", "tiff"}:
        return [ext]

    return [ext]


def find_image_files(
    image_dir: str,
    base_name: str = "",
    extension: str = "all",
) -> List[str]:
    """
    Find calibration images in image_dir.

    By default, this loads all supported image files regardless of filename:
        jpg, jpeg, bmp, png, tif, tiff

    If base_name is provided, only files starting with that base name are loaded.
    If extension is provided as something other than "all" or "*", only that type is loaded.
    """
    exts = _normalize_extension(extension)
    files: List[str] = []

    for ext in exts:
        if base_name:
            pattern = f"{base_name}*.{ext}"
        else:
            pattern = f"*.{ext}"

        files.extend(glob.glob(str(Path(image_dir) / pattern)))
        files.extend(glob.glob(str(Path(image_dir) / pattern.upper())))

    return sorted(set(files), key=lambda p: (Path(p).suffix.lower(), Path(p).name.lower()))


def _read_image_gray(path: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for image reading/extraction. Run: pip install opencv-python"
        )

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


# public alias
read_image_gray = _read_image_gray


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


def _order_uv_points_projective(
    pts_xy: np.ndarray,
    expected_cols: int,
    expected_rows: int,
) -> Optional[np.ndarray]:
    """
    Order UV dot centroids when simple horizontal row grouping fails.

    pts_xy is [column, row]. The method tries plausible four-corner choices,
    maps the image quadrilateral to a rectangular grid with a homography, and
    keeps the mapping that assigns every dot to a unique integer grid location.

    It returns [column, row] points in x-major, y-minor order, matching Xt/Yt.
    """
    if cv2 is None:
        return None

    pts = np.asarray(pts_xy, dtype=np.float32)
    expected_points = expected_cols * expected_rows

    if pts.shape[0] != expected_points:
        return None

    # Candidate corner sets based on common image-quadrilateral extremes.
    # These do not assume rows are horizontal.
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    k = min(8, len(pts))

    cand_bl = np.argsort(d)[:k]      # low column, high row
    cand_tl = np.argsort(s)[:k]      # low column, low row
    cand_br = np.argsort(s)[-k:]     # high column, high row
    cand_tr = np.argsort(d)[-k:]     # high column, low row

    dst = np.array(
        [
            [0, 0],
            [expected_cols - 1, 0],
            [0, expected_rows - 1],
            [expected_cols - 1, expected_rows - 1],
        ],
        dtype=np.float32,
    )

    best = None

    for ibl in cand_bl:
        for itl in cand_tl:
            for ibr in cand_br:
                for itr in cand_tr:
                    if len({int(ibl), int(itl), int(ibr), int(itr)}) < 4:
                        continue

                    src = np.array(
                        [pts[ibl], pts[itl], pts[ibr], pts[itr]],
                        dtype=np.float32,
                    )

                    if abs(cv2.contourArea(src.reshape(-1, 1, 2))) < 50:
                        continue

                    try:
                        H = cv2.getPerspectiveTransform(src, dst)
                    except Exception:
                        continue

                    ph = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)]) @ H.T
                    if np.any(np.abs(ph[:, 2]) < 1e-9):
                        continue

                    uv = ph[:, :2] / ph[:, 2:3]
                    grid = np.rint(uv).astype(int)
                    err = np.linalg.norm(uv - grid, axis=1)

                    in_bounds = (
                        (grid[:, 0] >= 0)
                        & (grid[:, 0] < expected_cols)
                        & (grid[:, 1] >= 0)
                        & (grid[:, 1] < expected_rows)
                    )

                    pairs = [tuple(g) for g, ok in zip(grid, in_bounds) if ok]
                    unique_count = len(set(pairs))
                    in_count = int(np.sum(in_bounds))
                    median_err = float(np.median(err[in_bounds])) if in_count else 999.0
                    max_err = float(np.max(err[in_bounds])) if in_count else 999.0

                    # Prefer complete, unique assignments; then lower error.
                    score = (unique_count, in_count, -median_err, -max_err)
                    if best is None or score > best[0]:
                        best = (score, grid, err, in_bounds)

    if best is None:
        return None

    score, grid, err, in_bounds = best
    unique_count, in_count, neg_median_err, neg_max_err = score

    if unique_count < expected_points:
        print(
            f"  projective ordering incomplete: unique={unique_count}/{expected_points}, "
            f"in_bounds={in_count}/{expected_points}"
        )
        return None

    # Build output in x-major, y-minor order.
    assignment = {}
    for point, gij, ok, e in zip(pts, grid, in_bounds, err):
        if not ok:
            return None
        key = (int(gij[0]), int(gij[1]))
        if key in assignment:
            return None
        assignment[key] = point

    ordered = []
    for x in range(expected_cols):
        for y in range(expected_rows):
            key = (x, y)
            if key not in assignment:
                return None
            ordered.append(assignment[key])

    print(
        "  UV projective ordering OK "
        f"(median grid error={-neg_median_err:.3f}, max grid error={-neg_max_err:.3f})"
    )
    return np.asarray(ordered, dtype=float)


def _detect_grid_points(
    img: np.ndarray,
    n_sq_x: int,
    n_sq_y: int,
) -> Optional[np.ndarray]:
    """
    Detect calibration points.

    First tries normal checkerboard detection.
    If that fails, falls back to UV bright-dot marker detection.

    Returns points in MATLAB/OCamCalib-style [row, col] order.

    For the UV dot pattern used here:
      - n_sq_x=6 and n_sq_y=4 means 7 x 5 = 35 points.
      - The MATLAB log chooses a start point, a "right" point, and a "below"
        point. For your sample image, the physical x direction runs upward in
        the image and the physical y direction runs left-to-right.
      - Therefore the fallback orders points bottom-to-top, and within each
        row, left-to-right. This matches Xt/Yt creation:
            for x in range(n_sq_x + 1):
                for y in range(n_sq_y + 1):
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")

    expected_cols = n_sq_x + 1  # 7 for your setup
    expected_rows = n_sq_y + 1  # 5 for your setup
    expected_points = expected_cols * expected_rows

    # ------------------------------------------------------------
    # 1) Try normal checkerboard detection first
    # ------------------------------------------------------------
    pattern_size = (expected_cols, expected_rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    try:
        ok, corners = cv2.findChessboardCorners(img, pattern_size, flags)
    except Exception:
        ok, corners = False, None

    if not ok and hasattr(cv2, "findChessboardCornersSB"):
        try:
            ok, corners = cv2.findChessboardCornersSB(img, pattern_size, None)
        except Exception:
            ok, corners = False, None

    if ok and corners is not None:
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)

        try:
            term = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                1e-3,
            )
            corners_refined = cv2.cornerSubPix(
                img,
                corners.reshape(-1, 1, 2),
                (5, 5),
                (-1, -1),
                term,
            )
            corners = corners_refined.reshape(-1, 2)
        except Exception:
            pass

        # OpenCV gives row-major image order. Convert to the x-major order
        # used by Xt/Yt: x outer, y inner.
        pts = []
        for x in range(expected_cols):
            for y in range(expected_rows):
                cv_index = y * expected_cols + x
                col, row = corners[cv_index]
                pts.append([row, col])

        return np.asarray(pts, dtype=float)

    # ------------------------------------------------------------
    # 2) Try OpenCV symmetric circle-grid detection for UV dot grids.
    #    This is more reliable for tilted/rotated UV-dot images than
    #    grouping by nearly-horizontal image rows.
    # ------------------------------------------------------------
    def _make_blob_detector():
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 700
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        return cv2.SimpleBlobDetector_create(params)

    try:
        blob_detector = _make_blob_detector()
        circle_patterns = [
            (expected_cols, expected_rows),
            (expected_rows, expected_cols),
        ]

        for circle_pattern in circle_patterns:
            ok, centers = cv2.findCirclesGrid(
                img,
                circle_pattern,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=blob_detector,
            )

            if not ok or centers is None or len(centers) != expected_points:
                continue

            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
            pcols, prows = circle_pattern

            # Convert OpenCV row-major output to the same x-major, y-minor
            # order used by Xt/Yt: for x in 0..n_sq_x, for y in 0..n_sq_y.
            # If OpenCV found the transposed layout, transpose back.
            pts = []
            if pcols == expected_cols and prows == expected_rows:
                for x in range(expected_cols):
                    for y in range(expected_rows):
                        col, row = centers[y * expected_cols + x]
                        pts.append([row, col])
            else:
                # Transposed grid: OpenCV has expected_rows columns and
                # expected_cols rows. Use row-major groups directly as x-major
                # groups, because each group has expected_rows points.
                for x in range(expected_cols):
                    for y in range(expected_rows):
                        col, row = centers[x * expected_rows + y]
                        pts.append([row, col])

            print(f"  UV circle-grid detector OK with pattern {pcols}x{prows}")
            return np.asarray(pts, dtype=float)
    except Exception as exc:
        print(f"  UV circle-grid detector skipped: {exc}")

    # ------------------------------------------------------------
    # 3) UV bright-dot fallback
    # ------------------------------------------------------------
    img_u8 = img.astype(np.uint8)

    blur = cv2.GaussianBlur(img_u8, (3, 3), 0)

    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8)

    h, w = img_u8.shape[:2]
    image_area = h * w

    candidates = []

    for lab in range(1, num_labels):
        x, y, bw, bh, area = stats[lab]
        cx, cy = centroids[lab]

        # Reject noise, huge blobs, and edge artifacts.
        if area < 1:
            continue
        if area > max(300, image_area * 0.002):
            continue
        if bw > 35 or bh > 35:
            continue
        if cx < 1 or cy < 1 or cx > w - 2 or cy > h - 2:
            continue

        candidates.append([cx, cy, area])

    print(f"  UV fallback found {len(candidates)} candidate bright dots")

    if len(candidates) < expected_points:
        return None

    candidates = np.asarray(candidates, dtype=float)

    # If there are too many dots, keep the expected number. For these UV
    # images, the true markers are typically compact bright components.
    if len(candidates) > expected_points:
        candidates = candidates[np.argsort(candidates[:, 2])[-expected_points:]]

    pts_xy = candidates[:, :2]  # [column, row]

    def group_by_axis(points_xy: np.ndarray, axis: int, threshold: float):
        order = np.argsort(points_xy[:, axis])
        groups = []

        for p in points_xy[order]:
            placed = False

            for g in groups:
                current_mean = np.mean([q[axis] for q in g])
                if abs(current_mean - p[axis]) < threshold:
                    g.append(p)
                    placed = True
                    break

            if not placed:
                groups.append([p])

        groups.sort(key=lambda g: np.mean([q[axis] for q in g]))
        return groups

    # Estimate dot spacing from nearest-neighbor distances.
    dists = []
    for i in range(len(pts_xy)):
        diff = pts_xy - pts_xy[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist = dist[dist > 0]
        if len(dist):
            dists.append(np.min(dist))

    spacing_est = float(np.median(dists)) if dists else 10.0
    group_thresh = max(4.0, spacing_est * 0.45)

    row_groups = group_by_axis(pts_xy, axis=1, threshold=group_thresh)

    group_lengths = [len(g) for g in row_groups]
    print(f"  UV fallback row groups top-to-bottom: {group_lengths}")

    # Your sample image appears as 7 visible rows x 5 visible columns.
    # To match the MATLAB log's start/right/below convention and the Xt/Yt
    # creation order, use bottom-to-top rows, then left-to-right points.
    if len(row_groups) == expected_cols and all(len(g) == expected_rows for g in row_groups):
        ordered_xy = []
        for row_group in reversed(row_groups):  # bottom-to-top
            row_sorted = sorted(row_group, key=lambda p: p[0])  # left-to-right
            ordered_xy.extend(row_sorted)

    # If an image appears unrotated as 5 rows x 7 columns, order by columns
    # left-to-right, then top-to-bottom within each column. This still matches
    # x-major Xt/Yt ordering.
    elif len(row_groups) == expected_rows and all(len(g) == expected_cols for g in row_groups):
        col_groups = group_by_axis(pts_xy, axis=0, threshold=group_thresh)
        if (len(col_groups) != expected_cols
                or not all(len(g) == expected_rows for g in col_groups)):
            print("  UV fallback ordering failed: row/column grouping mismatch")
            return None

        ordered_xy = []
        for col_group in col_groups:  # left-to-right
            col_sorted = sorted(col_group, key=lambda p: p[1])  # top-to-bottom
            ordered_xy.extend(col_sorted)

    else:
        print(
            "  UV row grouping did not match a rectangular grid; "
            f"got row groups {group_lengths}. Trying projective ordering..."
        )

        ordered_xy = _order_uv_points_projective(
            pts_xy,
            expected_cols=expected_cols,
            expected_rows=expected_rows,
        )

        if ordered_xy is None:
            print(
                "  UV fallback ordering failed: expected either "
                f"{expected_cols}x{expected_rows} or {expected_rows}x{expected_cols}; "
                f"got row groups {group_lengths}"
            )
            return None

    ordered_xy = np.asarray(ordered_xy, dtype=float)

    if ordered_xy.shape[0] != expected_points:
        print(
            f"  UV fallback ordering failed: expected {expected_points}, "
            f"got {ordered_xy.shape[0]}"
        )
        return None

    # Convert [col, row] to MATLAB-style [row, col].
    pts_row_col = np.column_stack([ordered_xy[:, 1], ordered_xy[:, 0]])

    return pts_row_col


def get_corners(
    img: np.ndarray,
    board: LedGridBoard,
) -> Tuple[bool, Optional[np.ndarray], LedGridBoard]:
    """
    Detect the calibration pattern in one image.

    Mirrors ROS camera_calibration's ``get_corners`` return contract:
    ``(ok, corners, board)``. ``corners`` is ``(n_points, 2)`` in
    ``[row, col]`` order, x-major/y-minor -- aligned with the board's
    ``Xt``/``Yt`` generation order.
    """
    pts = _detect_grid_points(img, board.n_sq_x, board.n_sq_y)
    return (pts is not None), pts, board
