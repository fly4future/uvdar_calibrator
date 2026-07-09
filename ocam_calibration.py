"""
ocam_calibration_step.py

Single-file Python implementation of the OCamCalib workflow discussed:
  1) Read/import images
  2) Extract UV/checkerboard markers automatically
  3) Calibration with polynomial degree 4 by default
  4) Find center
  5) Calibration refinement / recompute corners
  6) Reproject on images
  7) Analyze error
  8) Show calibration results
  9) Save + export calib_results.txt

Required:
    pip install numpy matplotlib opencv-python

Optional, for MATLAB .mat output:
    pip install scipy

Example:
    python ./ocam_calibration.py --image_dir photos --gui

The image extractor supports: j, jpg, jpeg, bmp, png, tif, tiff.
"""

from __future__ import annotations

import argparse
import glob
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None


# -----------------------------------------------------------------------------
# Data structures
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


@dataclass
class CalibData:
    ima_proc: List[int] = field(default_factory=list)  # MATLAB-style 1-based image ids
    Xt: Optional[np.ndarray] = None
    Yt: Optional[np.ndarray] = None
    Xp_abs: Optional[np.ndarray] = None  # row coordinates, shape (points, 1, images)
    Yp_abs: Optional[np.ndarray] = None  # col coordinates, shape (points, 1, images)
    ocam_model: Optional[OCamModel] = None
    taylor_order_default: int = 4
    taylor_order: Optional[int] = None
    calibrated: bool = False
    RRfin: Optional[np.ndarray] = None  # shape (3, 3, images)
    n_ima: int = 0
    active_images: Optional[np.ndarray] = None
    ind_active: Optional[np.ndarray] = None
    I: List[np.ndarray] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    n_sq_x: int = 6
    n_sq_y: int = 4
    dX: float = 50.0
    dY: float = 50.0
    wintx: Optional[int] = None
    winty: Optional[int] = None
    no_image_file: int = 0


def _idx(i: int) -> int:
    """Convert MATLAB-style 1-based image number to Python index."""
    return int(i) - 1


def _as_col(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _require_calib(calib_data: CalibData) -> bool:
    if calib_data.n_ima == 0 or not calib_data.calibrated:
        print("\nNo calibration data available. You must first calibrate your camera.\n")
        return False
    return True


# -----------------------------------------------------------------------------
# Image import and marker extraction
# -----------------------------------------------------------------------------

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
        row, left-to-right. This matches Xt/Yt creation below:
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
        if len(col_groups) != expected_cols or not all(len(g) == expected_rows for g in col_groups):
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

def create_calib_data_from_images(
    image_dir: str,
    base_name: str = "i_",
    extension: str = "j",
    n_sq_x: int = 6,
    n_sq_y: int = 4,
    spacing_mm: float = 50.0,
    xc: Optional[float] = None,
    yc: Optional[float] = None,
) -> CalibData:
    files = find_image_files(image_dir, base_name, extension)

    if not files:
        raise FileNotFoundError(
            f"No images found in {image_dir!r} with base_name={base_name!r} "
            f"and extension={extension!r}. Supported extensions include "
            "jpg, jpeg, bmp, png, tif, tiff."
        )

    images = [_read_image_gray(p) for p in files]

    height, width = images[0].shape[:2]

    if xc is None:
        xc = height / 2.0

    if yc is None:
        yc = width / 2.0

    n_points = (n_sq_x + 1) * (n_sq_y + 1)

    # Object points: x-major, y-minor, matching MATLAB/OCamCalib indexing.
    # These are the expected board/world coordinates in millimeters.
    #
    # With n_sq_x=6, n_sq_y=4, spacing=50:
    #   point 1  -> Xt=0,   Yt=0
    #   point 2  -> Xt=0,   Yt=50
    #   ...
    #   point 5  -> Xt=0,   Yt=200
    #   point 6  -> Xt=50,  Yt=0
    #   ...
    Xt = []
    Yt = []

    for x in range(n_sq_x + 1):
        for y in range(n_sq_y + 1):
            Xt.append(x * spacing_mm)
            Yt.append(y * spacing_mm)

    Xt = np.asarray(Xt, dtype=float).reshape(-1, 1)
    Yt = np.asarray(Yt, dtype=float).reshape(-1, 1)

    Xp_abs = np.full((n_points, 1, len(images)), np.nan, dtype=float)
    Yp_abs = np.full((n_points, 1, len(images)), np.nan, dtype=float)

    ima_proc: List[int] = []

    preview_dir = Path(image_dir) / "detected_marker_previews"
    preview_dir.mkdir(exist_ok=True)

    print("Extracting UV/checkerboard markers...")
    print(f"pattern size {n_sq_x + 1}x{n_sq_y + 1}")

    for k, (path, img) in enumerate(zip(files, images), start=1):
        print(f"Processing image {k}: {Path(path).name}")
        print(f"  image reading successful: {img.shape[0]}x{img.shape[1]}")

        pts = _detect_grid_points(img, n_sq_x, n_sq_y)

        if pts is None:
            print(f"  image {k}: FAILED marker extraction -> {Path(path).name}")
            continue

        preview_path = preview_dir / f"{Path(path).stem}_detected.bmp"
        save_detected_marker_preview(img, pts, preview_path)
        print(f"  saved preview -> {preview_path}")

        debug_path = preview_dir / f"{Path(path).stem}_points.txt"

        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("index,row_px,col_px,Xt_mm,Yt_mm\n")
            for idx, (row, col) in enumerate(pts, start=1):
                f.write(
                    f"{idx},{row:.6f},{col:.6f},"
                    f"{Xt[idx - 1, 0]:.6f},{Yt[idx - 1, 0]:.6f}\n"
                )

        print(f"  saved point mapping -> {debug_path}")

        Xp_abs[:, 0, k - 1] = pts[:, 0]
        Yp_abs[:, 0, k - 1] = pts[:, 1]

        ima_proc.append(k)

        print(f"  image {k}: OK -> {Path(path).name}")

    if not ima_proc:
        raise RuntimeError(
            "No image markers were extracted successfully. Check image quality/pattern size."
        )

    calib_data = CalibData(
        ima_proc=ima_proc,
        Xt=Xt,
        Yt=Yt,
        Xp_abs=Xp_abs,
        Yp_abs=Yp_abs,
        ocam_model=OCamModel(
            xc=float(xc),
            yc=float(yc),
            width=int(width),
            height=int(height),
        ),
        n_ima=len(images),
        active_images=np.ones(len(images), dtype=int),
        ind_active=np.arange(1, len(images) + 1),
        I=images,
        image_paths=files,
        n_sq_x=n_sq_x,
        n_sq_y=n_sq_y,
        dX=spacing_mm,
        dY=spacing_mm,
    )

    return calib_data


# -----------------------------------------------------------------------------
# MATLAB conversions: projection and calibration
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
# Reprojection, center search, refinement, plots
# -----------------------------------------------------------------------------

def reprojectpoints(calib_data: CalibData) -> Tuple[np.ndarray, np.ndarray, float]:
    err = []
    stderr = []
    MSE = 0.0

    for i in calib_data.ima_proc:
        k = _idx(i)

        xx = calib_data.RRfin[:, :, k] @ np.vstack(
            [
                calib_data.Xt.ravel(),
                calib_data.Yt.ravel(),
                np.ones(calib_data.Xt.size),
            ]
        )

        xp, yp = omni3d2pixel(
            calib_data.ocam_model.ss,
            xx,
            calib_data.ocam_model.width,
            calib_data.ocam_model.height,
        )

        stt = np.sqrt(
            (calib_data.Xp_abs[:, 0, k] - calib_data.ocam_model.xc - xp) ** 2
            + (calib_data.Yp_abs[:, 0, k] - calib_data.ocam_model.yc - yp) ** 2
        )

        err.append(float(np.nanmean(stt)))
        stderr.append(float(np.nanstd(stt)))

        MSE += float(
            np.nansum(
                (calib_data.Xp_abs[:, 0, k] - calib_data.ocam_model.xc - xp) ** 2
                + (calib_data.Yp_abs[:, 0, k] - calib_data.ocam_model.yc - yp) ** 2
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

def calibration(
    calib_data: CalibData,
    taylor_order: Optional[int] = None,
    show_plots: bool = True,
) -> CalibData:
    if not calib_data.ima_proc or calib_data.Xp_abs is None:
        print("\nNo corner data available. Extract grid corners before calibrating.\n")
        return calib_data

    if taylor_order is None:
        taylor_order = calib_data.taylor_order_default

    calib_data.taylor_order = int(taylor_order)

    calib_data.ocam_model.c = 1.0
    calib_data.ocam_model.d = 0.0
    calib_data.ocam_model.e = 0.0

    RRfin, ss = calibrate(
        calib_data.Xt,
        calib_data.Yt,
        calib_data.Xp_abs,
        calib_data.Yp_abs,
        calib_data.ocam_model.xc,
        calib_data.ocam_model.yc,
        calib_data.taylor_order,
        calib_data.ima_proc,
    )

    if np.isscalar(RRfin) or np.size(RRfin) == 1:
        raise RuntimeError("Calibration failed while computing extrinsics.")

    calib_data.RRfin = RRfin
    calib_data.ocam_model.ss = np.asarray(ss, dtype=float)
    calib_data.calibrated = True

    reprojectpoints(calib_data)

    print("ss =")
    print(calib_data.ocam_model.ss)

    if show_plots:
        plot_calibration_results(calib_data)

    return calib_data

def findcenter(calib_data: CalibData) -> CalibData:
    if not calib_data.ima_proc or calib_data.Xp_abs is None:
        print("\nNo corner data available. Extract grid corners before calibrating.\n")
        return calib_data

    print("\nComputing center coordinates.\n")

    if calib_data.taylor_order is None:
        calib_data.taylor_order = calib_data.taylor_order_default

    pxc = calib_data.ocam_model.xc
    pyc = calib_data.ocam_model.yc

    width = calib_data.ocam_model.width
    height = calib_data.ocam_model.height

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
                    calib_data.Xt,
                    calib_data.Yt,
                    calib_data.Xp_abs,
                    calib_data.Yp_abs,
                    xc,
                    yc,
                    calib_data.taylor_order,
                    calib_data.ima_proc,
                )

                if np.isscalar(RRfin) or np.size(RRfin) == 1:
                    continue

                MSE = reprojectpoints_fun(
                    calib_data.Xt,
                    calib_data.Yt,
                    calib_data.Xp_abs,
                    calib_data.Yp_abs,
                    xc,
                    yc,
                    RRfin,
                    ss,
                    calib_data.ima_proc,
                    width,
                    height,
                )

                if not np.isnan(MSE):
                    MSEA[ic, jc] = MSE

        ind = np.unravel_index(np.argmin(MSEA), MSEA.shape)

        calib_data.ocam_model.xc = float(xreg[ind])
        calib_data.ocam_model.yc = float(yreg[ind])

        dx_reg = abs((xregstop - xregstart) / xceil)
        dy_reg = abs((yregstop - yregstart) / yceil)

        xregstart = calib_data.ocam_model.xc - dx_reg
        xregstop = calib_data.ocam_model.xc + dx_reg

        yregstart = calib_data.ocam_model.yc - dy_reg
        yregstop = calib_data.ocam_model.yc + dy_reg

        print(f"{glc}...", end="", flush=True)

    print("\n")

    calib_data.RRfin, calib_data.ocam_model.ss = calibrate(
        calib_data.Xt,
        calib_data.Yt,
        calib_data.Xp_abs,
        calib_data.Yp_abs,
        calib_data.ocam_model.xc,
        calib_data.ocam_model.yc,
        calib_data.taylor_order,
        calib_data.ima_proc,
    )

    reprojectpoints(calib_data)

    print("xc =", calib_data.ocam_model.xc)
    print("yc =", calib_data.ocam_model.yc)

    calib_data.calibrated = True

    return calib_data


def findcenter_fast(
    calib_data: CalibData,
    iterations: int = 4,
    grid_size: int = 3,
    start_radius_fraction: float = 0.20,
) -> CalibData:
    """
    Faster center search.

    The original MATLAB-style findcenter search is accurate but slow because it
    recalibrates many times. This fast version searches a smaller grid and
    shrinks the search radius after each pass.
    """

    if not calib_data.ima_proc or calib_data.Xp_abs is None:
        print("\nNo corner data available. Extract grid corners before calibrating.\n")
        return calib_data

    print("\nComputing center coordinates using FAST search.\n")

    if calib_data.taylor_order is None:
        calib_data.taylor_order = calib_data.taylor_order_default

    width = calib_data.ocam_model.width
    height = calib_data.ocam_model.height

    best_xc = float(calib_data.ocam_model.xc)
    best_yc = float(calib_data.ocam_model.yc)

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
                    calib_data.Xt,
                    calib_data.Yt,
                    calib_data.Xp_abs,
                    calib_data.Yp_abs,
                    float(xc),
                    float(yc),
                    calib_data.taylor_order,
                    calib_data.ima_proc,
                )

                if np.isscalar(RRfin) or np.size(RRfin) == 1:
                    continue

                mse = reprojectpoints_fun(
                    calib_data.Xt,
                    calib_data.Yt,
                    calib_data.Xp_abs,
                    calib_data.Yp_abs,
                    float(xc),
                    float(yc),
                    RRfin,
                    ss,
                    calib_data.ima_proc,
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
        return calib_data

    calib_data.ocam_model.xc = best_xc
    calib_data.ocam_model.yc = best_yc
    calib_data.RRfin = best_RRfin
    calib_data.ocam_model.ss = np.asarray(best_ss, dtype=float)
    calib_data.calibrated = True

    print("\nFast center search complete.")
    print("xc =", calib_data.ocam_model.xc)
    print("yc =", calib_data.ocam_model.yc)

    reprojectpoints(calib_data)

    return calib_data


def recomp_corner_calib(
    calib_data: CalibData,
    wintx: Optional[int] = None,
    winty: Optional[int] = None,
    min_movement_px: float = 0.05,
    max_movement_px: float = 8.0,
) -> CalibData:
    """
    Refine detected image points near their current measured locations.

    Important safety rule:
        Never replace measured points with the model reprojection itself.

    The older Python version used the current model projection as the initial
    point and, if refinement failed, wrote the projected points into Xp_abs/Yp_abs.
    That can make reprojection error become exactly 0.000000, because the code is
    then comparing the model against points generated by the model. This function
    avoids that by starting from the existing measured points and keeping the
    original measured points whenever refinement is suspicious or unavailable.
    """
    if not calib_data.ima_proc or calib_data.Xp_abs is None:
        print("\nNo corner data available. Extract grid corners before calibrating.\n")
        return calib_data

    if cv2 is None:
        print("OpenCV unavailable; keeping original detected points.")
        return calib_data

    if wintx is None:
        wintx = calib_data.wintx or max(
            round(calib_data.ocam_model.width / 128),
            round(calib_data.ocam_model.height / 96),
        )

    if winty is None:
        winty = calib_data.winty or wintx

    calib_data.wintx = int(round(wintx))
    calib_data.winty = int(round(winty))

    print(f"Window size = {2 * calib_data.wintx + 1}x{2 * calib_data.winty + 1}")
    print("Safe corner refinement: refining from measured UV points, not model projections.")

    changed_total = 0
    kept_total = 0

    for kk in calib_data.ima_proc:
        k = _idx(kk)
        I = calib_data.I[k].astype(np.uint8)

        # OpenCV cornerSubPix expects points as [col, row]. Existing data stores
        # Xp_abs=row and Yp_abs=col.
        original_cv = np.vstack(
            [
                calib_data.Yp_abs[:, 0, k],
                calib_data.Xp_abs[:, 0, k],
            ]
        ).T.astype(np.float32)

        if calib_data.wintx == 0 or calib_data.winty == 0:
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
                (calib_data.winty, calib_data.wintx),
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

            calib_data.Yp_abs[:, 0, k] = updated[:, 0]
            calib_data.Xp_abs[:, 0, k] = updated[:, 1]

            changed = int(np.sum((movement >= min_movement_px) & usable))
            kept = int(np.sum(~usable))
            changed_total += changed
            kept_total += kept

            print(
                f"  image {kk}: refined {changed}/{len(movement)} points "
                f"(median move={np.median(movement):.3f}px, max move={np.max(movement):.3f}px)"
            )

        except Exception as exc:
            print(f"  image {kk}: corner refinement failed ({exc}); keeping original detected points")
            kept_total += original_cv.shape[0]

    print(f"Corners recomputed safely. Changed points: {changed_total}, kept/rejected: {kept_total}\nDone")
    return calib_data

def check_active_images(calib_data: CalibData) -> CalibData:
    if calib_data.n_ima != 0:
        if calib_data.active_images is None:
            calib_data.active_images = np.ones(calib_data.n_ima, dtype=int)

        n_act = len(calib_data.active_images)

        if n_act < calib_data.n_ima:
            calib_data.active_images = np.concatenate(
                [
                    calib_data.active_images,
                    np.ones(calib_data.n_ima - n_act, dtype=int),
                ]
            )
        elif n_act > calib_data.n_ima:
            calib_data.active_images = calib_data.active_images[: calib_data.n_ima]

        calib_data.ind_active = np.where(calib_data.active_images != 0)[0] + 1

        if np.prod(calib_data.active_images == 0):
            print("Error: There is no active image. Run Add/Suppress images to add images")

    return calib_data


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


def reproject_calib(calib_data: CalibData) -> None:
    if not _require_calib(calib_data):
        return

    check_active_images(calib_data)

    colors = "brgkcm"

    if calib_data.ocam_model.ss is None:
        print("Need to calibrate before showing image reprojection.")
        return

    for kk in calib_data.ima_proc:
        k = _idx(kk)

        if k < len(calib_data.I):
            I = calib_data.I[k]
        else:
            I = 255 * np.ones(
                (
                    calib_data.ocam_model.height,
                    calib_data.ocam_model.width,
                )
            )

        xx = calib_data.RRfin[:, :, k] @ np.vstack(
            [
                calib_data.Xt.ravel(),
                calib_data.Yt.ravel(),
                np.ones(calib_data.Xt.size),
            ]
        )

        m = world2cam(xx, calib_data.ocam_model)

        xp = m[0, :]
        yp = m[1, :]

        plt.figure(5 + kk)
        plt.clf()

        plt.imshow(I, cmap="gray")

        plt.title(f"Image {kk} - Image points (+) and reprojected grid points (o)")

        plt.plot(
            calib_data.Yp_abs[:, 0, k],
            calib_data.Xp_abs[:, 0, k],
            "r+",
        )

        plt.plot(
            yp,
            xp,
            colors[(kk - 1) % 6] + "o",
            fillstyle="none",
        )

        plt.plot(
            calib_data.ocam_model.yc,
            calib_data.ocam_model.xc,
            "ro",
            fillstyle="none",
        )

        plt.axis(
            [
                1,
                calib_data.ocam_model.width,
                calib_data.ocam_model.height,
                1,
            ]
        )

        draw_axes(
            calib_data.Xp_abs[:, 0, k],
            calib_data.Yp_abs[:, 0, k],
            calib_data.n_sq_y,
        )

    plt.show()


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


def analyse_error(calib_data: CalibData) -> None:
    if not _require_calib(calib_data):
        return

    plt.figure(5)
    plt.clf()

    colors = "brgkcm"

    err = []
    stderr = []
    MSE = 0.0

    if (
        calib_data.ocam_model.c is None
        and calib_data.ocam_model.d is None
        and calib_data.ocam_model.e is None
    ):
        calib_data.ocam_model.c = 1.0
        calib_data.ocam_model.d = 0.0
        calib_data.ocam_model.e = 0.0

    for i in calib_data.ima_proc:
        k = _idx(i)

        xx = calib_data.RRfin[:, :, k] @ np.vstack(
            [
                calib_data.Xt.ravel(),
                calib_data.Yt.ravel(),
                np.ones(calib_data.Xt.size),
            ]
        )

        xp1, yp1 = omni3d2pixel(
            calib_data.ocam_model.ss,
            xx,
            calib_data.ocam_model.width,
            calib_data.ocam_model.height,
        )

        xp = (
            xp1 * calib_data.ocam_model.c
            + yp1 * calib_data.ocam_model.d
            + calib_data.ocam_model.xc
        )

        yp = (
            xp1 * calib_data.ocam_model.e
            + yp1
            + calib_data.ocam_model.yc
        )

        sqerr = (calib_data.Xp_abs[:, 0, k] - xp) ** 2 + (
            calib_data.Yp_abs[:, 0, k] - yp
        ) ** 2

        err.append(float(np.nanmean(np.sqrt(sqerr))))
        stderr.append(float(np.nanstd(np.sqrt(sqerr))))

        MSE += float(np.nansum(sqerr))

        plt.plot(
            calib_data.Xp_abs[:, 0, k] - xp,
            calib_data.Yp_abs[:, 0, k] - yp,
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
    print(calib_data.ocam_model.ss)

    plt.show()


def plot_calibration_results(calib_data: CalibData) -> None:
    ss = calib_data.ocam_model.ss

    if ss is None:
        raise ValueError("Cannot plot calibration results because ocam_model.ss is empty.")

    rho = np.arange(
        0,
        int(np.floor(calib_data.ocam_model.width / 2)) + 1,
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


def show_calib_results(calib_data: CalibData) -> None:
    if not _require_calib(calib_data):
        return

    M = np.column_stack(
        [
            calib_data.Xt.ravel(),
            calib_data.Yt.ravel(),
            np.zeros(calib_data.Xt.size),
        ]
    )

    reprojectpoints_adv(
        calib_data.ocam_model,
        calib_data.RRfin,
        calib_data.ima_proc,
        calib_data.Xp_abs,
        calib_data.Yp_abs,
        M,
    )

    print("ss =")
    print(calib_data.ocam_model.ss)

    print("xc =")
    print(calib_data.ocam_model.xc)

    print("yc =")
    print(calib_data.ocam_model.yc)

    plot_calibration_results(calib_data)


def show_extrinsic(calib_data: CalibData) -> None:
    if not _require_calib(calib_data):
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        calib_data.Xt.ravel(),
        calib_data.Yt.ravel(),
        np.zeros(calib_data.Xt.size),
        marker="o",
    )

    for i in calib_data.ima_proc:
        k = _idx(i)
        T = calib_data.RRfin[:, 2, k]

        ax.scatter(T[0], T[1], T[2], marker="^")
        ax.text(T[0], T[1], T[2], str(i))

    ax.set_title("Extrinsic approximation from RRfin translations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


# -----------------------------------------------------------------------------
# Save/export
# -----------------------------------------------------------------------------

def _serializable_calib_dict(calib_data: CalibData) -> dict:
    return {
        "Xt": calib_data.Xt,
        "Yt": calib_data.Yt,
        "Xp_abs": calib_data.Xp_abs,
        "Yp_abs": calib_data.Yp_abs,
        "RRfin": calib_data.RRfin,
        "ima_proc": np.asarray(calib_data.ima_proc),
        "taylor_order": calib_data.taylor_order,
        "xc": calib_data.ocam_model.xc,
        "yc": calib_data.ocam_model.yc,
        "width": calib_data.ocam_model.width,
        "height": calib_data.ocam_model.height,
        "c": calib_data.ocam_model.c,
        "d": calib_data.ocam_model.d,
        "e": calib_data.ocam_model.e,
        "ss": calib_data.ocam_model.ss,
        "invpol": calib_data.ocam_model.invpol,
    }


def saving_calib(
    calib_data: CalibData,
    output_dir: str = ".",
) -> None:
    if not _require_calib(calib_data):
        return

    check_active_images(calib_data)

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

    np.savez(npz_path, **_serializable_calib_dict(calib_data))

    try:
        # SciPy is optional. Import dynamically so editors do not warn if it
        # is not installed; the .npz file is always saved above.
        scipy_io = __import__("scipy.io", fromlist=["savemat"])
        scipy_io.savemat(
            out / "Omni_Calib_Results.mat",
            {"calib_data": _serializable_calib_dict(calib_data)},
        )
    except Exception:
        pass

    print("done")


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


def exportData2TXT(
    calib_data: CalibData,
    output_dir: str = ".",
) -> None:
    if not _require_calib(calib_data):
        return

    print('Exporting ocam_model to "calib_results.txt"')

    export_data(calib_data.ocam_model, output_dir=output_dir)

    print("done")


def export_calib_results_txt(
    calib_data: CalibData,
    output_dir: str = ".",
) -> None:
    exportData2TXT(calib_data, output_dir=output_dir)



# -----------------------------------------------------------------------------
# Calibration image coverage guidance
# -----------------------------------------------------------------------------

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


def _uvdar_sample_metric(calib_data: CalibData, image_number: int) -> Optional[dict]:
    k = _idx(image_number)
    if calib_data.Xp_abs is None or calib_data.Yp_abs is None:
        return None

    rows = calib_data.Xp_abs[:, 0, k]
    cols = calib_data.Yp_abs[:, 0, k]
    ok = np.isfinite(rows) & np.isfinite(cols)
    if np.sum(ok) < 4:
        return None

    rows = rows[ok]
    cols = cols[ok]
    height = float(calib_data.ocam_model.height)
    width = float(calib_data.ocam_model.width)

    x_center = float(np.mean(cols) / max(width, 1.0))
    y_center = float(np.mean(rows) / max(height, 1.0))
    bbox_w = float(np.max(cols) - np.min(cols))
    bbox_h = float(np.max(rows) - np.min(rows))
    size = float(np.sqrt(max(bbox_w * bbox_h, 0.0) / max(width * height, 1.0)))

    # Estimate perspective tilt/skew from the two grid axes. The data is stored
    # x-major/y-minor, so reshape as [x_index, y_index, row_col].
    skew = 0.0
    try:
        nx = calib_data.n_sq_x + 1
        ny = calib_data.n_sq_y + 1
        rc = np.column_stack([calib_data.Xp_abs[:, 0, k], calib_data.Yp_abs[:, 0, k]])
        rc = rc.reshape(nx, ny, 2)
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
        "image": image_number,
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


def compute_calibration_coverage(calib_data: CalibData, target_images: int = 25) -> dict:
    metrics = []
    for i in calib_data.ima_proc:
        m = _uvdar_sample_metric(calib_data, i)
        if m is not None:
            metrics.append(m)

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

    filled = (
        len(required_x & x_bins)
        + len(required_y & y_bins)
        + len(required_size & size_bins)
        + len(required_skew & skew_bins)
        + len(required_quadrants & quadrants)
    )
    total = len(required_x) + len(required_y) + len(required_size) + len(required_skew) + len(required_quadrants)
    count_score = min(len(metrics), target_images) / max(target_images, 1)
    coverage_score = filled / max(total, 1)
    overall = 100.0 * min(count_score, coverage_score)
    complete = len(metrics) >= target_images and all(len(v) == 0 for v in missing.values())

    return {
        "metrics": metrics,
        "target_images": int(target_images),
        "accepted_images": len(metrics),
        "filled": filled,
        "total": total,
        "overall": overall,
        "complete": complete,
        "missing": missing,
        "sets": {
            "x": x_bins,
            "y": y_bins,
            "size": size_bins,
            "tilt": skew_bins,
            "quadrants": quadrants,
        },
    }


def _coverage_suggestions(report: dict) -> List[str]:
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
    if report["accepted_images"] < report["target_images"]:
        suggestions.append(
            f"add at least {report['target_images'] - report['accepted_images']} more accepted image(s)"
        )
    return suggestions


def format_calibration_coverage(report: dict) -> str:
    sets = report["sets"]
    missing = report["missing"]
    lines = []
    lines.append("\nStep 2b: Calibration image coverage report")
    lines.append("Note: calibration has no universal exact image count; this uses a coverage stop condition.")
    lines.append(f"Accepted images: {report['accepted_images']} / target {report['target_images']}")
    lines.append(f"Coverage score: {report['overall']:.1f}%")
    lines.append(_coverage_bar("count", min(report['accepted_images'], report['target_images']), report['target_images']))
    lines.append(_coverage_bar("x position", len(sets["x"]), len(COVERAGE_X_BINS)))
    lines.append(_coverage_bar("y position", len(sets["y"]), len(COVERAGE_Y_BINS)))
    lines.append(_coverage_bar("size", len(sets["size"]), len(COVERAGE_SIZE_BINS)))
    lines.append(_coverage_bar("tilt", len(sets["tilt"] & {"front-on", "moderately tilted"}), 2))
    lines.append(_coverage_bar("quadrants", len(sets["quadrants"]), 4))

    if report["complete"]:
        lines.append("Status: COMPLETE. You have enough varied images to run calibration.")
    else:
        lines.append("Status: NOT COMPLETE. Add more varied images before trusting final calibration.")
        suggestions = _coverage_suggestions(report)
        if suggestions:
            lines.append("Next images to capture:")
            for s in suggestions:
                lines.append(f"  - {s}")

    lines.append("\nPer-image coverage metrics:")
    lines.append("image,x_bin,y_bin,size_bin,tilt_bin,x_norm,y_norm,size,tilt")
    for m in report["metrics"]:
        lines.append(
            f"{m['image']},{m['x_bin']},{m['y_bin']},{m['size_bin']},{m['skew_bin']},"
            f"{m['x']:.3f},{m['y']:.3f},{m['size']:.3f},{m['skew']:.3f}"
        )
    return "\n".join(lines)


def print_calibration_coverage(
    calib_data: CalibData,
    target_images: int = 25,
    output_dir: str = ".",
    save_report: bool = True,
) -> dict:
    report = compute_calibration_coverage(calib_data, target_images=target_images)
    text = format_calibration_coverage(report)
    print(text)
    if save_report:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "calibration_coverage.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved coverage report to {path}")
    return report


def plot_calibration_coverage(calib_data: CalibData, target_images: int = 25) -> None:
    report = compute_calibration_coverage(calib_data, target_images=target_images)
    metrics = report["metrics"]
    if not metrics:
        return

    xs = [m["x"] for m in metrics]
    ys = [m["y"] for m in metrics]
    sizes = [m["size"] * 450.0 + 20.0 for m in metrics]

    plt.figure("UV-DAR calibration coverage")
    plt.clf()
    plt.scatter(xs, ys, s=sizes, alpha=0.65)
    for m in metrics:
        plt.text(m["x"], m["y"], str(m["image"]), fontsize=8)
    plt.axvline(0.33, linestyle="--", linewidth=1)
    plt.axvline(0.67, linestyle="--", linewidth=1)
    plt.axhline(0.33, linestyle="--", linewidth=1)
    plt.axhline(0.67, linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.xlabel("horizontal board location in image")
    plt.ylabel("vertical board location in image")
    plt.title(f"Calibration coverage: {report['overall']:.1f}%")
    plt.grid(True)
    plt.show()



# -----------------------------------------------------------------------------
# Tkinter coverage GUI
# -----------------------------------------------------------------------------

def launch_coverage_gui(
    image_dir: str = "photos",
    base_name: str = "i_",
    extension: str = "bmp",
    n_sq_x: int = 6,
    n_sq_y: int = 4,
    spacing_mm: float = 50.0,
    taylor_order: int = 4,
    output_dir: str = ".",
    target_images: int = 25,
    slow_find_center: bool = True,
) -> None:
    """Launch a ROS camera_calibration-style offline GUI for UV-DAR images."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError("Tkinter is required for --gui. Install python-tk/tkinter.") from exc

    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")

    class UVDARCalibrationGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("UV-DAR / OCamCalib Calibration Assistant")
            self.root.geometry("1180x760")
            self.root.minsize(1000, 650)

            self.image_dir = tk.StringVar(value=image_dir)
            self.base_name = tk.StringVar(value="")
            self.extension = tk.StringVar(value="all")
            self.n_sq_x = tk.IntVar(value=n_sq_x)
            self.n_sq_y = tk.IntVar(value=n_sq_y)
            self.spacing_mm = tk.DoubleVar(value=spacing_mm)
            self.taylor_order = tk.IntVar(value=taylor_order)
            self.output_dir = tk.StringVar(value=output_dir)
            self.target_images = tk.IntVar(value=target_images)
            self.slow_find_center = tk.BooleanVar(value=slow_find_center)
            self.no_plots = tk.BooleanVar(value=True)
            self.require_coverage = tk.BooleanVar(value=False)

            self.calib_data = None
            self.coverage_report = None
            self.current_image_index = 0
            self.photo_ref = None

            self._build_widgets()
            self._set_status("Choose a folder and click Load / Analyze Images.")

        def _build_widgets(self):
            top = ttk.Frame(self.root, padding=8)
            top.pack(side=tk.TOP, fill=tk.X)

            ttk.Label(top, text="Image folder:").grid(row=0, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.image_dir, width=45).grid(row=0, column=1, sticky="ew", padx=4)
            ttk.Button(top, text="Browse", command=self._browse_images).grid(row=0, column=2, padx=4)

            ttk.Label(top, text="Base:").grid(row=0, column=3, sticky="e")
            ttk.Entry(top, textvariable=self.base_name, width=8).grid(row=0, column=4, padx=4)
            ttk.Label(top, text="Ext:").grid(row=0, column=5, sticky="e")
            ttk.Entry(top, textvariable=self.extension, width=7).grid(row=0, column=6, padx=4)
            ttk.Label(top, text="Target:").grid(row=0, column=7, sticky="e")
            ttk.Spinbox(top, from_=8, to=200, textvariable=self.target_images, width=5).grid(row=0, column=8, padx=4)
            ttk.Button(top, text="Load / Analyze Images", command=self.load_images).grid(row=0, column=9, padx=8)
            top.columnconfigure(1, weight=1)

            opts = ttk.Frame(self.root, padding=(8, 0, 8, 6))
            opts.pack(side=tk.TOP, fill=tk.X)
            ttk.Label(opts, text="Grid squares X:").pack(side=tk.LEFT)
            ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_x, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Y:").pack(side=tk.LEFT)
            ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_y, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Spacing mm:").pack(side=tk.LEFT, padx=(12, 0))
            ttk.Entry(opts, textvariable=self.spacing_mm, width=7).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Taylor:").pack(side=tk.LEFT, padx=(12, 0))
            ttk.Spinbox(opts, from_=4, to=10, textvariable=self.taylor_order, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Checkbutton(opts, text="MATLAB-style slow Find Center", variable=self.slow_find_center).pack(side=tk.LEFT, padx=12)
            ttk.Checkbutton(opts, text="No plots during calibration", variable=self.no_plots).pack(side=tk.LEFT, padx=8)
            ttk.Checkbutton(opts, text="Require coverage before calibrating", variable=self.require_coverage).pack(side=tk.LEFT, padx=8)

            main = ttk.Frame(self.root, padding=8)
            main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            left = ttk.Frame(main)
            left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            right = ttk.Frame(main, width=330)
            right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

            self.image_canvas = tk.Canvas(left, bg="#202020", highlightthickness=1, highlightbackground="#888")
            self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            nav = ttk.Frame(left, padding=(0, 8, 0, 0))
            nav.pack(side=tk.BOTTOM, fill=tk.X)
            ttk.Button(nav, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
            ttk.Button(nav, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=4)
            self.image_label = ttk.Label(nav, text="No image loaded")
            self.image_label.pack(side=tk.LEFT, padx=12)

            title = ttk.Label(right, text="Calibration Coverage", font=("Segoe UI", 14, "bold"))
            title.pack(anchor="w")
            self.score_label = ttk.Label(right, text="Coverage: --", font=("Segoe UI", 11, "bold"))
            self.score_label.pack(anchor="w", pady=(8, 2))
            self.status_label = ttk.Label(right, text="Status: not analyzed", wraplength=310)
            self.status_label.pack(anchor="w", pady=(0, 10))

            self.bar_canvas = tk.Canvas(right, width=310, height=205, bg="white", highlightthickness=1, highlightbackground="#ccc")
            self.bar_canvas.pack(anchor="w", pady=(0, 10))

            ttk.Label(right, text="Next images to capture:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            self.suggestion_box = tk.Text(right, height=8, width=42, wrap="word")
            self.suggestion_box.pack(anchor="w", fill=tk.X, pady=(4, 10))
            self.suggestion_box.configure(state="disabled")

            ttk.Label(right, text="Details:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            self.details_box = tk.Text(right, height=10, width=42, wrap="none")
            self.details_box.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(4, 10))
            self.details_box.configure(state="disabled")

            actions = ttk.Frame(right)
            actions.pack(side=tk.BOTTOM, fill=tk.X)
            self.calibrate_button = ttk.Button(actions, text="CALIBRATE", command=self.calibrate, state="disabled")
            self.calibrate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            ttk.Button(actions, text="Save Report", command=self.save_report).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Button(actions, text="Exit", command=self.root.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

            self.bottom_status = ttk.Label(self.root, text="", relief=tk.SUNKEN, anchor="w", padding=4)
            self.bottom_status.pack(side=tk.BOTTOM, fill=tk.X)

        def _browse_images(self):
            folder = filedialog.askdirectory(initialdir=self.image_dir.get() or ".")
            if folder:
                self.image_dir.set(folder)

        def _set_status(self, text):
            self.bottom_status.configure(text=text)
            self.root.update_idletasks()

        def load_images(self):
            try:
                self._set_status("Loading images and extracting UV/checkerboard markers...")
                self.calib_data = create_calib_data_from_images(
                    image_dir=self.image_dir.get(),
                    base_name=self.base_name.get(),
                    extension=self.extension.get(),
                    n_sq_x=int(self.n_sq_x.get()),
                    n_sq_y=int(self.n_sq_y.get()),
                    spacing_mm=float(self.spacing_mm.get()),
                )
                self.coverage_report = compute_calibration_coverage(
                    self.calib_data,
                    target_images=int(self.target_images.get()),
                )
                self.current_image_index = 0
                self._update_coverage_panel()
                self._show_current_image()
                self._set_status("Analysis complete. Capture the suggested missing poses or run calibration.")
            except Exception as exc:
                messagebox.showerror("Load/analyze failed", str(exc))
                self._set_status("Load/analyze failed.")

        def _update_coverage_panel(self):
            report = self.coverage_report
            if not report:
                return
            self.score_label.configure(text=f"Coverage: {report['overall']:.1f}%")
            status = "READY TO CALIBRATE" if report["complete"] else "NOT COMPLETE"
            self.status_label.configure(text=f"Status: {status}   ({report['accepted_images']} / {report['target_images']} accepted images)")
            allow = report["complete"] or not self.require_coverage.get()
            self.calibrate_button.configure(state=("normal" if allow else "disabled"))
            self._draw_bars(report)

            suggestions = _coverage_suggestions(report)
            if not suggestions:
                suggestions = ["Coverage complete. You can calibrate now."]
            self._write_text(self.suggestion_box, "\n".join(f"• {s}" for s in suggestions))
            self._write_text(self.details_box, format_calibration_coverage(report))

        def _draw_bars(self, report):
            c = self.bar_canvas
            c.delete("all")
            sets = report["sets"]
            rows = [
                ("Images", min(report["accepted_images"], report["target_images"]), report["target_images"]),
                ("X", len(sets["x"]), len(COVERAGE_X_BINS)),
                ("Y", len(sets["y"]), len(COVERAGE_Y_BINS)),
                ("Size", len(sets["size"]), len(COVERAGE_SIZE_BINS)),
                ("Skew", len(sets["tilt"] & {"front-on", "moderately tilted"}), 2),
                ("Quadrants", len(sets["quadrants"]), 4),
            ]
            y = 18
            for label, val, total in rows:
                total = max(total, 1)
                frac = max(0, min(1, val / total))
                fill = "#55b96f" if val >= total else "#e0b642"
                c.create_text(10, y, text=label, anchor="w", font=("Segoe UI", 9, "bold"))
                c.create_rectangle(95, y - 9, 285, y + 9, outline="#999", fill="#eee")
                c.create_rectangle(95, y - 9, 95 + 190 * frac, y + 9, outline="", fill=fill)
                c.create_text(295, y, text=f"{val}/{total}", anchor="e", font=("Segoe UI", 9))
                y += 31

        def _write_text(self, widget, text):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

        def _show_current_image(self):
            if not self.calib_data or not self.calib_data.ima_proc:
                return
            i = self.calib_data.ima_proc[self.current_image_index]
            k = _idx(i)
            img = self.calib_data.I[k]
            if img.ndim == 2:
                preview = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                preview = img.copy()
            if self.calib_data.Xp_abs is not None:
                pts = np.column_stack([self.calib_data.Xp_abs[:, 0, k], self.calib_data.Yp_abs[:, 0, k]])
                for idx, (row, col) in enumerate(pts, start=1):
                    if not np.isfinite(row) or not np.isfinite(col):
                        continue
                    cv2.circle(preview, (int(round(col)), int(round(row))), 5, (0, 0, 255), 1)
                    cv2.putText(preview, str(idx), (int(round(col)) + 5, int(round(row)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

            cw = max(50, self.image_canvas.winfo_width())
            ch = max(50, self.image_canvas.winfo_height())
            h, w = preview.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(preview, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".ppm", rgb)
            if not ok:
                return
            import tkinter as tk
            self.photo_ref = tk.PhotoImage(data=buf.tobytes())
            self.image_canvas.delete("all")
            self.image_canvas.create_image(cw // 2, ch // 2, image=self.photo_ref, anchor="center")
            path = Path(self.calib_data.image_paths[k]).name if k < len(self.calib_data.image_paths) else f"image {i}"
            metric = _uvdar_sample_metric(self.calib_data, i)
            extra = ""
            if metric:
                extra = f"   X:{metric['x_bin']} Y:{metric['y_bin']} Size:{metric['size_bin']} Skew:{metric['skew_bin']}"
            self.image_label.configure(text=f"Image {self.current_image_index + 1}/{len(self.calib_data.ima_proc)}: {path}{extra}")

        def next_image(self):
            if not self.calib_data or not self.calib_data.ima_proc:
                return
            self.current_image_index = (self.current_image_index + 1) % len(self.calib_data.ima_proc)
            self._show_current_image()

        def prev_image(self):
            if not self.calib_data or not self.calib_data.ima_proc:
                return
            self.current_image_index = (self.current_image_index - 1) % len(self.calib_data.ima_proc)
            self._show_current_image()

        def save_report(self):
            if not self.coverage_report:
                messagebox.showinfo("No report", "Load/analyze images first.")
                return
            out = Path(self.output_dir.get())
            out.mkdir(parents=True, exist_ok=True)
            path = out / "calibration_coverage.txt"
            path.write_text(format_calibration_coverage(self.coverage_report) + "\n", encoding="utf-8")
            messagebox.showinfo("Saved", f"Saved coverage report to:\n{path}")

        def calibrate(self):
            if self.coverage_report and self.require_coverage.get() and not self.coverage_report["complete"]:
                messagebox.showwarning("Coverage incomplete", "Coverage is not complete yet. Add the suggested images first.")
                return
            if self.coverage_report and not self.coverage_report["complete"]:
                proceed = messagebox.askyesno("Coverage incomplete", "Coverage is not complete. Calibrate anyway?")
                if not proceed:
                    return
            try:
                self._set_status("Running full UV-DAR calibration. This can take a while...")
                run_full_calibration_workflow(
                    image_dir=self.image_dir.get(),
                    base_name=self.base_name.get(),
                    extension=self.extension.get(),
                    n_sq_x=int(self.n_sq_x.get()),
                    n_sq_y=int(self.n_sq_y.get()),
                    spacing_mm=float(self.spacing_mm.get()),
                    taylor_order=int(self.taylor_order.get()),
                    output_dir=self.output_dir.get(),
                    do_plots=not self.no_plots.get(),
                    do_find_center=True,
                    fast_find_center=not self.slow_find_center.get(),
                    do_corner_refinement=False,
                    target_images=int(self.target_images.get()),
                    coverage_only=False,
                    require_coverage=self.require_coverage.get(),
                    show_coverage=False,
                )
                self._set_status("Calibration complete. Saved Omni_Calib_Results and calib_results.txt.")
                messagebox.showinfo("Calibration complete", "Calibration finished and exported calib_results.txt.")
            except Exception as exc:
                messagebox.showerror("Calibration failed", str(exc))
                self._set_status("Calibration failed.")

    root = tk.Tk()
    app = UVDARCalibrationGUI(root)
    root.bind("<Left>", lambda _e: app.prev_image())
    root.bind("<Right>", lambda _e: app.next_image())
    root.mainloop()


# -----------------------------------------------------------------------------
# Full workflow / CLI
# -----------------------------------------------------------------------------

def run_full_calibration_workflow(
    image_dir: str,
    base_name: str = "i_",
    extension: str = "j",
    n_sq_x: int = 6,
    n_sq_y: int = 4,
    spacing_mm: float = 50.0,
    taylor_order: int = 4,
    output_dir: str = ".",
    do_plots: bool = True,
    do_find_center: bool = True,
    fast_find_center: bool = True,
    do_corner_refinement: bool = False,
    target_images: int = 25,
    coverage_only: bool = False,
    require_coverage: bool = False,
    show_coverage: bool = False,
) -> CalibData:
    calib_data = create_calib_data_from_images(
        image_dir=image_dir,
        base_name=base_name,
        extension=extension,
        n_sq_x=n_sq_x,
        n_sq_y=n_sq_y,
        spacing_mm=spacing_mm,
    )

    coverage_report = print_calibration_coverage(
        calib_data,
        target_images=target_images,
        output_dir=output_dir,
        save_report=True,
    )

    if show_coverage:
        plot_calibration_coverage(calib_data, target_images=target_images)

    if coverage_only:
        print("\nCoverage-only mode complete. No calibration was run.")
        return calib_data

    if require_coverage and not coverage_report["complete"]:
        print("\nStopping because --require_coverage was used and coverage is not complete.")
        print("Add the suggested images above, then rerun calibration.")
        return calib_data

    print("\nStep 3: Initial calibration")
    calib_data = calibration(
        calib_data,
        taylor_order=taylor_order,
        show_plots=False,
    )

    if do_find_center:
        print("\nStep 4: Find center")
        if fast_find_center:
            calib_data = findcenter_fast(calib_data)
        else:
            calib_data = findcenter(calib_data)
    else:
        print("\nStep 4: Skipping Find center")

    if do_corner_refinement:
        print("\nStep 5: Calibration refinement / recompute corners")
        calib_data = recomp_corner_calib(calib_data)
    else:
        print("\nStep 5: Skipping corner recomputation for UV dot pattern")
        print("Reason: UV-dot extraction already gives measured marker centers; recomputing corners is optional.")

    print("\nStep 5b: Final calibration after center finding")
    calib_data = calibration(
        calib_data,
        taylor_order=taylor_order,
        show_plots=False,
    )

    if do_plots:
        print("\nStep 6: Reproject on images")
        reproject_calib(calib_data)

        print("\nStep 7: Analyze error")
        analyse_error(calib_data)

        print("\nStep 8: Show calibration results")
        show_calib_results(calib_data)

        print("\nStep 8b: Show extrinsic")
        show_extrinsic(calib_data)

    print("\nStep 9: Save calibration")
    saving_calib(calib_data, output_dir=output_dir)

    print("\nStep 9b: Export calib_results.txt")
    exportData2TXT(calib_data, output_dir=output_dir)

    print("\nCalibration workflow complete.")
    return calib_data


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run OCamCalib-style camera calibration from images."
    )

    p.add_argument(
        "--image_dir",
        default=".",
        help="Folder containing calibration images.",
    )

    p.add_argument(
        "--base_name",
        default="",
        help="Image filename prefix. Empty means use all names.")
    
    p.add_argument(
        "--extension",
        default="all",
        help="Image extension. Use 'all' to load jpg/jpeg/bmp/png/tif/tiff.")

    p.add_argument(
        "--n_sq_x",
        type=int,
        default=6,
        help="Squares along x direction. Default: 6.",
    )

    p.add_argument(
        "--n_sq_y",
        type=int,
        default=4,
        help="Squares along y direction. Default: 4.",
    )

    p.add_argument(
        "--spacing_mm",
        type=float,
        default=50.0,
        help="Grid spacing in millimeters. Default: 50.",
    )

    p.add_argument(
        "--taylor_order",
        type=int,
        default=4,
        help="Polynomial degree. Default: 4.",
    )

    p.add_argument(
        "--output_dir",
        default=".",
        help="Folder for Omni_Calib_Results and calib_results.txt.",
    )

    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Run calibration/export without showing plots.",
    )

    p.add_argument(
        "--skip_find_center",
        action="store_true",
        help="Skip the Find center step for a quick test run.",
    )

    p.add_argument(
        "--slow_find_center",
        action="store_true",
        help="Use the original slower MATLAB-style Find center search.",
    )

    p.add_argument(
        "--refine_corners",
        action="store_true",
        help="Enable experimental corner recomputation. Default is off to avoid fake zero reprojection error for UV dot patterns.",
    )

    p.add_argument(
        "--target_images",
        type=int,
        default=25,
        help="Target accepted images for the coverage interface. Default: 25.",
    )

    p.add_argument(
        "--coverage_only",
        action="store_true",
        help="Only extract markers and print coverage guidance; do not calibrate.",
    )

    p.add_argument(
        "--require_coverage",
        action="store_true",
        help="Stop before calibration unless coverage guidance says the image set is complete.",
    )

    p.add_argument(
        "--show_coverage",
        action="store_true",
        help="Show a coverage scatter plot of board locations in the image.",
    )

    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch a ROS camera_calibration-style UV-DAR coverage GUI.",
    )

    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.gui:
        launch_coverage_gui(
            image_dir=args.image_dir,
            base_name=args.base_name,
            extension=args.extension,
            n_sq_x=args.n_sq_x,
            n_sq_y=args.n_sq_y,
            spacing_mm=args.spacing_mm,
            taylor_order=args.taylor_order,
            output_dir=args.output_dir,
            target_images=args.target_images,
            slow_find_center=args.slow_find_center,
        )
        return

    files = find_image_files(
        args.image_dir,
        args.base_name,
        args.extension,
    )

    if not files:
        print("No calibration images were found.")
        print("Example command:")
        print("  python ocam_calibration_step.py --image_dir photos --base_name i_ --extension bmp")
        print("Supported extensions: j, jpg, jpeg, bmp, png, tif, tiff, all")
        return

    print(f"Found {len(files)} image(s). Starting full calibration workflow...")

    run_full_calibration_workflow(
        image_dir=args.image_dir,
        base_name=args.base_name,
        extension=args.extension,
        n_sq_x=args.n_sq_x,
        n_sq_y=args.n_sq_y,
        spacing_mm=args.spacing_mm,
        taylor_order=args.taylor_order,
        output_dir=args.output_dir,
        do_plots=not args.no_plots,
        do_find_center=not args.skip_find_center,
        fast_find_center=not args.slow_find_center,
        do_corner_refinement=args.refine_corners,
        target_images=args.target_images,
        coverage_only=args.coverage_only,
        require_coverage=args.require_coverage,
        show_coverage=args.show_coverage,
    )


if __name__ == "__main__":
    main()