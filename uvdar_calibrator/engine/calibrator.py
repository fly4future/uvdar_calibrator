"""
GUI-agnostic calibration engine.

``Calibrator`` mirrors the shape of ROS image_pipeline's
``Calibrator``/``MonoCalibrator``: a small engine class holding the
accepted-sample database (``db``), readiness (``goodenough``) and
calibration state, with ``handle_frame`` as the direct analogue of ROS's
``handle_msg`` -- one photo == one "frame". The OCamCalib solver math lives
in :mod:`uvdar_calibrator.engine.ocam_model`; sample selection lives in
:mod:`uvdar_calibrator.engine.coverage`.

Behavioral note (intentional change from the old single-file tool): images
that are too similar to an already-accepted sample are *rejected* rather
than used unconditionally, so the accepted sample count for a folder is
typically lower than the number of successfully-detected images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import coverage
from .board import LedGridBoard
from .detection import get_corners, save_detected_marker_preview
from .ocam_model import (
    calibrate,
    export_data,
    findcenter,
    findcenter_fast,
    OCamModel,
    recomp_corner_calib,
    reprojectpoints,
    saving_calib,
)


@dataclass
class Sample:
    """One accepted calibration view."""

    params: List[float]            # [p_x, p_y, p_size, skew]
    image: np.ndarray              # grayscale image
    corners: np.ndarray            # (n_points, 2) [row, col], flat x-major/y-minor
    image_path: str


@dataclass
class FrameResult:
    """Outcome of feeding one photo to Calibrator.handle_frame."""

    image_path: str
    detected: bool                             # markers found at all
    accepted: bool                             # added to the sample db
    reason: str                                # human-readable feedback line
    params: Optional[List[float]] = None       # [p_x, p_y, p_size, skew] if detected
    similar_to: Optional[int] = None           # 1-based index of nearest db sample if rejected
    corners: Optional[np.ndarray] = None       # detected points [row, col] if detected
    progress: List[Tuple[str, float, float, float]] = field(default_factory=list)
    goodenough: bool = False


class Calibrator:
    """Incremental sample collection + OCamCalib solve over accepted samples."""

    def __init__(
        self,
        board: LedGridBoard,
        taylor_order: int = 4,
        preview_dir: Optional[str] = None,
        sample_threshold: float = coverage.DEFAULT_SAMPLE_THRESHOLD,
        param_ranges=coverage.DEFAULT_PARAM_RANGES,
        min_db_size: int = coverage.DEFAULT_MIN_DB_SIZE,
        save_previews_for_rejected: bool = True,
        fov_radius_frac: Optional[float] = None,
    ):
        self.board = board
        self.taylor_order = int(taylor_order)
        self.preview_dir = Path(preview_dir) if preview_dir else None
        # Batch mode keeps the historical behavior (a preview file for every
        # *detected* image, accepted or not). The live node sets this False so
        # a camera stream full of rejected near-duplicates doesn't flood the
        # preview directory with files.
        self.save_previews_for_rejected = bool(save_previews_for_rejected)
        self.sample_threshold = float(sample_threshold)
        self.param_ranges = tuple(param_ranges)
        self.min_db_size = int(min_db_size)
        # Fraction of min(width, height)/2 covered by the camera's usable
        # (e.g. fisheye) image circle, assumed centered on the frame. None
        # (default) means "use the full rectangular frame" -- see
        # coverage.get_parameters' valid_region for why this matters.
        self.fov_radius_frac = float(fov_radius_frac) if fov_radius_frac is not None else None

        self.db: List[Sample] = []
        self.goodenough = False
        self.calibrated = False
        self.image_size: Optional[Tuple[int, int]] = None  # (width, height)

        self.last_ocam_model: Optional[OCamModel] = None
        self.RRfin: Optional[np.ndarray] = None
        self.reprojection_err: Optional[np.ndarray] = None
        self.reprojection_stderr: Optional[np.ndarray] = None
        self.reprojection_mse: Optional[float] = None

        # Board/world coordinates in millimeters, x-major/y-minor
        # (MATLAB/OCamCalib indexing; must match detector point order).
        Xt = []
        Yt = []
        for x in range(board.n_sq_x + 1):
            for y in range(board.n_sq_y + 1):
                Xt.append(x * board.spacing_mm)
                Yt.append(y * board.spacing_mm)
        self.Xt = np.asarray(Xt, dtype=float).reshape(-1, 1)
        self.Yt = np.asarray(Yt, dtype=float).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def db_params(self) -> List[List[float]]:
        return [s.params for s in self.db]

    def valid_region_px(self) -> Optional[Tuple[float, float, float]]:
        """(cx, cy, radius) in pixels, or None for the full rectangular frame."""
        if self.fov_radius_frac is None or self.image_size is None:
            return None
        width, height = self.image_size
        cx, cy = width / 2.0, height / 2.0
        radius = self.fov_radius_frac * min(width, height) / 2.0
        return (cx, cy, radius)

    def handle_frame(self, image: np.ndarray, image_path: str = "") -> FrameResult:
        """
        Detect markers, decide accept/reject, update self.db.

        One image == one "frame" (analogue of ROS handle_msg).
        """
        name = Path(image_path).name if image_path else "<array>"

        height, width = image.shape[:2]
        if self.image_size is None:
            self.image_size = (int(width), int(height))
        elif self.image_size != (int(width), int(height)):
            return self._result(
                image_path,
                detected=False,
                accepted=False,
                reason=(
                    f"{name}: skipped -- image size {width}x{height} does not match "
                    f"first image {self.image_size[0]}x{self.image_size[1]}"
                ),
            )

        ok, corners, _ = get_corners(image, self.board)

        if not ok or corners is None:
            return self._result(
                image_path,
                detected=False,
                accepted=False,
                reason=f"{name}: no markers detected",
            )

        params = coverage.get_parameters(
            corners, self.board, self.image_size, valid_region=self.valid_region_px()
        )

        if not coverage.is_good_sample(params, self.db_params(), self.sample_threshold):
            if self.preview_dir is not None and self.save_previews_for_rejected:
                self._save_preview(image, corners, image_path)
            distances = [coverage.param_distance(params, p) for p in self.db_params()]
            nearest = int(np.argmin(distances)) + 1
            return self._result(
                image_path,
                detected=True,
                accepted=False,
                params=params,
                similar_to=nearest,
                corners=corners,
                reason=(
                    f"{name}: rejected -- too similar to sample {nearest} "
                    f"({Path(self.db[nearest - 1].image_path).name}, "
                    f"distance {min(distances):.3f} <= {self.sample_threshold})"
                ),
            )

        if self.preview_dir is not None:
            self._save_preview(image, corners, image_path)

        self.db.append(Sample(params=params, image=image, corners=corners, image_path=image_path))
        self.calibrated = False  # db changed; any previous solve is stale

        p_str = ", ".join(f"{v:.2f}" for v in params)
        return self._result(
            image_path,
            detected=True,
            accepted=True,
            params=params,
            corners=corners,
            reason=f"{name}: added as sample {len(self.db)}, p=[{p_str}]",
        )

    def _result(
        self, image_path, detected, accepted, reason,
        params=None, similar_to=None, corners=None,
    ) -> FrameResult:
        self.goodenough, progress = coverage.compute_goodenough(
            self.db_params(), self.param_ranges, self.min_db_size
        )
        return FrameResult(
            image_path=image_path,
            detected=detected,
            accepted=accepted,
            reason=reason,
            params=params,
            similar_to=similar_to,
            corners=corners,
            progress=progress,
            goodenough=self.goodenough,
        )

    def _save_preview(self, image, corners, image_path) -> None:
        try:
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem if image_path else f"frame_{len(self.db) + 1}"

            preview_path = self.preview_dir / f"{stem}_detected.bmp"
            save_detected_marker_preview(image, corners, preview_path)

            debug_path = self.preview_dir / f"{stem}_points.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("index,row_px,col_px,Xt_mm,Yt_mm\n")
                for idx, (row, col) in enumerate(corners, start=1):
                    f.write(
                        f"{idx},{row:.6f},{col:.6f},"
                        f"{self.Xt[idx - 1, 0]:.6f},{self.Yt[idx - 1, 0]:.6f}\n"
                    )
        except Exception as exc:
            print(f"  preview save failed for {image_path}: {exc}")

    # ------------------------------------------------------------------
    # Calibration over the accepted db
    # ------------------------------------------------------------------

    def _assemble(self):
        """Assemble solver arrays from the accepted samples only."""
        n = len(self.db)
        n_points = self.board.n_points

        Xp_abs = np.full((n_points, 1, n), np.nan, dtype=float)
        Yp_abs = np.full((n_points, 1, n), np.nan, dtype=float)

        for k, sample in enumerate(self.db):
            Xp_abs[:, 0, k] = sample.corners[:, 0]
            Yp_abs[:, 0, k] = sample.corners[:, 1]

        ima_proc = list(range(1, n + 1))  # MATLAB-style 1-based sample numbers

        return Xp_abs, Yp_abs, ima_proc

    def cal_fromcorners(
        self,
        do_find_center: bool = True,
        fast_find_center: bool = True,
        refine_corners: bool = False,
    ) -> OCamModel:
        """
        Run the OCamCalib solve over self.db and set self.calibrated.

        Steps mirror the original workflow: initial calibration -> find
        center -> optional corner refinement -> final calibration.
        """
        if not self.db:
            raise RuntimeError("No accepted samples. Feed images through handle_frame first.")

        if self.image_size is None:
            raise RuntimeError("Image size unknown; no frames were processed.")

        width, height = self.image_size
        Xp_abs, Yp_abs, ima_proc = self._assemble()

        model = OCamModel(
            xc=height / 2.0,
            yc=width / 2.0,
            width=int(width),
            height=int(height),
            c=1.0,
            d=0.0,
            e=0.0,
        )

        print("\nStep 3: Initial calibration")
        RRfin, ss = calibrate(
            self.Xt, self.Yt, Xp_abs, Yp_abs,
            model.xc, model.yc, self.taylor_order, ima_proc,
        )
        if np.isscalar(RRfin) or np.size(RRfin) == 1:
            raise RuntimeError("Calibration failed while computing extrinsics.")

        model.ss = np.asarray(ss, dtype=float)
        reprojectpoints(model, RRfin, ima_proc, self.Xt, self.Yt, Xp_abs, Yp_abs)
        print("ss =")
        print(model.ss)

        if do_find_center:
            print("\nStep 4: Find center")
            if fast_find_center:
                new_RRfin = findcenter_fast(
                    model, self.Xt, self.Yt, Xp_abs, Yp_abs, self.taylor_order, ima_proc
                )
                if new_RRfin is not None:
                    RRfin = new_RRfin
            else:
                RRfin = findcenter(
                    model, self.Xt, self.Yt, Xp_abs, Yp_abs, self.taylor_order, ima_proc
                )
        else:
            print("\nStep 4: Skipping Find center")

        if refine_corners:
            print("\nStep 5: Calibration refinement / recompute corners")
            recomp_corner_calib(
                model, [s.image for s in self.db], Xp_abs, Yp_abs, ima_proc
            )
        else:
            print("\nStep 5: Skipping corner recomputation for UV dot pattern")
            print(
                "Reason: UV-dot extraction already gives measured marker centers; "
                "recomputing corners is optional."
            )

        print("\nStep 5b: Final calibration after center finding")
        RRfin, ss = calibrate(
            self.Xt, self.Yt, Xp_abs, Yp_abs,
            model.xc, model.yc, self.taylor_order, ima_proc,
        )
        if np.isscalar(RRfin) or np.size(RRfin) == 1:
            raise RuntimeError("Final calibration failed while computing extrinsics.")

        model.ss = np.asarray(ss, dtype=float)
        err, stderr, mse = reprojectpoints(
            model, RRfin, ima_proc, self.Xt, self.Yt, Xp_abs, Yp_abs
        )
        print("ss =")
        print(model.ss)

        self.last_ocam_model = model
        self.RRfin = RRfin
        self.reprojection_err = err
        self.reprojection_stderr = stderr
        self.reprojection_mse = float(mse)
        self.calibrated = True

        return model

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------

    def report(self) -> str:
        """
        Render the text report.

        Readiness progress + supplementary bin hints (+ reprojection error
        once calibrated).
        """
        goodenough, progress = coverage.compute_goodenough(
            self.db_params(), self.param_ranges, self.min_db_size
        )
        lines = [coverage.format_progress(progress, goodenough, len(self.db))]

        metrics = []
        for i, sample in enumerate(self.db, start=1):
            m = coverage.sample_metric(
                sample.corners,
                self.board,
                self.image_size,
                label=Path(sample.image_path).name or str(i),
                valid_region=self.valid_region_px(),
            )
            if m is not None:
                metrics.append(m)
        if metrics:
            lines.append("")
            lines.append(coverage.format_bin_coverage(coverage.compute_bin_coverage(metrics)))

        if self.calibrated and self.last_ocam_model is not None:
            avg = float(np.nanmean(self.reprojection_err))
            lines.append("")
            lines.append(
                f"Calibrated: avg reprojection error {avg:.3f} px, "
                f"center=({self.last_ocam_model.xc:.2f}, {self.last_ocam_model.yc:.2f})"
            )

        return "\n".join(lines)

    def save(self, output_dir: str = ".") -> None:
        """Save Omni_Calib_Results.npz (+ optional .mat) to output_dir."""
        if not self.calibrated:
            print("\nNo calibration data available. You must first calibrate your camera.\n")
            return

        Xp_abs, Yp_abs, ima_proc = self._assemble()
        saving_calib(
            self.last_ocam_model,
            self.RRfin,
            ima_proc,
            self.Xt,
            self.Yt,
            Xp_abs,
            Yp_abs,
            self.taylor_order,
            output_dir=output_dir,
        )

    def export_txt(self, output_dir: str = ".") -> None:
        """Export calib_results.txt (OCamCalib text format) to output_dir."""
        if not self.calibrated:
            print("\nNo calibration data available. You must first calibrate your camera.\n")
            return

        print('Exporting ocam_model to "calib_results.txt"')
        export_data(self.last_ocam_model, output_dir=output_dir)
        print("done")
