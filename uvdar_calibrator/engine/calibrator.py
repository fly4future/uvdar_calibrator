"""
GUI-agnostic calibration engine.

Calibrator stores accepted calibration samples, checks calibration readiness,
and runs the OCamCalib solve over accepted samples only.

One image/frame is processed with handle_frame(). Detected views that are too
similar to already accepted samples are rejected on purpose so the final
calibration set is diverse.

Fast center search is always used.
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
    findcenter_fast,
    OCamModel,
    recomp_corner_calib,
    reprojectpoints,
    saving_calib,
)


@dataclass
class Sample:
    """One accepted calibration view."""

    params: List[float]          # [p_x, p_y, p_size, skew]
    image: np.ndarray            # grayscale image
    corners: np.ndarray          # detected points, shape (n_points, 2), [row, col]
    image_path: str


@dataclass
class FrameResult:
    """Outcome of feeding one image/frame to Calibrator.handle_frame()."""

    image_path: str
    detected: bool
    accepted: bool
    reason: str
    params: Optional[List[float]] = None
    similar_to: Optional[int] = None
    corners: Optional[np.ndarray] = None
    progress: List[Tuple[str, float, float, float]] = field(default_factory=list)
    goodenough: bool = False


class Calibrator:
    """Incremental sample collection plus OCamCalib solve over accepted samples."""

    def __init__(
        self,
        board: LedGridBoard,
        taylor_order: int = 4,
        preview_dir: Optional[str] = None,
        sample_threshold: float = coverage.DEFAULT_SAMPLE_THRESHOLD,
        param_ranges=coverage.DEFAULT_PARAM_RANGES,
        min_db_size: int = coverage.DEFAULT_MIN_DB_SIZE,
        save_previews_for_rejected: bool = True,
    ):
        self.board = board
        self.taylor_order = int(taylor_order)
        self.preview_dir = Path(preview_dir) if preview_dir else None
        self.save_previews_for_rejected = bool(save_previews_for_rejected)

        self.sample_threshold = float(sample_threshold)
        self.param_ranges = tuple(param_ranges)
        self.min_db_size = int(min_db_size)

        self.db: List[Sample] = []
        self.goodenough = False
        self.calibrated = False
        self.image_size: Optional[Tuple[int, int]] = None  # (width, height)

        self.last_ocam_model: Optional[OCamModel] = None
        self.RRfin: Optional[np.ndarray] = None
        self.reprojection_err: Optional[np.ndarray] = None
        self.reprojection_stderr: Optional[np.ndarray] = None
        self.reprojection_mse: Optional[float] = None

        # Board/world coordinates in millimeters.
        # Order is x-major/y-minor and must match detector point order.
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

    def handle_frame(self, image: np.ndarray, image_path: str = "") -> FrameResult:
        """
        Detect markers, decide accept/reject, and update the accepted sample db.
        """

        name = Path(image_path).name if image_path else "<array>"

        height, width = image.shape[:2]

        if self.image_size is None:
            self.image_size = (int(width), int(height))

        elif self.image_size != (int(width), int(height)):
            return self._result(
                image_path=image_path,
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
                image_path=image_path,
                detected=False,
                accepted=False,
                reason=f"{name}: no markers detected",
            )

        params = coverage.get_parameters(corners, self.board, self.image_size)

        if not coverage.is_good_sample(params, self.db_params(), self.sample_threshold):
            if self.preview_dir is not None and self.save_previews_for_rejected:
                self._save_preview(image, corners, image_path)

            distances = [coverage.param_distance(params, p) for p in self.db_params()]
            nearest = int(np.argmin(distances)) + 1

            return self._result(
                image_path=image_path,
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

        self.db.append(
            Sample(
                params=params,
                image=image,
                corners=corners,
                image_path=image_path,
            )
        )

        # The accepted sample database changed, so any previous calibration is stale.
        self.calibrated = False

        p_str = ", ".join(f"{v:.2f}" for v in params)

        return self._result(
            image_path=image_path,
            detected=True,
            accepted=True,
            params=params,
            corners=corners,
            reason=f"{name}: added as sample {len(self.db)}, p=[{p_str}]",
        )

    def _result(
        self,
        image_path,
        detected,
        accepted,
        reason,
        params=None,
        similar_to=None,
        corners=None,
    ) -> FrameResult:
        metrics = []
        for i, sample in enumerate(self.db, start=1):
            m = coverage.sample_metric(
                sample.corners,
                self.board,
                self.image_size,
                label=Path(sample.image_path).name or str(i),
            )
            if m is not None:
                metrics.append(m)

        self.goodenough, progress, _ = coverage.compute_goodenough_with_bins(
            self.db_params(),
            metrics,
            self.param_ranges,
            self.min_db_size,
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
    # Calibration over accepted samples
    # ------------------------------------------------------------------

    def _assemble(self):
        """Assemble OCamCalib solver arrays from accepted samples only."""

        n = len(self.db)
        n_points = self.board.n_points

        Xp_abs = np.full((n_points, 1, n), np.nan, dtype=float)
        Yp_abs = np.full((n_points, 1, n), np.nan, dtype=float)

        for k, sample in enumerate(self.db):
            Xp_abs[:, 0, k] = sample.corners[:, 0]
            Yp_abs[:, 0, k] = sample.corners[:, 1]

        # MATLAB-style 1-based sample numbers.
        ima_proc = list(range(1, n + 1))

        return Xp_abs, Yp_abs, ima_proc

    def cal_fromcorners(
        self,
        do_find_center: bool = True,
        fast_find_center: bool = True,
        refine_corners: bool = False,
    ) -> OCamModel:
        """
        Run OCamCalib over accepted samples.

        Fast center search is always used. The fast_find_center argument is
        kept only for backward compatibility with older GUI/CLI calls.
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
            self.Xt,
            self.Yt,
            Xp_abs,
            Yp_abs,
            model.xc,
            model.yc,
            self.taylor_order,
            ima_proc,
        )

        if np.isscalar(RRfin) or np.size(RRfin) == 1:
            raise RuntimeError("Calibration failed while computing extrinsics.")

        model.ss = np.asarray(ss, dtype=float)

        print("Initial calibration error:")
        reprojectpoints(
            model,
            RRfin,
            ima_proc,
            self.Xt,
            self.Yt,
            Xp_abs,
            Yp_abs,
        )

        if do_find_center:
            print("\nStep 4: Find center using fast search")

            new_RRfin = findcenter_fast(
                model,
                self.Xt,
                self.Yt,
                Xp_abs,
                Yp_abs,
                self.taylor_order,
                ima_proc,
            )

            if new_RRfin is not None:
                RRfin = new_RRfin

        else:
            print("\nStep 4: Skipping Find center")

        if refine_corners:
            print("\nStep 5: Calibration refinement / recompute corners")

            recomp_corner_calib(
                model,
                [s.image for s in self.db],
                Xp_abs,
                Yp_abs,
                ima_proc,
            )

        else:
            print("\nStep 5: Skipping corner recomputation for UV dot pattern")

        print("\nStep 5b: Final calibration after center finding")

        RRfin, ss = calibrate(
            self.Xt,
            self.Yt,
            Xp_abs,
            Yp_abs,
            model.xc,
            model.yc,
            self.taylor_order,
            ima_proc,
        )

        if np.isscalar(RRfin) or np.size(RRfin) == 1:
            raise RuntimeError("Final calibration failed while computing extrinsics.")

        model.ss = np.asarray(ss, dtype=float)

        err, stderr, mse = reprojectpoints(
            model,
            RRfin,
            ima_proc,
            self.Xt,
            self.Yt,
            Xp_abs,
            Yp_abs,
        )

        self.last_ocam_model = model
        self.RRfin = RRfin
        self.reprojection_err = err
        self.reprojection_stderr = stderr
        self.reprojection_mse = float(mse)
        self.calibrated = True

        print("\nFinal calibration parameters:")
        print("ss =")
        print(model.ss)
        print(f"xc = {model.xc}")
        print(f"yc = {model.yc}")

        return model

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------

    def report(self) -> str:
        """
        Render readiness progress, supplementary coverage hints, and calibration
        summary once calibrated.
        """

        goodenough, progress = coverage.compute_goodenough(
            self.db_params(),
            self.param_ranges,
            self.min_db_size,
        )

        lines = [
            coverage.format_progress(
                progress,
                goodenough,
                len(self.db),
            )
        ]

        metrics = []
        for i, sample in enumerate(self.db, start=1):
            m = coverage.sample_metric(
                sample.corners,
                self.board,
                self.image_size,
                label=Path(sample.image_path).name or str(i),
            )

            if m is not None:
                metrics.append(m)

        if metrics:
            lines.append("")
            lines.append(
                coverage.format_bin_coverage(
                    coverage.compute_bin_coverage(metrics)
                )
            )

        if self.calibrated and self.last_ocam_model is not None:
            avg = float(np.nanmean(self.reprojection_err))

            lines.append("")
            lines.append(
                f"Calibrated: avg reprojection error {avg:.3f} px, "
                f"center=({self.last_ocam_model.xc:.2f}, {self.last_ocam_model.yc:.2f})"
            )

        return "\n".join(lines)

    def save(self, output_dir: str = ".") -> None:
        """Save Omni_Calib_Results.npz and optional .mat to output_dir."""

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
        """Export calib_results.txt in OCamCalib text format."""

        if not self.calibrated:
            print("\nNo calibration data available. You must first calibrate your camera.\n")
            return

        print('Exporting ocam_model to "calib_results.txt"')
        export_data(self.last_ocam_model, output_dir=output_dir)
        print("done")