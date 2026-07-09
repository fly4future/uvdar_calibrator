"""Command-line entry point.

Mirrors the option-parsing shape of ROS image_pipeline's
``nodes/cameracalibrator.py`` (board geometry flags, output dir, no-plots,
...), minus everything ROS-specific. Photos are fed one at a time into
:class:`~uvdar_calibrator.calibrator.Calibrator`, which may *reject* images
that are too similar to an already-accepted sample -- this is expected and
is what produces a diverse calibration set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .board import LedGridBoard
from .calibrator import Calibrator
from .detection import find_image_files, read_image_gray


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m uvdar_calibrator",
        description="OCamCalib-style omnidirectional camera calibration from UV LED grid photos.",
    )

    p.add_argument(
        "--image_dir",
        default=".",
        help="Folder containing calibration images.",
    )

    p.add_argument(
        "--base_name",
        default="",
        help="Image filename prefix. Empty means use all names.",
    )

    p.add_argument(
        "--extension",
        default="all",
        help="Image extension. Use 'all' to load jpg/jpeg/bmp/png/tif/tiff.",
    )

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
        help=(
            "Enable experimental corner recomputation. Default is off to avoid "
            "fake zero reprojection error for UV dot patterns."
        ),
    )

    p.add_argument(
        "--coverage_only",
        action="store_true",
        help="Only feed images through sample selection and print the readiness report; do not calibrate.",
    )

    p.add_argument(
        "--show_coverage",
        action="store_true",
        help="Show a coverage scatter plot of accepted board locations in the image.",
    )

    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the interactive Tkinter calibration GUI.",
    )

    return p


def run(
    image_dir: str,
    base_name: str = "",
    extension: str = "all",
    n_sq_x: int = 6,
    n_sq_y: int = 4,
    spacing_mm: float = 50.0,
    taylor_order: int = 4,
    output_dir: str = ".",
    do_plots: bool = True,
    do_find_center: bool = True,
    fast_find_center: bool = True,
    refine_corners: bool = False,
    coverage_only: bool = False,
    show_coverage: bool = False,
) -> Optional[Calibrator]:
    """Non-GUI workflow: feed photos one-by-one into a Calibrator, then solve."""
    files = find_image_files(image_dir, base_name, extension)

    if not files:
        print("No calibration images were found.")
        print("Example command:")
        print("  python -m uvdar_calibrator --image_dir photos --base_name i_ --extension bmp")
        print("Supported extensions: j, jpg, jpeg, bmp, png, tif, tiff, all")
        return None

    print(f"Found {len(files)} image(s).")

    board = LedGridBoard(n_sq_x=n_sq_x, n_sq_y=n_sq_y, spacing_mm=spacing_mm)
    cal = Calibrator(
        board,
        taylor_order=taylor_order,
        preview_dir=str(Path(image_dir) / "detected_marker_previews"),
    )

    print("\nStep 1: Feeding images through sample selection")
    print(f"pattern size {board.n_cols}x{board.n_rows}")
    print(
        "Note: images too similar to an already-accepted sample are rejected;\n"
        "this is expected and produces a diverse calibration set."
    )

    n_rejected = 0
    n_failed = 0

    for k, path in enumerate(files, start=1):
        print(f"Processing image {k}: {Path(path).name}")
        result = cal.handle_frame(read_image_gray(path), path)
        print(f"  {result.reason}")
        if result.detected and not result.accepted:
            n_rejected += 1
        elif not result.detected:
            n_failed += 1

    print(
        f"\nSample selection done: {len(cal.db)} accepted, "
        f"{n_rejected} rejected as near-duplicates, {n_failed} failed detection."
    )

    print("\nStep 2: Readiness report")
    report_text = cal.report()
    print(report_text)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "calibration_coverage.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"Saved readiness report to {report_path}")

    if show_coverage:
        from . import coverage as _coverage

        metrics = []
        for sample in cal.db:
            m = _coverage.sample_metric(
                sample.corners, board, cal.image_size, label=Path(sample.image_path).name
            )
            if m is not None:
                metrics.append(m)
        _coverage.plot_bin_coverage(metrics)

    if coverage_only:
        print("\nCoverage-only mode complete. No calibration was run.")
        return cal

    if not cal.db:
        print("\nNo accepted samples; cannot calibrate.")
        return cal

    if not cal.goodenough:
        print(
            "\nWARNING: sample variety is not yet 'good enough' by ROS-style range "
            "progress. Calibrating anyway -- treat the result as preliminary and "
            "capture the suggested missing views."
        )

    cal.cal_fromcorners(
        do_find_center=do_find_center,
        fast_find_center=fast_find_center,
        refine_corners=refine_corners,
    )

    if do_plots:
        from . import plots

        Xp_abs, Yp_abs, ima_proc = cal._assemble()

        print("\nStep 6: Reproject on images")
        plots.reproject_calib(
            cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt,
            Xp_abs, Yp_abs,
            images=[s.image for s in cal.db],
            n_sq_y=board.n_sq_y,
        )

        print("\nStep 7: Analyze error")
        plots.analyse_error(
            cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt, Xp_abs, Yp_abs
        )

        print("\nStep 8: Show calibration results")
        plots.show_calib_results(
            cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt, Xp_abs, Yp_abs
        )

        print("\nStep 8b: Show extrinsic")
        plots.show_extrinsic(cal.RRfin, ima_proc, cal.Xt, cal.Yt)

    print("\nStep 9: Save calibration")
    cal.save(output_dir=output_dir)

    print("\nStep 9b: Export calib_results.txt")
    cal.export_txt(output_dir=output_dir)

    print("\nCalibration workflow complete.")
    return cal


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.gui:
        from .gui import launch_gui

        launch_gui(
            image_dir=args.image_dir,
            base_name=args.base_name,
            extension=args.extension,
            n_sq_x=args.n_sq_x,
            n_sq_y=args.n_sq_y,
            spacing_mm=args.spacing_mm,
            taylor_order=args.taylor_order,
            output_dir=args.output_dir,
            slow_find_center=args.slow_find_center,
        )
        return

    run(
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
        refine_corners=args.refine_corners,
        coverage_only=args.coverage_only,
        show_coverage=args.show_coverage,
    )


if __name__ == "__main__":
    main()
