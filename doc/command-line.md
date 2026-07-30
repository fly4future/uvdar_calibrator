# Command-Line Usage

## Calibration Without the GUI

You can run calibration directly from the terminal:

```bash
python -m uvdar_calibrator --image_dir photos
```

The CLI feeds photos through the same accept/reject sample selection as the GUI,
prints a readiness report, and then calibrates. If the accepted samples do not yet
cover enough variation, it prints a warning and calibrates anyway — treat that
result as preliminary.

It writes `calib_results.txt` and `calibration_coverage.txt` to `--output_dir`
(default: the current directory).

## Diagnostic Plots

Plots are opt-in, as in the GUI:

```bash
python -m uvdar_calibrator --image_dir photos --plots
```

This opens one window with a tab per diagnostic — reprojection, error analysis,
projection function, and extrinsics — and blocks until you close it, before the
results are exported. See [The Diagnostics
Window](using-the-gui.md#the-diagnostics-window) for what each tab tells you.
Without `--plots`, the same per-sample and average errors are still printed to the
console.

## Coverage-Only Mode

To check image coverage without running full calibration:

```bash
python -m uvdar_calibrator --image_dir photos --coverage_only --show_coverage
```

This mode is useful after adding new images. It lets you check whether the current
photo set has enough variation before running the full calibration.
