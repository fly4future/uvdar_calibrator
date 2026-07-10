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

## Coverage-Only Mode

To check image coverage without running full calibration:

```bash
python -m uvdar_calibrator --image_dir photos --coverage_only --show_coverage
```

This mode is useful after adding new images. It lets you check whether the current
photo set has enough variation before running the full calibration.
