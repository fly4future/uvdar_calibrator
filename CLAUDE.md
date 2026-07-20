# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python implementation of the OCamCalib omnidirectional camera calibration workflow
(Scaramuzza's MATLAB toolbox, via the ctu-mrs/OCamCalib_UVDAR fork), adapted to calibrate
UV-sensitive cameras against a non-square UV LED grid pattern instead of a printed
checkerboard. The surrounding workflow — how photos become accepted calibration samples and
how readiness is judged — mirrors ROS image_pipeline's `camera_calibration` package
(`Calibrator`/`MonoCalibrator` + `is_good_sample`/`compute_goodenough`).

The code lives in the `uvdar_calibrator/` package. There is no test suite, linter config, or
CI — verify changes by running the CLI/GUI below against `example_images/` and checking that
sample selection, calibration, and export still succeed.

## Running it

Install deps: `pip install -r requirements.txt` (numpy, matplotlib, opencv-python; scipy is
optional and only used for `.mat` export).

```bash
# GUI (recommended entry point)
python -m uvdar_calibrator --image_dir photos --gui

# Full CLI calibration (no GUI)
python -m uvdar_calibrator --image_dir photos

# Sample-selection / readiness check only, no calibration
python -m uvdar_calibrator --image_dir photos --coverage_only --show_coverage

# Filter by filename prefix / extension
python -m uvdar_calibrator --image_dir photos --base_name example_ --extension bmp --gui
```

## Architecture

```
uvdar_calibrator/
├── __init__.py             # public API: Calibrator, LedGridBoard, OCamModel, FrameResult, Sample
├── __main__.py             # python -m uvdar_calibrator -> apps.cli.main()
├── engine/                 # GUI/ROS-agnostic core -- no tkinter, no rclpy, no matplotlib
│   ├── board.py              # LedGridBoard target geometry (analogous to ROS ChessboardInfo)
│   ├── ocam_model.py         # OCamModel dataclass + pure Scaramuzza solver math (faithful MATLAB port)
│   ├── detection.py          # pattern detection: chessboard -> circle grid -> UV bright-dot fallback
│   ├── coverage.py           # get_parameters/is_good_sample/compute_goodenough (+ bin-based hint report)
│   └── calibrator.py         # Calibrator engine: db/goodenough/handle_frame/cal_fromcorners/export
├── diagnostics/
│   └── plots.py             # matplotlib diagnostics (block on window close)
└── apps/                   # the three ways to drive the engine
    ├── gui.py                 # Tkinter app: feeds photos (or live frames) into Calibrator, live range bars
    ├── cli.py                  # argparse entry point (offline/batch, calibrate_offline console_script)
    └── live_node.py            # ROS 2 node: live topic capture (cameracalibrator console_script)
```

### The engine (`engine/calibrator.Calibrator`)

The direct analogue of ROS's `MonoCalibrator`. One photo == one "frame":

- `handle_frame(image, path)` detects markers (`detection.get_corners`), reduces the view to
  four normalized numbers `[p_x, p_y, p_size, skew]` (`coverage.get_parameters`), and only
  appends it to the sample database `db` if its L1 distance from *every* already-accepted
  sample exceeds a threshold (`coverage.is_good_sample`). **Rejecting near-duplicate photos is
  intentional behavior**, not a bug — it is what produces a diverse calibration set, and the
  accepted count for a folder is typically lower than the number of detectable images.
- Readiness (`goodenough`) comes from `coverage.compute_goodenough`: per-axis progress is the
  min/max range of accepted samples' params against `param_ranges`, complete when every axis
  reaches 1.0 or the db reaches `min_db_size`. This — not the bin report — gates readiness.
- `cal_fromcorners()` assembles `Xt/Yt/Xp_abs/Yp_abs` from the accepted `db` only and runs the
  OCamCalib solve: initial `calibrate()` → `findcenter_fast()` (or slow `findcenter`) →
  optional `recomp_corner_calib` → final `calibrate()` + `reprojectpoints()`.
- `save()`/`export_txt()` write `Omni_Calib_Results.npz` (+ optional `.mat`) and
  `calib_results.txt` (OCamCalib text format, computing `invpol` via `findinvpoly`).

The tuning constants in `engine/coverage.py` (`DEFAULT_SAMPLE_THRESHOLD = 0.2`,
`DEFAULT_PARAM_RANGES = (0.7, 0.7, 0.4, 0.5)`, `DEFAULT_MIN_DB_SIZE = 40`) are ROS
camera_calibration defaults carried over unchanged; they likely need retuning for this
smaller/farther-captured UV grid — see the note at their definition.

### The solver (`engine/ocam_model.py`)

Pure math over `(Xt, Yt, Xp_abs, Yp_abs, xc, yc, ...)` arrays — no knowledge of samples,
goodness, or GUI state. `ima_proc` holds **1-based** (MATLAB-style) image numbers and `idx()`
converts to 0-based numpy indices wherever an array is touched; it's public (not `_idx`)
specifically because `diagnostics/plots.py` needs the same conversion. Keep this convention when
adding code that walks `ima_proc` — mixing 0- and 1-based indices here is the easiest way to
silently corrupt results. `taylor_order` default is 4; `min_order` is hardcoded to 4 in
`omni_find_parameters_fun`.

`recomp_corner_calib` (off by default via `--refine_corners`) deliberately refines from the
*measured* UV points, not from model-reprojected points — an earlier version used model
projections as the refinement seed, which could drive reprojection error to exactly 0 because
it compared the model against its own output. Preserve this "never seed refinement from the
model" invariant if touching this function.

### Detection (`engine/detection.py`)

`get_corners(img, board)` returns `(ok, corners, board)` like ROS. Detection is tried in
order and falls back progressively:

1. OpenCV `findChessboardCorners`/`findChessboardCornersSB` (in case a real checkerboard is
   used instead of the LED grid).
2. OpenCV `findCirclesGrid` symmetric grid detection with a blob detector tuned for small
   bright dots.
3. A custom UV bright-dot fallback: Otsu threshold → connected components → group into
   rows/columns by spacing → `_order_uv_points_projective` as a last resort for rotated/skewed
   grids.

The `Calibrator` writes per-image previews/point dumps to
`<image_dir>/detected_marker_previews/` for every detected image (accepted or not).

### Coverage (`engine/coverage.py`)

Two layers, only the first of which gates anything:

- **ROS-ported sample selection**: `get_parameters`, `is_good_sample`, `compute_goodenough` —
  the single source of truth for accept/reject and calibrate-readiness.
- **Bin-based hint report** (`COVERAGE_*_BINS`, `sample_metric`, `compute_bin_coverage`,
  `coverage_suggestions`, `format_bin_coverage`, `plot_bin_coverage`): a supplementary
  "next images to capture" report driven off accepted samples. It must never gate the
  CALIBRATE button or the CLI.

### GUI (`apps/gui.py`)

Tkinter app that reuses the engine directly — when changing pipeline behavior, change
`Calibrator`/`coverage` once and both CLI and GUI pick it up. Load/Analyze runs
`handle_frame` per photo with a live accept/reject log; the four X/Y/Size/Skew range bars
mirror ROS's `redraw_monocular` bars (colored segment from min to max accepted value, green
at full progress). CALIBRATE is driven by `goodenough` (with an explicit confirm-to-override
when not ready); SAVE/EXPORT by `calibrated`.

### Coordinate/order conventions to keep straight

- Detected marker points are stored `[row, col]` (`Xp_abs`=row, `Yp_abs`=col) — the reverse of
  OpenCV's usual `[x, y]`/`[col, row]`. `coverage.get_parameters` flips to `(x, y)` internally
  for the outside-corner area/skew math.
- Board points `Xt`/`Yt` are generated x-major, y-minor (outer loop over x, inner loop over y);
  all detector paths must return points in that same order for `Xp_abs`/`Yp_abs` to line up
  with `Xt`/`Yt`. The outside-corner indices in `coverage.outside_corner_indices` are derived
  for this x-major flat layout (`idx(x, y) = x * n_rows + y`).
- `n_sq_x`/`n_sq_y` are square counts; the actual point grid is `(n_sq_x+1) x (n_sq_y+1)`
  (`LedGridBoard.n_cols`/`n_rows`).
