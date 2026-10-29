# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python implementation of the OCamCalib omnidirectional camera calibration workflow
(Scaramuzza's MATLAB toolbox, via the ctu-mrs/OCamCalib_UVDAR fork), adapted to calibrate
UV-sensitive cameras against a non-square UV LED grid pattern instead of a printed
checkerboard. The surrounding workflow — how photos become accepted calibration samples and
how readiness is judged — mirrors ROS image_pipeline's `camera_calibration` package
(`Calibrator`/`MonoCalibrator` + `is_good_sample`/`compute_goodenough`).

The code lives in the `uvdar_calibrator/` package. It runs **only in Docker**: the camera
wizard detects the camera, starts its driver and the live calibration window. There is no
offline/photo-folder mode and no non-Docker launch path; keep it that way.

## Running it

```bash
./docker-dev.sh          # dev image (Dockerfile target `dev`), checkout mounted, builds + starts the wizard
./docker-dev.sh bash     # shell in the same container
```

Production is the default `Dockerfile` stage, published to
`ghcr.io/fly4future/uvdar_calibrator` (see README). Both stages share every layer up to
the calibrator source, so keep slow, rarely changing steps above the `base` stage's end.

Tests are plain scripts (each has a `__main__`), run inside the dev container — the live
ones need `rclpy`/`cv_bridge`:

```bash
cd /ros2_ws/src/uvdar_calibrator
for t in test/test_*.py; do python3 "$t" || break; done
```

`test/test_live_endtoend.py` replays `example_images/` through `test/replay_publisher.py`
into the real subscriber and consumer thread (no GUI). The GUI tests need a display and
skip without one.

## Architecture

```
uvdar_calibrator/
├── __init__.py             # public API: Calibrator, LedGridBoard, OCamModel, FrameResult, Sample
├── engine/                 # GUI/ROS-agnostic core -- no tkinter, no rclpy, no matplotlib
│   ├── board.py              # LedGridBoard target geometry (analogous to ROS ChessboardInfo)
│   ├── ocam_model.py         # OCamModel dataclass + pure Scaramuzza solver math (faithful MATLAB port)
│   ├── detection.py          # pattern detection: UV LED grid (circle grid) or checkerboard
│   ├── coverage.py           # get_parameters/is_good_sample/compute_goodenough (+ bin-based hint report)
│   └── calibrator.py         # Calibrator engine: db/goodenough/handle_frame/cal_fromcorners/export
├── diagnostics/
│   ├── plots.py              # matplotlib diagnostics: build figures, never show them
│   └── plot_window.py        # one Tk window, one tab per figure (the only Tk outside apps/)
└── apps/
    ├── gui.py                 # Tkinter LiveCalibrationApp: live preview, range bars, CALIBRATE/SAVE
    ├── live_node.py            # ROS 2 node + argparse (cameracalibrator console_script)
    └── camera_wizard.py        # camera detection + config/tmux setup (docker/calibrate.sh)
```

### The engine (`engine/calibrator.Calibrator`)

The direct analogue of ROS's `MonoCalibrator`. One photo == one "frame":

- `handle_frame(image, path)` detects markers (`detection.get_corners`), reduces the view to
  four normalized numbers `[p_x, p_y, p_size, skew]` (`coverage.get_parameters`), and only
  appends it to the sample database `db` if its distance from *every* already-accepted
  sample exceeds a threshold (`coverage.is_good_sample`). That distance
  (`coverage.param_distance`) is an L1 sum in which each axis is first divided by its
  `param_ranges` span, so the threshold is dimensionless ("how many target-spans apart")
  and means the same thing on every axis and every rig. **Rejecting near-duplicate photos is
  intentional behavior**, not a bug — it is what produces a diverse calibration set, and the
  accepted count for a folder is typically lower than the number of detectable images.
- Readiness (`goodenough`) comes from `coverage.compute_goodenough`: per-axis progress is the
  min/max range of accepted samples' params against `param_ranges`. Ready requires **both**
  every axis at 1.0 **and** at least `min_db_size` accepted samples — an AND, unlike ROS's
  OR, because a live stream trivially supplies many near-identical frames and sample count
  alone is not evidence of variety. This — not the bin report — gates readiness.
- `cal_fromcorners()` assembles `Xt/Yt/Xp_abs/Yp_abs` from the accepted `db` only and runs the
  OCamCalib solve: initial `calibrate()` → `findcenter()` (OCamCalib's 9-round grid search;
  a coarser "fast" search was removed after it showed ~50% more reprojection error) →
  optional `recomp_corner_calib` → final `calibrate()` + `reprojectpoints()`.
- `Calibrator.board` is a property: assigning a new `LedGridBoard` rebuilds `Xt`/`Yt` and
  marks a solved model stale. `assemble()` raises a clear error when a sample's point count
  differs from the board's, instead of letting the solver fail later with a numpy broadcast
  error. Every board swap must go through the instance (`cal.board = ...`), never by
  rebuilding `Xt`/`Yt` at a call site.
- `export_txt()` writes `calib_results.txt` (OCamCalib text format, computing `invpol` via
  `findinvpoly`) and returns the path. It takes either an `output_dir` or a full
  `path` (the GUI asks the user via `asksaveasfilename`). `.npz`/`.mat` output was removed
  — the text file is the only artifact, and that removed the package's only scipy use.

The tuning constants in `engine/coverage.py` (`DEFAULT_SAMPLE_THRESHOLD = 0.30`, in units of
target spans — not raw parameter units,
`DEFAULT_PARAM_RANGES = (0.6, 0.6, 0.3, 0.45)`, `DEFAULT_MIN_DB_SIZE = 20`) started as ROS
camera_calibration defaults but have been retuned for this smaller/farther-captured UV grid —
see the note at their definition. They are overridable per-run through `CalibratorConfig`, the
`--sample_threshold` / `--param_ranges` / `--min_db_size` flags of the live node
(the GUI does not show them).

### The solver (`engine/ocam_model.py`)

Pure math over `(Xt, Yt, Xp_abs, Yp_abs, xc, yc, ...)` arrays — no knowledge of samples,
goodness, or GUI state. `ima_proc` holds **1-based** (MATLAB-style) image numbers and `idx()`
converts to 0-based numpy indices wherever an array is touched; it's public (not `_idx`)
specifically because `diagnostics/plots.py` needs the same conversion. Keep this convention when
adding code that walks `ima_proc` — mixing 0- and 1-based indices here is the easiest way to
silently corrupt results. `taylor_order` default is 4; `min_order` is hardcoded to 4 in
`omni_find_parameters_fun`.

`recomp_corner_calib` (off by default) deliberately refines from the
*measured* UV points, not from model-reprojected points — an earlier version used model
projections as the refinement seed, which could drive reprojection error to exactly 0 because
it compared the model against its own output. Preserve this "never seed refinement from the
model" invariant if touching this function.

### Detection (`engine/detection.py`)

`get_corners(img, board, pattern)` returns `(ok, corners, board)` like ROS. `pattern`
(`CalibratorConfig.pattern`, `--pattern`, the GUI's Target toggle) selects one
detector; there is no fallback chain:

- `"led"` (default): OpenCV `findCirclesGrid` (symmetric grid, both orientations) with a
  blob detector tuned for small bright dots.
- `"chessboard"`: a printed checkerboard whose inner corners form the same point grid.
  `findChessboardCorners` **must** keep `CALIB_CB_FAST_CHECK`: without it a frame with no
  board costs a full search, which measured 80 s on one noisy 1280x800 frame on a Pi 5.

Detection used to try the checkerboard first on every frame and fall back to a custom
bright-dot grouper; the first never matched an LED grid and stalled live mode, the second
rescued 2 of 26 example images. Don't reintroduce either as an automatic fallback.

The `Calibrator` writes per-image previews/point dumps to
`<output_dir>/detected_marker_previews/` for accepted samples.

### Coverage (`engine/coverage.py`)

Two layers, only the first of which gates anything:

- **ROS-ported sample selection**: `get_parameters`, `is_good_sample`, `compute_goodenough` —
  the single source of truth for accept/reject and calibrate-readiness.
- **Bin-based hint report** (`COVERAGE_*_BINS`, `sample_metric`, `compute_bin_coverage`,
  `coverage_suggestions`, `format_bin_coverage`, `plot_bin_coverage`): a supplementary
  "next images to capture" report driven off accepted samples. It must never gate the
  CALIBRATE button.
- **The guide** (`next_capture_guide`): one "hold the grid here" target for the box, the
  graph and the headline text. It returns None exactly when `compute_goodenough` is ready
  -- never because the bins are full, since bins measure size/skew differently from the
  readiness ranges and fill earlier. Order: empty frame cell, then the least-covered
  readiness range (Size: closer; Skew: tilt; X/Y: toward the uncovered edge), then the
  sparsest cell while only the count is short.

Both layers share one coordinate frame: `get_parameters`/`sample_metric` reduce a view to
coordinates *relative to the camera's usable region* — the full frame, or the fisheye's
valid circle when `fov_radius_frac` is set. Anything that draws those coordinates back
onto the image must undo the same mapping (`coverage.normalized_to_pixel` /
`coverage.normalized_length_scale`, used by `gui.guide_box_rect`); scaling them by the
raw image size instead puts the "put the grid here" box on the wrong spot — up to 15% of
the image width off on a lens with an FOV radius fraction.

### GUI (`apps/gui.py`)

One class, `LiveCalibrationApp`, that reuses the engine directly — change pipeline
behavior in `Calibrator`/`coverage`, not in the GUI. It drains two queues from
`live_node`: an unthrottled preview queue for the image and a throttled result queue for
accept/reject. The four X/Y/Size/Skew range bars
mirror ROS's `redraw_monocular` bars. X/Y show the span the accepted samples cover; Size and
Skew are scored against a target, so they fill by progress instead. Green at full progress.
CALIBRATE is driven by `goodenough` (with an explicit confirm-to-override
when not ready); Show plots and Export calibration by `calibrated`. Everything is on the
main window -- grid size/spacing, Target and Mirror image in the second row, all applied
to the running session; there is no settings dialog or hidden dev mode. The sample log is
built from each `FrameResult` (not its `reason` string) and collapses repeated "no grid" /
"too similar" lines into one counted line, since a live stream produces one per frame.

The LED-grid row is editable: X/Y counts and pitch are applied
to the running `Calibrator` on commit (Enter, focus-out, or a spinbox arrow). A count
change swaps the board and *drops* the accepted samples, because they index the old
point grid; a pitch-only change keeps them and just marks the solved model stale. The
`Calibrator.board` setter rebuilds `Xt`/`Yt`, so the solver always uses the new grid
(assigning the board without rebuilding them yields a shape-mismatch broadcast error at
CALIBRATE). The consumer is paused and its queues are drained around a count change so a
frame detected with the old board cannot land after the clear.
`test/test_runtime_board.py` covers both paths, the world-point rebuild, and the revert on
invalid input.

The on-image **"put the grid here" box** (`gui.guide_box_rect` + `_draw_position_guide_on_image`)
marks the frame cell the next photo should go into. It is drawn on the live preview and the
sample view from the same guide the sidebar text and the
coverage graph use, so the three cannot disagree. `test/test_guide_overlay.py` pins the
geometry down (full-frame placement unchanged; circle-normalized placement lands in the cell
it names).

The **`Forward view (undistorted)`** checkbox (enabled only once calibrated) remaps the live
preview through `ocam_model.build_forward_view_maps` — a rectilinear/pinhole rendering of the
solved model. It is a *sanity check on the calibration*: the fisheye bows straight world lines,
and a correct model straightens them, so the LED grid should look like a clean perspective
rectangle rather than a barrel-bowed one. `test/test_forward_view.py` pins that down
quantitatively (0.725 px → 0.157 px per-line collinearity error on `example_images/`).

Do **not** try to widen this view to show the whole frame. A rectilinear projection sends a ray
at 90° off-axis to infinity, and this lens's border rays reach 125° — so no finite FOV contains
the rim, while the magnification penalty `1/tan(hfov/2)` shrinks the pattern you're inspecting
(0.18× at 160°, which renders as an hourglass with a smudge in the middle). Hence the fixed
`DEFAULT_FORWARD_VIEW_HFOV_DEG = 90.0`; showing everything would need a non-rectilinear
projection, which would not straighten lines and so would not serve the check.

### Camera setup (`apps/camera_wizard.py`)

Backs `docker/calibrate.sh` (which is now only a `python -m` shim with
`--repo-root`) and carries the logic that used to live in bash there. It finds
the attached camera, fills the templates under `config/<type>/`, and starts
`config/run-tmux.sh`. The templates remain the **source of truth** for driver
parameters, FOV/mirroring and everything else that is a property of the rig;
the wizard only writes what a template cannot know (serial number, device-tree
path, topic suffix).

Detection identifies a *device and its driver*, never the lens, and it is
conservative on purpose: exactly one camera is used without a prompt; several
of one family or several families require `--device`/`--camera` (or an
interactive answer); with `--yes` it fails instead of guessing. A wrongly
configured camera calibrates silently, so an explicit `--device` is also
allowed to name an undetected camera (e.g. a GigE Basler not visible without
host networking); `kind_for_device` lets a *detected* identifier pick the
family when `--camera` is `auto`. `detect_cameras` takes its sysfs root, device-tree prefix and
Basler enumerator as arguments so tests never need real hardware; the
libcamera probe keeps the sysfs `i2c ∩ media` heuristic and falls back to all
device-tree nodes when nothing pairs up. `test/test_camera_wizard.py` checks
the probe, the selection rules and that generated configs still match the
templates (including placeholder substitution).

Note `n_sq_x`/`n_sq_y` in the board config are unrelated to the camera side
configs `config/<camera>/`; don't confuse the two directories.

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
- Following from the `[row, col]` point convention above, `OCamModel.xc`/`.ss` "X" throughout
  the solver (`calibrate`, `omni_find_parameters_fun`, `reprojectpoints`, `omni3d2pixel`) means
  the **row** axis, and `.yc` means the **col** axis — the reverse of the naive width/height
  pairing. `Calibrator.cal_fromcorners`'s initial guess is correctly `xc=height/2.0,
  yc=width/2.0`, not the other way around; this has been verified by tracing the pairing through
  every solver function that reads it.
- **Mirroring the live input** (on by default: `--mirror`/`--no-mirror`, or the GUI's "Mirror image" switch)
  flips the frame in `live_node.mkgray`, *before* it reaches either queue, so the preview, the
  detection, the coverage advice and the saved model all describe the same (mirrored) view.
  The detector labels the grid in image order, so a mirrored frame reads the board with the
  axis that runs horizontally in that photo reversed — one of the four symmetries of the point
  grid, which the per-image rotation `R` in the solve absorbs. A *transposed* reading would
  not be absorbable (the 6×4 board is not a 4×6 board) and would show up as a large
  reprojection error. `detection.mirror_horizontal` uses `cv2.flip` rather than `img[:, ::-1]`
  on purpose: the numpy view has negative strides (which OpenCV refuses) and still aliases the
  caller's buffer — and the live path shares one array between the preview queue, the
  detection queue and `Sample.image`.
