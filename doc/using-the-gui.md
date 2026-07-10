# Using the GUI

## Running It

Install the required Python packages and launch the GUI from the repository root:

```bash
pip install -r requirements.txt
python -m uvdar_calibrator --image_dir photos --gui
```

## Step by Step

1. Click **Load / Analyze Images**. Photos are analyzed one at a time; the
   sample log shows, for each image, whether it was **added** as a sample,
   **rejected** because it is too similar to an already-accepted sample, or
   failed marker detection.
2. Check that the UV markers are detected correctly (browse the accepted
   samples with Previous/Next).
3. Review the four **X / Y / Size / Skew** progress bars. Each bar shows the
   range of that parameter covered by the accepted samples; it turns green
   when the covered range is wide enough.
4. Add more photos if the status says NOT READY (the "Next images to
   capture" box gives hints about what is missing).
5. Once the status says READY TO CALIBRATE, click **CALIBRATE**. (You can
   also calibrate earlier after confirming a warning, but treat the result
   as preliminary.)
6. Review the reprojection error. Once calibrated, the **Forward view
   (undistorted)** checkbox becomes available — toggle it to preview a
   cropped, forward-facing perspective crop generated from the calibrated
   model instead of the raw fisheye frame, as a visual sanity check (works
   both while browsing accepted samples with Previous/Next and, in live
   mode, on the incoming stream).
7. Click **SAVE / EXPORT** to write `Omni_Calib_Results.npz` and
   `calib_results.txt`.

Important: **some photos being rejected is normal and correct.** Two photos
of the grid in nearly the same position, size, and tilt add no new
information, so only the first one is kept. The goal is not to collect many
images — it is to collect images that cover the full camera field of view
with varied positions, sizes, and tilts.

![alt text](../figures/.figgui_interface.png)

## Understanding the Coverage Graph

With `--show_coverage`, a scatter graph shows where each **accepted** sample
places the UV LED grid in the camera image.

The x-axis shows the horizontal board location in the image; the y-axis shows the
vertical board location in the image. Each labeled point corresponds to one accepted
calibration sample.

A good calibration usually has an average reprojection error below 1.0 px. Lower is
better — values around 0.3-0.5 px are generally good.

## What Reprojection Error Means

Reprojection error is the difference between the detected UV marker location and the
model-predicted UV marker location. It is **not** the distance from the image center.

The error is computed approximately as:

```text
error = sqrt((detected_row - projected_row)^2 + (detected_col - projected_col)^2)
```

For each image, the displayed error is the average error across all detected UV
markers in that image. The center-to-point distance is useful for coverage and
field-of-view analysis, but it is not calibration error.

## Note

Some graphs and calibration information open in separate plot windows. The program
may pause until the current plot window is closed. To continue to the next graph or
calibration step, close the current plot tab/window first.
