# Using the GUI

The window opens once the camera wizard has started the driver (see the
[README](../README.md#getting-started)). It subscribes to the camera's image topic and
captures automatically.

![Calibration GUI screenshot](../.fig/gui_screenshot.png)

## The Window

- **Top rows:** the image topic with Start/Stop Capture; below it the LED grid size and
  spacing, the target (UV LEDs or a checkerboard) and **Mirror image**. All of them apply
  to the running session.
- **Left:** the live camera image with the "PUT LED GRID HERE" guide. Previous/Next
  browse the accepted samples, and Delete Sample removes a bad one (while capture is
  stopped).
- **Right:** whether you can calibrate and what to do next, the X / Y / Size / Skew
  range bars, a map of where the accepted samples sit in the frame, and the sample log.
- **Bottom right:** CALIBRATE, Show plots and Export calibration.

## Step by Step

1. Move the UV LED grid around in front of the camera. Frames are processed at about
   2 per second; the sample log shows each one as **added**, **too similar** to an
   earlier sample, or **no grid detected** (repeats are counted on one line).
2. Follow the guide: the box on the image and the text at the top of the sidebar say
   where to put the grid next. They stay until the samples are good enough.
3. Once the sidebar says **Ready to calibrate**, press **CALIBRATE**. Capture pauses
   while it solves. (You can also calibrate earlier after confirming a warning, but
   treat the result as preliminary.)
4. Check the result: the reprojection error shown after the solve, **Show plots**, and
   the forward view below.
5. Press **Export calibration**. The save dialog suggests `calib_results_<camera>.txt`
   in `/calibration_output`, which is the `calibration_output/` folder on the host. That
   text file is the only output.

Each range bar shows the span of that parameter the accepted samples cover; it turns
green when the span is wide enough. Readiness needs all four bars full **and** at least
20 accepted samples.

Important: **some photos being rejected is normal and correct.** Two photos
of the grid in nearly the same position, size, and tilt add no new
information, so only the first one is kept. The goal is not to collect many
images — it is to collect images that cover the full camera field of view
with varied positions, sizes, and tilts.

## The Forward View

Once calibrated, the **Forward view (undistorted)** checkbox below the image re-renders
it the way an ordinary perspective camera would have seen it, using the model just
solved. This is a sanity check: the fisheye bows the grid's rows and columns, and a
correct model straightens them again, so the LED grid should look like a clean
perspective rectangle.

The view shows the central 90° by default (the FOV box widens it). A perspective
projection cannot represent a ray at or past 90° from the optical axis, so a fisheye's
outer rim is unreachable at *any* setting, and widening the view shrinks the pattern
you are trying to inspect.

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

## The Diagnostics Window

After calibrating, **Show plots** opens a window with one tab per diagnostic. It does
not block the GUI. Each tab has the standard matplotlib toolbar, so use the
magnifier to zoom; with 20+ samples the reprojection panels are necessarily small.

| Tab | What it shows | What to look for |
| --- | --- | --- |
| **Reprojection** | One panel per accepted sample: detected markers (`+`) against the model's reprojected grid (`o`) | The two should sit on top of each other. A panel where they drift apart is a bad sample. |
| **Error analysis** | Reprojection error of every marker, per sample | A tight cluster near the origin. A sample that fans out wide is dragging the solve. |
| **Projection function** | The solved polynomial: image radius vs ray angle | A smooth monotonic curve. Wiggles or a fold-back mean the polynomial is overfitted. |
| **Extrinsics** | 3D scatter of where the board was for each sample, in mm relative to the camera | Spread in all directions and a wide range of distances. Points clustered at one distance mean your Size variety is not real; a point behind the camera or absurdly far away is a failed pose that will degrade the whole solve. |

The console also prints the per-sample and average reprojection error after every
solve.

## Changing the LED Grid

The grid fields (squares in X and Y, LED spacing in mm) apply as soon as you press
Enter, leave the field, or use a spinbox arrow. Changing a count changes the point grid
the detector looks for, so the samples captured for the old grid are dropped and
capture starts over. Changing only the spacing keeps the samples and asks you to press
CALIBRATE again.

## Mirrored Cameras

Some drivers deliver a left-right flipped image. **Mirror image** (on by default) flips the frames back
before the preview, the detection and the saved model see them. Switching it drops the
samples captured so far, since they describe the other orientation.

## Checkerboard Target

Instead of the UV LED grid you can calibrate against a printed checkerboard: set
**Target** to **Checkerboard**. Its inner corners take the place of the LEDs, so set the
grid fields to match (a 6 x 4 grid is a checkerboard of 7 x 5 squares). Switching drops
the samples captured so far.
