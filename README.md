# Camera_Calibration_UVDAR
A Python-based UV-DAR camera calibration tool based on [Davide Scaramuzza's OCamCalib model](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab/ocamcalib-toolbox-download-page?authuser=0), adapted for UV-sensitive cameras using an LED grid calibration pattern.

This tool is intended for calibrating the UV-sensitive cameras. The calibration pattern should be a non-square UV LED grid, where the LED markers act like the internal corners of a checkerboard grid.

The tool follows the architecture of ROS `camera_calibration` (image_pipeline): photos are fed one at a time into a calibration engine that decides whether to *accept* each view as a calibration sample. Views that are too similar to an already-accepted sample are **rejected as near-duplicates** — this is expected and is what produces a diverse calibration set. Readiness is reported as X / Y / Size / Skew range progress bars, exactly like the ROS tool.

The grid should be non-square, and composed of LEDs emitting in the 395nm range.
An example of such grid:
![alt text](figures/.figpattern_off.jpeg)
![alt text](figures/.figpattern_on.jpeg)
![alt text](figures/.figpattern_backside.jpeg)

If the camera is constructed correctly, the output image of the pattern should look like this:
 
![alt text](figures/.figcamera_view.jpg)

## Overview

The full calibration workflow is:

1. Capture UV LED grid images using any suitable camera capture method such as Arducam on Raspberry Pi, Pylon Viewer, ROS/rosbags, etc.
2. Move the captured images into a folder called `photos`. The images can have any filename and may be any supported image type.
3. Run the Python calibration GUI.
4. Check that UV markers are detected correctly.
5. Use the coverage graph to determine whether more images are needed.
6. Calibrate the camera.
7. Export the final OCamCalib parameters to `calib_results.txt`.

By default, the calibration script loads every supported image in the `photos` folder.

Supported image types:

```text
jpg, jpeg, bmp, png, tif, tiff
```

## 1. Capture Calibration Images (Ex: Arducam on Raspberry Pi)

The following Raspberry Pi / Arducam process is only an example. It is not required if you are using another camera system or capture tool.

 Before running the Python calibration tool, capture calibration images using the Arducam connected to the Raspberry Pi.

Use the same exposure, gain, image size, and resolution settings for every calibration image.

Use this command to capture one image:

```bash
rpicam-still --shutter 1 -t 5000 -o center_close.bmp --encoding bmp --gain 0.05 --width 960 --height 600
```

The output filename can be anything. For example:
```text
rpicam-still --shutter 1 -t 5000 -o center_close.bmp --encoding bmp --gain 0.05 --width 960 --height 600
rpicam-still --shutter 1 -t 5000 -o left_edge.bmp --encoding bmp --gain 0.05 --width 960 --height 600
rpicam-still --shutter 1 -t 5000 -o top_corner.bmp --encoding bmp --gain 0.05 --width 960 --height 600
```

BMP is recommended for Raspberry Pi capture because it avoids extra compression artifacts, but the Python calibration script can also read other supported image types if they are placed in the photos folder.

Recommended capture settings:
```text
shutter:  1
timeout:  5000 ms
encoding: bmp recommended
gain:     0.05
width:    960
height:   600
```
## 2. Capture Good Calibration Images

For good calibration, the UV LED grid must appear in many different parts of the image.

Capture images where:

- the full UV LED grid is visible
- all LEDs are detected clearly
- LED blobs are small and not saturated
- the grid appears near the image center
- the grid appears near the left side of the image
- the grid appears near the right side of the image
- the grid appears near the top of the image
- the grid appears near the bottom of the image
- the grid appears near the image corners
- the grid appears at different distances from the camera
- the grid is tilted in different directions

A good starting point is usually:
15-25 usable images

More images can help, but image diversity is more important than the raw image count. The GUI will help determine whether enough images have been captured.

## 3. Move Images into the Python Calibration Folder

On the computer where you run the Python calibration code, create a folder named:

```text
photos
```

Place all captured calibration images inside this folder.

The expected folder structure is:

```text
uvdar_calibrator_repo/
├── uvdar_calibrator/        # the calibration package
│   ├── board.py             # LED grid target geometry
│   ├── detection.py         # marker detection (chessboard -> circle grid -> UV dots)
│   ├── ocam_model.py        # OCamCalib/Scaramuzza solver math
│   ├── coverage.py          # sample selection + readiness scoring
│   ├── calibrator.py        # Calibrator engine (accept/reject, solve, export)
│   ├── plots.py             # matplotlib diagnostics
│   ├── gui.py               # Tkinter GUI
│   └── cli.py               # command-line entry point
├── photos/
│   ├── i_1.bmp
│   ├── center_close.bmp
│   ├── left_edge.png
│   ├── top_corner.jpg
│   └── calibration_view_12.tiff
```

The Python calibration script reads every supported image in the `photos` folder by default.  The images do **not** need to follow a specific naming pattern.

Valid example filenames:

```text
i_1.bmp
left_corner.png
center_close.jpg
calibration_view_12.tiff
robofly_test_image.bmp
image001.jpeg

The only requirements are:
1. The images are inside the photos folder.
2. The images are one of the supported file types.
3. The UV LED grid is visible in the image.

Optional filtering is still available. To use only files beginning with a specific prefix, use `--base_name`. To use only one image type, use `--extension`. For example:

```bash
python -m uvdar_calibrator --image_dir photos --base_name i_ --extension bmp --gui
```

## 4. Install Python Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## 5. Run the UV-DAR Calibration GUI

From the repository root, run:

```bash
python -m uvdar_calibrator --image_dir photos --gui
```

## 6. Using the GUI

In the GUI:

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
6. Review the reprojection error.
7. Click **SAVE / EXPORT** to write `Omni_Calib_Results.npz` and
   `calib_results.txt`.

Important: **some photos being rejected is normal and correct.** Two photos
of the grid in nearly the same position, size, and tilt add no new
information, so only the first one is kept. The goal is not to collect many
images — it is to collect images that cover the full camera field of view
with varied positions, sizes, and tilts.

![alt text](figures/.figgui_interface.png)

## 7. Understanding the Coverage Graph

With `--show_coverage`, a scatter graph shows where each **accepted** sample
places the UV LED grid in the camera image.

The x-axis shows:

```text
horizontal board location in image
```

The y-axis shows:

```text
vertical board location in image
```

Each labeled point corresponds to one accepted calibration sample.

A good calibration usually has:
average reprojection error < 1.0 px

Lower is better. Values around:
0.3-0.5 px
are generally good.

## 8. What Reprojection Error Means

Reprojection error is the difference between the detected UV marker location and the model-predicted UV marker location.

It is not the distance from the image center.

The error is computed approximately as:

```text
error = sqrt((detected_row - projected_row)^2 + (detected_col - projected_col)^2)
```

For each image, the displayed error is the average error across all detected UV markers in that image. The center-to-point distance is useful for coverage and field-of-view analysis, but it is not calibration error.

## 9. Command-Line Calibration Without the GUI

You can also run calibration directly from the terminal.

```bash
python -m uvdar_calibrator --image_dir photos
```

The CLI feeds photos through the same accept/reject sample selection as the
GUI, prints a readiness report, and then calibrates. If the accepted samples
do not yet cover enough variation, it prints a warning and calibrates
anyway — treat that result as preliminary.

## 10. Coverage-Only Mode

To check image coverage without running full calibration:

```bash
python -m uvdar_calibrator --image_dir photos --coverage_only --show_coverage
```

This mode is useful after adding new images. It lets you check whether the current photo set has enough variation before running the full calibration.

## Note
Some graphs and calibration information open in separate plot windows. The program may pause until the current plot window is closed. To continue to the next graph or calibration step, close the current plot tab/window first.