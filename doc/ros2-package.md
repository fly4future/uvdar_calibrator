# Running as a ROS 2 Package

The repository is also a colcon-buildable `ament_python` package
(`package.xml`/`setup.py`/`setup.cfg`), providing two `ros2 run` entry points. This is
purely additive — `python -m uvdar_calibrator` keeps working standalone, no ROS
required.

Build it into a colcon workspace:

```bash
# from <colcon_ws>/src
git clone <this repo> uvdar_calibrator
cd <colcon_ws>
colcon build --packages-select uvdar_calibrator
source install/setup.bash
```

## Offline/Batch Mode

Equivalent to `python -m uvdar_calibrator`:

```bash
ros2 run uvdar_calibrator calibrate_offline --image_dir photos --gui
```

## Live Mode

A node that subscribes to a `sensor_msgs/Image` topic and feeds frames into the same
calibration engine one at a time, run the same way as upstream ROS's own
`camera_calibration` node:

```bash
ros2 run uvdar_calibrator cameracalibrator \
  --n_sq_x 6 --n_sq_y 4 --spacing_mm 50 image:=/camera/image_raw
```

This opens the same Tkinter GUI, but instead of loading a photo folder it captures
frames live from the given topic (remap `image:=` to your camera's topic name), at a
throttled processing rate (`--rate_hz`, default 2 Hz) so a live stream doesn't flood
detection with near-duplicate frames. Accept/reject, the readiness bars, CALIBRATE,
SAVE/EXPORT, and the Forward view toggle all work identically to the batch GUI.

There is no `SetCameraInfo` upload step: `sensor_msgs/CameraInfo`'s distortion models
have no slot for the OCamCalib polynomial, so results stay as local files
(`Omni_Calib_Results.npz`/`.mat` + `calib_results.txt`), same as the offline path.
