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
have no slot for the OCamCalib polynomial, so the result stays a local file
(`calib_results.txt`, written wherever the save dialog points), same as the offline
path.

### Mirrored cameras

Some drivers deliver a left-right flipped image. `--mirror` flips the incoming frames
back before the preview, the detection and the saved model ever see them, so the
calibration describes the camera rather than the flipped stream:

```bash
ros2 run uvdar_calibrator cameracalibrator --mirror image:=/camera/image_raw
```

The same switch is in the GUI's Advanced Settings ("Mirror incoming frames"), so it
can be flipped while a session is running. Flipping it discards the samples captured
so far: they describe the other orientation.

### Changing the LED grid during a session

The grid geometry is not baked into the node. The `LED grid: 6 x 4, 50 mm
spacing (change)` button opens the X/Y square counts and the pitch, and they
apply to the running calibrator as soon as you press Enter, leave the field, or
use a spinbox arrow. Changing a count changes the point grid the detector must
find, so the samples captured for the old board are dropped and capture
restarts from an empty set. Changing only the pitch keeps the samples -- they
describe the same points -- and marks the solved model stale, so press
CALIBRATE again to re-solve in the new units.

## Camera Setup and Launch

`docker/calibrate.sh` is not the only way to get from "camera is plugged in" to a
running calibrator. The same detection and the same `config/<type>/` templates back a
launch file that starts the driver and the GUI directly, without Docker or tmux:

```bash
ros2 launch uvdar_calibrator calibrate.launch.py camera:=auto
ros2 launch uvdar_calibrator calibrate.launch.py camera:=libcamera suffix:=down
ros2 launch uvdar_calibrator calibrate.launch.py camera:=basler device:=12345678
```

It detects the camera (exactly like the Docker wizard), writes the generated configs
under `~/.ros/uvdar_calibrator/config` (the installed share tree is never modified),
then starts `libcamera_ros_driver` or `pylon_ros2_camera_wrapper` with those configs
and `cameracalibrator` remapped to the topic that driver publishes. Exports default
to `~/.ros/uvdar_calibrator/output`; override with `config_dir:=` / `output_dir:=`.

`camera:=auto` never prompts: it picks the only camera it finds and fails, listing
what it saw, when the choice is not unique. Pass `camera:=` and/or `device:=` then.
A detected `device:=` implies its family, so a serial is enough even with two camera
kinds plugged in; an undetected device (a GigE Basler without host networking, for
instance) needs `camera:=` too, because nothing else can know what it is. For a driver that
delivers a mirrored image, add `mirror:=true`.

The two driver packages are environment-specific, so they are not ROS dependencies of
this package: install `ros-<distro>-libcamera-ros-driver` for libcamera, or build
[ctu-mrs/pylon-ros-camera](https://github.com/ctu-mrs/pylon-ros-camera) (branch
`ros2`) for Basler. The launch file also starts no RMW daemon — use the ROS
environment you normally run. Inside the Docker image, `docker/calibrate.sh` remains
the entry point and additionally manages the tmux session.
