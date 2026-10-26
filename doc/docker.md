# Running in Docker

For easier setup we have a docker image that already contains camera drivers and UVDAR calibrator prebuilt.
On start it runs a setup wizard that detects the connected camera (asking only when it cannot
pick one automatically), writes its config, and launches the driver and the calibrator.

## Use the prebuilt image

```bash
docker pull ghcr.io/fly4future/uvdar_calibrator:latest

xhost +local:root # For Xorg GUI passthrough
mkdir -p calibration_output # This is where the generated calibration file will be stored

docker run -it --rm --network host --privileged -e DISPLAY \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev \
  -v "$PWD/calibration_output:/calibration_output" \
  ghcr.io/fly4future/uvdar_calibrator:latest
```

Parameters:

- `--rm` deletes the container when it exits
- `--network host` puts the container on the host's network, which is needed to connect to Basler GigE cameras.
- `--privileged` gives the container access to all host devices (USB, video4linux, media, I2C) and to `/sys`, which the wizard reads to detect which camera is connected. Without this we would need to pass each device separately.
- `-e DISPLAY` and `-v /tmp/.X11-unix:/tmp/.X11-unix` are needed for GUI from inside docker
- `-v /dev:/dev` shows the host's current device nodes inside the container, including cameras plugged in after the container started.
- `-e HOST_UID=... -e HOST_GID=...` makes the generated files in `calibration_output` belong to you instead of root.
  The container runs as root (it needs device and X11 access), so when the wizard exits it changes the owner of every
  root-owned file in `/calibration_output` to this user and group. Without these variables it uses the owner of the
  mounted directory. This only works if you create the directory before starting the container, because otherwise Docker
  creates it as root. If the container is killed instead of exiting normally (e.g. `docker kill`), the files stay root-owned.

## Camera detection and selection

The wizard (`docker/calibrate.sh`, which runs
`uvdar_calibrator/apps/camera_wizard.py`) finds the connected camera, writes its
driver/calibration/session configs from the templates in `config/<type>/`, and
starts the calibration session.

What it can detect:

- **libcamera / Raspberry Pi sensors** from the kernel device tree (works even
  before a driver is bound): a sensor is a node that appears both as an i2c
  client and as a media/video node. Driver: `libcamera_ros_driver`.
- **Basler** cameras from `pypylon`, by serial number. Driver:
  `pylon_ros2_camera_wrapper` (the ctu-mrs fork, cloned by the `Dockerfile`).

What it cannot detect is the **optics**: focal length, FOV radius, mirroring
and which position on the drone the camera points at are properties of the rig,
not of the device. Those stay in the templates / `--suffix`.

Selection rules — it never guesses:

- exactly one camera → used automatically, no prompt;
- several cameras of one family → pick one (prompt, or `--device`);
- several families → pass `--camera` (an explicit `--device` also disambiguates
  when its identifier is reported by exactly one family).

For unattended use (and for scripts) everything is flag-driven:

```bash
# what is connected?
python3 -m uvdar_calibrator.apps.camera_wizard --list

# hands-off when exactly one camera is attached
python3 -m uvdar_calibrator.apps.camera_wizard --yes

# explicit choice; --device also configures a camera the probe cannot see
python3 -m uvdar_calibrator.apps.camera_wizard --camera basler --device 12345678
python3 -m uvdar_calibrator.apps.camera_wizard --camera libcamera \
    --device /soc/i2c@80000/imx477@36 --suffix front

# write the configs without starting the tmux session
python3 -m uvdar_calibrator.apps.camera_wizard --camera libcamera --no-launch
```

`--suffix` is the libcamera image-topic suffix (`rpi_camera_<suffix>`,
default `front`) and is how several libcamera sensors on one drone are kept
apart. Each camera is calibrated in its own run; the session name and tmux
socket are fixed, so runs are sequential, not concurrent.

The same module is installed as `setup_camera`, so it also works outside the
container (with the Python deps and `pypylon` if a Basler is used). Without
Docker, `ros2 launch uvdar_calibrator calibrate.launch.py camera:=auto` runs the
same detection and starts the driver and the calibrator directly; see
[ros2-package.md](ros2-package.md).

## Develop / build locally

```bash
git clone git@github.com:fly4future/uvdar_calibrator.git
cd uvdar_calibrator
./docker-dev.sh
```

`docker-dev.sh` builds the development image (`Dockerfile.dev`) and opens a shell in it via `compose.dev.yaml`,
with the same flags as above. Your checkout is mounted at `/ros2_ws/src/uvdar_calibrator` and is not built
automatically. Inside the container (in `/ros2_ws`):

```bash
colcon build --symlink-install   # build the mounted calibrator
source install/setup.bash        # only needed after the first build
src/uvdar_calibrator/docker/calibrate.sh   # the camera calibration wizard
```

To build and run the production image locally instead, use `docker build -t uvdar_calibrator .` and the `docker run`
command above with `uvdar_calibrator` as the image.
