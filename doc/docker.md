# Running in Docker

For easier setup we have a docker image that already contains camera drivers and UVDAR calibrator prebuilt.
On start it runs a setup wizard to select correct camera and then launches the driver driver and calibrator.

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
