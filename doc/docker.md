# Camera Setup and Docker

The calibrator only runs in Docker: the image contains ROS 2, both camera drivers and the calibrator.
On start, the camera wizard detects the connected cameras, writes its driver configs from the templates and starts a tmux session with the driver and the calibration window.
Closing the window ends the session and the container.

## `docker run` flags

```bash
docker run -it --rm --network host --privileged -e DISPLAY \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev \
  -v "$PWD/calibration_output:/calibration_output" \
  ghcr.io/fly4future/uvdar_calibrator:latest
```

- `--network host` is needed to access network cameras (Basler). Also gives access to ROS topics from host if that is needed
- `--privileged` gives access to all host devices (USB, video4linux, media, I2C) and to `/sys`, which the wizard reads to detect the camera
- `-e DISPLAY` and `-v /tmp/.X11-unix:/tmp/.X11-unix` let you run GUI applications from docker
- `-v /dev:/dev` forwards all devices connected to host into the container
- `-e HOST_UID=... -e HOST_GID=...`: which UID and GID should own files in `/calibration_output` (otherwise it would be root)

`./docker-dev.sh` passes the same flags through `compose.dev.yaml`.

## Choosing the camera

What the wizard detects cameras:

- **libcamera** from the device tree: a sensor is a node that appears both as an I2C client and as a
  media/video node
- **Basler** from `pypylon`, by serial number.

## Building the images

There is a `Dockerfile` with two targets that share every layer up to the
calibrator's own source:

- the default (last) stage copies the source in and builds it. GitHub Actions
  publishes it as `ghcr.io/fly4future/uvdar_calibrator` on every push to `main`. To
  build it locally: `docker build -t uvdar_calibrator .`
- the `dev` stage, used by `./docker-dev.sh`, mounts the code at
  `/ros2_ws/src/uvdar_calibrator` and builds it on start
