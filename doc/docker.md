# Running in Docker

For easier setup we have a docker image that already contains camera drivers and UVDAR calibrator prebuilt.
On start it runs a setup wizard to select correct camera and then launches the driver driver and calibrator.

## Use the prebuilt image

```bash
docker pull ghcr.io/fly4future/uvdar_calibrator:latest

xhost +local:root # For Xorg GUI passthrough
mkdir -p calibration_output # This is where the generated calibration file will be stored

docker run -it --rm --network host --privileged -e DISPLAY \
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

## Develop / build locally

```bash
git clone git@github.com:fly4future/uvdar_calibrator.git
cd uvdar_calibrator
./docker-dev.sh
```

`docker-dev.sh` builds the image (`docker build -t uvdar_calibrator .`) and runs it with the same flags as above.
