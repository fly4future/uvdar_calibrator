docker build -t uvdar_calibrator .

xhost +local:root
mkdir -p calibration_output

docker run -it --rm --network host --privileged -e DISPLAY \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev \
  -v "$PWD/calibration_output:/calibration_output" uvdar_calibrator
