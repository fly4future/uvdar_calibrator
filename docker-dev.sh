docker build -t uvdar_calibrator .

xhost +local:root

docker run -it --rm --network host --privileged -e DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  -v "$PWD/calibration_output:/calibration_output" \
  -v "$PWD:/ros2_ws" \
  uvdar_calibrator