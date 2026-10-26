#!/usr/bin/env bash
set -e

xhost +local:root
mkdir -p calibration_output

# Do not mount the checkout over /ros2_ws: that is the MRS workspace inside the
# image, and its entrypoint sources /ros2_ws/install. Hiding it is what made the
# container fail with an entrypoint error. compose.dev.yaml mounts the source
# one level deeper and keeps build/install/log in seeded volumes.
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose -f compose.dev.yaml run --rm --build dev
