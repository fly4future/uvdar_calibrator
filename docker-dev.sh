#!/usr/bin/env bash
# Build the development image and run it with this checkout mounted.
#
#   ./docker-dev.sh          build the calibrator and start the camera wizard
#   ./docker-dev.sh bash     open a shell instead
set -e
cd "$(dirname "$(readlink -f "$0")")"

xhost +local:root
mkdir -p calibration_output

HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
  docker compose -f compose.dev.yaml run --rm --build dev "$@"
