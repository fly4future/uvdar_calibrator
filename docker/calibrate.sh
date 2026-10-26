#!/bin/bash
#
# UVDAR camera calibration wizard.
#
# Entrance point of the Docker image. The detection/config logic lives in
# uvdar_calibrator/apps/camera_wizard.py, which is importable and usable
# outside the container too:
#
#   python -m uvdar_calibrator.apps.camera_wizard --list
#   python -m uvdar_calibrator.apps.camera_wizard --camera basler --device 12345678
#   python -m uvdar_calibrator.apps.camera_wizard --yes   # hands-off, single camera
#
# All arguments are passed through unchanged. This script stays tiny on
# purpose: the wizard is Python so it can be tested without bash, Docker or a
# camera, and the same module backs setup_camera and the ROS launch file.
set -euo pipefail

SCRIPT=$(readlink -f "$0")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT")")

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m uvdar_calibrator.apps.camera_wizard --repo-root "$REPO_ROOT" "$@"
