xhost +local:root
mkdir -p calibration_output

HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose -f compose.dev.yaml run --rm --build dev
