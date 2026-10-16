#!/bin/bash
#
# UVDAR camera calibration wizard.
# Asks for the camera type and model, writes the driver configs and starts the tmux session via config/run-tmux.sh.
#
set -euo pipefail

if [[ ! -t 0 ]]; then
  echo "ERROR: no terminal attached — run the container with 'docker run -it ...'." >&2
  exit 1
fi

SCRIPT=$(readlink -f "$0")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT")")
CONFIG_DIR="$REPO_ROOT/config"

# set +u
# if [[ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]]; then
#   # shellcheck disable=SC1091
#   source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
# fi
# for ws in /ros2_ws/install/setup.bash "$REPO_ROOT/../../install/setup.bash"; do
#   if [[ -f "$ws" ]]; then
#     # shellcheck disable=SC1091
#     source "$ws"
#     break
#   fi
# done
# set -u

echo "Select camera type:"
select CAMERA_TYPE in libcamera basler; do
  [[ -n "$CAMERA_TYPE" ]] && break
done
export CAMERA_TYPE

generate_from_template() {
  local generated="$1" template="$2"
  [[ -f "$template" ]] || { echo "ERROR: template not found: $template" >&2; exit 1; }
  cp "$template" "$generated"
}

# Helper to replace the value of a top-level-ish `key:` line, preserving indentation and trailing comments 
# Usage: set_yaml_value <file> <key> <new-value> (new-value already quoted if the file uses quotes for that key)
set_yaml_value() {
  local file="$1" key="$2" value="$3"
  python3 - "$file" "$key" "$value" <<'EOF'
import re, sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
pat = re.compile(r'^(\s*' + re.escape(key) + r'\s*:\s*)("[^"]*"|\'[^\']*\'|[^\s#]+)(.*)$')
updated = False
out = []
for line in open(path).read().splitlines():
    m = pat.match(line)
    if m and not updated:
        line = m.group(1) + value + m.group(3)
        updated = True
    out.append(line)
if not updated:
    sys.exit(f"key '{key}' not found in {path}")
open(path, 'w').write('\n'.join(out) + '\n')
EOF
}

get_yaml_value() {
  local file="$1" key="$2"
  grep -E "^[[:space:]]*$key:" "$file" 2>/dev/null | head -1 | sed -E "s/^[[:space:]]*$key:[[:space:]]*[\"']?([^\"'#]+)[\"']?.*/\1/" | xargs || true
}

# Substitute wizard-chosen values into a generated session file.
# Usage: fill_session <session> <camera-id> [<driver-topic>]
fill_session() {
  local session="$1" camera_id="$2" driver_topic="${3:-}"
  python3 - "$session" "$camera_id" "$driver_topic" <<'EOF'
import sys
path, camera_id, driver_topic = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
if "__CAMERA_ID__" not in text:
    sys.exit(f"placeholder __CAMERA_ID__ not found in {path}")
text = text.replace("__CAMERA_ID__", camera_id)
if driver_topic:
    if "__DRIVER_IMAGE_TOPIC__" not in text:
        sys.exit(f"placeholder __DRIVER_IMAGE_TOPIC__ not found in {path}")
    text = text.replace("__DRIVER_IMAGE_TOPIC__", driver_topic)
open(path, "w").write(text)
EOF
}

# --- libcamera: pick a camera by device-tree path ------------------------
# Cameras are listed via sysfs device-tree links, one per line: "<dev>|<device-tree-path>|<compatible>".
#
# The kernel device tree is always visible through /sys (privileged container with /dev passthrough).
#
# Only sensor nodes are reported: a DT path must show up BOTH as an i2c client node (the sensor chip) AND 
# as a media/video node (its v4l2 subdev). This filters out i2c buses, CSI receivers, ISPs and codecs.
list_dt_cameras() {
  local sysfs="${SYSFS_ROOT:-/sys}"
  local dt_prefix="${FIRMWARE_DT_PREFIX:-/sys/firmware/devicetree}"
  local dev node target path compat cls
  local all="" sensors=""
  for dev in "$sysfs"/bus/i2c/devices/* "$sysfs"/class/video4linux/* "$sysfs"/class/media/*; do
    [[ -e "$dev" ]] || continue
    node=""
    if [[ -L "$dev/of_node" ]]; then
      node="$dev/of_node"
    elif [[ -L "$dev/device/of_node" ]]; then
      node="$dev/device/of_node"
    else
      continue
    fi
    target=$(readlink -f "$node" 2>/dev/null) || continue
    case "$target" in
      "$dt_prefix"/*) path="${target#$dt_prefix}" ;;
      *) continue ;;
    esac
    case "$dev" in
      "$sysfs"/bus/i2c/devices/*) cls="i2c" ;;
      *) cls="media" ;;
    esac
    compat=""
    if [[ -f "$node/compatible" ]]; then
      compat=$(tr '\0' ' ' < "$node/compatible" 2>/dev/null)
    fi
    case "$all" in
      # Skip exact duplicates only; the same path MUST be recorded once per
      # device class, otherwise the i2c/media intersection below breaks.
      *"${cls}|$(basename "$dev")|${path}|"*) ;;
      *) all+="${cls}|$(basename "$dev")|${path}|${compat}"$'\n' ;;
    esac
  done
  local c d p co
  # Paths present as both an i2c client and a media/video node = sensors.
  while IFS='|' read -r c d p co; do
    [[ -n "$p" && "$c" == "i2c" ]] || continue
    if awk -F'|' -v path="$p" '$1 == "media" && $3 == path { found=1; exit } END { exit !found }' <<< "$all"; then
      sensors+="${d}|${p}|${co}"$'\n'
    fi
  done <<< "$all"
  if [[ -n "$sensors" ]]; then
    printf '%s' "$sensors" | sort -u
  else
    awk -F'|' 'NF >= 3 { print $2 "|" $3 "|" $4 }' <<< "$all" | sort -u
  fi
}

setup_libcamera() {
  local custom_cfg="$CONFIG_DIR/libcamera/camera_custom_config.yaml"
  local calib_cfg="$CONFIG_DIR/libcamera/camera_calib.yaml"
  local session="$CONFIG_DIR/libcamera/session.yaml"
  generate_from_template "$custom_cfg" "$custom_cfg.template"
  generate_from_template "$calib_cfg" "$calib_cfg.template"
  generate_from_template "$session" "$session.template"
  echo
  echo "== libcamera setup =="
  local paths=() details=()
  add_candidate() { # $1=device-tree path, $2=human-readable source detail
    local have
    for have in "${paths[@]}"; do [[ "$have" == "$1" ]] && return 0; done
    paths+=("$1")
    details+=("$2")
  }

  local dev path compat
  while IFS='|' read -r dev path compat; do
    [[ -n "$path" ]] && add_candidate "$path" "sysfs $dev${compat:+ [$compat]}"
  done < <(list_dt_cameras)

  local camera_name=""
  if ((${#paths[@]} > 0)); then
    echo
    echo "Select the camera to calibrate:"
    select camera_name in "${paths[@]}"; do
      [[ -n "$camera_name" ]] && break
    done
    echo "camera_name: $camera_name"
  else
    echo "ERROR: no cameras detected."
    echo "Enter the camera device-tree path (or unique part, e.g. i2c@80000)."
    read -rp "camera_name: " camera_name
  fi
  [[ -z "$camera_name" ]] && { echo "ERROR: camera_name cannot be empty." >&2; exit 1; }

  # camera_name is set both in the driver config and in the calib file.
  set_yaml_value "$custom_cfg" "camera_name" "\"$camera_name\""
  set_yaml_value "$calib_cfg" "camera_name" "$camera_name"
  echo

  fill_session "$session" "$camera_name"
}

# --- basler: pick a camera by serial number --------------------------------
setup_basler() {
  local cam_cfg="$CONFIG_DIR/basler/camera.yaml"
  local session="$CONFIG_DIR/basler/session.yaml"
  local session_tpl="$session.template"
  generate_from_template "$cam_cfg" "$cam_cfg.template"
  echo
  echo "== Basler setup =="
  local serials=()
  python3 >/tmp/basler_list.txt 2>/tmp/basler_err.txt <<'EOF' || true
try:
    from pypylon import pylon
except ImportError:
    raise SystemExit("no pypylon")
tlf = pylon.TlFactory.GetInstance()
devs = tlf.EnumerateDevices()
if not devs:
    raise SystemExit("no devices")
for d in devs:
    print(f"{d.GetSerialNumber()}|{d.GetModelName()}|{d.GetUserDefinedName()}")
EOF


  local serial=""
  if ((${#serials[@]} > 0)); then
    local labels=() line ser model user label
    for line in "${serials[@]}"; do
      IFS='|' read -r ser model user <<< "$line"
      label="$ser"
      if [[ -n "$model" && -n "$user" ]]; then
        label+=" ($model, user '$user')"
      elif [[ -n "$model" ]]; then
        label+=" ($model)"
      elif [[ -n "$user" ]]; then
        label+=" (user '$user')"
      fi
      labels+=("$label")
    done
    echo
    echo "Select the camera to calibrate:"
    local choice=""
    select choice in "${labels[@]}"; do
      if [[ -n "$choice" ]]; then
        serial="${serials[$((REPLY-1))]}"
        serial="${serial%%|*}" # raw entries are "serial|model|user" — keep the serial
        break
      fi
    done
  else
    echo "Enter the camera serial number (read from sticker on the camera or use pylon viewer)."
    read -rp "device_serial_number: " serial
  fi
  [[ -z "$serial" ]] && { echo "ERROR: serial number cannot be empty." >&2; exit 1; }

  set_yaml_value "$cam_cfg" "device_user_id" '""'
  set_yaml_value "$cam_cfg" "device_serial_number" "\"$serial\""

  local cam_id node_name driver_topic
  cam_id=$(get_yaml_value "$cam_cfg" "camera_id")
  node_name=$(get_yaml_value "$cam_cfg" "node_name")
  cam_id="${cam_id:-uvdar_calibrator}"
  node_name="${node_name:-camera}"
  driver_topic="/$cam_id/$node_name/image_raw"

  generate_from_template "$session" "$session_tpl"

  fill_session "$session" "$serial" "$driver_topic"
}

case "$CAMERA_TYPE" in
  libcamera) setup_libcamera ;;
  basler) setup_basler ;;
  *) echo "ERROR: unknown camera type '$CAMERA_TYPE'." >&2; exit 1 ;;
esac

# Restart a stale session from a previous run, if any.
tmux -L f4f kill-session -t uvdar_calibrator >/dev/null 2>&1 || true

exec "$CONFIG_DIR/run-tmux.sh"
