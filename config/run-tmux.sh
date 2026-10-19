#!/bin/bash

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in
SCRIPT_PATH=$(dirname "$SCRIPT")
cd "$SCRIPT_PATH" || exit

SESSION_FILE="$SCRIPT_PATH/$CAMERA_TYPE/session.yaml"
if [[ ! -f "$SESSION_FILE" && -f "$SESSION_FILE.template" ]]; then
  if grep -q "__[A-Z_]*__" "$SESSION_FILE.template"; then
    echo "ERROR: $SESSION_FILE does not exist yet and its template needs camera-specific values - run docker/calibrate.sh first" >&2
    exit 1
  fi
  cp "$SESSION_FILE.template" "$SESSION_FILE"
fi
if [[ ! -f "$SESSION_FILE" ]]; then
  echo "ERROR: session file not found: $SESSION_FILE" >&2
  exit 1
fi

TMUX_SESSION_NAME=uvdar_calibrator
TMUX_SOCKET_NAME=f4f

tmuxinator start -p "$SESSION_FILE" --no-attach

# if we are not in tmux attach to the session, otherwise switch to it
if [ -z "${TMUX:-}" ]; then
  tmux -L "$TMUX_SOCKET_NAME" a -t "$TMUX_SESSION_NAME"
else
  tmux detach-client -E "tmux -L \"$TMUX_SOCKET_NAME\" a -t \"$TMUX_SESSION_NAME\""
fi
