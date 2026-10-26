"""Slider/settings helpers for the live camera exposure control.

``libcamera_ros_driver`` (the F4F fork) subscribes to ``~/expose_us``
(``std_msgs/Int32``, microseconds) and applies it to the running camera, so the
GUI only has to publish there. This module holds the pure parts -- the
logarithmic slider mapping, the remembered bounds/value -- with no ROS or Tk
dependency so they are testable off the drone.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

#: A short exposure keeps the UV LEDs crisp; 10 us is the usual sensor minimum.
DEFAULT_MIN_US = 10.0
#: Long enough for a dim rig, short enough that the 30 fps stream still moves.
DEFAULT_MAX_US = 100000.0
DEFAULT_EXPOSURE_US = 500.0
#: Slider travel; the value itself is logarithmic between the bounds.
SLIDER_STEPS = 1000

SETTINGS_ENV = "UVDAR_CALIBRATOR_SETTINGS"
SETTINGS_MIN_KEY = "exposure_min_us"
SETTINGS_MAX_KEY = "exposure_max_us"
SETTINGS_VALUE_KEY = "exposure_us"


def format_exposure(value: float) -> str:
    """Render microseconds without trailing ``.0`` or float noise."""
    value = float(value)
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def sanitize_bounds(minimum: float, maximum: float) -> tuple[float, float]:
    """Validate slider bounds; swaps a reversed pair, rejects nonsense."""
    lo, hi = float(minimum), float(maximum)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("exposure bounds must be finite numbers")
    if lo <= 0 or hi <= 0:
        raise ValueError("exposure bounds must be positive")
    return (lo, hi) if lo <= hi else (hi, lo)


def slider_value(position: float, minimum: float, maximum: float,
                 steps: int = SLIDER_STEPS) -> float:
    """Exposure at a slider position (0..steps), logarithmic between bounds."""
    lo, hi = sanitize_bounds(minimum, maximum)
    position = min(max(float(position), 0.0), float(steps))
    log_lo, log_hi = math.log(lo), math.log(hi)
    return math.exp(log_lo + (log_hi - log_lo) * position / steps)


def slider_position(value: float, minimum: float, maximum: float,
                    steps: int = SLIDER_STEPS) -> float:
    """Inverse of :func:`slider_value`, clamped to the slider's travel."""
    lo, hi = sanitize_bounds(minimum, maximum)
    log_lo, log_hi = math.log(lo), math.log(hi)
    if log_hi <= log_lo:
        return 0.0
    ratio = (math.log(min(max(float(value), lo), hi)) - log_lo) / (log_hi - log_lo)
    return min(max(ratio, 0.0), 1.0) * steps


def exposure_topic_for(image_topic: str) -> str:
    """Driver exposure topic for an ``.../image_raw`` image topic.

    ``libcamera_ros_driver`` names it after the camera node, i.e. the image
    topic's parent: ``/ns/rpi_camera_front/image_raw`` ->
    ``/ns/rpi_camera_front/expose_us``. Any other topic (a webcam, a replayed
    folder) has no such control, so the result is empty.
    """
    topic = str(image_topic or "").rstrip("/")
    suffix = "/image_raw"
    if not topic.endswith(suffix):
        return ""
    return topic[: -len(suffix)] + "/expose_us"


def settings_path() -> Path:
    """Where the GUI remembers exposure settings (``$UVDAR_CALIBRATOR_SETTINGS``)."""
    override = os.environ.get(SETTINGS_ENV)
    if override:
        return Path(override)
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "uvdar_calibrator" / "settings.json"


def load_settings() -> dict:
    """Remembered GUI settings; ``{}`` on a missing or unreadable file."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(settings: dict) -> None:
    """Persist GUI settings, creating the directory if needed."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def exposure_bounds(settings: dict | None = None) -> tuple[float, float]:
    """Remembered slider bounds, falling back to the defaults."""
    if settings is None:
        settings = load_settings()
    lo = settings.get(SETTINGS_MIN_KEY, DEFAULT_MIN_US)
    hi = settings.get(SETTINGS_MAX_KEY, DEFAULT_MAX_US)
    try:
        return sanitize_bounds(lo, hi)
    except (TypeError, ValueError):
        return DEFAULT_MIN_US, DEFAULT_MAX_US


def initial_exposure(bounds: tuple[float, float] | None = None,
                     settings: dict | None = None) -> float:
    """Remembered exposure (clamped to the slider bounds), else the default."""
    if settings is None:
        settings = load_settings()
    if bounds is None:
        bounds = exposure_bounds(settings)
    value = settings.get(SETTINGS_VALUE_KEY, DEFAULT_EXPOSURE_US)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_EXPOSURE_US
    if not math.isfinite(value) or value <= 0:
        value = DEFAULT_EXPOSURE_US
    return min(max(value, bounds[0]), bounds[1])


def remember_bounds(minimum: float, maximum: float) -> None:
    """Store the slider bounds; a read-only home directory is not an error."""
    settings = load_settings()
    settings[SETTINGS_MIN_KEY] = float(minimum)
    settings[SETTINGS_MAX_KEY] = float(maximum)
    try:
        save_settings(settings)
    except OSError:
        pass


def remember_exposure(value: float) -> None:
    """Store the last applied exposure; a read-only home directory is fine."""
    settings = load_settings()
    settings[SETTINGS_VALUE_KEY] = float(value)
    try:
        save_settings(settings)
    except OSError:
        pass
