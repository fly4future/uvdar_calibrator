"""Camera setup wizard: find one camera, fill its config, start the session.

Backs ``docker/calibrate.sh`` and is usable standalone::

    python -m uvdar_calibrator.apps.camera_wizard --list
    python -m uvdar_calibrator.apps.camera_wizard --camera basler --device 12345678
    python -m uvdar_calibrator.apps.camera_wizard --yes      # hands-off, single camera

The per-camera templates under ``config/<type>/`` stay the source of truth
for driver parameters; this module only decides *which* camera is being
configured and fills the values a template cannot know: the serial number,
the device-tree path and the image-topic suffix.

Detection is deliberately conservative. It can identify a *device* and its
driver, never the lens in front of it, so when the choice is not unique it
asks (or, with ``--yes``, fails) instead of guessing -- calibrating the
wrong camera leaves no visible trace except a model that is subtly wrong.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_SYSFS_ROOT = "/sys"
DEFAULT_DT_PREFIX = "/sys/firmware/devicetree"
DEFAULT_OUTPUT_DIR = "/calibration_output"

# Placeholders the session templates use; the wizard substitutes them.
_PLACEHOLDER_CAMERA_ID = "__CAMERA_ID__"
_PLACEHOLDER_DRIVER_TOPIC = "__DRIVER_IMAGE_TOPIC__"
_PLACEHOLDER_SUFFIX = "__DRIVER_SUFFIX__"


@dataclass(frozen=True)
class CameraDevice:
    """One physical camera found on this machine."""

    kind: str
    identifier: str
    label: str
    detail: str = ""


@dataclass
class Detection:
    """What one camera family's probe found (and why it found nothing)."""

    kind: str
    devices: List[CameraDevice]
    error: Optional[str] = None


@dataclass(frozen=True)
class PreparedCamera:
    """Generated configs for one camera, plus the topic its driver publishes."""

    kind: str
    camera_id: str
    config_dir: Path
    driver_config: Path
    calib_config: Optional[Path]
    session: Path
    topic: str


@dataclass(frozen=True)
class _DtNode:
    """A device-tree node seen through one of the sysfs views."""

    cls: str
    dev: str
    path: str
    compat: str


# ---------------------------------------------------------------------------
# libcamera: sensors are the device-tree nodes that appear both as an i2c
# client (the sensor chip) and as a media/video node (its v4l2 subdev).
# ---------------------------------------------------------------------------


def _scan_dt_nodes(sysfs_root: str, dt_prefix: str) -> List[_DtNode]:
    """Collect device-tree nodes visible through sysfs, without resolving models."""
    root = Path(sysfs_root)
    prefix = str(Path(dt_prefix).resolve())
    entries: List[_DtNode] = []
    seen = set()
    patterns = (
        ("bus/i2c/devices/*", "i2c"),
        ("class/video4linux/*", "media"),
        ("class/media/*", "media"),
    )
    for pattern, cls in patterns:
        for dev in sorted(root.glob(pattern)):
            node = None
            for candidate in (dev / "of_node", dev / "device" / "of_node"):
                if candidate.is_symlink():
                    node = candidate
                    break
            if node is None:
                continue
            target = str(node.resolve())
            if not target.startswith(prefix + os.sep):
                continue
            path = target[len(prefix):]
            compat = ""
            try:
                raw = (node / "compatible").read_bytes()
                compat = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except OSError:
                pass
            key = (cls, dev.name, path)
            if key in seen:
                continue
            seen.add(key)
            entries.append(_DtNode(cls, dev.name, path, compat))
    return entries


def _sensor_devices(entries: Sequence[_DtNode]) -> List[Tuple[str, str, str]]:
    """Reduce sysfs entries to sensors; falls back to every node when none pair up."""
    sensors = [
        node for node in entries
        if node.cls == "i2c" and any(
            other.path == node.path and other.cls == "media" for other in entries
        )
    ]
    if not sensors:
        sensors = list(entries)
    unique = {(node.dev, node.path, node.compat) for node in sensors}
    return sorted(unique)


def list_libcamera_devices(
    sysfs_root: str = DEFAULT_SYSFS_ROOT,
    dt_prefix: str = DEFAULT_DT_PREFIX,
) -> List[CameraDevice]:
    """Find libcamera sensors via the kernel device tree."""
    devices: Dict[str, CameraDevice] = {}
    for dev, path, compat in _sensor_devices(_scan_dt_nodes(sysfs_root, dt_prefix)):
        if path in devices:
            continue
        detail = f"sysfs {dev}" + (f" [{compat}]" if compat else "")
        devices[path] = CameraDevice("libcamera", path, path, detail)
    return list(devices.values())


# ---------------------------------------------------------------------------
# Basler: the serial number is the stable identity, and pypylon knows how to
# enumerate devices without opening them.
# ---------------------------------------------------------------------------


def _pypylon_enumerate() -> List[Tuple[str, str, str]]:
    """Return ``(serial, model, user_id)`` for every camera pypylon can see."""
    from pypylon import pylon

    rows = []
    for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
        user = dev.GetUserDefinedName() if dev.IsUserDefinedNameAvailable() else ""
        rows.append((dev.GetSerialNumber(), dev.GetModelName(), user))
    return rows


def list_basler_devices(
    enumerate_fn: Optional[Callable[[], Sequence[Tuple[str, str, str]]]] = None,
) -> Tuple[List[CameraDevice], Optional[str]]:
    """Find Basler cameras, returning ``(devices, error_reason)``."""
    if enumerate_fn is None:
        enumerate_fn = _pypylon_enumerate
    try:
        rows = list(enumerate_fn())
    except ImportError as exc:
        return [], f"pypylon is not installed ({exc})"
    except Exception as exc:  # noqa: BLE001 - the SDK raises its own types
        return [], str(exc) or exc.__class__.__name__

    devices = []
    for row in rows:
        serial, model, user = (list(row) + ["", "", ""])[:3]
        if not serial:
            continue
        label = f"{serial} {model}".strip()
        if user:
            label += f" (user id: {user})"
        devices.append(CameraDevice("basler", serial, label))
    return devices, None


# ---------------------------------------------------------------------------
# Detection and selection
# ---------------------------------------------------------------------------


def detect_cameras(
    sysfs_root: str = DEFAULT_SYSFS_ROOT,
    dt_prefix: str = DEFAULT_DT_PREFIX,
    basler_enumerate_fn: Optional[Callable[[], Sequence[Tuple[str, str, str]]]] = None,
) -> Dict[str, Detection]:
    """Probe every known camera family. Probing never modifies anything."""
    basler_devices, basler_error = list_basler_devices(basler_enumerate_fn)
    return {
        "libcamera": Detection("libcamera", list_libcamera_devices(sysfs_root, dt_prefix)),
        "basler": Detection("basler", basler_devices, basler_error),
    }


def resolve_kind(
    detections: Dict[str, Detection], requested: str = "auto",
) -> Tuple[Optional[str], Optional[str]]:
    """Pick the camera family; returns ``(kind, error)`` (``None`` means "ask")."""
    if requested and requested != "auto":
        if requested not in detections:
            known = ", ".join(sorted(detections))
            return None, f"unknown camera type '{requested}' (known: {known})"
        return requested, None

    kinds = [kind for kind, det in detections.items() if det.devices]
    if len(kinds) == 1:
        return kinds[0], None
    if len(kinds) > 1:
        return None, (
            f"cameras of more than one type detected ({', '.join(sorted(kinds))}); "
            "pass --camera"
        )
    return None, None


def resolve_device(
    kind: str,
    devices: Sequence[CameraDevice],
    requested: Optional[str],
    allow_unlisted: bool = False,
) -> Tuple[Optional[CameraDevice], Optional[str]]:
    """Pick one device; returns ``(device, error)`` (``None`` means "ask")."""
    if requested:
        for device in devices:
            if device.identifier == requested:
                return device, None
        if allow_unlisted:
            return CameraDevice(kind, requested, requested), None
        found = ", ".join(device.identifier for device in devices) or "none"
        return None, f"no {kind} camera with identifier '{requested}' (found: {found})"

    if len(devices) == 1:
        return devices[0], None
    if len(devices) > 1:
        return None, f"{len(devices)} {kind} cameras detected; pass --device <identifier>"
    return None, None


def kind_for_device(
    detections: Dict[str, Detection], identifier: str,
) -> Optional[str]:
    """Camera family that lists ``identifier``; ``None`` if none or several.

    Lets an explicit device select the family ("auto" with a serial), which is
    what somebody who knows the serial and has two camera kinds plugged in
    expects. It is not a guess: the identifier belongs to exactly one family.
    """
    kinds = [
        kind for kind, det in detections.items()
        if any(device.identifier == identifier for device in det.devices)
    ]
    return kinds[0] if len(kinds) == 1 else None


# ---------------------------------------------------------------------------
# Config materialisation (same files the tmuxinator sessions read)
# ---------------------------------------------------------------------------


def _copy_template(path: Path) -> None:
    template = path.with_name(path.name + ".template")
    if not template.is_file():
        raise FileNotFoundError(f"template not found: {template}")
    shutil.copyfile(template, path)


def _set_yaml_value(path: Path, key: str, value: str) -> None:
    """Replace the value of the first ``key:`` line, keeping indentation/comment."""
    pattern = re.compile(
        r"^(?P<pre>\s*" + re.escape(key) + r"\s*:\s*)"
        r"(?P<val>\"[^\"]*\"|'[^']*'|[^\s#]+)(?P<post>.*)$"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            lines[index] = match.group("pre") + value + match.group("post")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise KeyError(f"key '{key}' not found in {path}")


def _get_yaml_value(path: Path, key: str) -> str:
    """Read a scalar ``key:`` value, stripping quotes and trailing comments."""
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*:\s*(?P<rest>.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group("rest").split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return ""


def _fill_session(path: Path, replacements: Dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        if placeholder not in text:
            raise KeyError(f"placeholder {placeholder} not found in {path}")
        text = text.replace(placeholder, value)
    path.write_text(text, encoding="utf-8")


def libcamera_image_topic(suffix: str) -> str:
    """Topic ``libcamera_ros_driver`` publishes for ``uav_name:=uvdar_calibrator``."""
    return f"/uvdar_calibrator/rpi_camera_{suffix}/image_raw"


def prepare_libcamera(
    config_root: Path, device: CameraDevice, suffix: str = "front",
) -> PreparedCamera:
    """Write the libcamera driver + calibration + session configs."""
    config_dir = Path(config_root) / "libcamera"
    custom_cfg = config_dir / "camera_custom_config.yaml"
    calib_cfg = config_dir / "camera_calib.yaml"
    session = config_dir / "session.yaml"

    _copy_template(custom_cfg)
    _copy_template(calib_cfg)
    _copy_template(session)

    topic = libcamera_image_topic(suffix)
    _set_yaml_value(custom_cfg, "camera_name", f'"{device.identifier}"')
    _set_yaml_value(calib_cfg, "camera_name", device.identifier)
    _fill_session(
        session,
        {
            _PLACEHOLDER_CAMERA_ID: device.identifier,
            _PLACEHOLDER_SUFFIX: suffix,
            _PLACEHOLDER_DRIVER_TOPIC: topic,
        },
    )
    return PreparedCamera(
        "libcamera", device.identifier, config_dir, custom_cfg, calib_cfg, session, topic,
    )


def prepare_basler(config_root: Path, device: CameraDevice) -> PreparedCamera:
    """Write the Basler driver + session configs."""
    config_dir = Path(config_root) / "basler"
    camera_cfg = config_dir / "camera.yaml"
    session = config_dir / "session.yaml"

    _copy_template(camera_cfg)
    _copy_template(session)

    _set_yaml_value(camera_cfg, "device_user_id", '""')
    _set_yaml_value(camera_cfg, "device_serial_number", f'"{device.identifier}"')

    camera_id = _get_yaml_value(camera_cfg, "camera_id") or "uvdar_calibrator"
    node_name = _get_yaml_value(camera_cfg, "node_name") or "camera"
    topic = f"/{camera_id}/{node_name}/image_raw"
    _fill_session(
        session,
        {
            _PLACEHOLDER_CAMERA_ID: device.identifier,
            _PLACEHOLDER_DRIVER_TOPIC: topic,
        },
    )
    return PreparedCamera(
        "basler", device.identifier, config_dir, camera_cfg, None, session, topic,
    )


# ---------------------------------------------------------------------------
# Prompting / launching
# ---------------------------------------------------------------------------


def _select(message: str, options: Sequence[str]) -> int:
    """Numbered prompt; returns the chosen index. Only call with a tty."""
    print(message)
    for index, option in enumerate(options, 1):
        print(f"  [{index}] {option}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        if raw in options:
            return list(options).index(raw)
        print("Enter the number of one of the options.")


def format_detections(detections: Dict[str, Detection]) -> str:
    """Human-readable summary of a probe result."""
    lines = ["Detected cameras:"]
    for kind, det in detections.items():
        if det.devices:
            for device in det.devices:
                detail = f"  ({device.detail})" if device.detail else ""
                lines.append(f"  {kind}: {device.label}{detail}")
        else:
            reason = f"  ({det.error})" if det.error else ""
            lines.append(f"  {kind}: none detected{reason}")
    return "\n".join(lines)


def _print_detections(detections: Dict[str, Detection]) -> None:
    print(format_detections(detections))


def _tty_hint() -> str:
    """Explain what to do when a choice is needed but nothing can be asked."""
    if sys.stdin.isatty():
        return ""
    return (
        " (no terminal attached: run with 'docker run -it' "
        "or pass --camera/--device)"
    )


def _manual_device(kind: str, det: Detection) -> CameraDevice:
    """Ask for the identifier when the probe found nothing (or is unavailable)."""
    if det.error:
        print(f"{kind}: {det.error}", file=sys.stderr)
    if kind == "basler":
        print(
            "A GigE camera also needs host networking (docker run --network host).",
            file=sys.stderr,
        )
        identifier = input("Camera serial number: ").strip()
        return CameraDevice(kind, identifier, identifier)
    identifier = input(
        "Camera device-tree path (or a unique part, e.g. i2c@80000): "
    ).strip()
    return CameraDevice(kind, identifier, identifier)


def _handle_termination(signum, frame) -> None:
    """Docker stop / SIGTERM: tear the session down before exiting."""
    _kill_stale_session()
    raise SystemExit(130)


def _env_id(name: str, fallback: int) -> int:
    """Integer value of an environment variable, or ``fallback``."""
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else fallback
    except ValueError:
        return fallback


def _fix_output_ownership(output_dir: Optional[Path] = None) -> None:
    """Hand root-created files to the host user (containers run as root).

    The container is root because it needs device and X11 access, so anything
    written into the mounted output directory is root-owned on the host.
    ``HOST_UID``/``HOST_GID`` (set by ``docker-dev.sh``) say who it belongs to;
    without them the owner of the mounted directory is used. A no-op anywhere
    but Linux, where this problem exists.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    target = Path(output_dir or os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    try:
        stat_result = target.stat()
    except OSError:
        return
    uid = _env_id("HOST_UID", stat_result.st_uid)
    gid = _env_id("HOST_GID", stat_result.st_gid)
    if uid == 0:
        print(
            f"NOTE: {target} is owned by root; pass -e HOST_UID=$(id -u) "
            "-e HOST_GID=$(id -g) to get files owned by you.",
            file=sys.stderr,
        )
        return
    for path in (target, *target.rglob("*")):
        try:
            if path.lstat().st_uid != 0:
                continue
            os.chown(path, uid, gid, follow_symlinks=False)
        except OSError:
            continue


def _kill_stale_session() -> None:
    try:
        subprocess.run(
            ["tmux", "-L", "f4f", "kill-session", "-t", "uvdar_calibrator"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        pass


def _run_session(repo_root: Path, kind: str) -> int:
    script = Path(repo_root) / "config" / "run-tmux.sh"
    if not script.is_file():
        print(f"ERROR: session launcher not found: {script}", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["CAMERA_TYPE"] = kind
    try:
        return subprocess.run(["bash", str(script)], env=env).returncode
    except KeyboardInterrupt:
        _kill_stale_session()
        print("Aborted.", file=sys.stderr)
        return 130


def _default_repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / "config" / "run-tmux.sh").is_file():
            return candidate
    return candidates[0]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvdar_calibrator camera setup",
        description=(
            "Detect a connected camera, fill its driver/calibration/session configs "
            "and start the calibration session. One camera is selected automatically; "
            "pass --camera/--device to disambiguate."
        ),
    )
    parser.add_argument(
        "--camera", choices=("auto", "libcamera", "basler"), default="auto",
        help="Camera family. Default: auto (detect; asks if ambiguous).",
    )
    parser.add_argument(
        "--device", default=None,
        help=(
            "Camera identifier: serial number (Basler) or device-tree path "
            "(libcamera). Also configures a camera the probe cannot see."
        ),
    )
    parser.add_argument(
        "--suffix", default="front",
        help=(
            "libcamera image-topic suffix (topic becomes .../rpi_camera_<suffix>/"
            "image_raw). Default: front."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Only print the detected cameras and exit.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Never prompt: fail if the choice is not unique.",
    )
    parser.add_argument(
        "--no-launch", action="store_true",
        help="Write the configs but do not start the tmux session.",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Repository root holding config/ (default: detected).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _build_arg_parser().parse_args(argv)
    try:
        signal.signal(signal.SIGTERM, _handle_termination)
    except (AttributeError, ValueError, OSError):
        pass
    repo_root = Path(args.repo_root) if args.repo_root else _default_repo_root()
    detections = detect_cameras(
        sysfs_root=os.environ.get("SYSFS_ROOT", DEFAULT_SYSFS_ROOT),
        dt_prefix=os.environ.get("FIRMWARE_DT_PREFIX", DEFAULT_DT_PREFIX),
    )

    if args.list:
        _print_detections(detections)
        return 0

    interactive = sys.stdin.isatty() and not args.yes

    requested_kind = args.camera
    if args.device and requested_kind == "auto":
        requested_kind = kind_for_device(detections, args.device) or requested_kind

    try:
        kind, error = resolve_kind(detections, requested_kind)
        if kind is None:
            if not interactive:
                _print_detections(detections)
                print(
                    f"ERROR: {error or 'no cameras detected'}{_tty_hint()}",
                    file=sys.stderr,
                )
                return 1
            _print_detections(detections)
            kind = list(detections)[_select(
                "Select camera type:", list(detections),
            )]

        detection = detections[kind]
        device, error = resolve_device(
            kind, detection.devices, args.device, allow_unlisted=bool(args.device),
        )
        if device is not None and args.device and all(
            d.identifier != args.device for d in detection.devices
        ):
            print(
                f"WARNING: '{args.device}' was not detected; configuring it anyway.",
                file=sys.stderr,
            )
        if device is None:
            if detection.devices:
                if not interactive:
                    _print_detections(detections)
                    print(f"ERROR: {error}{_tty_hint()}", file=sys.stderr)
                    return 1
                index = _select(
                    f"Select the {kind} camera to calibrate:",
                    [
                        d.label + (f"  ({d.detail})" if d.detail else "")
                        for d in detection.devices
                    ],
                )
                device = detection.devices[index]
            else:
                if not interactive:
                    _print_detections(detections)
                    print(
                        f"ERROR: no {kind} camera detected. Connect the camera or "
                        f"pass --device.{_tty_hint()}",
                        file=sys.stderr,
                    )
                    return 1
                device = _manual_device(kind, detection)

        if not device.identifier:
            print("ERROR: the camera identifier cannot be empty.", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        _kill_stale_session()
        print("Aborted.", file=sys.stderr)
        return 130

    config_root = repo_root / "config"
    if kind == "libcamera":
        prepared = prepare_libcamera(config_root, device, suffix=args.suffix)
    else:
        prepared = prepare_basler(config_root, device)

    print(f"Camera: {kind} {device.identifier}")
    print(f"Config: {prepared.config_dir}")
    if args.no_launch:
        print(f"Session: {prepared.session}")
        return 0

    _kill_stale_session()
    print("Starting calibration session...")
    try:
        return _run_session(repo_root, kind)
    except KeyboardInterrupt:
        _kill_stale_session()
        print("Aborted.", file=sys.stderr)
        return 130
    finally:
        _fix_output_ownership()


if __name__ == "__main__":
    sys.exit(main())
