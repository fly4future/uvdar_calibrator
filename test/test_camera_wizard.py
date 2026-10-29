"""Assert-based checks for the camera setup wizard.

Run directly: python3 test/test_camera_wizard.py

These cover the two things that can go wrong without a visible symptom: the
wizard silently picking the wrong camera (so a model gets calibrated for a
lens it was not taken with) and the generated configs drifting from the
templates the tmuxinator sessions read.
"""

from pathlib import Path
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uvdar_calibrator.apps import camera_wizard as cw  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER = re.compile(r"__[A-Z_]+__")


def _node(cls, dev, path, compat=""):
    return cw._DtNode(cls, dev, path, compat)


def _quiet(function, *args, **kwargs):
    """Run with captured output; returns (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = function(*args, **kwargs)
    return rc, out.getvalue(), err.getvalue()


def test_sensor_devices_need_both_views():
    """A sensor is an i2c client that is also a media node; buses are not."""
    entries = [
        _node("i2c", "0-0036", "/soc/i2c@80000/sensor@36", "sony,imx477"),
        _node("media", "video0", "/soc/i2c@80000/sensor@36", "sony,imx477"),
        _node("i2c", "0-0050", "/soc/i2c@80000/eeprom@50", "atmel,24c32"),
        _node("media", "media0", "/soc/csi@880000", "arm,scmi"),
    ]
    rows = cw._sensor_devices(entries)
    assert rows == [("0-0036", "/soc/i2c@80000/sensor@36", "sony,imx477")], rows

    # Nothing pairs up (old kernels, unusual drivers): show every node, so a
    # real camera is never hidden behind a failed heuristic.
    fallback = cw._sensor_devices([
        _node("media", "video1", "/soc/i2c@80001/sensor@10", "ovti,ov5647"),
    ])
    assert fallback == [("video1", "/soc/i2c@80001/sensor@10", "ovti,ov5647")], fallback
    print("ok: only i2c+media nodes count as sensors, with a fallback")


def test_scan_follows_of_node_symlinks():
    """The sysfs scan resolves of_node links, reads compatible, dedupes by path."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dt = root / "device-tree"
        sysfs = root / "sys"
        targets = {
            "i2c@80000": dt / "soc" / "i2c@80000" / "sensor@36",
            "i2c@80001": dt / "soc" / "i2c@80001" / "sensor@10",
        }
        for name, target in targets.items():
            target.mkdir(parents=True)
            compat = "sony,imx477" if name == "i2c@80000" else "ovti,ov5647"
            (target / "compatible").write_bytes(compat.encode() + b"\0")

        links = [
            (sysfs / "bus/i2c/devices/0-0036/of_node", targets["i2c@80000"]),
            (sysfs / "class/video4linux/video0/of_node", targets["i2c@80000"]),
            (sysfs / "class/media/media0/of_node", targets["i2c@80000"]),
            (sysfs / "bus/i2c/devices/1-0010/of_node", targets["i2c@80001"]),
            (sysfs / "class/video4linux/video1/device/of_node", targets["i2c@80001"]),
        ]
        try:
            for link, target in links:
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(target), str(link), target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            print(f"skip: cannot create symlinks here ({exc})")
            return

        devices = cw.list_libcamera_devices(str(sysfs), str(dt))
        identifiers = sorted(device.identifier for device in devices)
        assert identifiers == [
            "/soc/i2c@80000/sensor@36",
            "/soc/i2c@80001/sensor@10",
        ], identifiers
        by_id = {device.identifier: device for device in devices}
        detail = by_id["/soc/i2c@80000/sensor@36"].detail
        assert "sony,imx477" in detail, detail
    print("ok: sysfs scan finds the sensors behind the of_node links")


def test_basler_listing_never_raises():
    """pypylon may be missing, or the SDK may fail; both become a reason."""
    devices, error = cw.list_basler_devices(
        lambda: [("12345678", "acA1300-60gm", ""), ("", "broken", "")]
    )
    assert error is None and len(devices) == 1, (devices, error)
    assert devices[0].identifier == "12345678"
    assert devices[0].label == "12345678 acA1300-60gm", devices[0].label

    devices, error = cw.list_basler_devices(
        lambda: [("1", "acA1300", "front"), ("2", "acA1300", "")]
    )
    assert devices[1].label == "2 acA1300"
    assert "user id: front" in devices[0].label, devices[0].label

    def missing():
        raise ImportError("No module named 'pypylon'")

    devices, error = cw.list_basler_devices(missing)
    assert devices == [] and "pypylon" in error, (devices, error)

    def broken():
        raise RuntimeError("no transport layer")

    devices, error = cw.list_basler_devices(broken)
    assert devices == [] and "transport" in error, (devices, error)
    print("ok: Basler enumeration degrades to a reason, never a crash")


def test_selection_prefers_unambiguous_answers():
    """Exactly one device is automatic; two is an explicit choice, never a guess."""
    one = cw.CameraDevice("basler", "1", "1 acA1300")
    two = cw.CameraDevice("basler", "2", "2 acA1300")
    detections = {
        "libcamera": cw.Detection("libcamera", []),
        "basler": cw.Detection("basler", [one, two]),
    }

    kind, error = cw.resolve_kind(detections)
    assert kind == "basler" and error is None, (kind, error)

    kind, error = cw.resolve_kind(
        {"libcamera": cw.Detection("libcamera", [cw.CameraDevice(
            "libcamera", "/soc/cam", "/soc/cam")]), "basler": detections["basler"]}
    )
    assert kind is None and "--camera" in error, (kind, error)

    kind, error = cw.resolve_kind(detections, "libcamera")
    assert kind == "libcamera" and error is None, (kind, error)
    kind, error = cw.resolve_kind(detections, "usb")
    assert kind is None and "unknown camera type" in error, (kind, error)

    device, error = cw.resolve_device("basler", [one, two], None)
    assert device is None and "--device" in error, (device, error)
    device, error = cw.resolve_device("basler", [one], None)
    assert device is one and error is None
    device, error = cw.resolve_device("basler", [one, two], "2")
    assert device is two and error is None

    device, error = cw.resolve_device("basler", [one], "999")
    assert device is None and "found: 1" in error, (device, error)
    # An explicit identifier may name a camera the probe could not see, which
    # is how a GigE camera on a non-host network still gets configured.
    device, error = cw.resolve_device("basler", [], "999", allow_unlisted=True)
    assert device is not None and device.identifier == "999" and error is None

    # An explicit identifier also picks the family, so "auto" plus a serial is
    # enough when two camera kinds are plugged in.
    both = {
        "libcamera": cw.Detection("libcamera", [cw.CameraDevice(
            "libcamera", "/soc/cam", "/soc/cam")]),
        "basler": cw.Detection("basler", [one]),
    }
    assert cw.kind_for_device(both, "1") == "basler"
    assert cw.kind_for_device(both, "/soc/cam") == "libcamera"
    assert cw.kind_for_device(both, "999") is None
    print("ok: unique camera auto-selects, ambiguous ones require explicit flags")


def test_generated_configs_match_the_templates():
    """The wizard writes exactly the files the tmuxinator sessions read."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        shutil.copytree(_REPO_ROOT / "config", repo / "config")

        lib = cw.CameraDevice("libcamera", "/soc/i2c@80000/sensor@36", "cam")
        prepared = cw.prepare_libcamera(repo / "config", lib, suffix="down")
        session = prepared.session
        cfg_dir = repo / "config" / "libcamera"
        custom = (cfg_dir / "camera_custom_config.yaml").read_text()
        calib = (cfg_dir / "camera_calib.yaml").read_text()
        session_text = session.read_text()

        assert 'camera_name: "/soc/i2c@80000/sensor@36"' in custom, custom
        assert "camera_name: /soc/i2c@80000/sensor@36" in calib, calib
        assert not _PLACEHOLDER.search(session_text), session_text
        assert "suffix:=down" in session_text
        assert prepared.topic == "/uvdar_calibrator/rpi_camera_down/image_raw"
        assert f"image:={prepared.topic}" in session_text
        assert prepared.driver_config == cfg_dir / "camera_custom_config.yaml"
        assert prepared.calib_config == cfg_dir / "camera_calib.yaml"
        assert "--camera_id \"/soc/i2c@80000/sensor@36\"" in session_text

        bas = cw.CameraDevice("basler", "12345678", "12345678 acA1300")
        prepared = cw.prepare_basler(repo / "config", bas)
        session_text = prepared.session.read_text()
        camera = prepared.driver_config.read_text()
        assert prepared.topic == "/uvdar_calibrator/camera/image_raw"
        assert f"image:={prepared.topic}" in session_text

        assert re.search(r'^\s*device_user_id: ""$', camera, re.MULTILINE), camera
        assert re.search(
            r'^\s*device_serial_number: "12345678"(?:\s+#.*)?$', camera, re.MULTILINE
        ), camera
        assert not _PLACEHOLDER.search(session_text), session_text
        assert "12345678" in session_text

        # Re-running resets the generated file from the template instead of
        # mutating yesterday's values.
        prepared.driver_config.write_text("camera_id: broken\n", encoding="utf-8")
        cw.prepare_basler(repo / "config", bas)
        assert prepared.driver_config.read_text() == camera
    print("ok: configs are regenerated from the templates and fully filled")


def test_main_hands_off_and_refuses_ambiguity():
    """--yes must never prompt: one camera configures, two abort with advice."""
    two = {"basler": cw.Detection("basler", [
        cw.CameraDevice("basler", "1", "1 acA1300"),
        cw.CameraDevice("basler", "2", "2 acA1300"),
    ])}
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        shutil.copytree(_REPO_ROOT / "config", repo / "config")

        original = cw.detect_cameras
        cw.detect_cameras = lambda **kwargs: two
        try:
            rc, out, err = _quiet(
                cw.main, ["--yes", "--no-launch", "--repo-root", str(repo)]
            )
            assert rc == 1, rc
            assert "--device" in err and "Camera: " not in out, (out, err)
            assert not (repo / "config" / "basler" / "camera.yaml").exists()

            # Naming one of the two is enough; the family is inferred from it.
            rc, out, err = _quiet(
                cw.main,
                ["--yes", "--no-launch", "--device", "2", "--repo-root", str(repo)],
            )
            assert rc == 0, (rc, err)
            assert "Camera: basler 2" in out, out
            camera = (repo / "config" / "basler" / "camera.yaml").read_text()
            assert 'device_serial_number: "2"' in camera, camera
        finally:
            cw.detect_cameras = original  # noqa: B010
    print("ok: --yes auto-configures a single named camera and refuses a crowd")


def test_list_is_a_pure_query():
    """--list must never write configs, even when nothing is detectable."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SYSFS_ROOT"] = str(Path(tmp) / "empty-sys")
        try:
            rc, out, err = _quiet(cw.main, ["--list", "--repo-root", tmp])
        finally:
            del os.environ["SYSFS_ROOT"]
        assert rc == 0, (rc, err)
        assert "libcamera:" in out and "basler:" in out, out
        assert not any((Path(tmp) / "config").glob("*/*.yaml"))
    print("ok: --list only reports")


def test_docker_images_start_the_wizard():
    """Both Docker images must start the Python wizard, not a script of their own.

    A merge once brought back an old bash wizard as the entry point: the image
    stopped detecting cameras and libcamera sessions kept literal
    __DRIVER_SUFFIX__/__DRIVER_IMAGE_TOPIC__ values, so no images ever reached
    the GUI. Nothing failed loudly, so pin the hand-off down here.
    """
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmds = [line for line in dockerfile.splitlines() if line.startswith("CMD ")]
    assert len(cmds) == 2, cmds
    for cmd in cmds:
        assert "uvdar_calibrator.apps.camera_wizard" in cmd, cmd
    print("ok: both docker images start the wizard")


if __name__ == "__main__":
    test_sensor_devices_need_both_views()
    test_scan_follows_of_node_symlinks()
    test_basler_listing_never_raises()
    test_selection_prefers_unambiguous_answers()
    test_generated_configs_match_the_templates()
    test_main_hands_off_and_refuses_ambiguity()
    test_list_is_a_pure_query()
    test_docker_images_start_the_wizard()
    print("all checks passed")
