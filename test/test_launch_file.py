"""Assert-based checks for launch/calibrate.launch.py.

Run directly: python3 test/test_launch_file.py

There is no `launch`/`launch_ros` here (and no camera), so the ROS modules are
replaced by tiny recording stubs and detection is faked. That verifies *our*
wiring -- which driver launch, which generated config, which topic, which
node arguments -- but not the real ROS signatures; those mirror the tmux
session templates, which remain the source of truth.
"""

from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCH_FILE = _REPO_ROOT / "launch" / "calibrate.launch.py"

_SHARE = {"path": None}


class _Config:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context.get(self.name, "")


class _Declare:
    def __init__(self, name, default_value="", description=""):
        self.name = name
        self.default_value = default_value


class _Opaque:
    def __init__(self, function):
        self.function = function


class _Include:
    def __init__(self, source, launch_arguments=()):
        self.source = source
        self.launch_arguments = dict(launch_arguments)


class _Source:
    def __init__(self, path):
        self.path = path


class _Node:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LaunchDescription:
    def __init__(self, actions=()):
        self.actions = list(actions)


_STUB_NAMES = (
    "ament_index_python",
    "ament_index_python.packages",
    "launch",
    "launch.actions",
    "launch.launch_description_sources",
    "launch.substitutions",
    "launch_ros",
    "launch_ros.actions",
)


def _install_stubs():
    saved = {name: sys.modules.get(name) for name in _STUB_NAMES}

    def module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    launch = module("launch")
    launch.LaunchDescription = _LaunchDescription
    actions = module("launch.actions")
    actions.DeclareLaunchArgument = _Declare
    actions.IncludeLaunchDescription = _Include
    actions.OpaqueFunction = _Opaque
    sources = module("launch.launch_description_sources")
    sources.PythonLaunchDescriptionSource = _Source
    substitutions = module("launch.substitutions")
    substitutions.LaunchConfiguration = _Config
    module("launch_ros")
    ros_actions = module("launch_ros.actions")
    ros_actions.Node = _Node
    module("ament_index_python")
    packages = module("ament_index_python.packages")
    packages.get_package_share_directory = (
        lambda name: str(Path(_SHARE["path"]) / name)
    )
    return saved


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("uvdar_calibrate_launch", _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare(module, detections, arguments):
    original = module.wizard.detect_cameras
    module.wizard.detect_cameras = lambda: detections
    try:
        description = module.generate_launch_description()
        opaque = next(a for a in description.actions if isinstance(a, _Opaque))
        context = {
            "camera": "auto",
            "device": "",
            "suffix": "front",
            "mirror": "false",
            "config_dir": arguments["config_dir"],
            "output_dir": arguments["output_dir"],
        }
        context.update(arguments)
        return opaque.function(context)
    finally:
        module.wizard.detect_cameras = original  # noqa: B010


def _basler(serial="12345678"):
    return {"basler": types.SimpleNamespace(
        devices=[types.SimpleNamespace(
            kind="basler",
            identifier=serial,
            label=f"{serial} acA1300",
            detail="",
        )],
        error=None,
    ), "libcamera": types.SimpleNamespace(devices=[], error=None)}


def _libcamera(path="/soc/i2c@80000/imx477@36"):
    return {"libcamera": types.SimpleNamespace(
        devices=[types.SimpleNamespace(
            kind="libcamera", identifier=path, label=path, detail="sysfs video0",
        )],
        error=None,
    ), "basler": types.SimpleNamespace(devices=[], error=None)}


def test_basler_launch_wiring():
    saved = _install_stubs()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            share = Path(tmp) / "share"
            shutil.copytree(
                _REPO_ROOT / "config", share / "uvdar_calibrator" / "config"
            )
            _SHARE["path"] = str(share)
            module = _load_launch_module()
            config_dir = Path(tmp) / "runtime" / "config"
            output_dir = Path(tmp) / "runtime" / "output"
            driver, calibrator = _prepare(module, _basler(), {
                "config_dir": str(config_dir),
                "output_dir": str(output_dir),
            })

            assert isinstance(driver, _Include)
            assert "pylon_ros2_camera_wrapper" in driver.source.path
            assert driver.source.path.endswith("pylon_ros2_camera.launch.py"), (
                driver.source.path
            )
            driver_config = Path(driver.launch_arguments["config_file"])
            assert driver_config == config_dir / "basler" / "camera.yaml"
            assert 'device_serial_number: "12345678"' in driver_config.read_text()

            assert isinstance(calibrator, _Node)
            kwargs = calibrator.kwargs
            assert kwargs["package"] == "uvdar_calibrator"
            assert kwargs["executable"] == "cameracalibrator"
            assert kwargs["remappings"] == [
                ("image", "/uvdar_calibrator/camera/image_raw")
            ], kwargs["remappings"]
            assert "--camera_id" in kwargs["arguments"]
            assert "12345678" in kwargs["arguments"]
            assert "--mirror" not in kwargs["arguments"]
            assert output_dir.is_dir()
        print("ok: basler launch starts the wrapper + calibrator on the wrapper topic")
    finally:
        for name in _STUB_NAMES:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]


def test_libcamera_launch_wiring_and_mirror():
    saved = _install_stubs()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            share = Path(tmp) / "share"
            shutil.copytree(
                _REPO_ROOT / "config", share / "uvdar_calibrator" / "config"
            )
            _SHARE["path"] = str(share)
            module = _load_launch_module()
            config_dir = Path(tmp) / "runtime" / "config"
            driver, calibrator = _prepare(module, _libcamera(), {
                "camera": "libcamera",
                "device": "/soc/i2c@80000/imx477@36",
                "suffix": "down",
                "mirror": "true",
                "config_dir": str(config_dir),
                "output_dir": str(Path(tmp) / "runtime" / "output"),
            })

            assert "libcamera_ros_driver" in driver.source.path
            assert driver.source.path.endswith("camera.launch.py"), driver.source.path
            args = driver.launch_arguments
            assert args["uav_name"] == "uvdar_calibrator"
            assert args["suffix"] == "down"
            assert Path(args["custom_config"]) == (
                config_dir / "libcamera" / "camera_custom_config.yaml"
            )
            assert args["calib_url"].startswith("file://"), args["calib_url"]

            assert calibrator.kwargs["remappings"] == [
                ("image", "/uvdar_calibrator/rpi_camera_down/image_raw")
            ]
            assert "--mirror" in calibrator.kwargs["arguments"]
        print("ok: libcamera launch honours suffix/mirror and the driver topic")
    finally:
        for name in _STUB_NAMES:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]


def test_launch_auto_refuses_ambiguity():
    saved = _install_stubs()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            share = Path(tmp) / "share"
            shutil.copytree(
                _REPO_ROOT / "config", share / "uvdar_calibrator" / "config"
            )
            _SHARE["path"] = str(share)
            module = _load_launch_module()

            # A named device selects its family, so "auto" plus a serial works
            # when both a libcamera sensor and a Basler are plugged in.
            combined = _basler()
            combined["libcamera"] = types.SimpleNamespace(
                devices=[types.SimpleNamespace(
                    kind="libcamera", identifier="/soc/cam", label="/soc/cam",
                    detail="",
                )],
                error=None,
            )
            driver, calibrator = _prepare(module, combined, {
                "device": "12345678",
                "config_dir": str(Path(tmp) / "named"),
                "output_dir": str(Path(tmp) / "named-out"),
            })
            assert driver.source.path.endswith("pylon_ros2_camera.launch.py")

            two = _basler()
            two["basler"].devices.append(types.SimpleNamespace(
                kind="basler", identifier="87654321", label="87654321", detail="",
            ))
            try:
                _prepare(module, two, {
                    "config_dir": str(Path(tmp) / "config"),
                    "output_dir": str(Path(tmp) / "output"),
                })
            except RuntimeError as exc:
                assert "--device" in str(exc), str(exc)
            else:
                raise AssertionError("an ambiguous camera set must not be guessed")

            try:
                _prepare(module, {"libcamera": types.SimpleNamespace(
                    devices=[], error=None,
                ), "basler": types.SimpleNamespace(devices=[], error=None)}, {
                    "config_dir": str(Path(tmp) / "config"),
                    "output_dir": str(Path(tmp) / "output"),
                })
            except RuntimeError as exc:
                assert "no cameras detected" in str(exc), str(exc)
            else:
                raise AssertionError("no camera must fail, not prompt")
        print("ok: camera:=auto fails with the detection summary instead of guessing")
    finally:
        for name in _STUB_NAMES:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]


if __name__ == "__main__":
    test_basler_launch_wiring()
    test_libcamera_launch_wiring_and_mirror()
    test_launch_auto_refuses_ambiguity()
    print("all checks passed")
