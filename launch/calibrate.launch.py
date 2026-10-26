"""Launch a camera driver and cameracalibrator for one prepared camera.

    ros2 launch uvdar_calibrator calibrate.launch.py camera:=auto
    ros2 launch uvdar_calibrator calibrate.launch.py camera:=libcamera suffix:=down
    ros2 launch uvdar_calibrator calibrate.launch.py camera:=basler device:=12345678

Non-Docker sibling of ``docker/calibrate.sh``: the same detection, templates
and generated configs, but two processes instead of a tmux session. The
configs are written under ``config_dir`` (default
``~/.ros/uvdar_calibrator/config``), never into the installed share tree.

This starts neither a RMW daemon nor a specific RMW -- use the environment you
normally run ROS in. ``camera:=auto`` picks the only detected camera and fails
instead of prompting when the choice is not unique, because a launch file must
not wait on stdin.
"""

from pathlib import Path
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from uvdar_calibrator.apps import camera_wizard as wizard

PACKAGE = "uvdar_calibrator"
DEFAULT_ROOT = Path.home() / ".ros" / "uvdar_calibrator"


def _default_config_dir() -> str:
    return str(DEFAULT_ROOT / "config")


def _default_output_dir() -> str:
    return str(DEFAULT_ROOT / "output")


def _libcamera_driver(prepared, suffix: str):
    launch_file = os.path.join(
        get_package_share_directory("libcamera_ros_driver"),
        "launch",
        "camera.launch.py",
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            "uav_name": "uvdar_calibrator",
            "suffix": suffix,
            "custom_config": str(prepared.driver_config),
            "calib_url": Path(prepared.calib_config).as_uri(),
        }.items(),
    )


def _basler_driver(prepared):
    launch_file = os.path.join(
        get_package_share_directory("pylon_ros2_camera_wrapper"),
        "launch",
        "pylon_ros2_camera.launch.py",
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={"config_file": str(prepared.driver_config)}.items(),
    )


def _prepare(context):
    camera = LaunchConfiguration("camera").perform(context)
    requested_device = LaunchConfiguration("device").perform(context)
    suffix = LaunchConfiguration("suffix").perform(context)
    mirror = LaunchConfiguration("mirror").perform(context).lower()
    config_dir = Path(
        LaunchConfiguration("config_dir").perform(context)
    ).expanduser().resolve()
    output_dir = Path(
        LaunchConfiguration("output_dir").perform(context)
    ).expanduser().resolve()

    detections = wizard.detect_cameras()
    if requested_device and camera == "auto":
        camera = wizard.kind_for_device(detections, requested_device) or camera
    kind, error = wizard.resolve_kind(detections, camera)
    if kind is None:
        raise RuntimeError(
            f"{error or 'no cameras detected'}\n"
            f"{wizard.format_detections(detections)}"
        )
    found, error = wizard.resolve_device(
        kind,
        detections[kind].devices,
        requested_device or None,
        allow_unlisted=bool(requested_device),
    )
    if found is None:
        raise RuntimeError(
            f"{error or f'no {kind} camera detected'}\n"
            f"{wizard.format_detections(detections)}"
        )
    if requested_device and all(
        device.identifier != requested_device for device in detections[kind].devices
    ):
        print(
            f"WARNING: '{requested_device}' was not detected; configuring it anyway.",
            file=sys.stderr,
        )

    wizard.seed_templates(
        config_dir,
        Path(get_package_share_directory(PACKAGE)) / "config",
    )
    if kind == "libcamera":
        prepared = wizard.prepare_libcamera(config_dir, found, suffix=suffix)
        driver = _libcamera_driver(prepared, suffix)
    else:
        prepared = wizard.prepare_basler(config_dir, found)
        driver = _basler_driver(prepared)

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        "--output_dir", str(output_dir),
        "--camera_id", prepared.camera_id,
    ]
    if mirror in ("1", "true", "yes", "on"):
        arguments.append("--mirror")
    calibrator = Node(
        package=PACKAGE,
        executable="cameracalibrator",
        arguments=arguments,
        remappings=[("image", prepared.topic)],
        output="screen",
    )
    return [driver, calibrator]


def generate_launch_description():
    """Entry point used by ``ros2 launch``."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera",
            default_value="auto",
            description="auto, libcamera or basler; auto fails when not unique.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="",
            description=(
                "Serial number (Basler) or device-tree path (libcamera). May name "
                "a camera the probe cannot see."
            ),
        ),
        DeclareLaunchArgument(
            "suffix",
            default_value="front",
            description="libcamera topic suffix (rpi_camera_<suffix>).",
        ),
        DeclareLaunchArgument(
            "mirror",
            default_value="false",
            description="Flip incoming frames left-right (driver delivers mirrored).",
        ),
        DeclareLaunchArgument(
            "config_dir",
            default_value=_default_config_dir(),
            description="Where the generated driver configs are written.",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=_default_output_dir(),
            description="Initial directory for calibration exports.",
        ),
        OpaqueFunction(function=_prepare),
    ])
