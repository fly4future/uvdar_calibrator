"""
Replay a folder of calibration photos onto a ROS 2 image topic.

Exists so live mode can be exercised end to end without a UV camera on the
bench. The live path has failure modes the offline path structurally cannot
have -- QoS mismatch, queue starvation, encoding conversion, GUI/spin thread
interaction -- and none of them are reachable by feeding ``Calibrator``
directly, which is what every other check in ``test/`` does.

Publishes with sensor-data QoS (BEST_EFFORT), deliberately: that is what real
camera drivers use, and it is the case a RELIABLE subscriber silently fails to
match. Replaying over RELIABLE would let a regression in
``CalibrationSubscriberNode``'s QoS pass unnoticed.

Run against the bundled photos::

    ros2 run uvdar_calibrator replay_images --image_dir example_images
    # ... in another shell:
    ros2 run uvdar_calibrator cameracalibrator image:=/image

Or without a ROS install, from the package root::

    python -m uvdar_calibrator.apps.replay_publisher --image_dir example_images
"""

import argparse
import sys
from typing import List, Optional

import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg

from ..engine.detection import find_image_files

DEFAULT_RATE_HZ = 5.0


class ImageReplayPublisher(Node):
    """Publish images from a directory onto an Image topic, one per tick."""

    def __init__(
        self,
        files: List[str],
        topic: str = "image",
        rate_hz: float = DEFAULT_RATE_HZ,
        loop: bool = True,
        frame_id: str = "camera",
    ):
        """Publish ``files`` on ``topic`` at ``rate_hz``, restarting if ``loop``."""
        super().__init__("uvdar_replay_publisher")
        self.files = files
        self.loop = loop
        self.frame_id = frame_id
        self.bridge = cv_bridge.CvBridge()
        self.index = 0
        self.published = 0

        self._pub = self.create_publisher(
            sensor_msgs.msg.Image, topic, qos_profile_sensor_data,
        )
        self._timer = self.create_timer(1.0 / float(rate_hz), self._tick)
        self.get_logger().info(
            f"Replaying {len(files)} image(s) on {self._pub.topic_name} "
            f"at {rate_hz} Hz (loop={loop})"
        )

    @property
    def topic(self) -> str:
        """Fully resolved name of the topic being published on."""
        return self._pub.topic_name

    def _tick(self) -> None:
        if self.index >= len(self.files):
            if not self.loop:
                self.get_logger().info(
                    f"Replayed {self.published} image(s); done."
                )
                self._timer.cancel()
                return
            self.index = 0

        path = self.files[self.index]
        self.index += 1

        # Grayscale to match what the calibrator works in, and to exercise the
        # mono8 branch of the node's mkgray.
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().warning(f"Could not read {path}; skipping")
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self._pub.publish(msg)
        self.published += 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay calibration photos onto a sensor_msgs/Image topic so live "
            "mode can be tested without a camera."
        ),
    )
    p.add_argument("--image_dir", required=True,
                   help="Directory of images to replay.")
    p.add_argument("--base_name", default="",
                   help="Only replay files starting with this prefix.")
    p.add_argument("--extension", default="all",
                   help="Image extension. Use 'all' for jpg/jpeg/bmp/png/tif/tiff.")
    p.add_argument("--topic", default="image",
                   help="Topic to publish on (default: image).")
    p.add_argument("--rate_hz", type=float, default=DEFAULT_RATE_HZ,
                   help=f"Publish rate (default: {DEFAULT_RATE_HZ}).")
    p.add_argument("--frame_id", default="camera",
                   help="frame_id stamped on each message.")
    p.add_argument("--once", action="store_true",
                   help="Publish each image once and stop, instead of looping.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``replay_images`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Same split as live_node: strip ROS's own args (and remappings) before
    # argparse sees them, so `image:=/foo` and `--ros-args` do not error out.
    ros_args = [a for a in argv if a == "--ros-args" or ":=" in a]
    own_args = [a for a in argv if a not in ros_args]

    args = _build_arg_parser().parse_args(own_args)

    files = find_image_files(args.image_dir, args.base_name, args.extension)
    if not files:
        print(
            f"No images found in {args.image_dir!r} "
            f"(base_name={args.base_name!r}, extension={args.extension!r})",
            file=sys.stderr,
        )
        return 1

    rclpy.init(args=ros_args)
    node = ImageReplayPublisher(
        files,
        topic=args.topic,
        rate_hz=args.rate_hz,
        loop=not args.once,
        frame_id=args.frame_id,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
