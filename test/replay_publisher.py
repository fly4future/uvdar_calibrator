"""
Test helper: replay calibration photos onto a ROS 2 image topic.

Exists so live mode can be exercised end to end without a UV camera on the
bench. The live path has failure modes -- QoS mismatch, queue starvation,
encoding conversion, GUI/spin thread interaction -- that none of the checks
feeding ``Calibrator`` directly can reach.

Publishes with sensor-data QoS (BEST_EFFORT), deliberately: that is what real
camera drivers use, and it is the case a RELIABLE subscriber silently fails to
match. Replaying over RELIABLE would let a regression in
``CalibrationSubscriberNode``'s QoS pass unnoticed.
"""

from typing import List

import cv2
import cv_bridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg

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
