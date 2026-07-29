"""Check the live subscription uses sensor-data QoS.

A RELIABLE subscriber does not match a BEST_EFFORT publisher, and DDS reports
that as silence rather than an error -- the GUI just shows no frames forever.
Run: python3 test/test_live_qos.py
"""

from pathlib import Path
from queue import Queue
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy  # noqa: E402
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy  # noqa: E402

from uvdar_calibrator.apps.live_node import CalibrationSubscriberNode  # noqa: E402


def test_subscription_qos_is_best_effort():
    rclpy.init()
    try:
        node = CalibrationSubscriberNode(Queue(), Queue(), image_topic="image")
        qos = node._sub.qos_profile
        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT, qos.reliability
        assert qos.durability == DurabilityPolicy.VOLATILE, qos.durability
        node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    test_subscription_qos_is_best_effort()
    print("OK: live subscription uses BEST_EFFORT/VOLATILE sensor-data QoS")
