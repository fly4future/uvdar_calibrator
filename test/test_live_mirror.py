"""Check that the live node really flips the frames it hands on.

The engine-side checks live in ``test/test_mirror.py``; this one drives the
actual ROS node, because the flip has to happen in ``mkgray`` -- before the
frame reaches the preview queue, the detection queue and ``Sample.image``,
which all share one array.

Run: PYTHONPATH=.:$PYTHONPATH python test/test_live_mirror.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import sensor_msgs.msg  # noqa: E402

from uvdar_calibrator.apps.live_node import (  # noqa: E402
    BufferQueue,
    CalibrationSubscriberNode,
)


def _image_msg(array: np.ndarray) -> sensor_msgs.msg.Image:
    """A mono8 Image message carrying ``array``."""
    msg = sensor_msgs.msg.Image()
    msg.height, msg.width = array.shape[:2]
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = array.shape[1]
    msg.data = array.tobytes()
    return msg


def _asymmetric_frame() -> np.ndarray:
    """A frame whose left edge is bright and whose right edge is dark."""
    frame = np.zeros((4, 6), dtype=np.uint8)
    frame[:, 0] = 255
    return frame


def test_mirror_flag_flips_incoming_frames():
    """--mirror flips the frames, and the switch can be flipped at runtime."""
    rclpy.init()
    node = None
    try:
        node = CalibrationSubscriberNode(
            BufferQueue(maxsize=1), BufferQueue(maxsize=1), image_topic="image",
            mirror=True,
        )
        frame = _asymmetric_frame()

        mirrored = node.mkgray(_image_msg(frame))
        assert mirrored.shape == frame.shape, mirrored.shape
        assert mirrored[0][-1] == 255, "the frame was not mirrored"
        assert mirrored[0][0] == 0, "the frame was not mirrored"

        # The same node, with the switch turned off again.
        node.set_mirror(False)
        plain = node.mkgray(_image_msg(frame))
        assert plain[0][0] == 255, "the frame was mirrored when it should not be"
        assert plain[0][-1] == 0

        # And back on -- the switch is not one-shot.
        node.set_mirror(True)
        assert node.mkgray(_image_msg(frame))[0][-1] == 255
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    print("ok: the live node flips the incoming frames on demand")


if __name__ == "__main__":
    test_mirror_flag_flips_incoming_frames()
    print("all checks passed")
