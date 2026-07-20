"""
Live ROS 2 calibration node.

This node subscribes to a sensor_msgs/Image topic and feeds frames to the
UV-DAR / OCamCalib Tkinter GUI.

Important live-mode behavior:

- A raw preview queue updates the GUI camera view as fast as possible.
- A separate processing queue feeds Calibrator.handle_frame() at a limited rate.
- The default processing rate is 5 Hz, which means one calibration sample attempt
  every 200 ms.
- Queues are bounded, so old frames are dropped instead of building up lag.
- Incoming frames are resized to 960x600 by default, matching the desired
  calibration image size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
import sensor_msgs.msg

from .gui import launch_live_gui
from ..engine.board import LedGridBoard
from ..engine.calibrator import Calibrator


DEFAULT_RATE_HZ = 5.0
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 600


class BufferQueue(Queue):
    """
    Queue that discards the oldest element when full.

    This keeps the consumer and GUI looking at the freshest frames instead of
    working through an old backlog.
    """

    def put(self, item, *args, **kwargs):
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class SpinThread(threading.Thread):
    """Run rclpy.spin(node) in the background; Tkinter owns the main thread."""

    def __init__(self, node: Node):
        super().__init__(name="rclpy-spin", daemon=True)
        self.node = node

    def run(self):
        try:
            rclpy.spin(self.node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass
        except rclpy.exceptions.InvalidHandle:
            pass


class FrameConsumerThread(threading.Thread):
    """
    Pull raw frames off a queue and run them through the Calibrator.

    This thread is throttled by rate_hz. The GUI preview is not throttled by
    this thread; preview frames use a separate queue.
    """

    def __init__(
        self,
        raw_queue: Queue,
        result_queue: Queue,
        calibrator: Calibrator,
        rate_hz: float = DEFAULT_RATE_HZ,
    ):
        super().__init__(name="frame-consumer", daemon=True)
        self.raw_queue = raw_queue
        self.result_queue = result_queue
        self.calibrator = calibrator
        self.min_period = (1.0 / float(rate_hz)) if rate_hz > 0 else 0.0

        self.capturing = threading.Event()
        self.capturing.set()

        self.handle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_processed = 0.0
        self._seq = 0

    def stop(self):
        self._stop_event.set()

    def wait_until_idle(self):
        """Block until no handle_frame call is in flight."""
        with self.handle_lock:
            pass

    def run(self):
        while not self._stop_event.is_set():
            try:
                gray = self.raw_queue.get(timeout=0.1)
            except Empty:
                continue

            if not self.capturing.is_set():
                continue

            now = time.monotonic()
            if now - self._last_processed < self.min_period:
                continue

            self._last_processed = now
            self._seq += 1

            with self.handle_lock:
                result = self.calibrator.handle_frame(
                    gray,
                    f"live_frame_{self._seq:05d}",
                )

            self.result_queue.put((result, gray))


class CalibrationSubscriberNode(Node):
    """
    Subscribe to a sensor_msgs/Image topic and queue grayscale frames.

    preview_queue receives every freshest resized frame for display.
    raw_queue receives frames for throttled calibration processing.
    """

    def __init__(
        self,
        raw_queue: Queue,
        preview_queue: Queue,
        image_topic: str = "image",
        queue_size: int = 10,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ):
        super().__init__("uvdar_cameracalibrator")

        self.raw_queue = raw_queue
        self.preview_queue = preview_queue
        self.queue_size = int(queue_size)
        self.width = int(width)
        self.height = int(height)

        self.bridge = cv_bridge.CvBridge()
        self._sub = None
        self.subscribe(image_topic)

    def subscribe(self, image_topic: str) -> str:
        """Subscribe or resubscribe to image_topic; returns the resolved topic."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)

        self._sub = self.create_subscription(
            sensor_msgs.msg.Image,
            image_topic,
            self._on_image,
            self.queue_size,
        )

        self.get_logger().info(f"Subscribed to {self._sub.topic_name}")
        return self._sub.topic_name

    @property
    def resolved_topic(self) -> str:
        return self._sub.topic_name if self._sub is not None else "image"

    def _on_image(self, msg: sensor_msgs.msg.Image) -> None:
        try:
            gray = self.mkgray(msg)
        except Exception as exc:
            self.get_logger().warning(f"Could not convert image message: {exc}")
            return

        # GUI preview gets the freshest frame.
        self.preview_queue.put(gray)

        # Calibration processing also gets the freshest frame, but is throttled
        # later by FrameConsumerThread.
        self.raw_queue.put(gray)

    def mkgray(self, msg: sensor_msgs.msg.Image) -> np.ndarray:
        """
        Convert Image message to 8-bit grayscale and resize to the configured size.
        """

        dtype = self.bridge.encoding_to_dtype_with_channels(msg.encoding)[0]

        if dtype in ["uint16", "int16"]:
            mono16 = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            gray = np.array(np.clip(mono16, 0, 255), dtype=np.uint8)

        elif "FC1" in msg.encoding:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            _, max_val, _, _ = cv2.minMaxLoc(img)

            if max_val > 0:
                scale = 255.0 / max_val
                gray = (img * scale).astype(np.uint8)
            else:
                gray = img.astype(np.uint8)

        else:
            gray = self.bridge.imgmsg_to_cv2(msg, "mono8")

        if gray.shape[1] != self.width or gray.shape[0] != self.height:
            gray = cv2.resize(
                gray,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )

        return gray


def _split_ros_args(argv: Sequence[str]) -> Tuple[List[str], Optional[List[str]]]:
    """
    Split argv into app args and args for rclpy.init.

    Bare remap tokens like image:=/camera/image_raw are accepted.
    """

    argv = list(argv)
    ros_args: List[str] = []

    if "--ros-args" in argv:
        i = argv.index("--ros-args")
        ros_args = argv[i:]
        argv = argv[:i]

    app_args: List[str] = []
    remaps: List[str] = []

    for arg in argv:
        if ":=" in arg and not arg.startswith("-"):
            remaps.append(arg)
        else:
            app_args.append(arg)

    if remaps:
        if not ros_args:
            ros_args = ["--ros-args"]
        for remap in remaps:
            ros_args += ["-r", remap]

    return app_args, (ros_args if ros_args else None)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ros2 run uvdar_calibrator cameracalibrator",
        description=(
            "Live OCamCalib-style calibration from a sensor_msgs/Image topic. "
            "Subscribes to 'image' by default. Remap with image:=/camera/image_raw."
        ),
    )

    p.add_argument(
        "--n_sq_x",
        type=int,
        default=6,
        help="Squares along x direction. Default: 6.",
    )

    p.add_argument(
        "--n_sq_y",
        type=int,
        default=4,
        help="Squares along y direction. Default: 4.",
    )

    p.add_argument(
        "--spacing_mm",
        type=float,
        default=50.0,
        help="Grid spacing in millimeters. Default: 50.",
    )

    p.add_argument(
        "--taylor_order",
        type=int,
        default=4,
        help="Polynomial degree. Default: 4.",
    )

    p.add_argument(
        "--output_dir",
        default=".",
        help="Folder for Omni_Calib_Results, calib_results.txt, and previews.",
    )

    p.add_argument(
        "--rate_hz",
        type=float,
        default=DEFAULT_RATE_HZ,
        help=(
            "Maximum calibration-processing rate. "
            f"Default: {DEFAULT_RATE_HZ} Hz, or one processed frame every 200 ms."
        ),
    )

    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Resize incoming frames to this width. Default: {DEFAULT_WIDTH}.",
    )

    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Resize incoming frames to this height. Default: {DEFAULT_HEIGHT}.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    app_args, ros_args = _split_ros_args(argv)

    parser = _build_arg_parser()
    args = parser.parse_args(app_args)

    board = LedGridBoard(
        n_sq_x=args.n_sq_x,
        n_sq_y=args.n_sq_y,
        spacing_mm=args.spacing_mm,
    )

    calibrator = Calibrator(
        board,
        taylor_order=args.taylor_order,
        preview_dir=str(Path(args.output_dir) / "detected_marker_previews"),
        save_previews_for_rejected=False,
    )

    raw_queue: Queue = BufferQueue(maxsize=1)
    preview_queue: Queue = BufferQueue(maxsize=1)
    result_queue: Queue = BufferQueue(maxsize=1)

    rclpy.init(args=ros_args)

    node = CalibrationSubscriberNode(
        raw_queue=raw_queue,
        preview_queue=preview_queue,
        image_topic="image",
        width=args.width,
        height=args.height,
    )

    spin_thread = SpinThread(node)
    spin_thread.start()

    consumer = FrameConsumerThread(
        raw_queue=raw_queue,
        result_queue=result_queue,
        calibrator=calibrator,
        rate_hz=args.rate_hz,
    )
    consumer.start()

    try:
        launch_live_gui(
            calibrator=calibrator,
            result_queue=result_queue,
            preview_queue=preview_queue,
            consumer=consumer,
            subscribe_fn=node.subscribe,
            initial_topic=node.resolved_topic,
            output_dir=args.output_dir,
        )

    finally:
        consumer.stop()

        try:
            rclpy.shutdown()
        except Exception:
            pass

        spin_thread.join(timeout=2.0)
        consumer.join(timeout=2.0)


if __name__ == "__main__":
    main()