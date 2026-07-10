"""
Live ROS 2 calibration node.

Mirrors the shape of ROS image_pipeline's ``CalibrationNode`` /
``cameracalibrator`` entry point, simplified for monocular-only use and
adapted to hand work off to the Tkinter GUI instead of a cv2.imshow loop:

- ``CalibrationSubscriberNode`` subscribes to a ``sensor_msgs/Image``
  topic (default name ``image``; remap with ``image:=/camera/image_raw``)
  and pushes grayscale frames onto a :class:`BufferQueue`.
- ``FrameConsumerThread`` pulls frames off that queue -- throttled to at
  most ``rate_hz`` calls per second, since consecutive live frames are
  near-duplicates that would pay full detection cost only to be rejected
  by ``is_good_sample`` -- and feeds them to
  :meth:`~uvdar_calibrator.calibrator.Calibrator.handle_frame`, pushing
  each result onto a second queue for the Tkinter app to drain.
- ``rclpy.spin`` runs on a background :class:`SpinThread`; Tkinter owns
  the main thread.

Run it the same way as upstream's cameracalibrator::

    ros2 run uvdar_calibrator cameracalibrator --n_sq_x 6 --n_sq_y 4
        --spacing_mm 50 image:=/camera/image_raw

There is deliberately no ``SetCameraInfo`` upload here:
``sensor_msgs/CameraInfo`` distortion models have no slot for the
OCamCalib polynomial, so results are persisted locally via SAVE/EXPORT
(``Omni_Calib_Results.npz`` + ``calib_results.txt``) instead.
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

from .board import LedGridBoard
from .calibrator import Calibrator
from .gui import launch_live_gui

DEFAULT_RATE_HZ = 2.0


class BufferQueue(Queue):
    """
    Queue that discards the oldest element when full.

    Same behavior as upstream camera_calibration's ``BufferQueue``: the
    consumer always sees the freshest frames instead of an ever-growing
    backlog.
    """

    def put(self, item, *args, **kwargs):
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class SpinThread(threading.Thread):
    """Run ``rclpy.spin(node)`` in the background; Tkinter owns the main thread."""

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

    Processing is throttled to at most ``rate_hz`` ``handle_frame`` calls
    per second (frames arriving faster are dropped before paying any
    detection cost). Each processed frame's ``FrameResult`` is pushed,
    together with the frame itself, onto ``result_queue`` for the Tkinter
    side to drain via ``root.after`` polling.
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

        #: Cleared to pause capture (frames are still received but dropped).
        self.capturing = threading.Event()
        self.capturing.set()
        #: Held while handle_frame runs; lets the GUI wait out an in-flight
        #: frame before starting the solver (see wait_until_idle).
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
                continue  # throttle: drop frames arriving faster than rate_hz
            self._last_processed = now

            self._seq += 1
            with self.handle_lock:
                result = self.calibrator.handle_frame(gray, f"live_frame_{self._seq:05d}")
            self.result_queue.put((result, gray))


class CalibrationSubscriberNode(Node):
    """Subscribe to a sensor_msgs/Image topic and queue grayscale frames."""

    def __init__(self, raw_queue: Queue, image_topic: str = "image", queue_size: int = 10):
        super().__init__("uvdar_cameracalibrator")
        self.raw_queue = raw_queue
        self.queue_size = int(queue_size)
        self.bridge = cv_bridge.CvBridge()
        self._sub = None
        self.subscribe(image_topic)

    def subscribe(self, image_topic: str) -> str:
        """(Re)subscribe to ``image_topic``; returns the resolved topic name."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)
        self._sub = self.create_subscription(
            sensor_msgs.msg.Image, image_topic, self._on_image, self.queue_size
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
        self.raw_queue.put(gray)

    def mkgray(self, msg: sensor_msgs.msg.Image) -> np.ndarray:
        """
        Convert an Image message into an 8-bit 1-channel monochrome image.

        Ported from upstream camera_calibration's ``Calibrator.mkgray`` --
        handles 16-bit and floating-point encodings robustly, not just mono8.
        """
        # as cv_bridge automatically scales, we need to remove that behavior
        if self.bridge.encoding_to_dtype_with_channels(msg.encoding)[0] in ["uint16", "int16"]:
            mono16 = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            mono8 = np.array(np.clip(mono16, 0, 255), dtype=np.uint8)
            return mono8
        elif "FC1" in msg.encoding:
            # floating point image handling
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            _, max_val, _, _ = cv2.minMaxLoc(img)
            if max_val > 0:
                scale = 255.0 / max_val
                mono_img = (img * scale).astype(np.uint8)
            else:
                mono_img = img.astype(np.uint8)
            return mono_img
        else:
            return self.bridge.imgmsg_to_cv2(msg, "mono8")


def _split_ros_args(argv: Sequence[str]) -> Tuple[List[str], Optional[List[str]]]:
    """
    Split argv into (app args, args for rclpy.init).

    Everything from ``--ros-args`` on goes to rclpy, and bare ``name:=value``
    remap tokens are accepted anywhere (upstream cameracalibrator style, so
    ``image:=/camera/image_raw`` works without an explicit ``--ros-args -r``).
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
            "Subscribes to 'image' (remap with image:=/camera/image_raw) and feeds "
            "frames into the same Calibrator engine as the offline batch tool."
        ),
    )
    p.add_argument("--n_sq_x", type=int, default=6,
                   help="Squares along x direction. Default: 6.")
    p.add_argument("--n_sq_y", type=int, default=4,
                   help="Squares along y direction. Default: 4.")
    p.add_argument("--spacing_mm", type=float, default=50.0,
                   help="Grid spacing in millimeters. Default: 50.")
    p.add_argument("--taylor_order", type=int, default=4,
                   help="Polynomial degree. Default: 4.")
    p.add_argument("--output_dir", default=".",
                   help="Folder for Omni_Calib_Results, calib_results.txt and previews.")
    p.add_argument("--rate_hz", type=float, default=DEFAULT_RATE_HZ,
                   help=(
                       "Maximum frame-processing rate. Frames arriving faster are "
                       "dropped before detection; consecutive live frames are "
                       f"near-duplicates anyway. Default: {DEFAULT_RATE_HZ}."
                   ))
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    app_args, ros_args = _split_ros_args(argv)

    parser = _build_arg_parser()
    args = parser.parse_args(app_args)

    board = LedGridBoard(
        n_sq_x=args.n_sq_x, n_sq_y=args.n_sq_y, spacing_mm=args.spacing_mm
    )
    calibrator = Calibrator(
        board,
        taylor_order=args.taylor_order,
        preview_dir=str(Path(args.output_dir) / "detected_marker_previews"),
        # A live stream produces an endless supply of detected-but-rejected
        # near-duplicates; only write preview files for accepted samples.
        save_previews_for_rejected=False,
    )

    raw_queue: Queue = BufferQueue(maxsize=1)
    result_queue: Queue = Queue()

    rclpy.init(args=ros_args)
    node = CalibrationSubscriberNode(raw_queue, image_topic="image")

    spin_thread = SpinThread(node)
    spin_thread.start()
    consumer = FrameConsumerThread(
        raw_queue, result_queue, calibrator, rate_hz=args.rate_hz
    )
    consumer.start()

    try:
        launch_live_gui(
            calibrator,
            result_queue,
            consumer,
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
