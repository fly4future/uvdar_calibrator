"""End-to-end live check: publisher -> ROS -> subscriber -> Calibrator.

Every other check in this directory feeds ``Calibrator`` directly, which
cannot reach the failure modes that only exist once ROS is in the loop: a QoS
mismatch (frames silently never arrive), a broken Image->ndarray conversion, or
frames that reach the queue but never become accepted samples. This drives the
real ``CalibrationSubscriberNode`` and ``FrameConsumerThread`` -- everything
``cameracalibrator`` runs except the Tkinter window.

Run: PYTHONPATH=.:$PYTHONPATH python test/test_live_endtoend.py
"""

from pathlib import Path
from queue import Queue
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402

from uvdar_calibrator.apps.live_node import (  # noqa: E402
    BufferQueue,
    CalibrationSubscriberNode,
    FrameConsumerThread,
)
from uvdar_calibrator.apps.replay_publisher import ImageReplayPublisher  # noqa: E402
from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.calibrator import Calibrator, CalibratorConfig  # noqa: E402
from uvdar_calibrator.engine.detection import find_image_files  # noqa: E402

IMAGE_DIR = Path(__file__).resolve().parents[1] / "example_images"
TIMEOUT_S = 90.0
WANT_SAMPLES = 3


def test_live_pipeline_accepts_samples(tmp_preview_dir="/tmp/uvdar_live_e2e"):
    files = find_image_files(str(IMAGE_DIR))
    assert files, f"no example images in {IMAGE_DIR}"

    board = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)
    calibrator = Calibrator(board, CalibratorConfig(
        preview_dir=tmp_preview_dir,
        save_previews_for_rejected=False,
    ))

    raw_queue = BufferQueue(maxsize=1)
    preview_queue = BufferQueue(maxsize=1)
    result_queue: Queue = Queue()

    rclpy.init()
    consumer = None
    try:
        sub = CalibrationSubscriberNode(
            raw_queue, preview_queue, image_topic="image",
        )
        pub = ImageReplayPublisher(
            files, topic="image", rate_hz=20.0, loop=True,
        )

        consumer = FrameConsumerThread(
            raw_queue, result_queue, calibrator, rate_hz=20.0,
        )
        consumer.start()

        executor = SingleThreadedExecutor()
        executor.add_node(pub)
        executor.add_node(sub)

        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline and len(calibrator.db) < WANT_SAMPLES:
            executor.spin_once(timeout_sec=0.05)

        published = pub.published
        accepted = len(calibrator.db)
        image_size = calibrator.image_size
        executor.shutdown()
        pub.destroy_node()
        sub.destroy_node()
    finally:
        # Join, don't just signal: stop() only sets an event, and the thread
        # can still be inside handle_frame. Letting it outlive rclpy.shutdown()
        # aborts the process during interpreter teardown.
        if consumer is not None:
            consumer.stop()
            consumer.join(timeout=10.0)
        try:
            rclpy.shutdown()
        except Exception:
            pass

    # A QoS mismatch shows up exactly here: the publisher happily publishes and
    # the subscriber never receives, so image_size stays None.
    assert published > 0, "replay publisher published nothing"
    assert image_size is not None, (
        f"published {published} frames but the subscriber received none "
        "(QoS mismatch, or wrong topic)"
    )
    assert accepted >= WANT_SAMPLES, (
        f"only {accepted} sample(s) accepted from {published} published frames "
        f"in {TIMEOUT_S}s; expected >= {WANT_SAMPLES}"
    )
    print(
        f"ok: live pipeline accepted {accepted} sample(s) from {published} "
        f"published frames at {image_size[0]}x{image_size[1]}"
    )


if __name__ == "__main__":
    test_live_pipeline_accepts_samples()
    print("all checks passed")
