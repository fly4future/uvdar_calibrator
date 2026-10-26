"""Checks for the live exposure control (libcamera).

``libcamera_ros_driver`` applies ``~/expose_us`` (``std_msgs/Int32``) to the
running camera, so the GUI only publishes there. These checks pin down the
pieces that must not drift: the topic is derived from the image topic, the log
slider round-trips, the bounds/value are remembered, and the card only appears
when a publisher was wired in.

Run: python3 test/test_exposure.py
"""

from pathlib import Path
import json
import os
import queue
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uvdar_calibrator.apps import exposure  # noqa: E402
from uvdar_calibrator.engine.board import LedGridBoard  # noqa: E402
from uvdar_calibrator.engine.calibrator import (  # noqa: E402
    Calibrator,
    CalibratorConfig,
)

BOARD = LedGridBoard(n_sq_x=6, n_sq_y=4, spacing_mm=50.0)


def test_exposure_topic_is_derived_from_the_image_topic():
    assert exposure.exposure_topic_for(
        "/uvdar_calibrator/rpi_camera_front/image_raw"
    ) == "/uvdar_calibrator/rpi_camera_front/expose_us"
    assert exposure.exposure_topic_for(
        "/uvdar_calibrator/rpi_camera_front/image_raw/"
    ) == "/uvdar_calibrator/rpi_camera_front/expose_us"
    # A replayed folder or a webcam has no runtime exposure control.
    assert exposure.exposure_topic_for("/image") == ""
    assert exposure.exposure_topic_for("image") == ""
    assert exposure.exposure_topic_for("") == ""
    print("ok: the exposure topic is the camera node next to image_raw")


def test_log_slider_round_trips():
    lo, hi = 10.0, 100000.0
    assert abs(exposure.slider_value(0, lo, hi) - lo) < 1e-9
    assert abs(exposure.slider_value(exposure.SLIDER_STEPS, lo, hi) - hi) < 1e-6

    for value in (10.0, 42.0, 500.0, 8000.0, 99999.0):
        position = exposure.slider_position(value, lo, hi)
        back = exposure.slider_value(position, lo, hi)
        assert abs(back - value) <= max(1e-6, value * 1e-6), (value, back)

    # A log slider must split the travel evenly across decades, not values.
    mid = exposure.slider_value(exposure.SLIDER_STEPS / 2, lo, hi)
    assert abs(mid - (lo * hi) ** 0.5) < 1e-6, mid

    assert exposure.sanitize_bounds(1000, 10) == (10.0, 1000.0)
    for bad in ((0, 10), (-5, 10), (float("nan"), 10)):
        try:
            exposure.sanitize_bounds(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad bounds accepted: {bad}")
    assert exposure.format_exposure(100000.0) == "100000"
    assert exposure.format_exposure(12.5) == "12.5"
    print("ok: the log slider round-trips and swaps/clamps its bounds")


def test_bounds_and_value_are_remembered():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ[exposure.SETTINGS_ENV] = str(Path(tmp) / "settings.json")
        try:
            assert exposure.load_settings() == {}
            assert exposure.exposure_bounds() == (
                exposure.DEFAULT_MIN_US, exposure.DEFAULT_MAX_US
            )
            assert exposure.initial_exposure() == exposure.DEFAULT_EXPOSURE_US

            exposure.remember_bounds(25.0, 50000.0)
            exposure.remember_exposure(1234.0)
            assert exposure.exposure_bounds() == (25.0, 50000.0)
            assert exposure.initial_exposure() == 1234.0

            # The remembered value is clamped to the remembered bounds.
            exposure.remember_exposure(1.0)
            assert exposure.initial_exposure() == 25.0

            Path(os.environ[exposure.SETTINGS_ENV]).write_text(
                "not json", encoding="utf-8"
            )
            assert exposure.load_settings() == {}
            assert exposure.exposure_bounds() == (
                exposure.DEFAULT_MIN_US, exposure.DEFAULT_MAX_US
            )
            assert exposure.initial_exposure() == exposure.DEFAULT_EXPOSURE_US
        finally:
            del os.environ[exposure.SETTINGS_ENV]
    print("ok: exposure bounds/value persist and a corrupt file falls back")


class _StubConsumer:
    def __init__(self):
        self.capturing = threading.Event()
        self.capturing.set()
        self.raw_queue = queue.Queue()

    def wait_until_idle(self):
        pass


def test_live_exposure_card():
    try:
        import tkinter as tk
    except Exception:
        print("ok: live exposure card (skipped: no tkinter)")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ok: live exposure card (skipped: no display -- {exc})")
        return

    from uvdar_calibrator.apps.gui import LiveCalibrationApp

    root.withdraw()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ[exposure.SETTINGS_ENV] = str(Path(tmp) / "settings.json")
        try:
            calls = []
            app = LiveCalibrationApp(
                root,
                calibrator=Calibrator(BOARD, CalibratorConfig()),
                result_queue=queue.Queue(),
                preview_queue=queue.Queue(),
                consumer=_StubConsumer(),
                subscribe_fn=lambda topic: topic,
                exposure_initial=500.0,
                request_exposure_fn=calls.append,
            )
            lo, hi = app._exposure_bounds
            assert lo <= 500.0 <= hi, (lo, hi)
            assert app.exposure_value.get() == "500"
            assert app._exposure_applied == 500.0

            # Dragging only updates the readout; letting go publishes once.
            app.exposure_position.set(exposure.SLIDER_STEPS)
            app._on_exposure_slider(None)
            assert calls == [], "a drag must not publish"
            assert app.exposure_value.get() == exposure.format_exposure(hi)
            app._on_exposure_release()
            assert len(calls) == 1 and abs(calls[0] - hi) <= hi * 1e-9, calls
            assert app._exposure_applied == calls[0]
            saved = json.loads(
                Path(os.environ[exposure.SETTINGS_ENV]).read_text(encoding="utf-8")
            )
            assert saved[exposure.SETTINGS_VALUE_KEY] == calls[0], saved
            assert "Exposure set to" in app.exposure_status.cget("text")

            # Letting go again without moving must not publish the same value.
            app._on_exposure_release()
            assert len(calls) == 1, calls

            # Bounds are sanitised (swapped here) and remembered.
            app.exposure_min.set("80000")
            app.exposure_max.set("100")
            app._on_exposure_bounds()
            assert app._exposure_bounds == (100.0, 80000.0)
            saved = json.loads(
                Path(os.environ[exposure.SETTINGS_ENV]).read_text(encoding="utf-8")
            )
            assert saved[exposure.SETTINGS_MIN_KEY] == 100.0, saved
            assert saved[exposure.SETTINGS_MAX_KEY] == 80000.0, saved

            # Typing a value outside the bounds clamps instead of sending it.
            app.exposure_value.set("5")
            app._on_exposure_entry()
            assert calls[-1] == 100.0, calls[-1]

            # The remembered value is sent once the first frame proves the
            # camera is up; repeated polls must not resend it.
            synced = LiveCalibrationApp(
                root,
                calibrator=Calibrator(BOARD, CalibratorConfig()),
                result_queue=queue.Queue(),
                preview_queue=queue.Queue(),
                consumer=_StubConsumer(),
                subscribe_fn=lambda topic: topic,
                exposure_initial=500.0,
                request_exposure_fn=calls.append,
            )
            assert synced._exposure_synced is False
            before = len(calls)
            synced._sync_exposure_once()
            synced._sync_exposure_once()
            assert calls[before:] == [500.0], calls[before:]
            assert "sent to the camera" in synced.exposure_status.cget("text")

            plain = LiveCalibrationApp(
                root,
                calibrator=Calibrator(BOARD, CalibratorConfig()),
                result_queue=queue.Queue(),
                preview_queue=queue.Queue(),
                consumer=_StubConsumer(),
                subscribe_fn=lambda topic: topic,
            )
            assert not hasattr(plain, "exposure_scale"), (
                "the card must only appear when an exposure topic is wired in"
            )
        finally:
            root.destroy()
            del os.environ[exposure.SETTINGS_ENV]
    print("ok: live exposure card publishes on release and remembers the value")


if __name__ == "__main__":
    test_exposure_topic_is_derived_from_the_image_topic()
    test_log_slider_round_trips()
    test_bounds_and_value_are_remembered()
    test_live_exposure_card()
    print("all checks passed")
