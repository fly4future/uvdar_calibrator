"""
Tkinter live calibration GUI.

Feeds frames from the ROS 2 subscriber in :mod:`uvdar_calibrator.apps.live_node`
into :class:`~uvdar_calibrator.engine.calibrator.Calibrator` and shows:

- a live log of per-frame accept/reject decisions,
- four ROS-style range progress bars (X, Y, Size, Skew), each drawn as a
  track with a colored segment from the min to the max accepted parameter
  value (green when that axis has enough variation),
- the supplementary bin-based "next images to capture" hints, driven off
  the accepted samples,
- CALIBRATE gated by ``calibrator.goodenough`` (with confirm-to-override)
  and SAVE/EXPORT gated by ``calibrator.calibrated``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import math
import queue
import re
import time
from typing import Optional, Tuple

import numpy as np

from ..engine import coverage
from ..engine import ocam_model
from ..engine.board import LedGridBoard
from ..engine.calibrator import Calibrator, CalibratorConfig

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
except Exception:  # pragma: no cover - headless environments
    tk = None
    filedialog = None
    tkfont = None
    messagebox = None
    ttk = None

#: Keep at most this many lines in the sample log (relevant for live mode,
#: where rejected-frame lines keep arriving for as long as capture runs).
MAX_LOG_LINES = 1000


def _sanitize_filename_part(text: str) -> str:
    """Make free-form text (a ROS topic, a folder name) safe for a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("._")


def camera_id_from_config(identifier: str) -> str:
    """
    Derive a filename-safe camera identifier from a driver config value.

    """
    return _sanitize_filename_part((identifier or "").strip())


def _require_gui_deps() -> None:
    if tk is None:
        raise RuntimeError("Tkinter is required for the GUI. Install python-tk/tkinter.")
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")


def guide_box_rect(
    guide: dict,
    image_size: Tuple[int, int],
    valid_region: Optional[Tuple[float, float, float]] = None,
) -> Tuple[int, int, int, int, int, int]:
    """
    Pixel geometry of the on-image "put the grid here" box.

    Returns ``(left, top, right, bottom, cx, cy)``: the (clipped) rectangle
    to stroke and the crosshair centre to draw on it.

    ``guide`` holds normalized coordinates in the same frame of reference as
    ``coverage.get_parameters`` -- the camera's usable region (the full
    frame, or the fisheye's valid circle when it has one). Scaling them by
    the raw image size would put the box on a different spot than the number
    means, so the mapping is undone here, once, for every caller.

    Kept as a module-level function (no Tk state) so it can be checked
    without opening a window.
    """
    width, height = image_size
    px, py = coverage.normalized_to_pixel(
        guide["x"], guide["y"], image_size, valid_region,
    )
    cx, cy = int(round(px)), int(round(py))

    scale = coverage.normalized_length_scale(image_size, valid_region)
    box = int(round(guide["size"] * scale * 2.2))
    box = max(60, min(box, int(0.85 * min(width, height))))

    left = max(0, cx - box // 2)
    top = max(0, cy - box // 2)
    right = min(width - 1, cx + box // 2)
    bottom = min(height - 1, cy + box // 2)
    return left, top, right, bottom, cx, cy


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
#
# One palette and one font family for every widget and every canvas drawing.
# Previously the font name was hardcoded as "Segoe UI" in a dozen places,
# which silently fell back to whatever Tk defaults to on the Ubuntu images
# the UAVs run -- so the app looked different on the drone than on the
# laptop. ``apply_theme`` resolves the family once, from what is actually
# installed, and hands it to everyone.

#: Font families in preference order. The first *installed* one wins; Tk
#: falls back silently for a family that isn't there, so probing is the only
#: way to get a deliberate choice instead of an accidental one.
UI_FONT_CANDIDATES = (
    "Segoe UI",
    "Cantarell",
    "Ubuntu",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Helvetica",
)


@dataclass(frozen=True)
class Theme:
    """Colors plus the resolved font family, shared by widgets and drawings."""

    family: str = "TkDefaultFont"
    bg: str = "#eef0f3"          # window / panel background
    surface: str = "#ffffff"     # cards: canvases, text boxes, entry fields
    border: str = "#cfd4da"
    text: str = "#1d2125"
    muted: str = "#6a7178"
    good: str = "#2e9e4f"
    bad: str = "#d9534f"
    image_bg: str = "#202020"    # backdrop behind the camera image

    def font(self, size: int = 9, weight: str = ""):
        """Font tuple for canvas items, e.g. ``theme.font(10, "bold")``."""
        return (self.family, size, weight) if weight else (self.family, size)


def apply_theme(root) -> Theme:
    """
    Install the app-wide ttk style and return the resolved :class:`Theme`.

    "clam" is requested because it is the one built-in theme that honors
    ``background``/``fieldbackground``, so the flat palette below actually
    shows up; the native themes on Windows/macOS would ignore most of it.
    """
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    installed = set(tkfont.families(root))
    family = next((f for f in UI_FONT_CANDIDATES if f in installed), "TkDefaultFont")
    theme = Theme(family=family)

    style.configure(".", font=(family, 9), background=theme.bg, foreground=theme.text)
    style.configure("TFrame", background=theme.bg)
    style.configure("TLabel", background=theme.bg, foreground=theme.text)
    style.configure("TCheckbutton", background=theme.bg, foreground=theme.text)
    style.configure("TButton", padding=(10, 5), background=theme.surface)
    style.configure("TEntry", fieldbackground=theme.surface, padding=3)
    style.configure("TSpinbox", fieldbackground=theme.surface, padding=3)

    return theme


class LiveCalibrationApp:
    """
    Live-topic calibration app driven by the ROS 2 subscriber in ``live_node``.

    The top row holds an image-topic field and a Start/Stop capture toggle.
    Two queues are drained independently: ``preview_queue`` (unthrottled --
    always the freshest camera frame, for a responsive live view) and
    ``result_queue`` (throttled by ``--rate_hz`` -- processed
    ``FrameResult``s, for the sample log/accept-reject/progress panel), so
    the camera view stays live regardless of detection cost.
    Previous/Next browse the accepted samples.
    """

    POLL_MS = 50

    def __init__(
        self,
        root,
        calibrator: Calibrator,
        result_queue,
        preview_queue,
        consumer,
        subscribe_fn,
        initial_topic: str = "image",
        camera_id: str = "",
        mirror: bool = False,
        set_mirror_fn=None,
        output_dir: str = ".",
        slow_find_center: bool = False,
        dev_mode: bool = False,
    ):
        board = calibrator.board
        config = CalibratorConfig.from_calibrator(calibrator)

        self.result_queue = result_queue
        self.preview_queue = preview_queue
        self.consumer = consumer
        self.subscribe_fn = subscribe_fn
        self.topic = tk.StringVar(value=initial_topic)
        self._subscribed_topic = initial_topic
        # Identifier from the driver config (wizard's --camera_id); used for
        # the suggested SAVE filename.
        self.camera_id = camera_id or ""
        self.n_rejected = 0
        self.n_failed = 0
        self._latest_preview = None
        self._latest_corners = None
        self._latest_outcome = "waiting for first processed frame"
        self._last_frame_t = time.monotonic()
        self._fps = 0.0
        self._stale_warned = False

        self.root = root
        self.root.title("UV-DAR / OCamCalib Calibration Assistant")
        self.theme = apply_theme(self.root)

        # Fit the window to the screen's work area instead of forcing a fixed
        # 1180x980 that overran smaller displays (pushing the CALIBRATE/SAVE
        # buttons off the bottom). Bounded by the design size on large screens,
        # by the available screen on small ones.
        screen_w = max(100, root.winfo_screenwidth())
        screen_h = max(100, root.winfo_screenheight())
        design_w, design_h = 1180, 980
        w = min(design_w, screen_w - 40)
        h = min(design_h, screen_h - 60)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(860, min(600, h))

        self.n_sq_x = tk.IntVar(value=board.n_sq_x)
        self.n_sq_y = tk.IntVar(value=board.n_sq_y)
        self.spacing_mm = tk.DoubleVar(value=board.spacing_mm)
        self.taylor_order = tk.IntVar(value=config.taylor_order)
        self.output_dir = tk.StringVar(value=output_dir)
        self.slow_find_center = tk.BooleanVar(value=slow_find_center)
        # Launch-time settings of the already-built Calibrator, shown
        # read-only in the "Advanced Settings..." dialog.
        self.calib_config = config
        self.show_plots = tk.BooleanVar(value=False)
        self.forward_view_var = tk.BooleanVar(value=False)
        # Flip the incoming frames left-right. Seeded from --mirror and
        # switchable at runtime in the Advanced Settings dialog; the live node
        # owns the actual flip.
        self.mirror = tk.BooleanVar(value=bool(mirror))
        self.set_mirror_fn = set_mirror_fn
        # Rectilinear FOV. Wider shows more of the frame but adds black corners:
        # a rectilinear projection breaks down past ~90 deg off-axis.
        self.forward_view_hfov = tk.DoubleVar(
            value=ocam_model.DEFAULT_FORWARD_VIEW_HFOV_DEG
        )

        # Dev mode hides everything that is diagnostic rather than actionable:
        # the bars keep whatever they were drawing and the log keeps its history.
        self.dev_mode = bool(dev_mode)
        # (parent, [(widget, pack_kwargs, dev_only), ...]) in display order.
        # Toggling re-packs each row from scratch in this order, because pack()
        # appends to a parent's stacking order.
        self._layout_rows = []

        self.calibrator = calibrator
        self.current_sample_index = 0
        # The position guide is drawn on every rendered preview frame but only
        # depends on the accepted-sample db, so recompute it when the db
        # changes rather than at frame rate.
        self._guide_cache = None
        self._guide_db_len = -1
        self.photo_ref = None
        self._forward_view_cache = None        # (map_x, map_y) or None
        self._forward_view_model_id = None     # id(last built-from model)
        # The cache is also keyed on mode+hfov, both of which can now
        # change without touching the calibrator, so a stale map must not be
        # reused
        self._forward_view_key = None

        self._build_widgets()
        self.root.bind("<Left>", lambda _e: self.prev_sample())
        self.root.bind("<Right>", lambda _e: self.next_sample())
        self.root.bind("<Control-d>", lambda _e: self._toggle_dev_mode())
        self.root.bind("<Control-D>", lambda _e: self._toggle_dev_mode())

        self._refresh_capture_button()
        self._set_status(f"Live capture running on topic '{initial_topic}'.")
        self._update_board_button()
        self.root.after(self.POLL_MS, self._poll_queue)

    # ------------------------------------------------------------------
    # Dev mode
    # ------------------------------------------------------------------

    def _register_row(self, parent, items) -> None:
        """
        Record one parent's child order, then pack it.

        ``items`` is ``[(widget, pack_kwargs, dev_only), ...]`` in the order the
        widgets should appear. Marking a widget dev-only hides it unless
        ``self.dev_mode``; see ``_layout_rows``.
        """
        self._layout_rows.append((parent, items))
        self._pack_row(parent, items)

    def _pack_row(self, parent, items) -> None:
        for widget, pack_kwargs, dev_only in items:
            widget.pack_forget()
            if dev_only and not self.dev_mode:
                continue
            widget.pack(**pack_kwargs)

    def _apply_dev_mode(self) -> None:
        for parent, items in self._layout_rows:
            self._pack_row(parent, items)
        self.dev_button.configure(
            text=("Hide dev" if self.dev_mode else "Dev"),
            style=("Dev.TButton" if self.dev_mode else "TButton"),
        )
        self._update_progress_panel()

    def _toggle_dev_mode(self) -> None:
        self.dev_mode = not self.dev_mode
        self._apply_dev_mode()
        self._set_status(
            f"Dev mode {'on' if self.dev_mode else 'off'} "
            f"(Ctrl+D toggles)."
        )

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _build_widgets(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self._build_source_controls(top)

        # What stays permanently visible: the source row above, and one button
        # that reports the LED-grid geometry the app will assume. The three
        # fields behind it (count x count, pitch) are wrong for nobody who uses
        # the default board, and the two checkboxes that used to sit here
        # (MATLAB-style slow Find Center, show plots) were pure expert knobs --
        # all of them now live in the Advanced Settings dialog.
        # Always shown: the board can be swapped while the stream is running.
        opts = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        opts.pack(side=tk.TOP, fill=tk.X)
        self.board_option_widgets = []

        self._board_panel_visible = False
        self.board_button = ttk.Button(
            opts, text="", command=self._toggle_board_panel
        )
        self.board_button.pack(side=tk.LEFT)
        ttk.Button(
            opts, text="Advanced Settings...", command=self._open_advanced_settings
        ).pack(side=tk.RIGHT)

        self.board_panel = ttk.Frame(opts, padding=(10, 0, 0, 0))
        ttk.Label(self.board_panel, text="Grid squares X:").pack(side=tk.LEFT)
        w = ttk.Spinbox(self.board_panel, from_=1, to=30, textvariable=self.n_sq_x, width=5)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Label(self.board_panel, text="Y:").pack(side=tk.LEFT)
        w = ttk.Spinbox(self.board_panel, from_=1, to=30, textvariable=self.n_sq_y, width=5)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Label(self.board_panel, text="Spacing mm:").pack(side=tk.LEFT, padx=(12, 0))
        w = ttk.Entry(self.board_panel, textvariable=self.spacing_mm, width=7)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)

        ttk.Label(
            self.board_panel, text="applies immediately",
            font=self.theme.font(8), foreground=self.theme.muted,
        ).pack(side=tk.LEFT, padx=(12, 0))

        # The button doubles as the readout of the current geometry, so a
        # collapsed panel still tells the user what the app is assuming.
        for var in (self.n_sq_x, self.n_sq_y, self.spacing_mm):
            var.trace_add("write", lambda *_a: self._update_board_button())
        self._update_board_button()

        # Commit on Enter, focus-out or a spinbox arrow -- not on every
        # keystroke of a half-typed number.
        for widget in self.board_option_widgets:
            widget.bind("<Return>", self._on_board_fields_committed)
            widget.bind("<KP_Enter>", self._on_board_fields_committed)
            widget.bind("<FocusOut>", self._on_board_fields_committed)
            if isinstance(widget, ttk.Spinbox):
                widget.configure(command=self._on_board_fields_committed)

        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # The right sidebar's fixed-height content (range bars, coverage
        # graph, log, suggestions) can exceed a short screen. It is therefore
        # wrapped in a scrollable region, while the CALIBRATE / SAVE-EXPORT
        # button row is pinned below it (outside the scroll region) so the
        # buttons are ALWAYS visible regardless of window height.
        right = ttk.Frame(main, width=340)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right.pack_propagate(False)

        self.right_scroll = tk.Canvas(
            right, highlightthickness=0, background=self.theme.bg
        )
        self.right_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.right_scrollbar = ttk.Scrollbar(
            right, orient="vertical", command=self.right_scroll.yview
        )
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_scroll.configure(yscrollcommand=self.right_scrollbar.set)
        self._right_scroll_frame = ttk.Frame(self.right_scroll)
        self._right_scroll_window = self.right_scroll.create_window(
            (0, 0), window=self._right_scroll_frame, anchor="nw"
        )
        self._right_scroll_frame.bind(
            "<Configure>",
            lambda _e: self.right_scroll.configure(
                scrollregion=self.right_scroll.bbox("all")
            ),
        )
        self.right_scroll.bind(
            "<Configure>",
            lambda e: self.right_scroll.itemconfigure(
                self._right_scroll_window, width=e.width
            ),
        )
        self.right_scroll.bind_all(
            "<MouseWheel>",
            lambda e: self._on_mousewheel(e),
            add="+",
        )

        self.image_canvas = tk.Canvas(
            left, bg=self.theme.image_bg,
            highlightthickness=1, highlightbackground=self.theme.border,
        )
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav = ttk.Frame(left, padding=(0, 8, 0, 0))
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        prev_button = ttk.Button(nav, text="Previous", command=self.prev_sample)
        next_button = ttk.Button(nav, text="Next", command=self.next_sample)
        self.delete_button = ttk.Button(
            nav, text="Delete Sample", command=self.delete_sample
        )
        self.forward_view_toggle = ttk.Checkbutton(
            nav,
            text="Forward view (undistorted)",
            variable=self.forward_view_var,
            command=self._on_forward_view_toggle,
            state="disabled",
        )
        # Spinbox: Upper bound is the measured limit: past ~144 deg the mapping
        # samples outside the sensor and the view is mostly black.
        self.forward_view_fov_box = ttk.Spinbox(
            nav,
            from_=60.0, to=140.0, increment=5.0,
            textvariable=self.forward_view_hfov,
            width=5,
            command=self._on_forward_view_toggle,
            state="disabled",
        )
        self.forward_view_fov_label = ttk.Label(nav, text="FOV")
        self.image_label = ttk.Label(nav, text="No accepted sample loaded")
        # Only the "which sample am I looking at" caption is always needed.
        # Browsing, deleting a bad sample and the forward-view check are all
        # inspection tools, so they live in dev mode.
        self._register_row(nav, [
            (prev_button, {"side": tk.LEFT}, True),
            (next_button, {"side": tk.LEFT, "padx": 4}, True),
            (self.delete_button, {"side": tk.LEFT, "padx": 4}, True),
            (self.image_label, {"side": tk.LEFT, "padx": 12}, False),
            (self.forward_view_fov_label, {"side": tk.RIGHT}, True),
            (self.forward_view_fov_box, {"side": tk.RIGHT, "padx": (0, 8)}, True),
            (self.forward_view_toggle, {"side": tk.RIGHT}, True),
        ])

        sidebar = self._right_scroll_frame

        progress_title = ttk.Label(
            sidebar, text="Calibration Progress", font=self.theme.font(14, "bold")
        )
        # Three lines instead of one "Status: NOT READY -- need more varied
        # views (8 accepted samples)" label"
        self.readiness_verdict = ttk.Label(
            sidebar, text="Waiting for frames", font=self.theme.font(13, "bold"),
            wraplength=340, justify="center",
        )
        self.readiness_action = ttk.Label(
            sidebar,
            text="Press Start Capture to begin collecting photos from the topic.",
            font=self.theme.font(10), wraplength=340, justify="center",
        )
        self.readiness_detail = ttk.Label(
            sidebar, text="", font=self.theme.font(8), foreground=self.theme.muted,
            wraplength=340, justify="center",
        )

        # Four ROS-style range bars: X, Y, Size, Skew.
        self.bar_canvas = self._card_canvas(sidebar, height=140)
        coverage_title = ttk.Label(
            sidebar, text="Board position coverage", font=self.theme.font(10, "bold")
        )
        self.coverage_canvas = self._card_canvas(sidebar, height=180)
        log_title = ttk.Label(
            sidebar, text="Sample log", font=self.theme.font(10, "bold")
        )
        self.log_box = self._text_card(sidebar, height=10)
        suggestion_title = ttk.Label(
            sidebar, text="Next image to capture", font=self.theme.font(10, "bold")
        )
        self.suggestion_box = self._text_card(sidebar, height=4)

        # The three readiness lines are the whole of the minimal sidebar.
        # (_text_card packs the card frame and returns the Text, so
        # .master is what has to be hidden.)
        self._register_row(sidebar, [
            (progress_title, {"anchor": "center"}, True),
            (self.readiness_verdict, {"anchor": "center", "pady": (8, 2)}, False),
            (self.readiness_action, {"anchor": "center"}, False),
            (self.readiness_detail, {"anchor": "center", "pady": (4, 10)}, False),
            (self.bar_canvas, {"anchor": "center", "pady": (0, 10)}, True),
            (coverage_title, {"anchor": "center"}, True),
            (self.coverage_canvas, {"anchor": "center", "pady": (4, 10)}, True),
            (log_title, {"anchor": "w"}, True),
            (self.log_box.master, {"anchor": "w", "fill": "x", "pady": (4, 10)}, True),
            (suggestion_title, {"anchor": "w"}, True),
            (self.suggestion_box.master,
             {"anchor": "w", "fill": "x", "pady": (4, 10)}, True),
        ])
        self._draw_bars([])
        self._draw_coverage_graph()

        actions = ttk.Frame(right)
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        self.calibrate_button = ttk.Button(
            actions, text="CALIBRATE", command=self.calibrate, state="disabled"
        )
        self.calibrate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.save_button = ttk.Button(
            actions, text="SAVE / EXPORT", command=self.save_and_export, state="disabled"
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # A hairline plus muted text instead of a sunken groove: the status
        # line is the app's only persistent feedback surface, so it should
        # read as a quiet strip rather than a second toolbar. The Dev toggle
        # lives here because it is the one control that must stay reachable in
        # the minimal layout without adding anything to the main window.
        footer = ttk.Frame(self.root)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(footer).pack(side=tk.TOP, fill=tk.X)
        self.bottom_status = ttk.Label(
            footer, text="", anchor="w", padding=(10, 4), foreground=self.theme.muted,
        )
        self.bottom_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.dev_button = ttk.Button(
            footer,
            text=("Hide dev" if self.dev_mode else "Dev"),
            command=self._toggle_dev_mode,
            width=6,
            style=("Dev.TButton" if self.dev_mode else "TButton"),
        )
        self.dev_button.pack(side=tk.RIGHT, padx=4, pady=2)
        ttk.Style(self.root).configure("Dev.TButton", foreground=self.theme.good)

    def _card_canvas(self, parent, height: int):
        """A white drawing surface that matches the rest of the palette."""
        return tk.Canvas(
            parent, width=320, height=height,
            background=self.theme.surface, highlightthickness=1,
            highlightbackground=self.theme.border,
        )

    def _text_card(self, parent, height: int, pady=(4, 10)):
        """
        A flat, scrollable read-only text card, already packed.

        The log and the suggestions are plain ``tk.Text``, which by default
        brings its own 3D border and a white background that clashes with the
        surrounding cards; styling them here keeps both identical and gives
        the log a scrollbar it previously lacked. The card *frame* is what
        gets packed -- returning the Text to pack instead would leave it
        inside an unmanaged frame, and it would never be displayed.
        """
        card = tk.Frame(parent, background=self.theme.border)
        card.pack(anchor="w", fill=tk.X, pady=pady)
        text = tk.Text(
            card, height=height, width=44, wrap="word",
            background=self.theme.surface, foreground=self.theme.text,
            font=self.theme.font(9), relief=tk.FLAT, borderwidth=1,
            highlightthickness=0, padx=6, pady=4, spacing1=1, spacing3=1,
            state="disabled",
        )
        scrollbar = ttk.Scrollbar(card, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return text

    def _toggle_board_panel(self):
        """Show/hide the LED-grid fields (grid size, pitch) behind one button."""
        self._board_panel_visible = not self._board_panel_visible
        if self._board_panel_visible:
            self.board_panel.pack(side=tk.LEFT, fill=tk.X, expand=True)
        else:
            self.board_panel.pack_forget()
        self._update_board_button()

    def _update_board_button(self):
        """
        Label the toggle with the geometry it represents.

        A bare "Settings" button would leave the user guessing which board the
        app is assuming; showing "7 x 5, 15 mm" means the default works for
        everybody and the button only matters to somebody whose board differs.
        """
        try:
            text = (
                f"LED grid: {self.n_sq_x.get()} x {self.n_sq_y.get()}, "
                f"{self.spacing_mm.get():g} mm spacing"
            )
        except tk.TclError:
            # Mid-edit in the spinbox ("" or "1e"): keep the old label.
            return
        # Wording rather than a triangle glyph: U+25B8/U+25BE are missing from
        # some of the candidate UI fonts, and a tofu box on the one button that
        # reveals the settings is worse than no arrow at all.
        action = "hide" if self._board_panel_visible else "change"
        self.board_button.configure(text=f"{text}  ({action})")

    def _on_mousewheel(self, event):
        # Only scroll when the pointer is over the sidebar's scroll region,
        # so the image canvas (which shows Previous/Next-bound samples) isn't
        # hijacked by the wheel.
        try:
            x = self.right_scroll.winfo_pointerx() - self.root.winfo_rootx()
            y = self.right_scroll.winfo_pointery() - self.root.winfo_rooty()
        except Exception:
            return
        if not (0 <= x <= self.right_scroll.winfo_width() and
                0 <= y <= self.right_scroll.winfo_height()):
            return
        delta = int(-1 * (event.delta / 120))
        self.right_scroll.yview_scroll(delta, "units")

    def _open_advanced_settings(self):
        """
        Dialog for the settings that are not on the main window.

        The Calibrator's launch-time configuration (taylor_order,
        fov_radius_frac, sample_threshold, param_ranges, min_db_size,
        max_accepted_samples, save_previews_for_rejected) is shown read-only:
        the running Calibrator can't be reconfigured mid-capture. The switches
        that act outside it -- mirroring, slow Find Center, plots -- stay
        editable.
        """
        cfg = self.calib_config

        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        def _fmt(value):
            return "" if value is None else str(value)

        def _value_row(row, label, value, hint):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(body, text=_fmt(value)).grid(row=row, column=1, sticky="w", padx=6)
            ttk.Label(
                body, text=hint, font=self.theme.font(8), foreground=self.theme.muted,
            ).grid(row=row, column=2, sticky="w")

        def _section(row, title):
            ttk.Label(
                body, text=title, font=self.theme.font(9, "bold"),
            ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))

        def _separator(row):
            ttk.Separator(body, orient=tk.HORIZONTAL).grid(
                row=row, column=0, columnspan=3, sticky="ew", pady=8
            )

        _section(0, "Calibration model and sample selection (set at launch)")
        rows = [
            ("Taylor order:", self.taylor_order.get(),
             "polynomial degree of the omnidirectional model; 4 is the OCamCalib default"),
            ("FOV radius fraction:", cfg.fov_radius_frac,
             "fisheye usable-circle radius, fraction of min(w,h)/2; blank = full frame"),
            ("Sample threshold:", cfg.sample_threshold,
             "min distance for a new sample to be accepted, in target spans"),
            ("Param ranges:",
             "  ".join(f"{n}: {v}" for n, v in zip(coverage.PARAM_NAMES, cfg.param_ranges)),
             "per-axis span the samples must cover"),
            ("Min DB size:", cfg.min_db_size,
             "accepted-sample count required for readiness"),
            ("Max accepted samples:", cfg.max_accepted_samples,
             "hard cap on retained samples; blank = unbounded"),
            ("Previews for rejected:", "yes" if cfg.save_previews_for_rejected else "no",
             "also save previews of rejected frames"),
        ]
        row = 1
        for label, value, hint in rows:
            _value_row(row, label, value, hint)
            row += 1

        _separator(row)
        row += 1
        _section(row, "Solve")
        row += 1
        ttk.Checkbutton(
            body, text="MATLAB-style slow Find Center (slower, more accurate)",
            variable=self.slow_find_center,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Checkbutton(
            body, text="Show diagnostics plots after calibration",
            variable=self.show_plots,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        _separator(row)
        row += 1
        _section(row, "Incoming frames")
        row += 1
        ttk.Checkbutton(
            body,
            text="Mirror incoming frames (flip left-right)",
            variable=self.mirror,
            command=self._on_mirror_toggled,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Label(
            body,
            text=(
                "for cameras whose driver delivers a mirrored image; the "
                "preview, the detection and the saved model all use the "
                "mirrored view"
            ),
            font=self.theme.font(8), foreground=self.theme.muted, wraplength=360,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        ttk.Button(body, text="Close", command=dialog.destroy).grid(
            row=row, column=0, columnspan=3, sticky="e", pady=(12, 0)
        )

        dialog.grab_set()
        dialog.wait_window()

    # ------------------------------------------------------------------
    # Mirroring the incoming frames
    # ------------------------------------------------------------------

    def _apply_mirror(self, mirrored: bool) -> None:
        """Hand the new orientation to whoever produces the frames."""
        if self.set_mirror_fn is not None:
            self.set_mirror_fn(mirrored)

    def _discard_samples(self, reason: str, detail: str = "") -> None:
        """Drop every accepted sample after the view or the target changed."""
        cal = self.calibrator
        if cal is None:
            return

        n_discarded = len(cal.db)
        cal.db.clear()
        self._invalidate_calibration()
        if n_discarded:
            line = f"{reason}: dropped {n_discarded} sample(s)"
            if detail:
                line = f"{line} {detail}"
            self._append_log(line)
        self._update_progress_panel()

    def _discard_samples_for_orientation_change(self, mirrored: bool) -> None:
        """Drop the accepted samples: they describe the other orientation."""
        self._discard_samples(
            f"mirror {'on' if mirrored else 'off'}",
            "captured in the other orientation",
        )

    def _invalidate_calibration(self, reset_view: bool = True) -> None:
        """Make any solved model stale without touching the accepted samples."""
        cal = self.calibrator
        if cal is None:
            return
        cal.calibrated = False
        if reset_view:
            self.current_sample_index = 0
        # The guide is cached per db length, and a discarded db can leave the
        # length unchanged (delete one, add one), so the cache is dropped
        # explicitly.
        self._guide_db_len = -1
        self._forward_view_cache = None
        self._forward_view_model_id = None
        self._forward_view_key = None
        self.save_button.configure(state="disabled")
        self.forward_view_var.set(False)
        self._sync_forward_view_controls()

    def _set_status(self, text):
        # No update_idletasks() here: these are called from inside after()
        # callbacks, and pumping the event loop there can reenter the live
        # queue poll. Callers that block the loop (image load, the solve) ask
        # for a repaint explicitly.
        self.bottom_status.configure(text=text)

    def _append_log(self, line):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        n_lines = int(self.log_box.index("end-1c").split(".")[0])
        if n_lines > MAX_LOG_LINES:
            self.log_box.delete("1.0", f"{n_lines - MAX_LOG_LINES + 1}.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _write_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # ------------------------------------------------------------------
    # Progress panel
    # ------------------------------------------------------------------

    def _plain_readiness(self, goodenough, progress, n, min_db_size, guide):
        """
        The (verdict, action, detail) triple shown above the range bars.

        Three jobs that used to share one "Status: ..." string, split so each
        can do its own job: answers the question - can i calibrate?
        """
        if n == 0:
            # Must stand on its own: the sample log that would explain a
            # rejection is hidden in the minimal layout, so point at the cause
            # rather than at a panel that may not be on screen.
            return (
                "No photos accepted yet",
                "Check the LED grid is in view, well lit, and fills enough of "
                "the frame -- then capture again.",
                "",
            )

        axes_short = [name for name, _lo, _hi, p in progress if p < 1.0]
        count_short = n < min_db_size

        if goodenough:
            return (
                "Ready to calibrate",
                "Press CALIBRATE.",
                f"{n} photos accepted, all four views covered",
            )

        # One action, chosen by what is actually blocking: a thin database is
        # fixed by taking more photos, a coverage gap by moving the board. When
        # both apply, the position advice is the more specific of the two, so
        # it leads and the count is appended.
        if axes_short and guide is not None:
            action = guide["text"] + "."
            if count_short:
                action += f" Keep going -- {n} of {min_db_size} photos so far."
        elif count_short:
            action = f"Add more photos -- {n} of {min_db_size} so far."
        elif axes_short:
            names = ", ".join(axes_short)
            action = f"Vary the view: the {names} range is still too narrow."
        else:
            action = "Add more photos."

        detail = f"{n} of {min_db_size} photos" if count_short else f"{n} photos"
        for name, _lo, _hi, p in progress:
            if p < 1.0:
                detail += f" - {name}: {100.0 * p:.0f}% coverage"
            else:
                detail += f" - {name}: full coverage"

        return ("Not ready yet", action, detail)

    def _update_progress_panel(self):
        cal = self.calibrator
        if cal is None:
            return

        goodenough, progress = coverage.compute_goodenough(
            cal.db_params(), cal.param_ranges, cal.min_db_size
        )
        self._draw_bars(progress)
        self._draw_coverage_graph()

        n = len(cal.db)
        guide = self._coverage_overlay_guide()
        verdict, action, detail = self._plain_readiness(
            goodenough, progress, n, cal.min_db_size, guide,
        )
        self.readiness_verdict.configure(
            text=verdict, foreground=(self.theme.good if goodenough else self.theme.text),
        )
        self.readiness_action.configure(text=action)
        self.readiness_detail.configure(text=detail)

        # Gate CALIBRATE on goodenough; allow an explicit override once
        # samples exist (confirmation dialog in calibrate()).
        self.calibrate_button.configure(state=("normal" if n > 0 else "disabled"))

        metrics = cal.db_metrics()
        if metrics:
            suggestions = coverage.coverage_suggestions(coverage.compute_bin_coverage(metrics))
            if not suggestions:
                suggestions = ["All coverage bins are represented."]
        else:
            suggestions = ["No accepted samples yet."]

        # The position guide used to be the first bullet here, but it is now the
        # headline action above
        self._write_text(self.suggestion_box, "\n".join(f"• {s}" for s in suggestions))

    def _progress_color(self, p: float) -> str:
        """Return the bar color: red -> yellow by progress, green when complete."""
        if p >= 1.0:
            return self.theme.good
        # interpolate red (low) to yellow (high)
        r1, g1, b1 = (0xd9, 0x53, 0x4f)
        r2, g2, b2 = (0xe0, 0xb6, 0x42)
        t = max(0.0, min(1.0, p))
        return "#%02x%02x%02x" % (
            round(r1 + (r2 - r1) * t),
            round(g1 + (g2 - g1) * t),
            round(b1 + (b2 - b1) * t),
        )

    def _draw_bars(self, progress):
        """Draw the four range bars (Size/Skew fill by progress to target)."""
        c = self.bar_canvas
        c.delete("all")

        x0, x1 = 60, 280
        y = 20

        rows = progress if progress else [(name, 0.0, 0.0, 0.0) for name in coverage.PARAM_NAMES]

        for name, lo, hi, p in rows:
            c.create_text(10, y, text=name, anchor="w", font=self.theme.font(9, "bold"))
            c.create_rectangle(
                x0, y - 8, x1, y + 8, outline=self.theme.border, fill="#f0f1f3"
            )
            lo_c = max(0.0, min(1.0, lo))
            hi_c = max(0.0, min(1.0, hi))
            # X/Y show the span the samples cover; Size/Skew are scored
            # against a target maximum, so fill them by progress instead.
            if str(name).strip().lower() in ("size", "skew"):
                if p > 0:
                    c.create_rectangle(
                        x0, y - 8, x0 + (x1 - x0) * min(p, 1.0), y + 8,
                        outline="", fill=self._progress_color(p),
                    )
            elif hi_c > lo_c:
                c.create_rectangle(
                    x0 + (x1 - x0) * lo_c,
                    y - 8,
                    x0 + (x1 - x0) * hi_c,
                    y + 8,
                    outline="",
                    fill=self._progress_color(p),
                )
            c.create_text(
                x1 + 10, y, text=f"{100.0 * p:.0f}% covered", anchor="w",
                font=self.theme.font(8), fill=self.theme.text,
            )
            y += 32

    def _skew_color(self, skew: float) -> str:
        """Interpolate a dot's fill color by skew: light blue (0) -> dark orange (1)."""
        s = max(0.0, min(1.0, skew))
        r1, g1, b1 = (0x9e, 0xca, 0xe1)  # light blue, low skew/tilt
        r2, g2, b2 = (0xe6, 0x55, 0x0d)  # dark orange, high skew/tilt
        return "#%02x%02x%02x" % (
            round(r1 + (r2 - r1) * s),
            round(g1 + (g2 - g1) * s),
            round(b1 + (b2 - b1) * s),
        )

    def _draw_coverage_graph(self):
        """
        Scatter plot of accepted samples' board position (x, y) in the
        image -- shows at a glance where the board has already been
        captured, complementing the range bars. Marker size ~ apparent
        board size; marker color ~ skew/tilt (light -> dark = low -> high),
        since skew has no natural x/y-position representation of its own.
        Reuses the params already computed by coverage.get_parameters on
        each accepted Sample, no extra computation needed.
        """
        c = self.coverage_canvas
        c.delete("all")

        width, height = 320, 180
        margin = 14
        margin_left = 24  # wider than `margin` so the vertical axis label fits
        x0, y0 = margin_left, margin
        x1, y1 = width - margin, height - margin

        c.create_rectangle(
            x0, y0, x1, y1, outline=self.theme.border, fill="#fafbfc"
        )
        for frac in (1 / 3, 2 / 3):
            x = x0 + frac * (x1 - x0)
            y = y0 + frac * (y1 - y0)
            c.create_line(x, y0, x, y1, fill="#e3e6ea", dash=(3, 3))
            c.create_line(x0, y, x1, y, fill="#e3e6ea", dash=(3, 3))

        c.create_text(
            (x0 + x1) / 2, y1 + 9, text="left \u2192 right (X)",
            font=self.theme.font(8), fill=self.theme.muted,
        )
        # anchor="w" at a fixed small x keeps this inside the canvas -- centering
        # it under x0 (as before) could push its bbox to a negative x and clip it.
        c.create_text(
            3, (y0 + y1) / 2, text="top\n\u2193\nbtm", font=self.theme.font(8),
            justify="center", anchor="w", fill=self.theme.muted,
        )

        # Skew color legend, top-left.
        c.create_oval(x0 + 2, 2, x0 + 10, 10, outline="", fill=self._skew_color(0.0))
        c.create_text(
            x0 + 14, 6, text="low skew", anchor="w",
            fill=self.theme.muted, font=self.theme.font(8),
        )
        c.create_oval(x0 + 72, 2, x0 + 80, 10, outline="", fill=self._skew_color(1.0))
        c.create_text(
            x0 + 84, 6, text="high skew", anchor="w",
            fill=self.theme.muted, font=self.theme.font(8),
        )

        cal = self.calibrator
        if cal is None or not cal.db:
            c.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2,
                text="No accepted samples yet", fill=self.theme.muted,
                font=self.theme.font(9),
            )
            return

        for sample in cal.db:
            px = max(0.0, min(1.0, float(sample.params[0])))
            py = max(0.0, min(1.0, float(sample.params[1])))
            psize = max(0.0, float(sample.params[2]))
            pskew = max(0.0, min(1.0, float(sample.params[3])))

            x = x0 + px * (x1 - x0)
            y = y0 + py * (y1 - y0)
            radius = max(3, min(11, 3 + 18 * psize))

            c.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="#1f77b4", fill=self._skew_color(pskew),
            )

        # Same target the on-image overlay draws, in graph space: it shows
        # where the next capture should land relative to what's already here.
        guide = self._coverage_overlay_guide()
        if guide is not None:
            gx = x0 + guide["x"] * (x1 - x0)
            gy = y0 + guide["y"] * (y1 - y0)
            c.create_oval(gx - 10, gy - 10, gx + 10, gy + 10, outline="#00a6a6", width=3)
            c.create_text(
                gx, gy - 18, text="next", fill="#008080",
                font=self.theme.font(8, "bold"),
            )

        c.create_text(
            x1, 6, text=f"{len(cal.db)} accepted",
            anchor="ne", fill=self.theme.muted, font=self.theme.font(8),
        )

    # ------------------------------------------------------------------
    # Frame rendering / sample browsing
    # ------------------------------------------------------------------

    def _forward_view_maps(self):
        """Return the cached forward-view remap LUT, rebuilding if stale."""
        cal = self.calibrator
        if cal is None or not cal.calibrated or cal.last_ocam_model is None:
            return None
        model = cal.last_ocam_model
        hfov = self.forward_view_hfov.get()
        key = round(hfov, 1)
        if self._forward_view_model_id != id(model) or self._forward_view_key != key:
            self._forward_view_cache = ocam_model.build_forward_view_maps(
                model, hfov_deg=hfov
            )
            self._forward_view_model_id = id(model)
            self._forward_view_key = key
        return self._forward_view_cache

    def _sync_forward_view_controls(self):
        """Enable the forward-view controls only once a model exists."""
        calibrated = (
            self.calibrator is not None
            and self.calibrator.calibrated
            and self.calibrator.last_ocam_model is not None
        )
        self.forward_view_toggle.configure(state="normal" if calibrated else "disabled")
        # The FOV only means something while the remap is active.
        active = calibrated and self.forward_view_var.get()
        self.forward_view_fov_box.configure(state="normal" if active else "disabled")
        self.forward_view_fov_label.configure(state="normal" if active else "disabled")

    def _apply_forward_view(self, image, corners):
        """Remap a raw frame to the forward view when a mode other than raw is on."""
        if not self.forward_view_var.get():
            return image, corners
        maps = self._forward_view_maps()
        if maps is None:
            return image, corners
        map_x, map_y = maps
        src = image.astype(np.uint8) if image.ndim == 2 else image
        remapped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return remapped, None  # corners are in raw-image coords; meaningless on the remap

    def _on_forward_view_toggle(self):
        self._sync_forward_view_controls()
        self._show_current_sample()

    def _coverage_overlay_guide(self):
        """Position guide, recomputed only when the accepted-sample db changes."""
        cal = self.calibrator
        n = len(cal.db) if cal is not None else -1
        if n != self._guide_db_len:
            self._guide_cache = self._compute_coverage_overlay_guide()
            self._guide_db_len = n
        return self._guide_cache

    def _compute_coverage_overlay_guide(self):
        """
        Where to put the board next, as normalized image coordinates.

        The on-image box and the coverage graph both draw from this, so they
        cannot disagree with the suggestion text. None when nothing needs
        improving, which is how callers know to draw no guidance.
        """
        cal = self.calibrator
        if cal is None:
            return None

        metrics = cal.db_metrics()
        report = coverage.compute_bin_coverage(metrics) if metrics else None
        suggestions = coverage.coverage_suggestions(report) if report else []
        if not report or not suggestions:
            return None

        progress_map = {}
        if cal.db:
            _goodenough, progress, _r = coverage.compute_goodenough_with_bins(
                cal.db_params(), metrics, cal.param_ranges, cal.min_db_size,
            )
            for name, lo, hi, p in progress:
                progress_map[str(name).strip().lower()] = (
                    float(lo), float(hi), float(p)
                )

        cell = coverage.next_target_cell(report)
        if cell is not None:
            x, y = coverage.cell_center(cell)
        else:
            # All cells filled: stay put and change angle or distance instead.
            x = float(np.mean([s.params[0] for s in cal.db]))
            y = float(np.mean([s.params[1] for s in cal.db]))

        size_info = progress_map.get("size")
        if size_info is not None and size_info[2] < 1.0:
            lo, hi, _p = size_info
            if lo > 0.12 and hi >= 0.22:
                size = 0.10
            else:
                size = 0.34
        elif cal.db:
            size = float(np.median([s.params[2] for s in cal.db]))
        else:
            size = 0.22

        return {
            "x": max(0.05, min(0.95, x)),
            "y": max(0.05, min(0.95, y)),
            "size": max(0.08, min(0.45, size)),
            "text": suggestions[0],
        }

    def _draw_position_guide_on_image(self, preview):
        """Overlay the "put the grid here" target box on a BGR preview."""
        guide = self._coverage_overlay_guide()
        if guide is None:
            return preview

        h, w = preview.shape[:2]
        cal = self.calibrator
        valid_region = cal.valid_region_px() if cal is not None else None
        x0, y0, x1, y1, cx, cy = guide_box_rect(
            guide, (w, h), valid_region,
        )

        color = (255, 255, 0)

        overlay = preview.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        cv2.line(overlay, (cx - 18, cy), (cx + 18, cy), color, 2)
        cv2.line(overlay, (cx, cy - 18), (cx, cy + 18), color, 2)
        preview = cv2.addWeighted(overlay, 0.75, preview, 0.25, 0)

        cv2.putText(
            preview, "PUT LED GRID HERE", (max(5, x0), max(22, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

        short_hint = guide["text"].replace("Move LED grid here: ", "")
        if len(short_hint) > 46:
            short_hint = short_hint[:43] + "..."
        cv2.putText(
            preview, short_hint, (max(5, x0), min(h - 12, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )

        return preview

    def _render_frame(self, image, corners, caption, show_guidance: bool = False):
        """Draw one grayscale frame (+ detected points, if any) on the canvas."""
        img = image
        if img.ndim == 2:
            preview = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            preview = img.copy()

        if corners is not None:
            for idx, (row, col) in enumerate(corners, start=1):
                if not np.isfinite(row) or not np.isfinite(col):
                    continue
                cv2.circle(preview, (int(round(col)), int(round(row))), 5, (0, 0, 255), 1)
                cv2.putText(preview, str(idx), (int(round(col)) + 5, int(round(row)) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # Guidance is drawn in raw sensor coordinates, so it must go on
        # before the display rescale. Skipped under the forward view, whose
        # remap would put the target box somewhere the board isn't.
        if show_guidance and not self.forward_view_var.get():
            preview = self._draw_position_guide_on_image(preview)

        cw = max(50, self.image_canvas.winfo_width())
        ch = max(50, self.image_canvas.winfo_height())
        h, w = preview.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(preview, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".ppm", rgb)
        if not ok:
            return
        self.photo_ref = tk.PhotoImage(data=buf.tobytes())
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=self.photo_ref, anchor="center")
        self.image_label.configure(text=caption)

    def _show_current_sample(self):
        cal = self.calibrator
        if cal is None or not cal.db:
            self.image_label.configure(text="No accepted sample loaded")
            return

        self.current_sample_index %= len(cal.db)
        sample = cal.db[self.current_sample_index]
        name = Path(sample.image_path).name
        p_str = ", ".join(f"{v:.2f}" for v in sample.params)
        img, pts = self._apply_forward_view(sample.image, sample.corners)
        n_samples = len(self.calibrator.db)
        self._render_frame(
            img,
            pts,
            f"Sample {self.current_sample_index + 1}/{n_samples}: {name}   p=[{p_str}]",
            show_guidance=True,
        )

    def next_sample(self):
        if self.calibrator and self.calibrator.db:
            self.current_sample_index = (self.current_sample_index + 1) % len(self.calibrator.db)
            self._show_current_sample()

    def prev_sample(self):
        if self.calibrator and self.calibrator.db:
            self.current_sample_index = (self.current_sample_index - 1) % len(self.calibrator.db)
            self._show_current_sample()

    def delete_sample(self):
        """
        Drop the browsed sample from the db.

        A mis-ordered UV-dot detection yields a plausible corners array that
        is_good_sample (which only sees four numbers) happily accepts, and it
        then poisons the solve; a live session has no other recovery.
        """
        cal = self.calibrator
        if cal is None or not cal.db:
            messagebox.showinfo("No samples", "No accepted samples to delete.")
            return

        removed = Path(cal.db[self.current_sample_index].image_path).name
        cal.db.pop(self.current_sample_index)
        # Deleting invalidates any previous solve, and the cached guide is
        # keyed on len(db) -- which a delete plus an accept leaves unchanged.
        cal.calibrated = False
        self.save_button.configure(state="disabled")
        self._guide_db_len = -1

        self.current_sample_index = (
            min(self.current_sample_index, len(cal.db) - 1) if cal.db else 0
        )

        self._append_log(f"deleted sample: {removed} ({len(cal.db)} remaining)")
        self._update_progress_panel()
        self._show_current_sample()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def calibrate(self):
        # Stop collecting before solving so the sample db cannot change
        # underneath cal_fromcorners; the user can restart capture after.
        if self.consumer.capturing.is_set():
            self.toggle_capture()
        self.consumer.wait_until_idle()

        cal = self.calibrator
        if cal is None or not cal.db:
            messagebox.showinfo("No samples", "No accepted samples yet.")
            return

        if not cal.goodenough:
            proceed = messagebox.askyesno(
                "Not enough variety yet",
                "The accepted samples do not yet cover enough X/Y/Size/Skew "
                "variation. Calibrate anyway?",
            )
            if not proceed:
                return

        try:
            # The solve blocks the Tk thread for seconds; without a busy
            # cursor and a forced repaint the window just looks frozen.
            self._set_status(
                f"Solving over {len(cal.db)} samples. This can take a while..."
            )
            self.root.configure(cursor="watch")
            self.root.update_idletasks()
            try:
                cal.cal_fromcorners(
                    do_find_center=True,
                    fast_find_center=not self.slow_find_center.get(),
                    refine_corners=False,
                )
            finally:
                self.root.configure(cursor="")
            self.save_button.configure(state="normal")
            self.forward_view_var.set(False)
            self._forward_view_cache = None
            self._forward_view_model_id = None
            self._forward_view_key = None
            self._sync_forward_view_controls()

            avg = float(np.nanmean(cal.reprojection_err))
            self._set_status(
                f"Calibration complete: avg reprojection error {avg:.3f} px. "
                "Use SAVE / EXPORT to write results."
            )
            messagebox.showinfo(
                "Calibration complete",
                f"Calibration finished over {len(cal.db)} accepted samples.\n"
                f"Average reprojection error: {avg:.3f} px\n"
                f"Center: ({cal.last_ocam_model.xc:.2f}, {cal.last_ocam_model.yc:.2f})",
            )

            if self.show_plots.get():
                from ..diagnostics import plot_window, plots

                Xp_abs, Yp_abs, ima_proc = cal.assemble()
                plots.print_calib_summary(
                    cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt,
                    Xp_abs, Yp_abs,
                )
                figures = plots.build_diagnostic_figures(
                    cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt,
                    Xp_abs, Yp_abs,
                    images=[s.image for s in cal.db],
                    n_sq_y=cal.board.n_sq_y,
                )
                # Toplevel, so live capture keeps running behind the plots.
                plot_window.show_diagnostics(figures, parent=self.root)
        except Exception as exc:
            messagebox.showerror("Calibration failed", str(exc))
            self._set_status("Calibration failed.")

    def _suffixed_save_filename(self, camera_id: str) -> str:
        """``calib_results_<camera_id>.txt``, or the plain default if empty."""
        camera_id = _sanitize_filename_part(camera_id)
        if not camera_id:
            return "calib_results.txt"
        return f"calib_results_{camera_id}.txt"

    def save_and_export(self):
        cal = self.calibrator
        if cal is None or not cal.calibrated:
            messagebox.showinfo("Not calibrated", "Run CALIBRATE first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save calibration results",
            initialdir=self.output_dir.get() or ".",
            initialfile=self._default_save_filename(),
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            self._set_status("Save cancelled.")
            return
        try:
            saved = cal.export_txt(path=path)
            self._set_status(f"Saved calibration results to {saved}")
            messagebox.showinfo("Saved", f"Saved:\n{saved}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _default_save_filename(self) -> str:
        # Named after the camera identifier from the driver config (set by
        # the wizard via --camera_id); empty there means the plain default.
        return self._suffixed_save_filename(
            camera_id_from_config(self.camera_id)
        )

    # ------------------------------------------------------------------
    # LED grid (runtime)
    # ------------------------------------------------------------------

    def _sync_board_fields(self) -> None:
        """Show the running Calibrator's geometry in the editable fields."""
        board = self.calibrator.board
        self.n_sq_x.set(int(board.n_sq_x))
        self.n_sq_y.set(int(board.n_sq_y))
        self.spacing_mm.set(float(board.spacing_mm))
        self._update_board_button()

    def _on_board_fields_committed(self, _event=None):
        """
        Apply an edited LED-grid geometry to the running Calibrator.

        A new square count changes the point grid the detector must find and
        the solver indexes points by, so the samples collected for the old
        grid are dropped and capture restarts from an empty database. A pitch
        change alone keeps the same physical points, so the samples survive --
        only the world scale changes and the user re-solves.
        """
        cal = self.calibrator
        if cal is None:
            return "break"

        try:
            board = LedGridBoard(
                n_sq_x=int(self.n_sq_x.get()),
                n_sq_y=int(self.n_sq_y.get()),
                spacing_mm=float(self.spacing_mm.get()),
            )
        except (tk.TclError, ValueError):
            board = None
        if board is None or not (
            1 <= board.n_sq_x <= 30
            and 1 <= board.n_sq_y <= 30
            and math.isfinite(board.spacing_mm)
            and board.spacing_mm > 0
        ):
            self._sync_board_fields()
            self._set_status(
                "LED grid unchanged: counts must be 1-30 and spacing above 0."
            )
            return "break"

        old = cal.board
        if board == old:
            self._update_board_button()
            return "break"

        if (board.n_sq_x, board.n_sq_y) != (old.n_sq_x, old.n_sq_y):
            # Pause first: the consumer converts frames independently of the
            # Tk thread, and one accepted after the clear would mix the two
            # point grids in a single solve.
            was_running = self.consumer.capturing.is_set()
            if was_running:
                self.consumer.capturing.clear()
            self.consumer.wait_until_idle()
            # Hold the conversion lock while the grid is replaced: a frame the
            # consumer already pulled off the queue must be detected with the
            # new board (or not at all), never accepted into the cleared db
            # under the old one.
            gate = getattr(self.consumer, "handle_lock", None) or nullcontext()
            with gate:
                self._drain_frame_queues()
                cal.board = board
                self._discard_samples(
                    f"LED grid {board.n_sq_x} x {board.n_sq_y}, "
                    f"{board.spacing_mm:g} mm",
                    "captured for the previous grid",
                )

            if was_running:
                self.consumer.capturing.set()
            self._set_status(
                f"LED grid {board.n_sq_x} x {board.n_sq_y}, "
                f"{board.spacing_mm:g} mm spacing: capture a new set."
            )
        else:
            # Same target, new pitch: keep the samples, which still describe
            # the same points; the solved model must be rebuilt in the new
            # units.
            cal.board = board
            self._invalidate_calibration(reset_view=False)
            self._update_progress_panel()
            self._set_status(
                f"LED grid spacing set to {board.spacing_mm:g} mm; samples "
                "kept -- press CALIBRATE to re-solve."
            )
        self._update_board_button()
        return "break"

    def _build_source_controls(self, parent):
        ttk.Label(parent, text="Image topic:").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.topic, width=45).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        self.capture_button = ttk.Button(parent, text="Stop Capture", command=self.toggle_capture)
        self.capture_button.grid(row=0, column=2, padx=8)
        parent.columnconfigure(1, weight=1)

    def _refresh_capture_button(self):
        running = self.consumer.capturing.is_set()
        self.capture_button.configure(text=("Stop Capture" if running else "Start Capture"))
        # FrameConsumerThread appends to db from another thread. Rather than
        # lock every db access, deletion is only offered while capture is
        # stopped -- the same thing calibrate() already requires.
        self.delete_button.configure(state="disabled" if running else "normal")

    def toggle_capture(self):
        if self.consumer.capturing.is_set():
            self.consumer.capturing.clear()
            self._set_status("Live capture paused. Previous/Next browse accepted samples.")
        else:
            topic = self.topic.get().strip() or "image"
            if topic != self._subscribed_topic:
                try:
                    resolved = self.subscribe_fn(topic)
                except Exception as exc:
                    messagebox.showerror("Subscribe failed", str(exc))
                    return
                self._subscribed_topic = resolved
                self.topic.set(resolved)
            self.consumer.capturing.set()
            self._set_status(f"Live capture running on topic '{self._subscribed_topic}'.")
        self._refresh_capture_button()

    def _on_mirror_toggled(self):
        """
        Switch the frame orientation with no frame of the old one left over.

        The consumer thread pulls frames off the raw queue independently of
        the GUI, so a frame converted before the switch could still be
        accepted afterwards and mix two orientations in a single solve.
        Capture is therefore paused for the switch, the queues are drained,
        and capture resumes.
        """
        mirrored = bool(self.mirror.get())
        was_running = self.consumer.capturing.is_set()
        if was_running:
            self.consumer.capturing.clear()
        self.consumer.wait_until_idle()
        self._drain_frame_queues()

        self._apply_mirror(mirrored)
        self._discard_samples_for_orientation_change(mirrored)

        if was_running:
            self.consumer.capturing.set()
        self._set_status(
            f"Live capture on '{self._subscribed_topic}': incoming frames are "
            f"{'mirrored' if mirrored else 'not mirrored'}."
        )

    def _drain_frame_queues(self) -> None:
        """Throw away frames converted under the previous orientation."""
        raw_queue = getattr(self.consumer, "raw_queue", None)
        for q in (raw_queue, self.result_queue, self.preview_queue):
            if q is None:
                continue
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        # Corners come from a throttled frame, so they may belong to a frame
        # that is no longer current.
        self._latest_corners = None
        self._latest_preview = None

    def _poll_queue(self):
        """
        Drain both queues independently every tick.

        result_queue (throttled by --rate_hz) drives accept/reject
        bookkeeping, the sample log, and the progress panel -- unchanged
        from before. preview_queue (unthrottled) drives what's rendered on
        the canvas, so the live view keeps updating every tick even when no
        new processed result has arrived yet.
        """
        got_result = False
        while True:
            try:
                result, _image = self.result_queue.get_nowait()
            except queue.Empty:
                break
            got_result = True
            self._latest_corners = result.corners
            if result.accepted:
                self._latest_outcome = f"ACCEPTED as sample {len(self.calibrator.db)}"
                # Keep browsing anchored to the newest accepted sample.
                self.current_sample_index = len(self.calibrator.db) - 1
            elif result.detected:
                self._latest_outcome = "detected but rejected (too similar)"
                self.n_rejected += 1
            else:
                self._latest_outcome = "no markers detected"
                self.n_failed += 1
            self._append_log(result.reason)

        if got_result:
            self._update_progress_panel()

        latest_preview = None
        while True:
            try:
                latest_preview = self.preview_queue.get_nowait()
            except queue.Empty:
                break
        if latest_preview is not None:
            self._latest_preview = latest_preview
            now = time.monotonic()
            dt = now - self._last_frame_t
            if dt > 0:
                # Exponential moving average; a raw 1/dt reading jitters too
                # much to be readable in a status bar.
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_frame_t = now
            self._stale_warned = False

        if self.consumer.capturing.is_set() and self._latest_preview is not None:
            img, pts = self._apply_forward_view(self._latest_preview, self._latest_corners)
            self._render_frame(
                img, pts, f"Live preview: {self._latest_outcome}", show_guidance=True,
            )
        # Outside the render branch on purpose: a stream that stopped
        # delivering has no preview to render, and that is exactly when the
        # user needs to be told. A wrong-QoS or wrong-topic subscription is
        # silent at the DDS layer, so silence has to be surfaced here.
        if self.consumer.capturing.is_set():
            if time.monotonic() - self._last_frame_t > 3.0:
                if not self._stale_warned:
                    self._set_status(
                        f"No frames on '{self._subscribed_topic}' -- "
                        "is the camera publishing?"
                    )
                    self._stale_warned = True
            else:
                self._set_status(
                    f"Live capture on '{self._subscribed_topic}' at "
                    f"{self._fps:.1f} fps: {len(self.calibrator.db)} accepted, "
                    f"{self.n_rejected} rejected, {self.n_failed} without detection."
                )

        self.root.after(self.POLL_MS, self._poll_queue)


def launch_live_gui(
    calibrator: Calibrator,
    result_queue,
    preview_queue,
    consumer,
    subscribe_fn,
    initial_topic: str = "image",
    camera_id: str = "",
    output_dir: str = ".",
    slow_find_center: bool = False,
    dev_mode: bool = False,
    mirror: bool = False,
    set_mirror_fn=None,
) -> None:
    """Launch the live (ROS 2 topic-driven) calibration GUI."""
    _require_gui_deps()
    root = tk.Tk()
    LiveCalibrationApp(
        root,
        calibrator=calibrator,
        result_queue=result_queue,
        preview_queue=preview_queue,
        consumer=consumer,
        subscribe_fn=subscribe_fn,
        initial_topic=initial_topic,
        camera_id=camera_id,
        output_dir=output_dir,
        slow_find_center=slow_find_center,
        dev_mode=dev_mode,
        mirror=mirror,
        set_mirror_fn=set_mirror_fn,
    )
    root.mainloop()
