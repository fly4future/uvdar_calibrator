"""
Tkinter calibration GUIs (batch and live).

Both apps feed frames into
:class:`~uvdar_calibrator.engine.calibrator.Calibrator` and share one panel
implementation (``_BaseCalibrationApp``):

- a live log of per-frame accept/reject decisions,
- four ROS-style range progress bars (X, Y, Size, Skew), each drawn as a
  track with a colored segment from the min to the max accepted parameter
  value (green when that axis has enough variation),
- the supplementary bin-based "next images to capture" hints, driven off
  the accepted samples,
- CALIBRATE gated by ``calibrator.goodenough`` (with confirm-to-override)
  and SAVE/EXPORT gated by ``calibrator.calibrated``.

``BatchCalibrationApp`` drives the panel from a folder of photos
("frames arrive one at a time" simulated over files);
``LiveCalibrationApp`` drives it from a queue of results produced by the
ROS 2 subscriber in :mod:`uvdar_calibrator.apps.live_node`. Only the frame
*source* differs between the two -- keep panel/progress logic in the base
class so the apps never diverge.
"""

from __future__ import annotations

from pathlib import Path
import queue

import numpy as np

from ..engine import coverage
from ..engine import ocam_model
from ..engine.board import LedGridBoard
from ..engine.calibrator import Calibrator
from ..engine.detection import find_image_files, read_image_gray

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - headless environments
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

#: Keep at most this many lines in the sample log (relevant for live mode,
#: where rejected-frame lines keep arriving for as long as capture runs).
MAX_LOG_LINES = 1000


def _require_gui_deps() -> None:
    if tk is None:
        raise RuntimeError("Tkinter is required for the GUI. Install python-tk/tkinter.")
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")


class _BaseCalibrationApp:
    """
    Shared calibration panel.

    Holds everything that only depends on ``Calibrator`` state, not on how
    frames arrive: widget layout, range bars, sample log, suggestion box,
    sample browsing, and the CALIBRATE / SAVE-EXPORT / report actions.
    Subclasses implement ``_build_source_controls`` (the top row deciding
    where frames come from) and feed results through ``_append_log`` /
    ``_update_progress_panel`` / ``_render_frame``.
    """

    def __init__(
        self,
        root,
        n_sq_x: int = 6,
        n_sq_y: int = 4,
        spacing_mm: float = 50.0,
        taylor_order: int = 4,
        output_dir: str = ".",
        slow_find_center: bool = False,
    ):
        self.root = root
        self.root.title("UV-DAR / OCamCalib Calibration Assistant")
        self.root.geometry("1180x980")
        self.root.minsize(1000, 870)

        self.n_sq_x = tk.IntVar(value=n_sq_x)
        self.n_sq_y = tk.IntVar(value=n_sq_y)
        self.spacing_mm = tk.DoubleVar(value=spacing_mm)
        self.taylor_order = tk.IntVar(value=taylor_order)
        self.output_dir = tk.StringVar(value=output_dir)
        self.slow_find_center = tk.BooleanVar(value=slow_find_center)
        self.no_plots = tk.BooleanVar(value=True)
        self.forward_view_var = tk.BooleanVar(value=False)

        self.calibrator: Calibrator | None = None
        self.current_sample_index = 0
        self.photo_ref = None
        self._forward_view_cache = None        # (map_x, map_y) or None
        self._forward_view_model_id = None     # id(last built-from model)

        self._build_widgets()
        self.root.bind("<Left>", lambda _e: self.prev_sample())
        self.root.bind("<Right>", lambda _e: self.next_sample())

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _build_source_controls(self, parent) -> None:
        """Build the top row deciding where frames come from (subclass hook)."""
        raise NotImplementedError

    def _build_widgets(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self._build_source_controls(top)

        opts = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        opts.pack(side=tk.TOP, fill=tk.X)
        self.board_option_widgets = []
        ttk.Label(opts, text="Grid squares X:").pack(side=tk.LEFT)
        w = ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_x, width=5)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Label(opts, text="Y:").pack(side=tk.LEFT)
        w = ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_y, width=5)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Label(opts, text="Spacing mm:").pack(side=tk.LEFT, padx=(12, 0))
        w = ttk.Entry(opts, textvariable=self.spacing_mm, width=7)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Label(opts, text="Taylor:").pack(side=tk.LEFT, padx=(12, 0))
        w = ttk.Spinbox(opts, from_=4, to=10, textvariable=self.taylor_order, width=5)
        w.pack(side=tk.LEFT, padx=4)
        self.board_option_widgets.append(w)
        ttk.Checkbutton(
            opts, text="MATLAB-style slow Find Center", variable=self.slow_find_center
        ).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(
            opts, text="No plots after calibration", variable=self.no_plots
        ).pack(side=tk.LEFT, padx=8)

        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(main, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.image_canvas = tk.Canvas(
            left, bg="#202020", highlightthickness=1, highlightbackground="#888"
        )
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav = ttk.Frame(left, padding=(0, 8, 0, 0))
        nav.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(nav, text="Previous", command=self.prev_sample).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next", command=self.next_sample).pack(side=tk.LEFT, padx=4)
        self.forward_view_toggle = ttk.Checkbutton(
            nav,
            text="Forward view (undistorted)",
            variable=self.forward_view_var,
            command=self._on_forward_view_toggle,
            state="disabled",
        )
        self.forward_view_toggle.pack(side=tk.RIGHT)
        self.image_label = ttk.Label(nav, text="No accepted sample loaded")
        self.image_label.pack(side=tk.LEFT, padx=12)

        title = ttk.Label(right, text="Calibration Progress", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w")
        self.status_label = ttk.Label(right, text="Status: no images analyzed", wraplength=320)
        self.status_label.pack(anchor="w", pady=(8, 8))

        # Four ROS-style range bars: X, Y, Size, Skew.
        self.bar_canvas = tk.Canvas(
            right, width=320, height=140, bg="white",
            highlightthickness=1, highlightbackground="#ccc",
        )
        self.bar_canvas.pack(anchor="w", pady=(0, 10))
        self._draw_bars([])

        ttk.Label(
            right, text="Board position coverage:", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.coverage_canvas = tk.Canvas(
            right, width=320, height=180, bg="white",
            highlightthickness=1, highlightbackground="#ccc",
        )
        self.coverage_canvas.pack(anchor="w", pady=(4, 10))
        self._draw_coverage_graph()

        ttk.Label(right, text="Sample log:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_box = tk.Text(right, height=10, width=44, wrap="word")
        self.log_box.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(4, 10))
        self.log_box.configure(state="disabled")

        ttk.Label(
            right, text="Next images to capture:", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.suggestion_box = tk.Text(right, height=6, width=44, wrap="word")
        self.suggestion_box.pack(anchor="w", fill=tk.X, pady=(4, 10))
        self.suggestion_box.configure(state="disabled")

        actions = ttk.Frame(right)
        actions.pack(side=tk.BOTTOM, fill=tk.X)
        self.calibrate_button = ttk.Button(
            actions, text="CALIBRATE", command=self.calibrate, state="disabled"
        )
        self.calibrate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.save_button = ttk.Button(
            actions, text="SAVE / EXPORT", command=self.save_and_export, state="disabled"
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(actions, text="Save Report", command=self.save_report).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(actions, text="Exit", command=self.root.destroy).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0)
        )

        self.bottom_status = ttk.Label(self.root, text="", relief=tk.SUNKEN, anchor="w", padding=4)
        self.bottom_status.pack(side=tk.BOTTOM, fill=tk.X)

    def _set_status(self, text):
        self.bottom_status.configure(text=text)
        self.root.update_idletasks()

    def _append_log(self, line):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        n_lines = int(self.log_box.index("end-1c").split(".")[0])
        if n_lines > MAX_LOG_LINES:
            self.log_box.delete("1.0", f"{n_lines - MAX_LOG_LINES + 1}.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.root.update_idletasks()

    def _write_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # ------------------------------------------------------------------
    # Progress panel
    # ------------------------------------------------------------------

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
        if goodenough:
            status = f"READY TO CALIBRATE ({n} accepted samples)"
        else:
            status = f"NOT READY -- need more varied views ({n} accepted samples)"
        self.status_label.configure(text=f"Status: {status}")

        # Gate CALIBRATE on goodenough; allow an explicit override once
        # samples exist (confirmation dialog in calibrate()).
        self.calibrate_button.configure(state=("normal" if n > 0 else "disabled"))

        metrics = []
        for sample in cal.db:
            m = coverage.sample_metric(
                sample.corners, cal.board, cal.image_size,
                label=Path(sample.image_path).name,
            )
            if m is not None:
                metrics.append(m)
        if metrics:
            suggestions = coverage.coverage_suggestions(coverage.compute_bin_coverage(metrics))
            if not suggestions:
                suggestions = ["All coverage bins are represented."]
        else:
            suggestions = ["No accepted samples yet."]
        self._write_text(self.suggestion_box, "\n".join(f"• {s}" for s in suggestions))

    def _progress_color(self, p: float) -> str:
        """Return the bar color: red -> yellow by progress, green when complete."""
        if p >= 1.0:
            return "#2e9e4f"
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
        """
        Draw the four ROS-style range bars.

        Each bar is a 0..1 track; the colored segment spans the min..max
        parameter values covered by the accepted samples (mirroring
        OpenCVCalibrationNode.redraw_monocular's bar drawing).
        """
        c = self.bar_canvas
        c.delete("all")

        x0, x1 = 60, 280
        y = 20

        rows = progress if progress else [(name, 0.0, 0.0, 0.0) for name in coverage.PARAM_NAMES]

        for name, lo, hi, p in rows:
            c.create_text(10, y, text=name, anchor="w", font=("Segoe UI", 9, "bold"))
            c.create_rectangle(x0, y - 8, x1, y + 8, outline="#999", fill="#eee")
            lo_c = max(0.0, min(1.0, lo))
            hi_c = max(0.0, min(1.0, hi))
            if hi_c > lo_c:
                c.create_rectangle(
                    x0 + (x1 - x0) * lo_c,
                    y - 8,
                    x0 + (x1 - x0) * hi_c,
                    y + 8,
                    outline="",
                    fill=self._progress_color(p),
                )
            c.create_text(x1 + 10, y, text=f"{100.0 * p:.0f}%", anchor="w", font=("Segoe UI", 9))
            y += 32

    def _draw_coverage_graph(self):
        """
        Scatter plot of accepted samples' board position (x, y) in the
        image, marker size ~ apparent board size -- shows at a glance where
        the board has already been captured, complementing the range bars.
        Reuses the params already computed by coverage.get_parameters on
        each accepted Sample, no extra computation needed.
        """
        c = self.coverage_canvas
        c.delete("all")

        width, height = 320, 180
        margin = 14
        x0, y0 = margin, margin
        x1, y1 = width - margin, height - margin

        c.create_rectangle(x0, y0, x1, y1, outline="#999", fill="#fafafa")
        for frac in (1 / 3, 2 / 3):
            x = x0 + frac * (x1 - x0)
            y = y0 + frac * (y1 - y0)
            c.create_line(x, y0, x, y1, fill="#ddd", dash=(3, 3))
            c.create_line(x0, y, x1, y, fill="#ddd", dash=(3, 3))

        c.create_text((x0 + x1) / 2, y1 + 8, text="left → right (X)", font=("Segoe UI", 7))
        c.create_text(x0 - 8, (y0 + y1) / 2, text="top\n↓\nbtm", font=("Segoe UI", 6), justify="center")

        cal = self.calibrator
        if cal is None or not cal.db:
            c.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2,
                text="No accepted samples yet", fill="#777", font=("Segoe UI", 9),
            )
            return

        for sample in cal.db:
            px = max(0.0, min(1.0, float(sample.params[0])))
            py = max(0.0, min(1.0, float(sample.params[1])))
            psize = max(0.0, float(sample.params[2]))

            x = x0 + px * (x1 - x0)
            y = y0 + py * (y1 - y0)
            radius = max(3, min(11, 3 + 18 * psize))

            c.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="#1f77b4", fill="#9ecae1",
            )

        c.create_text(
            x1, 6, text=f"{len(cal.db)} accepted",
            anchor="ne", fill="#333", font=("Segoe UI", 8),
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
        if self._forward_view_model_id != id(model):
            self._forward_view_cache = ocam_model.build_forward_view_maps(model)
            self._forward_view_model_id = id(model)
        return self._forward_view_cache

    def _apply_forward_view(self, image, corners):
        """Remap a raw frame to the forward view when the toggle is on."""
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
        self._show_current_sample()

    def _render_frame(self, image, corners, caption):
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
        self._render_frame(
            img,
            pts,
            f"Sample {self.current_sample_index + 1}/{len(cal.db)}: {name}   p=[{p_str}]",
        )

    def next_sample(self):
        if self.calibrator and self.calibrator.db:
            self.current_sample_index = (self.current_sample_index + 1) % len(self.calibrator.db)
            self._show_current_sample()

    def prev_sample(self):
        if self.calibrator and self.calibrator.db:
            self.current_sample_index = (self.current_sample_index - 1) % len(self.calibrator.db)
            self._show_current_sample()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def calibrate(self):
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
            self._set_status("Running UV-DAR calibration. This can take a while...")
            cal.cal_fromcorners(
                do_find_center=True,
                fast_find_center=not self.slow_find_center.get(),
                refine_corners=False,
            )
            self.save_button.configure(state="normal")
            self.forward_view_toggle.configure(state="normal")

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

            if not self.no_plots.get():
                from ..diagnostics import plots

                Xp_abs, Yp_abs, ima_proc = cal._assemble()
                plots.reproject_calib(
                    cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt,
                    Xp_abs, Yp_abs,
                    images=[s.image for s in cal.db],
                    n_sq_y=cal.board.n_sq_y,
                )
                plots.analyse_error(
                    cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt, Xp_abs, Yp_abs
                )
                plots.show_calib_results(
                    cal.last_ocam_model, cal.RRfin, ima_proc, cal.Xt, cal.Yt, Xp_abs, Yp_abs
                )
        except Exception as exc:
            messagebox.showerror("Calibration failed", str(exc))
            self._set_status("Calibration failed.")

    def save_and_export(self):
        cal = self.calibrator
        if cal is None or not cal.calibrated:
            messagebox.showinfo("Not calibrated", "Run CALIBRATE first.")
            return
        try:
            out = self.output_dir.get() or "."
            cal.save(output_dir=out)
            cal.export_txt(output_dir=out)
            out_abs = Path(out).resolve()
            self._set_status(f"Saved Omni_Calib_Results and calib_results.txt to {out_abs}")
            messagebox.showinfo(
                "Saved",
                f"Saved:\n{out_abs / 'Omni_Calib_Results.npz'}\n{out_abs / 'calib_results.txt'}",
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_report(self):
        cal = self.calibrator
        if cal is None:
            messagebox.showinfo("No report", "No samples collected yet.")
            return
        out = Path(self.output_dir.get() or ".")
        out.mkdir(parents=True, exist_ok=True)
        path = out / "calibration_coverage.txt"
        path.write_text(cal.report() + "\n", encoding="utf-8")
        messagebox.showinfo("Saved", f"Saved readiness report to:\n{path.resolve()}")


class BatchCalibrationApp(_BaseCalibrationApp):
    """Folder-based app: Load / Analyze runs handle_frame over each photo."""

    def __init__(
        self,
        root,
        image_dir: str = "photos",
        base_name: str = "",
        extension: str = "all",
        **kwargs,
    ):
        self.image_dir = tk.StringVar(value=image_dir)
        self.base_name = tk.StringVar(value=base_name)
        self.extension = tk.StringVar(value=extension)
        super().__init__(root, **kwargs)
        self._set_status("Choose a folder and click Load / Analyze Images.")

    def _build_source_controls(self, parent):
        ttk.Label(parent, text="Image folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.image_dir, width=45).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(parent, text="Browse", command=self._browse_images).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="Base:").grid(row=0, column=3, sticky="e")
        ttk.Entry(parent, textvariable=self.base_name, width=8).grid(row=0, column=4, padx=4)
        ttk.Label(parent, text="Ext:").grid(row=0, column=5, sticky="e")
        ttk.Entry(parent, textvariable=self.extension, width=7).grid(row=0, column=6, padx=4)
        ttk.Button(parent, text="Load / Analyze Images", command=self.load_images).grid(
            row=0, column=7, padx=8
        )
        parent.columnconfigure(1, weight=1)

    def _browse_images(self):
        folder = filedialog.askdirectory(initialdir=self.image_dir.get() or ".")
        if folder:
            self.image_dir.set(folder)

    # ------------------------------------------------------------------
    # Incremental load: one handle_frame per photo
    # ------------------------------------------------------------------

    def load_images(self):
        try:
            files = find_image_files(
                self.image_dir.get(),
                self.base_name.get(),
                self.extension.get(),
            )
            if not files:
                messagebox.showerror(
                    "No images",
                    f"No images found in {self.image_dir.get()!r} with "
                    f"base={self.base_name.get()!r} ext={self.extension.get()!r}.",
                )
                return

            board = LedGridBoard(
                n_sq_x=int(self.n_sq_x.get()),
                n_sq_y=int(self.n_sq_y.get()),
                spacing_mm=float(self.spacing_mm.get()),
            )
            self.calibrator = Calibrator(
                board,
                taylor_order=int(self.taylor_order.get()),
                preview_dir=str(Path(self.image_dir.get()) / "detected_marker_previews"),
            )
            self.current_sample_index = 0
            self.save_button.configure(state="disabled")
            self.forward_view_var.set(False)
            self.forward_view_toggle.configure(state="disabled")
            self._forward_view_cache = None
            self._forward_view_model_id = None
            self._write_text(self.log_box, "")

            n_rejected = 0
            n_failed = 0

            for k, path in enumerate(files, start=1):
                self._set_status(f"Analyzing image {k}/{len(files)}: {Path(path).name}")
                result = self.calibrator.handle_frame(read_image_gray(path), path)
                self._append_log(f"image {k}: {result.reason}")
                if result.detected and not result.accepted:
                    n_rejected += 1
                elif not result.detected:
                    n_failed += 1
                self._update_progress_panel()

            self._show_current_sample()
            self._set_status(
                f"Done: {len(self.calibrator.db)} accepted, {n_rejected} rejected as "
                f"near-duplicates, {n_failed} failed detection."
            )
        except Exception as exc:
            messagebox.showerror("Load/analyze failed", str(exc))
            self._set_status("Load/analyze failed.")


class LiveCalibrationApp(_BaseCalibrationApp):
    """
    Live-topic app driven by the ROS 2 subscriber in ``live_node``.

    Replaces the folder controls with an image-topic field and a
    Start/Stop capture toggle; frames are processed by a background
    consumer thread and their ``FrameResult``s drained here via
    ``root.after`` polling. The canvas follows the most recently
    processed live frame while capturing; Previous/Next still browse the
    accepted samples.
    """

    POLL_MS = 50

    def __init__(
        self,
        root,
        calibrator: Calibrator,
        result_queue,
        consumer,
        subscribe_fn,
        initial_topic: str = "image",
        **kwargs,
    ):
        self.result_queue = result_queue
        self.consumer = consumer
        self.subscribe_fn = subscribe_fn
        self.topic = tk.StringVar(value=initial_topic)
        self._subscribed_topic = initial_topic
        self.n_rejected = 0
        self.n_failed = 0
        super().__init__(root, **kwargs)

        self.calibrator = calibrator
        # Board geometry / Taylor order are fixed at node start (CLI flags);
        # the running Calibrator cannot be re-configured mid-capture.
        for widget in self.board_option_widgets:
            widget.configure(state="disabled")

        self._refresh_capture_button()
        self._set_status(f"Live capture running on topic '{initial_topic}'.")
        self.root.after(self.POLL_MS, self._poll_queue)

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

    def _poll_queue(self):
        latest = None
        while True:
            try:
                result, image = self.result_queue.get_nowait()
            except queue.Empty:
                break
            latest = (result, image)
            if result.detected and not result.accepted:
                self.n_rejected += 1
            elif not result.detected:
                self.n_failed += 1
            self._append_log(result.reason)

        if latest is not None:
            result, image = latest
            self._update_progress_panel()
            if self.consumer.capturing.is_set():
                if result.accepted:
                    outcome = f"ACCEPTED as sample {len(self.calibrator.db)}"
                elif result.detected:
                    outcome = "rejected (too similar)"
                else:
                    outcome = "no markers detected"
                img, pts = self._apply_forward_view(image, result.corners)
                self._render_frame(img, pts, f"Live frame: {outcome}")
                self._set_status(
                    f"Live capture on '{self._subscribed_topic}': "
                    f"{len(self.calibrator.db)} accepted, {self.n_rejected} rejected, "
                    f"{self.n_failed} without detection."
                )
                # Keep browsing anchored to the newest accepted sample.
                if result.accepted:
                    self.current_sample_index = len(self.calibrator.db) - 1

        self.root.after(self.POLL_MS, self._poll_queue)

    def calibrate(self):
        # Stop collecting before solving so the sample db cannot change
        # underneath cal_fromcorners; the user can restart capture after.
        if self.consumer.capturing.is_set():
            self.toggle_capture()
        self.consumer.wait_until_idle()
        super().calibrate()


def launch_gui(
    image_dir: str = "photos",
    base_name: str = "",
    extension: str = "all",
    n_sq_x: int = 6,
    n_sq_y: int = 4,
    spacing_mm: float = 50.0,
    taylor_order: int = 4,
    output_dir: str = ".",
    slow_find_center: bool = False,
) -> None:
    """Launch the interactive batch (folder-based) calibration GUI."""
    _require_gui_deps()
    root = tk.Tk()
    BatchCalibrationApp(
        root,
        image_dir=image_dir,
        base_name=base_name,
        extension=extension,
        n_sq_x=n_sq_x,
        n_sq_y=n_sq_y,
        spacing_mm=spacing_mm,
        taylor_order=taylor_order,
        output_dir=output_dir,
        slow_find_center=slow_find_center,
    )
    root.mainloop()


def launch_live_gui(
    calibrator: Calibrator,
    result_queue,
    consumer,
    subscribe_fn,
    initial_topic: str = "image",
    output_dir: str = ".",
    slow_find_center: bool = False,
) -> None:
    """Launch the live (ROS 2 topic-driven) calibration GUI."""
    _require_gui_deps()
    root = tk.Tk()
    LiveCalibrationApp(
        root,
        calibrator=calibrator,
        result_queue=result_queue,
        consumer=consumer,
        subscribe_fn=subscribe_fn,
        initial_topic=initial_topic,
        n_sq_x=calibrator.board.n_sq_x,
        n_sq_y=calibrator.board.n_sq_y,
        spacing_mm=calibrator.board.spacing_mm,
        taylor_order=calibrator.taylor_order,
        output_dir=output_dir,
        slow_find_center=slow_find_center,
    )
    root.mainloop()
