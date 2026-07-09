"""Tkinter calibration GUI.

Feeds photos one-by-one into :class:`~uvdar_calibrator.calibrator.Calibrator`
(simulating the "frames arrive one at a time" model of ROS
camera_calibration) and shows:

- a live log of per-image accept/reject decisions,
- four ROS-style range progress bars (X, Y, Size, Skew), each drawn as a
  track with a colored segment from the min to the max accepted parameter
  value (green when that axis has enough variation),
- the supplementary bin-based "next images to capture" hints, driven off
  the accepted samples.

The CALIBRATE gating follows ``calibrator.goodenough``; SAVE/EXPORT follows
``calibrator.calibrated``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import coverage
from .board import LedGridBoard
from .calibrator import Calibrator
from .detection import find_image_files, read_image_gray

try:
    import cv2
except Exception:
    cv2 = None


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
    """Launch the interactive calibration GUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError("Tkinter is required for --gui. Install python-tk/tkinter.") from exc

    if cv2 is None:
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python")

    class UVDARCalibrationGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("UV-DAR / OCamCalib Calibration Assistant")
            self.root.geometry("1180x760")
            self.root.minsize(1000, 650)

            self.image_dir = tk.StringVar(value=image_dir)
            self.base_name = tk.StringVar(value=base_name)
            self.extension = tk.StringVar(value=extension)
            self.n_sq_x = tk.IntVar(value=n_sq_x)
            self.n_sq_y = tk.IntVar(value=n_sq_y)
            self.spacing_mm = tk.DoubleVar(value=spacing_mm)
            self.taylor_order = tk.IntVar(value=taylor_order)
            self.output_dir = tk.StringVar(value=output_dir)
            self.slow_find_center = tk.BooleanVar(value=slow_find_center)
            self.no_plots = tk.BooleanVar(value=True)

            self.calibrator: Calibrator | None = None
            self.current_sample_index = 0
            self.photo_ref = None

            self._build_widgets()
            self._set_status("Choose a folder and click Load / Analyze Images.")

        # --------------------------------------------------------------
        # Widgets
        # --------------------------------------------------------------

        def _build_widgets(self):
            top = ttk.Frame(self.root, padding=8)
            top.pack(side=tk.TOP, fill=tk.X)

            ttk.Label(top, text="Image folder:").grid(row=0, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.image_dir, width=45).grid(row=0, column=1, sticky="ew", padx=4)
            ttk.Button(top, text="Browse", command=self._browse_images).grid(row=0, column=2, padx=4)

            ttk.Label(top, text="Base:").grid(row=0, column=3, sticky="e")
            ttk.Entry(top, textvariable=self.base_name, width=8).grid(row=0, column=4, padx=4)
            ttk.Label(top, text="Ext:").grid(row=0, column=5, sticky="e")
            ttk.Entry(top, textvariable=self.extension, width=7).grid(row=0, column=6, padx=4)
            ttk.Button(top, text="Load / Analyze Images", command=self.load_images).grid(row=0, column=7, padx=8)
            top.columnconfigure(1, weight=1)

            opts = ttk.Frame(self.root, padding=(8, 0, 8, 6))
            opts.pack(side=tk.TOP, fill=tk.X)
            ttk.Label(opts, text="Grid squares X:").pack(side=tk.LEFT)
            ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_x, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Y:").pack(side=tk.LEFT)
            ttk.Spinbox(opts, from_=1, to=30, textvariable=self.n_sq_y, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Spacing mm:").pack(side=tk.LEFT, padx=(12, 0))
            ttk.Entry(opts, textvariable=self.spacing_mm, width=7).pack(side=tk.LEFT, padx=4)
            ttk.Label(opts, text="Taylor:").pack(side=tk.LEFT, padx=(12, 0))
            ttk.Spinbox(opts, from_=4, to=10, textvariable=self.taylor_order, width=5).pack(side=tk.LEFT, padx=4)
            ttk.Checkbutton(opts, text="MATLAB-style slow Find Center", variable=self.slow_find_center).pack(side=tk.LEFT, padx=12)
            ttk.Checkbutton(opts, text="No plots after calibration", variable=self.no_plots).pack(side=tk.LEFT, padx=8)

            main = ttk.Frame(self.root, padding=8)
            main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            left = ttk.Frame(main)
            left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            right = ttk.Frame(main, width=340)
            right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

            self.image_canvas = tk.Canvas(left, bg="#202020", highlightthickness=1, highlightbackground="#888")
            self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            nav = ttk.Frame(left, padding=(0, 8, 0, 0))
            nav.pack(side=tk.BOTTOM, fill=tk.X)
            ttk.Button(nav, text="Previous", command=self.prev_sample).pack(side=tk.LEFT)
            ttk.Button(nav, text="Next", command=self.next_sample).pack(side=tk.LEFT, padx=4)
            self.image_label = ttk.Label(nav, text="No accepted sample loaded")
            self.image_label.pack(side=tk.LEFT, padx=12)

            title = ttk.Label(right, text="Calibration Progress", font=("Segoe UI", 14, "bold"))
            title.pack(anchor="w")
            self.status_label = ttk.Label(right, text="Status: no images analyzed", wraplength=320)
            self.status_label.pack(anchor="w", pady=(8, 8))

            # Four ROS-style range bars: X, Y, Size, Skew.
            self.bar_canvas = tk.Canvas(right, width=320, height=140, bg="white", highlightthickness=1, highlightbackground="#ccc")
            self.bar_canvas.pack(anchor="w", pady=(0, 10))
            self._draw_bars([])

            ttk.Label(right, text="Sample log:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            self.log_box = tk.Text(right, height=10, width=44, wrap="word")
            self.log_box.pack(anchor="w", fill=tk.BOTH, expand=True, pady=(4, 10))
            self.log_box.configure(state="disabled")

            ttk.Label(right, text="Next images to capture:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            self.suggestion_box = tk.Text(right, height=6, width=44, wrap="word")
            self.suggestion_box.pack(anchor="w", fill=tk.X, pady=(4, 10))
            self.suggestion_box.configure(state="disabled")

            actions = ttk.Frame(right)
            actions.pack(side=tk.BOTTOM, fill=tk.X)
            self.calibrate_button = ttk.Button(actions, text="CALIBRATE", command=self.calibrate, state="disabled")
            self.calibrate_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            self.save_button = ttk.Button(actions, text="SAVE / EXPORT", command=self.save_and_export, state="disabled")
            self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Button(actions, text="Save Report", command=self.save_report).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Button(actions, text="Exit", command=self.root.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

            self.bottom_status = ttk.Label(self.root, text="", relief=tk.SUNKEN, anchor="w", padding=4)
            self.bottom_status.pack(side=tk.BOTTOM, fill=tk.X)

        def _browse_images(self):
            folder = filedialog.askdirectory(initialdir=self.image_dir.get() or ".")
            if folder:
                self.image_dir.set(folder)

        def _set_status(self, text):
            self.bottom_status.configure(text=text)
            self.root.update_idletasks()

        def _append_log(self, line):
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
            self.root.update_idletasks()

        def _write_text(self, widget, text):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

        # --------------------------------------------------------------
        # Incremental load: one handle_frame per photo
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Progress panel
        # --------------------------------------------------------------

        def _update_progress_panel(self):
            cal = self.calibrator
            if cal is None:
                return

            goodenough, progress = coverage.compute_goodenough(
                cal.db_params(), cal.param_ranges, cal.min_db_size
            )
            self._draw_bars(progress)

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
            """Red -> yellow -> green by progress, green when complete."""
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
            """Draw the four ROS-style range bars.

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

        # --------------------------------------------------------------
        # Sample browsing
        # --------------------------------------------------------------

        def _show_current_sample(self):
            cal = self.calibrator
            if cal is None or not cal.db:
                self.image_label.configure(text="No accepted sample loaded")
                return

            self.current_sample_index %= len(cal.db)
            sample = cal.db[self.current_sample_index]

            img = sample.image
            if img.ndim == 2:
                preview = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                preview = img.copy()

            for idx, (row, col) in enumerate(sample.corners, start=1):
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
            import tkinter as tk
            self.photo_ref = tk.PhotoImage(data=buf.tobytes())
            self.image_canvas.delete("all")
            self.image_canvas.create_image(cw // 2, ch // 2, image=self.photo_ref, anchor="center")

            name = Path(sample.image_path).name
            p_str = ", ".join(f"{v:.2f}" for v in sample.params)
            self.image_label.configure(
                text=f"Sample {self.current_sample_index + 1}/{len(cal.db)}: {name}   p=[{p_str}]"
            )

        def next_sample(self):
            if self.calibrator and self.calibrator.db:
                self.current_sample_index = (self.current_sample_index + 1) % len(self.calibrator.db)
                self._show_current_sample()

        def prev_sample(self):
            if self.calibrator and self.calibrator.db:
                self.current_sample_index = (self.current_sample_index - 1) % len(self.calibrator.db)
                self._show_current_sample()

        # --------------------------------------------------------------
        # Actions
        # --------------------------------------------------------------

        def calibrate(self):
            cal = self.calibrator
            if cal is None or not cal.db:
                messagebox.showinfo("No samples", "Load/analyze images first.")
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
                    from . import plots

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
                messagebox.showinfo("No report", "Load/analyze images first.")
                return
            out = Path(self.output_dir.get() or ".")
            out.mkdir(parents=True, exist_ok=True)
            path = out / "calibration_coverage.txt"
            path.write_text(cal.report() + "\n", encoding="utf-8")
            messagebox.showinfo("Saved", f"Saved readiness report to:\n{path.resolve()}")

    root = tk.Tk()
    app = UVDARCalibrationGUI(root)
    root.bind("<Left>", lambda _e: app.prev_sample())
    root.bind("<Right>", lambda _e: app.next_sample())
    root.mainloop()
