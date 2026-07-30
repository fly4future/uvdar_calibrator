"""
One Tk window holding every calibration diagnostic as a tab.

The diagnostics used to arrive as N + 3 separate blocking matplotlib windows
-- one per accepted sample plus error/projection/extrinsics -- so a 23-sample
calibration meant closing 26 windows by hand. :mod:`plots` now builds bare
figures and this module shows them together.

Deliberately Tk-only, with no headless fallback: interactive matplotlib
already needed Tk in practice, so requiring it here loses nothing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk


def show_diagnostics(
    figures: Sequence[Tuple[str, Figure]],
    parent: Optional[tk.Misc] = None,
    title: str = "Calibration diagnostics",
) -> Optional[tk.Misc]:
    """
    Show ``(tab title, Figure)`` pairs as tabs of a single window.

    With ``parent`` given the window is a ``Toplevel``, so the caller's GUI
    keeps running and this returns immediately; without one it owns a fresh
    ``Tk`` root and blocks in ``mainloop`` until closed, which is what the CLI
    wants. Each tab carries the standard matplotlib toolbar -- that is what
    makes a 23-panel reprojection grid readable, since you can zoom into one
    panel instead of squinting at all of them.
    """
    if not figures:
        return None

    owns_root = parent is None
    window: tk.Misc = tk.Tk() if owns_root else tk.Toplevel(parent)
    window.title(title)
    window.geometry("1200x900")

    notebook = ttk.Notebook(window)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Keep canvas references alive: the tab frames hold the widgets, but the
    # FigureCanvasTkAgg objects themselves would otherwise be collectable.
    canvases: List[FigureCanvasTkAgg] = []

    for tab_title, figure in figures:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=tab_title)

        canvas = FigureCanvasTkAgg(figure, master=tab)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()

        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        canvases.append(canvas)

    window._uvdar_canvases = canvases  # type: ignore[attr-defined]

    if owns_root:
        window.mainloop()
        return None

    return window
