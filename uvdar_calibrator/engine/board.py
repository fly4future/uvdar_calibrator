"""
Calibration target descriptor.

Analogous to ROS camera_calibration's ``ChessboardInfo``, minus the
pattern/ArUco fields: there is exactly one pattern kind here, the UV LED
grid (with a checkerboard/circle-grid detection fallback tried first for
robustness, not as a user-selectable mode).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LedGridBoard:
    """
    Geometry of the UV LED grid calibration target.

    ``n_sq_x``/``n_sq_y`` are square counts (MATLAB/OCamCalib convention);
    the actual point grid is ``(n_sq_x + 1) x (n_sq_y + 1)``.
    """

    n_sq_x: int = 6
    n_sq_y: int = 4
    spacing_mm: float = 50.0

    @property
    def n_cols(self) -> int:
        return self.n_sq_x + 1

    @property
    def n_rows(self) -> int:
        return self.n_sq_y + 1

    @property
    def n_points(self) -> int:
        return self.n_cols * self.n_rows
