"""
OCamCalib-style omnidirectional camera calibration for UV LED grids.

Workflow architecture mirrors ROS image_pipeline's ``camera_calibration``
package (Calibrator engine + incremental sample selection); the solver is a
faithful port of Scaramuzza's OCamCalib (ctu-mrs/OCamCalib_UVDAR fork).
"""

from .engine.board import LedGridBoard
from .engine.calibrator import Calibrator, CalibratorConfig, FrameResult, Sample
from .engine.ocam_model import OCamModel

__all__ = [
    "Calibrator",
    "CalibratorConfig",
    "FrameResult",
    "LedGridBoard",
    "OCamModel",
    "Sample",
]
