"""Mevo shot detection via screenshot comparison + OCR."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from birdman_putting.config import MevoSettings
from birdman_putting.mevo.ocr import ROI, MevoOCR
from birdman_putting.mevo.screenshot import WindowCapture

logger = logging.getLogger(__name__)

# Valid ranges for shot metrics
_VALID_RANGES: dict[str, tuple[float, float]] = {
    "ball_speed": (5.0, 250.0),      # mph
    "launch_angle": (-15.0, 60.0),   # degrees VLA
    "launch_direction": (-45.0, 45.0),  # degrees HLA
    "spin_rate": (0.0, 15000.0),     # rpm
    "spin_axis": (-90.0, 90.0),      # degrees
    "club_speed": (0.0, 200.0),      # mph
}


@dataclass
class MevoShotData:
    """Full shot data from Mevo OCR."""

    ball_speed: float      # mph
    launch_angle: float    # degrees (VLA)
    launch_direction: float  # degrees (HLA)
    spin_rate: float       # rpm (total spin)
    spin_axis: float       # degrees
    club_speed: float      # mph (0 if not available)

    @property
    def back_spin(self) -> float:
        """Decompose total spin into back spin component."""
        return abs(self.spin_rate * math.cos(math.radians(self.spin_axis)))

    @property
    def side_spin(self) -> float:
        """Decompose total spin into side spin component."""
        return self.spin_rate * math.sin(math.radians(self.spin_axis))


def build_rois(roi_dict: dict[str, list[int]]) -> list[ROI]:
    """Convert config ROI dict to list of ROI objects.

    Args:
        roi_dict: Mapping of metric name to [x, y, width, height].
    """
    rois: list[ROI] = []
    for name, coords in roi_dict.items():
        if len(coords) != 4:
            logger.warning("ROI '%s' has %d coords (expected 4), skipping", name, len(coords))
            continue
        rois.append(ROI(name=name, x=coords[0], y=coords[1], width=coords[2], height=coords[3]))
    return rois


def _compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if a.shape != b.shape:
        return float("inf")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def _values_changed(
    prev: dict[str, float | None],
    curr: dict[str, float | None],
    tolerance: float = 0.1,
) -> bool:
    """Check if OCR values have changed beyond tolerance."""
    for key in curr:
        pv = prev.get(key)
        cv = curr.get(key)
        if pv is None and cv is not None:
            return True
        if pv is not None and cv is None:
            continue  # Don't trigger on lost readings
        if pv is not None and cv is not None and abs(pv - cv) > tolerance:
            return True
    return False


def _validate_metrics(metrics: dict[str, float | None]) -> bool:
    """Check that required metrics are present and within valid ranges."""
    required = ["ball_speed", "launch_angle", "launch_direction"]
    for key in required:
        value = metrics.get(key)
        if value is None:
            return False
        lo, hi = _VALID_RANGES.get(key, (float("-inf"), float("inf")))
        if not (lo <= value <= hi):
            logger.debug("Metric '%s' = %.2f out of range [%.1f, %.1f]", key, value, lo, hi)
            return False

    # Validate optional metrics if present
    for key in ("spin_rate", "spin_axis", "club_speed"):
        value = metrics.get(key)
        if value is not None:
            lo, hi = _VALID_RANGES[key]
            if not (lo <= value <= hi):
                logger.debug("Metric '%s' = %.2f out of range", key, value)
                return False
    return True


class MevoDetector:
    """Detects new Mevo shots via screenshot + OCR.

    Call ``poll()`` in a loop; it returns ``MevoShotData`` when a new shot
    is detected, or ``None`` otherwise.
    """

    def __init__(
        self,
        settings: MevoSettings,
        ocr: MevoOCR,
        capture: WindowCapture,
    ) -> None:
        self._settings = settings
        self._ocr = ocr
        self._capture = capture
        self._prev_frame: np.ndarray | None = None
        self._prev_metrics: dict[str, float | None] = {}

    def poll(self) -> MevoShotData | None:
        """Check for a new shot.

        Returns MevoShotData if a new shot is detected, None otherwise.
        """
        frame = self._capture.capture()
        if frame is None:
            return None

        # Check if display has changed
        if self._prev_frame is not None:
            mse = _compute_mse(frame, self._prev_frame)
            if mse < self._settings.mse_threshold:
                return None  # No change
            logger.debug("Display change detected (MSE=%.1f)", mse)

        self._prev_frame = frame.copy()

        # Run OCR
        metrics = self._ocr.read_metrics(frame)

        # Check if values are different from previous shot
        if self._prev_metrics and not _values_changed(self._prev_metrics, metrics):
            return None

        # Validate metrics
        if not _validate_metrics(metrics):
            logger.debug("Invalid metrics: %s", metrics)
            return None

        # Build shot data
        self._prev_metrics = metrics
        shot = MevoShotData(
            ball_speed=metrics["ball_speed"],  # type: ignore[arg-type]
            launch_angle=metrics["launch_angle"],  # type: ignore[arg-type]
            launch_direction=metrics["launch_direction"],  # type: ignore[arg-type]
            spin_rate=metrics.get("spin_rate") or 0.0,
            spin_axis=metrics.get("spin_axis") or 0.0,
            club_speed=metrics.get("club_speed") or 0.0,
        )
        logger.info(
            "New Mevo shot: %.1f mph, VLA=%.1f, HLA=%.1f, Spin=%d",
            shot.ball_speed, shot.launch_angle, shot.launch_direction, shot.spin_rate,
        )
        return shot
