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
    "ball_speed": (5.0, 250.0),        # mph
    "launch_angle": (-15.0, 60.0),     # degrees VLA
    "launch_direction": (-45.0, 45.0), # degrees HLA
    "spin_rate": (0.0, 15000.0),       # rpm
    "spin_axis": (-90.0, 90.0),        # degrees
    "club_speed": (0.0, 200.0),        # mph
    "smash_factor": (0.5, 2.0),
    "carry_distance": (0.0, 400.0),    # yards
    "total_distance": (0.0, 450.0),    # yards
    "apex_height": (0.0, 200.0),       # yards
    "flight_time": (0.0, 15.0),        # seconds
    "descent_angle": (0.0, 90.0),      # degrees
    "curve": (-100.0, 100.0),          # yards
    "roll_distance": (0.0, 100.0),     # yards
}


@dataclass
class MevoShotData:
    """Full shot data from Mevo OCR."""

    # Core (required)
    ball_speed: float      # mph
    launch_angle: float    # degrees (VLA)
    launch_direction: float  # degrees (HLA)

    # Standard optional
    spin_rate: float = 0.0       # rpm (total spin)
    spin_axis: float = 0.0       # degrees
    club_speed: float = 0.0      # mph (0 if not available)

    # Premium optional metrics
    smash_factor: float = 0.0
    carry_distance: float = 0.0  # yards
    total_distance: float = 0.0  # yards
    apex_height: float = 0.0     # yards
    flight_time: float = 0.0     # seconds
    descent_angle: float = 0.0   # degrees
    curve: float = 0.0           # yards
    roll_distance: float = 0.0   # yards

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
    for key in _VALID_RANGES:
        if key in required:
            continue
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
        self._prev_crops: dict[str, np.ndarray] | None = None
        self._prev_metrics: dict[str, float | None] = {}
        self._baseline_captured: bool = False

    def _compute_roi_mse(self, frame: np.ndarray) -> float:
        """Compute MSE over ROI regions only (faster than full-frame)."""
        if self._prev_crops is None:
            return float("inf")
        total_mse = 0.0
        total_pixels = 0
        for roi in self._ocr._rois:
            crop = MevoOCR._crop_roi(frame, roi)
            prev = self._prev_crops.get(roi.name)
            if prev is None or crop.shape != prev.shape:
                return float("inf")
            diff = crop.astype(np.float64) - prev.astype(np.float64)
            total_mse += float(np.sum(diff * diff))
            total_pixels += crop.size
        return total_mse / total_pixels if total_pixels > 0 else float("inf")

    def _store_crops(self, frame: np.ndarray) -> None:
        """Store cropped ROI regions for next comparison."""
        self._prev_crops = {
            roi.name: MevoOCR._crop_roi(frame, roi).copy()
            for roi in self._ocr._rois
        }

    def poll(self) -> MevoShotData | None:
        """Check for a new shot.

        Returns MevoShotData if a new shot is detected, None otherwise.
        """
        frame = self._capture.capture()
        if frame is None:
            return None

        # Check if display has changed (ROI-only for speed)
        if self._prev_crops is not None:
            mse = self._compute_roi_mse(frame)
            if mse < self._settings.mse_threshold:
                return None  # No change
            logger.debug("Display change detected (MSE=%.1f)", mse)
        elif self._prev_frame is not None:
            mse = _compute_mse(frame, self._prev_frame)
            if mse < self._settings.mse_threshold:
                return None  # No change
            logger.debug("Display change detected (MSE=%.1f)", mse)

        self._prev_frame = frame.copy()
        self._store_crops(frame)

        # Run OCR
        metrics = self._ocr.read_metrics(frame)

        # First poll captures baseline — don't fire stale data as a shot
        if not self._baseline_captured:
            self._baseline_captured = True
            if _validate_metrics(metrics):
                self._prev_metrics = metrics
                logger.info(
                    "Mevo baseline captured (stale): %.1f mph — waiting for new shot",
                    metrics.get("ball_speed", 0) or 0,
                )
            return None

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
            smash_factor=metrics.get("smash_factor") or 0.0,
            carry_distance=metrics.get("carry_distance") or 0.0,
            total_distance=metrics.get("total_distance") or 0.0,
            apex_height=metrics.get("apex_height") or 0.0,
            flight_time=metrics.get("flight_time") or 0.0,
            descent_angle=metrics.get("descent_angle") or 0.0,
            curve=metrics.get("curve") or 0.0,
            roll_distance=metrics.get("roll_distance") or 0.0,
        )
        logger.info(
            "New Mevo shot: %.1f mph, VLA=%.1f, HLA=%.1f, Spin=%d",
            shot.ball_speed, shot.launch_angle, shot.launch_direction, shot.spin_rate,
        )
        return shot
