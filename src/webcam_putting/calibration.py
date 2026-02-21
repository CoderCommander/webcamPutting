"""Auto-calibration: detect ball in full frame and build a detection zone around it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from webcam_putting.config import DetectionZone
from webcam_putting.detection import BallDetection

logger = logging.getLogger(__name__)

# Calibration constants
STABILITY_FRAMES = 20
STABILITY_TOLERANCE_PX = 5
TIMEOUT_FRAMES = 300  # ~5s at 60fps

# Zone margin constants
MARGIN_BEHIND = 80  # pixels behind the ball (away from roll direction)
MARGIN_AHEAD = 30   # pixels ahead of the ball (toward roll direction)
MARGIN_Y = 100       # pixels above and below


class CalibrationState(Enum):
    """Auto-calibration state machine states."""

    IDLE = "idle"
    SEARCHING = "searching"
    STABILIZING = "stabilizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class CalibrationResult:
    """Result of a successful auto-calibration."""

    zone: DetectionZone


class AutoCalibrator:
    """Detects a stationary ball and builds a DetectionZone around it.

    Feed BallDetection objects from full-frame detection. When the ball
    position stabilizes (STABILITY_FRAMES consecutive frames within
    STABILITY_TOLERANCE_PX), computes and returns a CalibrationResult.
    """

    def __init__(self, direction: str = "left_to_right") -> None:
        self._direction = direction
        self._state = CalibrationState.IDLE
        self._candidates: list[tuple[int, int]] = []
        self._frame_count: int = 0
        self._result: CalibrationResult | None = None

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def result(self) -> CalibrationResult | None:
        return self._result

    def start(self) -> None:
        """Begin calibration."""
        self._state = CalibrationState.SEARCHING
        self._candidates.clear()
        self._frame_count = 0
        self._result = None

    def cancel(self) -> None:
        """Cancel calibration."""
        self._state = CalibrationState.IDLE
        self._candidates.clear()
        self._frame_count = 0

    def update(
        self,
        detection: BallDetection | None,
        frame_width: int,
        frame_height: int,
    ) -> CalibrationResult | None:
        """Process one frame during calibration.

        Args:
            detection: Ball detection for this frame, or None if not found.
            frame_width: Width of the frame in pixels.
            frame_height: Height of the frame in pixels.

        Returns:
            CalibrationResult when calibration succeeds, None otherwise.
        """
        if self._state not in (CalibrationState.SEARCHING, CalibrationState.STABILIZING):
            return None

        self._frame_count += 1

        # Timeout check
        if self._frame_count > TIMEOUT_FRAMES:
            logger.warning("Calibration timed out after %d frames", self._frame_count)
            self._state = CalibrationState.FAILED
            return None

        if detection is None:
            # Lost the ball — stay in searching
            if self._state == CalibrationState.STABILIZING:
                self._state = CalibrationState.SEARCHING
                self._candidates.clear()
            return None

        x, y = detection.x, detection.y
        self._state = CalibrationState.STABILIZING
        self._candidates.append((x, y))

        # Keep buffer bounded
        max_buf = STABILITY_FRAMES * 2
        if len(self._candidates) > max_buf:
            self._candidates.pop(0)

        # Check stability
        if len(self._candidates) >= STABILITY_FRAMES:
            matching = sum(
                1 for cx, cy in self._candidates
                if abs(cx - x) <= STABILITY_TOLERANCE_PX
                and abs(cy - y) <= STABILITY_TOLERANCE_PX
            )

            if matching >= STABILITY_FRAMES:
                zone = self._compute_zone(x, y, frame_width, frame_height)
                self._result = CalibrationResult(zone=zone)
                self._state = CalibrationState.COMPLETE
                logger.info(
                    "Calibration complete: ball at (%d, %d), zone x=[%d, %d] y=[%d, %d]",
                    x, y, zone.start_x1, zone.start_x2, zone.y1, zone.y2,
                )
                return self._result

        return None

    def _compute_zone(
        self, ball_x: int, ball_y: int, frame_w: int, frame_h: int,
    ) -> DetectionZone:
        """Build a DetectionZone centered on the detected ball position."""
        if self._direction == "right_to_left":
            # Ball near LEFT edge of start zone; gateway goes left
            x1 = ball_x - MARGIN_AHEAD
            x2 = ball_x + MARGIN_BEHIND
        else:
            # Ball near RIGHT edge of start zone; gateway goes right
            x1 = ball_x - MARGIN_BEHIND
            x2 = ball_x + MARGIN_AHEAD

        y1 = ball_y - MARGIN_Y
        y2 = ball_y + MARGIN_Y

        # Clamp to frame bounds
        x1 = max(0, x1)
        x2 = min(frame_w, x2)
        y1 = max(0, y1)
        y2 = min(frame_h, y2)

        return DetectionZone(
            start_x1=x1,
            start_x2=x2,
            y1=y1,
            y2=y2,
            direction=self._direction,
        )
