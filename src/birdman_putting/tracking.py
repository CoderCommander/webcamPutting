"""Ball tracking state machine for shot detection."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from birdman_putting.config import BallSettings, DetectionZone, ShotSettings
from birdman_putting.detection import BallDetection
from birdman_putting.physics import pixel_to_mm_ratio

logger = logging.getLogger(__name__)


class ShotState(Enum):
    """Shot detection state machine states."""

    IDLE = "idle"                    # No ball detected
    BALL_DETECTED = "ball_detected"  # Ball found, accumulating start candidates
    STARTED = "started"              # Ball position confirmed stable in start zone
    ENTERED = "entered"              # Ball crossed into detection gateway
    LEFT = "left"                    # Ball exited detection gateway — shot complete


@dataclass
class ShotResult:
    """Data from a completed shot, ready for physics calculation."""

    start_position: tuple[int, int]
    end_position: tuple[int, int]
    start_radius: int
    entry_time: float
    exit_time: float
    px_mm_ratio: float
    positions: list[tuple[int, int, float]] = field(default_factory=list)


class BallTracker:
    """Tracks ball through the detection zone and detects complete shots.

    State machine: IDLE → BALL_DETECTED → STARTED → ENTERED → LEFT

    Improvements over original:
    - Proper enum-based state machine (replaces 3 boolean flags)
    - Position clustering with tolerance for start detection
    - Single reset() method (eliminates duplicated reset code)
    - Configurable thresholds (not hardcoded)
    """

    def __init__(
        self,
        zone: DetectionZone,
        ball_settings: BallSettings,
        shot_settings: ShotSettings,
    ):
        self.zone = zone
        self.ball_settings = ball_settings
        self.shot_settings = shot_settings

        self._state = ShotState.IDLE
        self._start_candidates: list[tuple[int, int]] = []
        self._start_circle: tuple[int, int, int] = (0, 0, 0)
        self._start_pos: tuple[int, int] = (0, 0)
        self._entry_pos: tuple[int, int] = (0, 0)
        self._entry_time: float = 0.0
        self._px_mm_ratio: float = 0.0
        self._positions: deque[tuple[int, int, float]] = deque(maxlen=64)
        self._shot_count: int = 0

        # Last shot data for UI display
        self.last_shot_speed: float = 0.0
        self.last_shot_hla: float = 0.0
        self.last_shot_start: tuple[int, int] = (0, 0)
        self.last_shot_end: tuple[int, int] = (0, 0)
        self.last_shot_positions: list[tuple[int, int]] = []

    @property
    def state(self) -> ShotState:
        return self._state

    @property
    def shot_count(self) -> int:
        return self._shot_count

    @property
    def start_circle(self) -> tuple[int, int, int]:
        return self._start_circle

    @property
    def positions(self) -> list[tuple[int, int, float]]:
        """Current tracked positions (for real-time trail display)."""
        return list(self._positions)

    @property
    def px_mm_ratio(self) -> float:
        return self._px_mm_ratio

    def reset(self) -> None:
        """Reset tracking state for next shot."""
        self._state = ShotState.IDLE
        self._start_candidates.clear()
        self._start_circle = (0, 0, 0)
        self._start_pos = (0, 0)
        self._entry_pos = (0, 0)
        self._entry_time = 0.0
        self._px_mm_ratio = 0.0
        self._positions.clear()

    @property
    def _is_rtl(self) -> bool:
        """Whether the ball rolls right-to-left."""
        return self.zone.direction == "right_to_left"

    def update(self, detection: BallDetection | None) -> ShotResult | None:
        """Process one frame's detection. Returns ShotResult when shot completes.

        Args:
            detection: Ball detection for this frame, or None if not found.

        Returns:
            ShotResult when a complete shot is detected, None otherwise.
        """
        # Compute gateway coordinates based on direction
        if self._is_rtl:
            # Gateway is to the LEFT of the start zone
            gateway_x2 = self.zone.start_x1 - self.zone.gateway_width
            gateway_x1 = gateway_x2 - self.zone.gateway_width
        else:
            # Gateway is to the RIGHT of the start zone (original behavior)
            gateway_x1 = self.zone.start_x2 + self.zone.gateway_width
            gateway_x2 = gateway_x1 + self.zone.gateway_width

        # Timeout: if we're in ENTERED state and too much time passed, reset
        if self._state == ShotState.ENTERED and self._entry_time > 0:
            elapsed = time.perf_counter() - self._entry_time
            if elapsed > self.shot_settings.min_time_seconds * 4:
                logger.debug("Timeout in ENTERED state, resetting")
                self.reset()

        if detection is None:
            return None

        x, y = detection.x, detection.y

        # --- IDLE / BALL_DETECTED: Looking for stable ball in start zone ---
        if self._state in (ShotState.IDLE, ShotState.BALL_DETECTED):
            if self.zone.start_x1 <= x <= self.zone.start_x2:
                self._state = ShotState.BALL_DETECTED
                self._start_candidates.append((x, y))

                # Keep buffer at max size
                max_candidates = self.ball_settings.start_stability_frames * 2
                if len(self._start_candidates) > max_candidates:
                    self._start_candidates.pop(0)

                # Check for stable position using clustering
                if len(self._start_candidates) >= self.ball_settings.start_stability_frames:
                    tolerance = self.ball_settings.start_position_tolerance
                    matching = sum(
                        1 for cx, cy in self._start_candidates
                        if abs(cx - x) <= tolerance and abs(cy - y) <= tolerance
                    )
                    logger.info(
                        "Stability: (%d,%d) candidates=%d matching=%d/%d tol=%d",
                        x, y, len(self._start_candidates), matching,
                        self.ball_settings.start_stability_frames, tolerance,
                    )

                    if matching >= self.ball_settings.start_stability_frames:
                        logger.info("New start found at (%d, %d) r=%d", x, y, detection.radius)
                        self._state = ShotState.STARTED
                        self._shot_count += 1
                        self._start_circle = (x, y, detection.radius)
                        self._start_pos = (x, y)
                        self._positions.clear()
                        self._positions.append((x, y, detection.timestamp))
                        self._start_candidates.clear()

                        # Calculate pixel-to-mm ratio
                        radius = self.ball_settings.fixed_radius or detection.radius
                        self._px_mm_ratio = pixel_to_mm_ratio(radius)

            return None

        # --- STARTED: Ball is stable, waiting for it to cross into gateway ---
        if self._state == ShotState.STARTED:
            # Check if ball moved into gateway
            entered = x <= gateway_x2 if self._is_rtl else x >= gateway_x1
            if entered:
                self._state = ShotState.ENTERED
                self._entry_time = detection.timestamp
                self._entry_pos = (x, y)
                self._positions.append((x, y, detection.timestamp))
                logger.info("Ball entered gateway at (%d, %d)", x, y)
            elif self.zone.start_x1 <= x <= self.zone.start_x2:
                # Ball still in start zone — check for repositioning (new stable pos)
                self._start_candidates.append((x, y))
                max_candidates = self.ball_settings.start_stability_frames * 2
                if len(self._start_candidates) > max_candidates:
                    self._start_candidates.pop(0)
                if len(self._start_candidates) >= self.ball_settings.start_stability_frames:
                    tolerance = self.ball_settings.start_position_tolerance
                    matching = sum(
                        1 for cx, cy in self._start_candidates
                        if abs(cx - x) <= tolerance and abs(cy - y) <= tolerance
                    )
                    if matching >= self.ball_settings.start_stability_frames:
                        logger.info("Re-start at (%d, %d) r=%d", x, y, detection.radius)
                        self._start_circle = (x, y, detection.radius)
                        self._start_pos = (x, y)
                        self._positions.clear()
                        self._positions.append((x, y, detection.timestamp))
                        self._start_candidates.clear()
                        radius = self.ball_settings.fixed_radius or detection.radius
                        self._px_mm_ratio = pixel_to_mm_ratio(radius)
            else:
                # Ball in transit between start zone and gateway — track, stay STARTED
                self._positions.append((x, y, detection.timestamp))

            return None

        # --- ENTERED: Ball crossed gateway, waiting for exit ---
        if self._state == ShotState.ENTERED:
            self._positions.append((x, y, detection.timestamp))

            # Check if ball has exited past the gateway with enough travel
            min_dist = self.shot_settings.min_exit_distance_px
            if self._is_rtl:
                exited = x < gateway_x1 and len(self._positions) >= 2
                if exited:
                    travel = self._entry_pos[0] - x
                    if travel >= min_dist:
                        self._state = ShotState.LEFT
                        result = ShotResult(
                            start_position=self._entry_pos,
                            end_position=(x, y),
                            start_radius=self._start_circle[2],
                            entry_time=self._entry_time,
                            exit_time=detection.timestamp,
                            px_mm_ratio=self._px_mm_ratio,
                            positions=list(self._positions),
                        )
                        logger.info("Ball left at (%d, %d), shot complete", x, y)
                        self.reset()
                        return result
            else:
                exited = x > gateway_x2 and len(self._positions) >= 2
                if exited:
                    travel = x - self._entry_pos[0]
                    if travel >= min_dist:
                        self._state = ShotState.LEFT
                        result = ShotResult(
                            start_position=self._entry_pos,
                            end_position=(x, y),
                            start_radius=self._start_circle[2],
                            entry_time=self._entry_time,
                            exit_time=detection.timestamp,
                            px_mm_ratio=self._px_mm_ratio,
                            positions=list(self._positions),
                        )
                        logger.info("Ball left at (%d, %d), shot complete", x, y)
                        self.reset()
                        return result

            return None

        return None
