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
        max_trail_points: int = 150,
    ):
        self.ball_settings = ball_settings
        self.shot_settings = shot_settings
        self._zone = zone
        self._update_gateway_coords()

        self._state = ShotState.IDLE
        self._start_candidates: deque[tuple[int, int]] = deque(
            maxlen=ball_settings.start_stability_frames * 2,
        )
        self._start_circle: tuple[int, int, int] = (0, 0, 0)
        self._start_pos: tuple[int, int] = (0, 0)
        self._entry_pos: tuple[int, int] = (0, 0)
        self._entry_time: float = 0.0
        self._px_mm_ratio: float = 0.0
        self._positions: deque[tuple[int, int, float]] = deque(maxlen=max_trail_points)
        self._shot_count: int = 0
        self._post_shot_cooldown_until: float = 0.0

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
    def positions(self) -> deque[tuple[int, int, float]]:
        """Current tracked positions (for real-time trail display)."""
        return self._positions

    @property
    def px_mm_ratio(self) -> float:
        return self._px_mm_ratio

    def reset(self, cooldown: bool = False) -> None:
        """Reset tracking state for next shot.

        Args:
            cooldown: If True, activate post-shot cooldown to prevent
                immediate re-arm.  Only shot completions should pass True;
                manual resets and watchdog resets should use the default.
        """
        self._state = ShotState.IDLE
        self._start_candidates.clear()
        self._start_circle = (0, 0, 0)
        self._start_pos = (0, 0)
        self._entry_pos = (0, 0)
        self._entry_time = 0.0
        self._px_mm_ratio = 0.0
        self._positions.clear()
        if cooldown and self.shot_settings.post_shot_cooldown > 0:
            self._post_shot_cooldown_until = (
                time.perf_counter() + self.shot_settings.post_shot_cooldown
            )

    @property
    def zone(self) -> DetectionZone:
        return self._zone

    @zone.setter
    def zone(self, value: DetectionZone) -> None:
        self._zone = value
        self._update_gateway_coords()

    def _update_gateway_coords(self) -> None:
        """Cache gateway coordinates from zone config."""
        if self._zone.direction == "right_to_left":
            self._gateway_x2 = self._zone.start_x1 - self._zone.gateway_width
            self._gateway_x1 = self._gateway_x2 - self._zone.gateway_width
        else:
            self._gateway_x1 = self._zone.start_x2 + self._zone.gateway_width
            self._gateway_x2 = self._gateway_x1 + self._zone.gateway_width
        self._is_rtl = self._zone.direction == "right_to_left"

    def update(self, detection: BallDetection | None) -> ShotResult | None:
        """Process one frame's detection. Returns ShotResult when shot completes.

        Args:
            detection: Ball detection for this frame, or None if not found.

        Returns:
            ShotResult when a complete shot is detected, None otherwise.
        """
        gateway_x1 = self._gateway_x1
        gateway_x2 = self._gateway_x2

        # Timeout: if we're in ENTERED state and too much time passed,
        # complete the shot with what we have rather than discarding it.
        # The ball likely went off-screen (common with fast putts).
        if self._state == ShotState.ENTERED and self._entry_time > 0:
            elapsed = time.perf_counter() - self._entry_time
            if elapsed > self.shot_settings.min_time_seconds * 4:
                if len(self._positions) >= 2:
                    last = self._positions[-1]
                    result = ShotResult(
                        start_position=self._entry_pos,
                        end_position=(last[0], last[1]),
                        start_radius=self._start_circle[2],
                        entry_time=self._entry_time,
                        exit_time=last[2],
                        px_mm_ratio=self._px_mm_ratio,
                        positions=list(self._positions),
                    )
                    logger.info(
                        "ENTERED timeout — completing shot with last pos (%d, %d)",
                        last[0], last[1],
                    )
                    self.reset(cooldown=True)
                    return result
                logger.debug("Timeout in ENTERED state, resetting")
                self.reset()

        if detection is None:
            return None

        x, y = detection.x, detection.y

        # --- IDLE / BALL_DETECTED: Looking for stable ball in start zone ---
        # Post-shot cooldown: ignore detections until timer expires
        if self._state in (ShotState.IDLE, ShotState.BALL_DETECTED):
            if self._post_shot_cooldown_until > 0:
                if time.perf_counter() < self._post_shot_cooldown_until:
                    return None
                self._post_shot_cooldown_until = 0.0
            if self.zone.start_x1 <= x <= self.zone.start_x2:
                self._state = ShotState.BALL_DETECTED
                self._start_candidates.append((x, y))

                # deque maxlen handles size cap automatically

                # Check for stable position using clustering
                if len(self._start_candidates) >= self.ball_settings.start_stability_frames:
                    tolerance = self.ball_settings.start_position_tolerance
                    matching = sum(
                        1 for cx, cy in self._start_candidates
                        if abs(cx - x) <= tolerance and abs(cy - y) <= tolerance
                    )
                    logger.debug(
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
            if entered and len(self._positions) < 2:
                # Only start position recorded — ball hasn't been seen moving
                # toward the gateway.  Don't enter the gateway yet (could be
                # noise), but still record the position so that the next frame
                # can proceed once we have 2+ data points.
                self._positions.append((x, y, detection.timestamp))
                return None
            if entered:
                # Check if ball jumped FAR past the gateway in one frame
                # (e.g. x=204 → x=632).  If so, complete the shot immediately
                # instead of entering ENTERED state and waiting for an exit
                # that will never come (ball is at frame edge or off-screen).
                min_dist = self.shot_settings.min_exit_distance_px
                if self._is_rtl:
                    past_gateway = x < gateway_x1 and self._start_pos[0] - x >= min_dist
                else:
                    past_gateway = x > gateway_x2 and x - self._start_pos[0] >= min_dist
                if past_gateway and len(self._positions) >= 2:
                    # Ball skipped the gateway entirely — complete shot now.
                    # Use the start position (where ball was at rest) as entry,
                    # not the second-to-last tracked position (which may be
                    # the same jumped position from the previous frame).
                    self._positions.append((x, y, detection.timestamp))
                    self._state = ShotState.LEFT
                    result = ShotResult(
                        start_position=self._start_pos,
                        end_position=(x, y),
                        start_radius=self._start_circle[2],
                        entry_time=self._positions[0][2],
                        exit_time=detection.timestamp,
                        px_mm_ratio=self._px_mm_ratio,
                        positions=list(self._positions),
                    )
                    logger.info(
                        "Ball jumped past gateway to (%d, %d), start=(%d,%d), shot complete",
                        x, y, *self._start_pos,
                    )
                    self.reset(cooldown=True)
                    return result

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
            # Continuity check: reject detections that contradict real
            # ball motion.  Once the real ball exits the frame, the
            # detector often latches onto orange noise at random
            # positions — we filter these by direction and Y drift.
            # We do NOT reject large X jumps because fast putts can
            # legitimately skip 200+px per frame.
            if self._positions:
                last_x, last_y, _ = self._positions[-1]
                dx = x - last_x
                dy = y - last_y

                # Reject positions moving against the putt direction.
                # Tolerance scales with time gap — a slow putt detected
                # every 0.5s can have a larger dx from natural roll
                # wobble than one detected every 0.016s.  Base tolerance
                # is 30px; we also accept larger backward motion if the
                # ball is still near the start (hasn't traveled far yet).
                start_x = self._start_pos[0]
                traveled = abs(x - start_x)
                backward_tol = 30
                if self._is_rtl and dx > backward_tol:
                    logger.debug(
                        "ENTERED: rejecting backward motion (%d,%d)->(%d,%d)",
                        last_x, last_y, x, y,
                    )
                    return None
                if not self._is_rtl and dx < -backward_tol:
                    logger.debug(
                        "ENTERED: rejecting backward motion (%d,%d)->(%d,%d)",
                        last_x, last_y, x, y,
                    )
                    return None

                # Reject large Y deviations — real ball rolls mostly
                # horizontally during a putt, but a slow/curving putt
                # with sparse detection can legitimately drift 80+px
                # vertically over the full roll.  Threshold scales with
                # how far the ball has traveled: 60px for close-in
                # jumps, up to 100px once ball is far from start.
                y_tol = 60 + min(40, traveled // 10)
                if abs(dy) > y_tol:
                    logger.debug(
                        "ENTERED: rejecting large Y drift (%d,%d)->(%d,%d) tol=%d",
                        last_x, last_y, x, y, y_tol,
                    )
                    return None

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
                        self.reset(cooldown=True)
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
                        self.reset(cooldown=True)
                        return result

            return None

        return None
