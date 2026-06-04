"""Ball tracking state machine for shot detection."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from birdman_putting.config import BallSettings, DetectionZone, ShotSettings
from birdman_putting.detection import BallDetection
from birdman_putting.physics import pixel_to_mm_ratio

logger = logging.getLogger(__name__)

# --- ENTERED-state forward-velocity sanity ---------------------------------
# Conversions used to turn a configured max putt speed (MPH) into a per-frame
# pixel-displacement cap, so a noise re-lock far ahead of the ball (implying a
# hundreds-of-MPH velocity) is rejected without ever rejecting a legitimate
# fast putt.
_MPH_TO_FPS = 5280.0 / 3600.0   # 1.4667 ft/s per MPH
_MM_PER_FOOT = 304.8            # exact
# Multiplier on shot_settings.max_speed_mph for the rejection threshold.  We
# want the cap to sit WELL above any real putt (which tops out ~20-25 MPH), so
# only physically-impossible jumps are dropped.  3x → ~75 MPH ceiling.
_ENTERED_MAX_SPEED_FACTOR = 3.0
# Fallback per-frame pixel cap used only when pixels-per-foot can't be derived
# (no px/mm ratio yet).  Derived from a generous max speed at a typical wide
# scale: a putt physically cannot cross most of the frame in one 60fps frame.
# 75 MPH ≈ 110 ft/s; at a low ~40 px/ft that's ~4400 px/s ≈ 73 px per 16ms.
# We use a flat, very generous 1500 px/frame so this never trips a real putt.
_ENTERED_FALLBACK_MAX_PX_PER_FRAME = 1500.0


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
        # Full-traversal tracking: once exit threshold is met, keep
        # accumulating positions until the ball is no longer detected
        # (or ENTERED-state timeout fires).  This gives the trajectory
        # fitter many more data points for a robust launch-velocity
        # estimate, instead of stopping at exit_x + min_exit_distance.
        self._exit_threshold_met: bool = False
        self._no_detection_streak: int = 0

        # Public "last meaningful activity" timestamp (perf_counter base).
        # Updated ONLY on genuine progress: state-machine transitions and
        # genuine re-starts (a new resting position).  A stationary ball
        # whose re-starts are suppressed does NOT bump this, so an external
        # watchdog (app.py) can measure time-since-real-activity and reset a
        # ball that is stuck in STARTED.  Starts at 0.0 (no activity yet).
        self.last_activity_time: float = 0.0
        # Set by _update_impl when a genuine (non-suppressed) re-start occurs,
        # so the update() wrapper bumps last_activity_time even though a
        # re-start is not a formal state transition.
        self._activity_bumped: bool = False

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
        self._exit_threshold_met = False
        self._no_detection_streak = 0
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

        Thin wrapper around :meth:`_update_impl` that maintains the public
        ``last_activity_time`` timestamp.  Activity advances on ANY state
        transition and on a genuine re-start (signalled by the inner method
        via ``self._activity_bumped``).  A stationary ball whose re-starts are
        suppressed produces neither, so ``last_activity_time`` stays frozen —
        letting the app watchdog reset a truly stuck ball.

        Args:
            detection: Ball detection for this frame, or None if not found.

        Returns:
            ShotResult when a complete shot is detected, None otherwise.
        """
        prev_state = self._state
        self._activity_bumped = False
        result = self._update_impl(detection)
        if self._state != prev_state or self._activity_bumped:
            # Use the detection's timestamp when available (same perf_counter
            # base the app watchdog compares against); otherwise fall back to
            # a fresh perf_counter() reading.
            self.last_activity_time = (
                detection.timestamp if detection is not None else time.perf_counter()
            )
        return result

    def _update_impl(self, detection: BallDetection | None) -> ShotResult | None:
        """Core state-machine step (see :meth:`update`)."""
        gateway_x1 = self._gateway_x1
        gateway_x2 = self._gateway_x2

        # Timeout / completion checks for ENTERED state.
        # We have two completion paths:
        #   1. Ball lost after exit threshold met — most common; ball
        #      rolled off-frame.  Complete after 3 consecutive None
        #      detections, ~50ms at 60fps.  Gives us all the trajectory
        #      data the camera could see.
        #   2. Hard timeout at 4× min_time_seconds — fallback for stuck
        #      states (e.g., ball stops in frame and detector keeps
        #      seeing it).
        if self._state == ShotState.ENTERED and self._entry_time > 0:
            elapsed = time.perf_counter() - self._entry_time

            # (1) Detection lost after exit threshold was met
            if (detection is None
                    and self._exit_threshold_met
                    and len(self._positions) >= 2):
                self._no_detection_streak += 1
                if self._no_detection_streak >= 3:
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
                        "Ball left frame after exit — shot complete with "
                        "%d trail points, last pos (%d, %d)",
                        len(self._positions), last[0], last[1],
                    )
                    self.reset(cooldown=True)
                    return result

            # (2) Hard timeout — ball never reached exit OR tracker stuck
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
                # Ball is INSIDE start zone.  Two sub-cases that need
                # different handling:
                #
                #  (a) Ball is currently in motion (frame-to-frame
                #      displacement > threshold) — this is the launch
                #      phase of a putt traversing the start zone.
                #      Append the position so launch-velocity can read
                #      it.  Without this, we miss the highest-velocity
                #      portion of the trail.
                #
                #  (b) Ball is stationary (or nearly so) at its current
                #      location — could be the original rest, or the
                #      user repositioning to a new spot in the zone.
                #      Run the stability check; on convergence, treat
                #      it as a new resting position (re-start).
                # Use a tight motion threshold so we catch the FIRST
                # frame of slow-stroke putts.  A higher threshold (e.g. 6
                # px) misses the 1-2 launch frames of slow strokes,
                # causing the window to start at already-decelerated
                # motion and under-read by ~30%.
                _IN_ZONE_FRAME_MOTION_THRESHOLD = 3  # px frame-to-frame

                # Use the LAST APPENDED POSITION (or rest_pos as fallback)
                # for frame-to-frame motion calc — NOT _start_candidates,
                # which we may have just cleared during motion.  Without
                # this dedicated reference, alternating frames get dropped:
                # frame-A appends + clears candidates, frame-B sees empty
                # candidates → frame_motion=0 → not appended, frame-C sees
                # candidate from B → appended, etc.
                if self._positions:
                    last_x, last_y, _ = self._positions[-1]
                else:
                    last_x, last_y = self._start_pos
                frame_motion = math.hypot(x - last_x, y - last_y)

                if frame_motion > _IN_ZONE_FRAME_MOTION_THRESHOLD:
                    # Active motion.  If this is the first motion frame,
                    # pull in the immediately-preceding stability candidate
                    # too — it may have been just-under-threshold motion
                    # rather than rest.  This gives launch-velocity an
                    # extra anchor frame that's closer to motion onset.
                    if (len(self._positions) == 1
                            and self._start_candidates):
                        prev_cx, prev_cy = self._start_candidates[-1]
                        # Only include if it's between rest and current —
                        # i.e. it was already-moving (not noise jitter)
                        rest_x, rest_y = self._start_pos
                        prev_dist = math.hypot(prev_cx - rest_x, prev_cy - rest_y)
                        if prev_dist > 1.5:  # 1+ px of motion from rest
                            # Use a timestamp halfway between rest and now
                            # — we don't have the exact prev timestamp, but
                            # half the current dt is a reasonable estimate
                            prev_t = (self._positions[0][2] + detection.timestamp) / 2
                            self._positions.append((int(prev_cx), int(prev_cy), prev_t))
                    self._positions.append((x, y, detection.timestamp))
                    self._start_candidates.clear()
                else:
                    # Stationary / settling — usual stability / re-start logic
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
                            # Suppress redundant re-starts: if the converged
                            # position is within tolerance of the CURRENT
                            # start, the ball simply never moved — this is the
                            # same rest, not a new one.  Re-emitting it floods
                            # the log with hundreds of "Re-start" lines and
                            # (because a re-start is not a state change) defeats
                            # the app watchdog, which then fires the
                            # "stuck in started >10s" auto-reset loop.  Reset
                            # the stability window so we go quiet, but do NOT
                            # log, re-emit, or bump activity.
                            same_rest = (
                                abs(x - self._start_pos[0]) <= tolerance
                                and abs(y - self._start_pos[1]) <= tolerance
                            )
                            self._start_candidates.clear()
                            if not same_rest:
                                logger.info(
                                    "Re-start at (%d, %d) r=%d", x, y, detection.radius
                                )
                                self._start_circle = (x, y, detection.radius)
                                self._start_pos = (x, y)
                                self._positions.clear()
                                self._positions.append((x, y, detection.timestamp))
                                radius = self.ball_settings.fixed_radius or detection.radius
                                self._px_mm_ratio = pixel_to_mm_ratio(radius)
                                # Genuine re-start = real activity; bump the
                                # watchdog timestamp via the update() wrapper.
                                self._activity_bumped = True
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
                last_x, last_y, last_t = self._positions[-1]
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

                # Forward-velocity sanity: reject a detection whose implied
                # frame-to-frame speed exceeds the max PLAUSIBLE putt speed.
                # Once the real ball exits the frame, the detector latches onto
                # noise blobs at random positions — some land far AHEAD with
                # near-zero Y drift (so the backward/Y checks above pass) yet
                # imply hundreds of MPH.  Appending them pollutes the
                # trajectory fit.  The cap is set generously (3x max_speed_mph)
                # so legitimate fast putts are never rejected.
                dt = detection.timestamp - last_t
                disp_px = math.hypot(dx, dy)
                max_mph = self.shot_settings.max_speed_mph * _ENTERED_MAX_SPEED_FACTOR
                ppf = self._px_mm_ratio * _MM_PER_FOOT  # px/mm → px/ft
                if ppf > 0 and dt > 0:
                    max_px = (max_mph * _MPH_TO_FPS) * ppf * dt
                elif dt > 0:
                    # No scale available — fall back to a flat, very generous
                    # per-frame pixel cap (documented assumption: a real putt
                    # cannot cross most of the frame in a single frame).
                    max_px = _ENTERED_FALLBACK_MAX_PX_PER_FRAME
                else:
                    max_px = float("inf")  # dt unknown → cannot judge; allow
                if disp_px > max_px:
                    logger.debug(
                        "ENTERED: rejecting impossible velocity "
                        "(%d,%d)->(%d,%d) disp=%.0fpx dt=%.4fs cap=%.0fpx",
                        last_x, last_y, x, y, disp_px, dt, max_px,
                    )
                    return None

            self._positions.append((x, y, detection.timestamp))
            # Reset no-detection counter — we just got a fresh detection
            self._no_detection_streak = 0

            # Mark exit threshold once met (ball past gateway with min
            # travel).  We DON'T complete the shot here — we keep
            # tracking until the ball is no longer detected (handled in
            # the timeout block at the top of update()).  This gives
            # the trajectory fitter all visible motion frames, not just
            # the early portion.
            min_dist = self.shot_settings.min_exit_distance_px
            if self._is_rtl:
                exit_met = (x < gateway_x1
                            and (self._entry_pos[0] - x) >= min_dist
                            and len(self._positions) >= 2)
            else:
                exit_met = (x > gateway_x2
                            and (x - self._entry_pos[0]) >= min_dist
                            and len(self._positions) >= 2)
            if exit_met and not self._exit_threshold_met:
                self._exit_threshold_met = True
                logger.debug(
                    "ENTERED: exit threshold met at (%d,%d), continuing "
                    "for trajectory data", x, y,
                )

            return None

        return None
