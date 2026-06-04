"""Tests for ball tracking state machine."""

import time

import pytest

from birdman_putting.config import BallSettings, DetectionZone, ShotSettings
from birdman_putting.detection import BallDetection
from birdman_putting.tracking import BallTracker, ShotState


def _det(x: int, y: int, t: float, r: int = 15) -> BallDetection:
    """Shorthand for creating a BallDetection."""
    return BallDetection(x=x, y=y, radius=r, contour_area=100.0, timestamp=t)


@pytest.fixture
def tracker(detection_zone, ball_settings, shot_settings) -> BallTracker:
    """Tracker with fast stabilization for tests."""
    return BallTracker(
        zone=detection_zone,
        ball_settings=ball_settings,
        shot_settings=shot_settings,
    )


class TestStateTransitions:
    def test_starts_idle(self, tracker: BallTracker):
        assert tracker.state == ShotState.IDLE

    def test_ball_detected_in_start_zone(self, tracker: BallTracker):
        """Ball in start zone transitions to BALL_DETECTED."""
        tracker.update(_det(50, 300, time.perf_counter()))
        assert tracker.state == ShotState.BALL_DETECTED

    def test_stable_ball_transitions_to_started(self, tracker: BallTracker):
        """Repeated stable detections transition to STARTED."""
        t = time.perf_counter()
        for i in range(10):
            result = tracker.update(_det(50, 300, t + i * 0.016))

        assert tracker.state == ShotState.STARTED
        assert result is None  # No shot yet

    def test_ball_entering_gateway(self, tracker: BallTracker):
        """Ball crossing gateway transitions to ENTERED."""
        t = time.perf_counter()
        # First stabilize
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))

        assert tracker.state == ShotState.STARTED

        # Ball moves through transit zone then past gateway (start_x2=180, gateway at 195)
        tracker.update(_det(185, 300, t + 0.4))  # transit between start zone and gateway
        tracker.update(_det(195, 300, t + 0.5))
        assert tracker.state == ShotState.ENTERED

    def test_full_shot_cycle(self, tracker: BallTracker):
        """Full shot: stable -> entered -> exited with enough distance."""
        zone = tracker.zone
        t = time.perf_counter()

        # Stabilize in start zone
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))

        # Transit through zone then enter gateway
        gateway_x1 = zone.start_x2 + zone.gateway_width
        tracker.update(_det(zone.start_x2 + 5, 300, t + 0.4))  # transit
        tracker.update(_det(gateway_x1 + 5, 300, t + 0.5))
        assert tracker.state == ShotState.ENTERED

        # Exit well past gateway (min_exit_distance_px=50).  With the
        # full-traversal change, shot doesn't complete on the exit
        # frame — it completes when detection is lost (ball leaves
        # frame).  Send the exit-frame detection, then 3 None frames.
        gateway_x2 = gateway_x1 + zone.gateway_width
        tracker.update(_det(gateway_x2 + 200, 295, t + 0.6))
        # Ball has now left the frame — feed Nones to trigger completion
        result = None
        for i in range(3):
            result = tracker.update(None)

        assert result is not None
        assert result.start_position[0] == gateway_x1 + 5
        assert result.end_position[0] == gateway_x2 + 200

    def test_none_detection_preserves_state(self, tracker: BallTracker):
        """None detection should not crash or change state."""
        result = tracker.update(None)
        assert result is None
        assert tracker.state == ShotState.IDLE

    def test_reset_clears_state(self, tracker: BallTracker):
        t = time.perf_counter()
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))

        assert tracker.state == ShotState.STARTED
        tracker.reset()
        assert tracker.state == ShotState.IDLE

    def test_ball_outside_start_zone_stays_idle(self, tracker: BallTracker):
        """Ball detected outside start zone should not trigger state change."""
        tracker.update(_det(500, 300, time.perf_counter()))
        assert tracker.state == ShotState.IDLE


class TestShotCounting:
    def test_shot_count_increments(self, tracker: BallTracker):
        assert tracker.shot_count == 0

        t = time.perf_counter()
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))

        assert tracker.shot_count == 1


class TestGradualMovement:
    """Tests for gradual ball movement — the bug that killed every shot."""

    def test_gradual_ltr_shot_completes(self, tracker: BallTracker) -> None:
        """Ball rolling gradually (5px steps) through gateway must complete."""
        t = time.perf_counter()

        # Stabilize at x=50
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        # Roll gradually from x=50 toward gateway (start_x2=180, gateway_x1=195)
        # This crosses >10px from start_pos, which used to reset to BALL_DETECTED
        x = 50
        frame = 10
        while x < 200:
            x += 5
            tracker.update(_det(x, 300, t + frame * 0.016))
            frame += 1
            # Must never reset to BALL_DETECTED or IDLE
            assert tracker.state in (ShotState.STARTED, ShotState.ENTERED), (
                f"State reset to {tracker.state} at x={x}"
            )

        # Should have entered gateway
        assert tracker.state == ShotState.ENTERED

        # Exit past gateway (gateway_x2=210, need 50px travel past entry).
        # Send exit detection, then 3 None frames for completion.
        tracker.update(_det(300, 298, t + frame * 0.016))
        result = None
        for _ in range(3):
            result = tracker.update(None)
        assert result is not None
        assert result.end_position[0] == 300

    def test_gradual_rtl_shot_completes(self) -> None:
        """RTL ball rolling gradually must complete."""
        zone = DetectionZone(
            start_x1=400, start_x2=570, y1=180, y2=450,
            direction="right_to_left",
        )
        tracker = BallTracker(
            zone=zone,
            ball_settings=BallSettings(start_stability_frames=3, start_position_tolerance=5),
            shot_settings=ShotSettings(),
        )
        t = time.perf_counter()

        # Stabilize at x=500
        for i in range(10):
            tracker.update(_det(500, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        # Roll gradually left: gateway_x2 = 400 - 15 = 385
        x = 500
        frame = 10
        while x > 380:
            x -= 5
            tracker.update(_det(x, 300, t + frame * 0.016))
            frame += 1
            assert tracker.state in (ShotState.STARTED, ShotState.ENTERED), (
                f"State reset to {tracker.state} at x={x}"
            )

        assert tracker.state == ShotState.ENTERED

        # Exit well past gateway (gateway_x1 = 385 - 15 = 370).
        # Send exit detection, then 3 None frames for completion.
        tracker.update(_det(200, 298, t + frame * 0.016))
        result = None
        for _ in range(3):
            result = tracker.update(None)
        assert result is not None

    def test_repositioning_updates_start(self, tracker: BallTracker) -> None:
        """Ball repositioning within start zone updates start pos without reset."""
        t = time.perf_counter()

        # Stabilize at x=50
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED
        assert tracker.start_circle[:2] == (50, 300)

        # Reposition to x=100 (still in start zone, start_x2=180)
        base = t + 0.5
        for i in range(10):
            tracker.update(_det(100, 300, base + i * 0.016))

        # Should still be STARTED, not reset
        assert tracker.state == ShotState.STARTED
        # Start position should have updated
        assert tracker.start_circle[:2] == (100, 300)

    def test_in_zone_motion_appended(self, tracker: BallTracker) -> None:
        """Ball moving rapidly through start zone (frame-to-frame > 6px)
        should be appended to positions, even before exiting the zone.

        Without this, the launch-velocity window misses the highest-
        velocity portion of the trail (ball decelerates as it traverses
        the start zone), causing a ~30% under-read on real putts.
        """
        t = time.perf_counter()

        # Stabilize at x=50 (rest)
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED
        rest_count = len(tracker.positions)

        # Simulate a putt: ball moves 15 px per frame through the start
        # zone (x=50 -> 65 -> 80 -> 95 -> 110 -> 125 -> 140 -> 155 -> 170)
        # All within start zone (start_x2=180 in fixture).
        base = t + 0.5
        for i in range(8):
            x = 65 + i * 15
            tracker.update(_det(x, 300, base + i * 0.016))

        # Each in-zone motion frame should have been appended (not just
        # the last one before zone exit).
        positions = tracker.positions
        # We expect rest_count + 8 motion frames (some, all, or near all
        # depending on threshold — at minimum >2 motion frames captured)
        in_zone_motion_xs = [p[0] for p in positions if 50 < p[0] <= 180]
        assert len(in_zone_motion_xs) >= 4, (
            f"only {len(in_zone_motion_xs)} in-zone motion frames captured; "
            f"positions: {list(positions)}"
        )

    def test_transit_positions_tracked(self, tracker: BallTracker) -> None:
        """Positions in transit zone (between start and gateway) are recorded."""
        t = time.perf_counter()

        # Stabilize at x=50
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        # Move into transit zone (past start_x2=180 but before gateway_x1=190)
        tracker.update(_det(185, 300, t + 0.5))
        assert tracker.state == ShotState.STARTED

        positions = tracker.positions
        # Should have start pos + transit pos
        assert any(p[0] == 185 for p in positions)

        # Enter and exit gateway to complete shot
        tracker.update(_det(195, 300, t + 0.6))
        assert tracker.state == ShotState.ENTERED
        # Exit detection — but completion now requires lost detections
        tracker.update(_det(300, 298, t + 0.7))
        result = None
        for _ in range(3):
            result = tracker.update(None)
        assert result is not None
        # Transit position should be in the result
        assert any(p[0] == 185 for p in result.positions)


class TestRightToLeft:
    """Tests for right-to-left ball roll direction."""

    @pytest.fixture
    def rtl_zone(self) -> DetectionZone:
        """RtL detection zone with ball starting on the right side."""
        return DetectionZone(
            start_x1=400, start_x2=570, y1=180, y2=450,
            direction="right_to_left",
        )

    @pytest.fixture
    def rtl_tracker(
        self, rtl_zone: DetectionZone, ball_settings: BallSettings,
        shot_settings: ShotSettings,
    ) -> BallTracker:
        return BallTracker(
            zone=rtl_zone,
            ball_settings=ball_settings,
            shot_settings=shot_settings,
        )

    def test_rtl_gateway_entry(self, rtl_tracker: BallTracker) -> None:
        """Ball moving left into gateway should transition to ENTERED."""
        zone = rtl_tracker.zone
        t = time.perf_counter()

        # Stabilize inside start zone
        for i in range(10):
            rtl_tracker.update(_det(500, 300, t + i * 0.016))

        assert rtl_tracker.state == ShotState.STARTED

        # Transit through zone then enter gateway (to the LEFT)
        gateway_x2 = zone.start_x1 - zone.gateway_width
        rtl_tracker.update(_det(zone.start_x1 - 5, 300, t + 0.4))  # transit
        rtl_tracker.update(_det(gateway_x2 - 5, 300, t + 0.5))
        assert rtl_tracker.state == ShotState.ENTERED

    def test_rtl_full_shot_cycle(self, rtl_tracker: BallTracker) -> None:
        """Full RtL shot: stable -> entered -> exited with enough distance."""
        zone = rtl_tracker.zone
        t = time.perf_counter()

        # Stabilize
        for i in range(10):
            rtl_tracker.update(_det(500, 300, t + i * 0.016))

        assert rtl_tracker.state == ShotState.STARTED

        # Transit then enter gateway (moving left past start_x1)
        gateway_x2 = zone.start_x1 - zone.gateway_width
        gateway_x1 = gateway_x2 - zone.gateway_width
        rtl_tracker.update(_det(zone.start_x1 - 5, 300, t + 0.4))  # transit
        rtl_tracker.update(_det(gateway_x2 - 5, 300, t + 0.5))
        assert rtl_tracker.state == ShotState.ENTERED

        # Exit well past gateway to the left.  Send exit detection,
        # then 3 None frames to trigger post-traversal completion.
        rtl_tracker.update(_det(gateway_x1 - 200, 295, t + 0.6))
        result = None
        for _ in range(3):
            result = rtl_tracker.update(None)

        assert result is not None
        assert result.end_position[0] == gateway_x1 - 200


class TestRestartSuppression:
    """A stationary ball that re-stabilizes repeatedly must not emit a flood
    of Re-start logs, and last_activity_time must reflect real activity."""

    def test_stationary_ball_single_restart_no_spam(
        self, tracker: BallTracker, caplog
    ) -> None:
        """Drive a stationary ball so it re-stabilizes many times within
        tolerance. At most ONE 'Re-start' should be emitted (the position
        equals the existing start within tolerance → suppressed)."""
        import logging

        t = time.perf_counter()
        # Stabilize at x=50 → STARTED (logs "New start", not "Re-start")
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        with caplog.at_level(logging.INFO, logger="birdman_putting.tracking"):
            # Keep feeding the SAME resting position (within tolerance).
            # Each batch would re-trigger the stability check; without
            # suppression this logs "Re-start" hundreds of times.
            base = t + 0.5
            for i in range(60):
                tracker.update(_det(50, 300, base + i * 0.016))

        assert tracker.state == ShotState.STARTED
        restart_logs = [
            r for r in caplog.records if "Re-start" in r.getMessage()
        ]
        assert len(restart_logs) == 0, (
            f"expected no redundant Re-start logs, got {len(restart_logs)}: "
            f"{[r.getMessage() for r in restart_logs]}"
        )

    def test_genuine_reposition_emits_one_restart_and_updates_activity(
        self, tracker: BallTracker, caplog
    ) -> None:
        """Moving the ball to a genuinely NEW rest spot emits exactly one
        Re-start and advances last_activity_time."""
        import logging

        t = time.perf_counter()
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        activity_before = tracker.last_activity_time

        with caplog.at_level(logging.INFO, logger="birdman_putting.tracking"):
            # Reposition to x=120 (far beyond tolerance) and settle there.
            base = t + 0.5
            for i in range(10):
                tracker.update(_det(120, 300, base + i * 0.016))

        assert tracker.state == ShotState.STARTED
        assert tracker.start_circle[:2] == (120, 300)
        restart_logs = [
            r for r in caplog.records if "Re-start" in r.getMessage()
        ]
        assert len(restart_logs) == 1, (
            f"expected exactly one Re-start, got {len(restart_logs)}"
        )
        # Activity timestamp must advance on a genuine re-start.
        assert tracker.last_activity_time > activity_before

    def test_last_activity_time_advances_on_state_transition(
        self, tracker: BallTracker
    ) -> None:
        """last_activity_time should update when the state machine
        transitions (IDLE→BALL_DETECTED→STARTED)."""
        t = time.perf_counter()
        # Before any update, attribute exists and is a float.
        assert isinstance(tracker.last_activity_time, float)

        tracker.update(_det(50, 300, t))
        after_detect = tracker.last_activity_time
        assert after_detect > 0

        for i in range(1, 10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED
        # Transition to STARTED must have advanced activity.
        assert tracker.last_activity_time >= after_detect

    def test_stationary_activity_does_not_advance_when_suppressed(
        self, tracker: BallTracker
    ) -> None:
        """A truly stationary ball (suppressed re-starts) must NOT keep
        bumping last_activity_time — otherwise the app watchdog can never
        fire on a stuck ball. Activity should stay frozen while nothing
        meaningful happens."""
        t = time.perf_counter()
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        frozen = tracker.last_activity_time

        # Feed the same rest position for a long stretch.
        base = t + 1.0
        for i in range(80):
            tracker.update(_det(50, 300, base + i * 0.016))

        assert tracker.state == ShotState.STARTED
        # No meaningful activity → timestamp must not have advanced.
        assert tracker.last_activity_time == frozen


class TestEnteredVelocitySanity:
    """ENTERED-state forward-velocity sanity: a noise blob far ahead in one
    frame implies an impossible velocity and must be rejected."""

    def test_impossible_forward_velocity_rejected(
        self, tracker: BallTracker
    ) -> None:
        """A forward detection implying an impossible frame-to-frame velocity
        is NOT appended to the trajectory."""
        zone = tracker.zone
        t = time.perf_counter()

        # Stabilize and enter the gateway normally.
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        gateway_x1 = zone.start_x2 + zone.gateway_width
        tracker.update(_det(zone.start_x2 + 5, 300, t + 0.40))  # transit
        tracker.update(_det(gateway_x1 + 5, 300, t + 0.42))     # enter
        assert tracker.state == ShotState.ENTERED

        positions_before = len(tracker.positions)
        last_x = tracker.positions[-1][0]

        # Noise blob jumps ~2000px forward in a single 16ms frame — far
        # beyond the frame and any conceivable putt velocity (hundreds of
        # MPH). Must be rejected regardless of the exact ppf derived from
        # the detected ball radius.
        result = tracker.update(_det(last_x + 2000, 300, t + 0.436))

        assert result is None
        assert len(tracker.positions) == positions_before, (
            "impossible-velocity detection should not be appended"
        )

    def test_legitimate_fast_putt_not_rejected(
        self, tracker: BallTracker
    ) -> None:
        """A fast but PLAUSIBLE putt (large dx, but within the velocity cap)
        is still accepted — we must not reject legitimate fast putts."""
        zone = tracker.zone
        t = time.perf_counter()

        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        gateway_x1 = zone.start_x2 + zone.gateway_width
        tracker.update(_det(zone.start_x2 + 5, 300, t + 0.40))
        tracker.update(_det(gateway_x1 + 5, 300, t + 0.42))
        assert tracker.state == ShotState.ENTERED

        positions_before = len(tracker.positions)
        last_x = tracker.positions[-1][0]

        # A ~20 MPH putt: ~29 ft/s ≈ 1700 px/s at 57px/ft → ~27px per 16ms.
        # Use 120px over 0.05s (a sparse-detection fast putt) ≈ 2400px/s
        # ≈ 42 ft/s ≈ 29 MPH — generous but plausible-ish; must be accepted
        # since the cap is set well above any real putt.
        result = tracker.update(_det(last_x + 120, 300, t + 0.47))
        assert result is None  # shot not complete yet, but…
        assert len(tracker.positions) == positions_before + 1, (
            "a fast-but-plausible putt detection must be appended"
        )


class TestPostShotCooldown:
    """Tests for post-shot cooldown timer."""

    def test_cooldown_blocks_rearm(self, detection_zone, ball_settings) -> None:
        """After shot completion, tracker ignores detections during cooldown."""
        shot_settings = ShotSettings(post_shot_cooldown=1.0)
        tracker = BallTracker(
            zone=detection_zone, ball_settings=ball_settings, shot_settings=shot_settings,
        )
        t = time.perf_counter()

        # Complete a full shot
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED
        tracker.update(_det(185, 300, t + 0.4))
        tracker.update(_det(195, 300, t + 0.5))
        assert tracker.state == ShotState.ENTERED
        gateway_x2 = detection_zone.start_x2 + 2 * detection_zone.gateway_width
        # Send exit detection, then None frames to complete shot
        tracker.update(_det(gateway_x2 + 200, 295, t + 0.6))
        result = None
        for _ in range(3):
            result = tracker.update(None)
        assert result is not None
        assert tracker.state == ShotState.IDLE

        # Immediately after shot: cooldown should block new detections
        tracker.update(_det(50, 300, t + 0.7))
        assert tracker.state == ShotState.IDLE  # Still idle, cooldown active

    def test_manual_reset_no_cooldown(self, tracker) -> None:
        """Manual reset() does not trigger cooldown."""
        t = time.perf_counter()
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))
        assert tracker.state == ShotState.STARTED

        tracker.reset()  # Manual reset, no cooldown
        assert tracker.state == ShotState.IDLE

        # Should immediately detect a new ball (no cooldown)
        tracker.update(_det(50, 300, t + 1.0))
        assert tracker.state == ShotState.BALL_DETECTED
