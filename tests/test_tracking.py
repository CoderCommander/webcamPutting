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

        # Ball moves past gateway (start_x2=180, gateway at 190)
        tracker.update(_det(195, 300, t + 0.5))
        assert tracker.state == ShotState.ENTERED

    def test_full_shot_cycle(self, tracker: BallTracker):
        """Full shot: stable -> entered -> exited with enough distance."""
        zone = tracker.zone
        t = time.perf_counter()

        # Stabilize in start zone
        for i in range(10):
            tracker.update(_det(50, 300, t + i * 0.016))

        # Enter gateway
        gateway_x1 = zone.start_x2 + zone.gateway_width
        tracker.update(_det(gateway_x1 + 5, 300, t + 0.5))
        assert tracker.state == ShotState.ENTERED

        # Exit well past gateway (min_exit_distance_px=50)
        gateway_x2 = gateway_x1 + zone.gateway_width
        result = tracker.update(_det(gateway_x2 + 200, 295, t + 0.6))

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

        # Gateway is to the LEFT: gateway_x2 = start_x1 - gw_width = 390
        gateway_x2 = zone.start_x1 - zone.gateway_width
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

        # Enter gateway (moving left past start_x1)
        gateway_x2 = zone.start_x1 - zone.gateway_width
        gateway_x1 = gateway_x2 - zone.gateway_width
        rtl_tracker.update(_det(gateway_x2 - 5, 300, t + 0.5))
        assert rtl_tracker.state == ShotState.ENTERED

        # Exit well past gateway to the left
        result = rtl_tracker.update(_det(gateway_x1 - 200, 295, t + 0.6))

        assert result is not None
        assert result.end_position[0] == gateway_x1 - 200
