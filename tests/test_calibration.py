"""Tests for auto-calibration module."""

from webcam_putting.calibration import (
    STABILITY_FRAMES,
    TIMEOUT_FRAMES,
    AutoCalibrator,
    CalibrationState,
)
from webcam_putting.detection import BallDetection


def _det(x: int, y: int, r: int = 15) -> BallDetection:
    """Shorthand for creating a BallDetection."""
    return BallDetection(x=x, y=y, radius=r, contour_area=100.0, timestamp=0.0)


FRAME_W, FRAME_H = 640, 360


class TestCalibrationStates:
    def test_starts_idle(self) -> None:
        cal = AutoCalibrator()
        assert cal.state == CalibrationState.IDLE

    def test_start_transitions_to_searching(self) -> None:
        cal = AutoCalibrator()
        cal.start()
        assert cal.state == CalibrationState.SEARCHING

    def test_cancel_returns_to_idle(self) -> None:
        cal = AutoCalibrator()
        cal.start()
        cal.update(_det(100, 200), FRAME_W, FRAME_H)
        cal.cancel()
        assert cal.state == CalibrationState.IDLE

    def test_detection_transitions_to_stabilizing(self) -> None:
        cal = AutoCalibrator()
        cal.start()
        cal.update(_det(100, 200), FRAME_W, FRAME_H)
        assert cal.state == CalibrationState.STABILIZING

    def test_lost_ball_returns_to_searching(self) -> None:
        cal = AutoCalibrator()
        cal.start()
        cal.update(_det(100, 200), FRAME_W, FRAME_H)
        assert cal.state == CalibrationState.STABILIZING
        cal.update(None, FRAME_W, FRAME_H)
        assert cal.state == CalibrationState.SEARCHING


class TestStability:
    def test_stable_ball_completes(self) -> None:
        """Ball at consistent position for STABILITY_FRAMES should complete."""
        cal = AutoCalibrator()
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 180), FRAME_W, FRAME_H)

        assert result is not None
        assert cal.state == CalibrationState.COMPLETE
        assert result.zone.start_x1 >= 0
        assert result.zone.start_x2 <= FRAME_W

    def test_within_tolerance_still_completes(self) -> None:
        """Ball positions within tolerance should still count as stable."""
        cal = AutoCalibrator()
        cal.start()

        for i in range(STABILITY_FRAMES):
            # Jitter within tolerance
            jitter = (i % 3) - 1  # -1, 0, 1
            result = cal.update(_det(200 + jitter, 180 + jitter), FRAME_W, FRAME_H)

        assert result is not None
        assert cal.state == CalibrationState.COMPLETE

    def test_unstable_ball_does_not_complete(self) -> None:
        """Ball jumping around beyond tolerance should not complete."""
        cal = AutoCalibrator()
        cal.start()

        for i in range(STABILITY_FRAMES):
            # Large jumps well beyond tolerance
            x = 100 + (i * 20)
            result = cal.update(_det(x, 180), FRAME_W, FRAME_H)

        assert result is None
        assert cal.state != CalibrationState.COMPLETE


class TestTimeout:
    def test_timeout_after_max_frames(self) -> None:
        """Calibration should fail after TIMEOUT_FRAMES with no stable position."""
        cal = AutoCalibrator()
        cal.start()

        for _ in range(TIMEOUT_FRAMES + 1):
            result = cal.update(None, FRAME_W, FRAME_H)

        assert result is None
        assert cal.state == CalibrationState.FAILED


class TestZoneComputationLtR:
    def test_ltr_ball_near_right_edge(self) -> None:
        """In LtR mode, ball should be near the right edge of the start zone."""
        cal = AutoCalibrator(direction="left_to_right")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 180), FRAME_W, FRAME_H)

        assert result is not None
        zone = result.zone
        # Ball at x=200: start_x1 = 200-80=120, start_x2 = 200+30=230
        assert zone.start_x1 == 120
        assert zone.start_x2 == 230
        assert zone.direction == "left_to_right"

    def test_ltr_y_range(self) -> None:
        """Y range should be ball_y +/- 100."""
        cal = AutoCalibrator(direction="left_to_right")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 180), FRAME_W, FRAME_H)

        assert result is not None
        zone = result.zone
        assert zone.y1 == 80
        assert zone.y2 == 280


class TestZoneComputationRtL:
    def test_rtl_ball_near_left_edge(self) -> None:
        """In RtL mode, ball should be near the left edge of the start zone."""
        cal = AutoCalibrator(direction="right_to_left")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 180), FRAME_W, FRAME_H)

        assert result is not None
        zone = result.zone
        # Ball at x=200: start_x1 = 200-30=170, start_x2 = 200+80=280
        assert zone.start_x1 == 170
        assert zone.start_x2 == 280
        assert zone.direction == "right_to_left"


class TestEdgeClamping:
    def test_clamp_left_edge(self) -> None:
        """Zone near left frame edge should be clamped to 0."""
        cal = AutoCalibrator(direction="left_to_right")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(30, 180), FRAME_W, FRAME_H)

        assert result is not None
        assert result.zone.start_x1 == 0

    def test_clamp_top_edge(self) -> None:
        """Zone near top frame edge should be clamped to 0."""
        cal = AutoCalibrator(direction="left_to_right")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 50), FRAME_W, FRAME_H)

        assert result is not None
        assert result.zone.y1 == 0

    def test_clamp_right_edge(self) -> None:
        """Zone near right frame edge should be clamped to frame width."""
        cal = AutoCalibrator(direction="right_to_left")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(600, 180), FRAME_W, FRAME_H)

        assert result is not None
        assert result.zone.start_x2 == FRAME_W

    def test_clamp_bottom_edge(self) -> None:
        """Zone near bottom frame edge should be clamped to frame height."""
        cal = AutoCalibrator(direction="left_to_right")
        cal.start()

        for _ in range(STABILITY_FRAMES):
            result = cal.update(_det(200, 320), FRAME_W, FRAME_H)

        assert result is not None
        assert result.zone.y2 == FRAME_H


class TestIdleIgnored:
    def test_update_in_idle_does_nothing(self) -> None:
        """Updates before start() should be ignored."""
        cal = AutoCalibrator()
        result = cal.update(_det(200, 180), FRAME_W, FRAME_H)
        assert result is None
        assert cal.state == CalibrationState.IDLE
