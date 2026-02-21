"""Tests for ball detection module."""

import time

import cv2
import numpy as np

from birdman_putting.color_presets import get_preset
from birdman_putting.detection import (
    BallDetector,
    generate_hsv_from_patch,
    resize_with_aspect_ratio,
)


class TestBallDetector:
    def _make_frame_with_ball(
        self,
        ball_bgr: tuple[int, int, int],
        center: tuple[int, int] = (100, 250),
        radius: int = 15,
        frame_size: tuple[int, int] = (360, 640),
    ) -> np.ndarray:
        """Create a synthetic frame with a colored circle."""
        frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        cv2.circle(frame, center, radius, ball_bgr, -1)
        return frame

    def test_detect_orange_ball(self):
        # Orange ball in BGR
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),  # Bright orange
            center=(100, 300),
            radius=15,
        )

        # Use orange2 preset (the most commonly used)
        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is not None
        assert abs(detection.x - 100) <= 2
        assert abs(detection.y - 300) <= 2
        assert detection.radius > 0

    def test_no_ball_returns_none(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)  # Black frame

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is None

    def test_ball_outside_zone_not_detected(self):
        # Ball at y=50, zone starts at y=100
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),
            center=(100, 50),
            radius=15,
        )

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=100, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is None

    def test_ball_too_small_not_detected(self):
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),
            center=(100, 250),
            radius=2,  # Very small
        )

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is None

    def test_radius_filtering(self):
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),
            center=(100, 250),
            radius=15,
        )

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        # Expect radius ~15, filter for radius ~100 — should not match
        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
            expected_radius=100,
            radius_tolerance=10,
        )

        assert detection is None

    def test_get_mask(self):
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),
            center=(100, 250),
            radius=15,
        )

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        mask = detector.get_mask(frame, 0, 640, 0, 360)
        assert mask.shape == (360, 640)
        # Mask should have some white pixels where the ball is
        assert np.sum(mask > 0) > 0


class TestGenerateHsvFromPatch:
    def test_solid_orange_patch(self):
        """Sampling a solid orange region should produce an HSV range containing orange."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill center with orange BGR (0, 140, 255)
        frame[40:60, 40:60] = (0, 140, 255)

        hsv_range = generate_hsv_from_patch(frame, 50, 50, patch_size=11)

        # Orange in HSV is roughly H=10-20, high S, high V
        # The range should encompass the actual hue of the orange pixel
        orange_hsv = cv2.cvtColor(
            np.array([[[0, 140, 255]]], dtype=np.uint8), cv2.COLOR_BGR2HSV
        )[0, 0]
        assert hsv_range.hmin <= orange_hsv[0] <= hsv_range.hmax
        assert hsv_range.smin <= orange_hsv[1] <= hsv_range.smax
        assert hsv_range.vmin <= orange_hsv[2] <= hsv_range.vmax

    def test_edge_clamping(self):
        """Sampling near the frame edge should not crash."""
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)

        # Corner — patch will be clamped
        hsv_range = generate_hsv_from_patch(frame, 0, 0, patch_size=21)
        assert 0 <= hsv_range.hmin <= hsv_range.hmax <= 179
        assert 0 <= hsv_range.smin <= hsv_range.smax <= 255
        assert 0 <= hsv_range.vmin <= hsv_range.vmax <= 255

    def test_range_bounds_valid(self):
        """All output values should be within valid HSV ranges."""
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        hsv_range = generate_hsv_from_patch(frame, 50, 50)
        assert 0 <= hsv_range.hmin <= 179
        assert 0 <= hsv_range.hmax <= 179
        assert 0 <= hsv_range.smin <= 255
        assert 0 <= hsv_range.smax <= 255
        assert 0 <= hsv_range.vmin <= 255
        assert 0 <= hsv_range.vmax <= 255

    def test_detector_uses_double_conversion_range(self):
        """A detector with a range matching the double-converted color space detects the ball."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Draw an orange ball
        cv2.circle(frame, (100, 200), 15, (0, 140, 255), -1)

        # Compute the double-converted HSV for this BGR color and build a range
        # BGR(0,140,255) → HSV(~13,255,255) → treat as BGR → HSV(~30,~242,255)
        # Use a range wide enough to cover blur effects
        from birdman_putting.color_presets import HSVRange

        hsv_range = HSVRange(hmin=20, smin=200, vmin=200, hmax=40, smax=255, vmax=255)

        detector = BallDetector(hsv_range=hsv_range, min_radius=5)
        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=0.0,
        )
        assert detection is not None
        assert abs(detection.x - 100) <= 2
        assert abs(detection.y - 200) <= 2


class TestResizeWithAspectRatio:
    def test_resize_by_width(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_with_aspect_ratio(img, width=320)
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240  # Maintained aspect ratio

    def test_resize_by_height(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_with_aspect_ratio(img, height=240)
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320

    def test_no_resize(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_with_aspect_ratio(img)
        assert result.shape == img.shape
