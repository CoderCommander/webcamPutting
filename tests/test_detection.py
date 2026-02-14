"""Tests for ball detection module."""

import time

import cv2
import numpy as np

from webcam_putting.color_presets import get_preset
from webcam_putting.detection import BallDetector, resize_with_aspect_ratio


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
