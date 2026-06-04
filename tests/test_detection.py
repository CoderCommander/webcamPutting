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

    def test_circularity_rejects_irregular_shape(self):
        """An elongated/irregular orange blob (like a hand) should be rejected."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Draw a tall narrow orange rectangle — low circularity
        cv2.rectangle(frame, (90, 200), (110, 340), (0, 140, 255), -1)

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
            min_circularity=0.5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is None

    def test_circularity_accepts_circle(self):
        """A round orange ball should pass the circularity filter."""
        frame = self._make_frame_with_ball(
            ball_bgr=(0, 140, 255),
            center=(100, 300),
            radius=15,
        )

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
            min_circularity=0.5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
        )

        assert detection is not None

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

    # --- Best-scoring contour selection (expected_pos) -----------------

    def test_no_expectation_returns_largest(self):
        """Regression: with NO expected_pos/expected_radius, two blobs →
        the LARGEST one wins (preserves original largest-area behavior).
        """
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Small round ball at (100, 300), r=12
        cv2.circle(frame, (100, 300), 12, (0, 140, 255), -1)
        # Larger round blob at (400, 300), r=30
        cv2.circle(frame, (400, 300), 30, (0, 140, 255), -1)

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
        # Largest blob (the r=30 one at x=400) must win
        assert abs(detection.x - 400) <= 3
        assert detection.radius > 20

    def test_expected_pos_does_not_override_largest_blob(self):
        """Scoring is reverted: even when expected_pos sits on a small ball,
        detect() returns the LARGEST passing orange contour, not the nearest.

        This restores the historical, projector-safe behavior. The overhead
        OBS tracer is projected onto the physical mat, so the camera sees
        near-the-ball false blobs; the old proximity scoring wrongly preferred
        those over the real moving ball and dropped motion frames. The real
        ball is the largest orange blob, so largest-wins is reliable.
        """
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Small blob near expected_pos (e.g. a projected-tracer segment)...
        cv2.circle(frame, (120, 300), 14, (0, 140, 255), -1)
        # ...and the LARGER real ball elsewhere in the zone.
        cv2.circle(frame, (480, 300), 35, (0, 140, 255), -1)

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
            expected_pos=(118, 300),
            expected_radius=14,
        )

        assert detection is not None
        # Largest blob wins regardless of expected_pos (scoring disabled).
        assert abs(detection.x - 480) <= 3
        assert detection.radius > 20

    def test_blurred_large_radius_ball_not_rejected(self):
        """A motion-blurred rolling ball's enclosing-circle radius balloons
        well beyond its rest radius. The radius gate must stay LOOSE so the
        blurred ball survives — rejecting it on a tight tolerance dropped
        nearly every motion frame in production (fit collapsed to 1-2 frames,
        breaking hard-putt speed). expected_radius biases scoring, it does
        NOT hard-reject a far-off radius."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # A single blurred-ball-sized blob, r≈30, while rest radius was ~12.
        cv2.circle(frame, (300, 300), 30, (0, 140, 255), -1)

        detector = BallDetector(
            hsv_range=get_preset("orange2"),
            min_radius=5,
        )

        detection = detector.detect(
            frame=frame,
            zone_x1=0, zone_x2_limit=640,
            zone_y1=0, zone_y2=360,
            timestamp=time.perf_counter(),
            expected_radius=12,
        )

        # Loose gate: the blurred ball is still detected, not dropped.
        assert detection is not None
        assert abs(detection.x - 300) <= 4


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

    def test_detector_uses_single_conversion_range(self):
        """A detector with an HSV range matching single BGR→HSV detects the ball."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Draw an orange ball
        cv2.circle(frame, (100, 200), 15, (0, 140, 255), -1)

        # BGR(0,140,255) → single HSV gives H≈13, S=255, V=255
        from birdman_putting.color_presets import HSVRange

        hsv_range = HSVRange(hmin=5, smin=200, vmin=200, hmax=25, smax=255, vmax=255)

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
