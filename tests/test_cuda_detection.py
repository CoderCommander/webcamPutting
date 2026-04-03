"""Tests for CUDA-accelerated ball detection.

Skipped automatically if CUDA is not available.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from birdman_putting.color_presets import get_preset
from birdman_putting.gpu import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="CUDA not available",
)


@pytest.fixture
def hsv_range():
    return get_preset("orange2")


@pytest.fixture
def synthetic_frame():
    """200x250 frame with an orange circle."""
    frame = np.zeros((250, 200, 3), dtype=np.uint8)
    frame[:] = (40, 60, 30)  # Dark green
    cv2.circle(frame, (100, 125), 10, (0, 140, 255), -1)  # Orange ball
    return frame


class TestCudaBallDetector:
    def test_detects_ball(self, hsv_range, synthetic_frame) -> None:
        from birdman_putting.cuda_detection import CudaBallDetector
        from birdman_putting.gpu import init_cuda

        init_cuda()
        detector = CudaBallDetector(hsv_range=hsv_range, min_radius=3)
        result = detector.detect(
            synthetic_frame, 0, 200, 0, 250, timestamp=0.0,
        )
        assert result is not None
        # Ball should be near center
        assert 80 < result.x < 120
        assert 105 < result.y < 145

    def test_matches_cpu_detector(self, hsv_range, synthetic_frame) -> None:
        from birdman_putting.cuda_detection import CudaBallDetector
        from birdman_putting.detection import BallDetector
        from birdman_putting.gpu import init_cuda

        init_cuda()
        cpu = BallDetector(hsv_range=hsv_range, min_radius=3)
        gpu = CudaBallDetector(hsv_range=hsv_range, min_radius=3)

        cpu_result = cpu.detect(synthetic_frame, 0, 200, 0, 250, timestamp=0.0)
        gpu_result = gpu.detect(synthetic_frame, 0, 200, 0, 250, timestamp=0.0)

        assert cpu_result is not None
        assert gpu_result is not None
        # Results should be close (minor floating-point differences possible)
        assert abs(cpu_result.x - gpu_result.x) <= 2
        assert abs(cpu_result.y - gpu_result.y) <= 2

    def test_no_ball_returns_none(self, hsv_range) -> None:
        from birdman_putting.cuda_detection import CudaBallDetector
        from birdman_putting.gpu import init_cuda

        init_cuda()
        # Plain black frame — no ball
        frame = np.zeros((250, 200, 3), dtype=np.uint8)
        detector = CudaBallDetector(hsv_range=hsv_range)
        result = detector.detect(frame, 0, 200, 0, 250, timestamp=0.0)
        assert result is None
