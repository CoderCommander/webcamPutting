"""Tests for camera module — frame validation and fallback behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from birdman_putting.camera import Camera
from birdman_putting.config import CameraSettings


class TestIsBlackFrame:
    """Test the static black frame detection helper."""

    def test_pure_black_is_detected(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert Camera._is_black_frame(frame) is True

    def test_near_black_is_detected(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 2
        assert Camera._is_black_frame(frame) is True

    def test_dim_frame_is_not_black(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 10
        assert Camera._is_black_frame(frame) is False

    def test_normal_frame_is_not_black(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        assert Camera._is_black_frame(frame) is False

    def test_threshold_boundary(self) -> None:
        # Just below threshold — should be black
        frame = np.full((100, 100, 3), 2, dtype=np.uint8)
        assert Camera._is_black_frame(frame) is True

        # Above threshold — should not be black
        frame = np.full((100, 100, 3), 4, dtype=np.uint8)
        assert Camera._is_black_frame(frame) is False


class TestValidateFrames:
    """Test the _validate_frames method with mocked capture."""

    def _make_camera(self, **kwargs: object) -> Camera:
        settings = CameraSettings(**kwargs)  # type: ignore[arg-type]
        return Camera(settings)

    @patch("birdman_putting.camera.time.sleep")
    def test_good_frames_pass(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        good_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, good_frame)
        camera._cap = mock_cap

        assert camera._validate_frames() is True

    @patch("birdman_putting.camera.time.sleep")
    def test_black_frames_fail(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, black_frame)
        camera._cap = mock_cap

        assert camera._validate_frames() is False

    @patch("birdman_putting.camera.time.sleep")
    def test_warmup_then_good_frame_passes(self, mock_sleep: MagicMock) -> None:
        """Simulate camera that produces black during warmup then good frames."""
        camera = self._make_camera()
        mock_cap = MagicMock()
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        good_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128

        # 5 warmup reads (discarded) + first validation read is good
        frames = [(True, black_frame)] * 5 + [(True, good_frame)]
        mock_cap.read.side_effect = frames
        camera._cap = mock_cap

        assert camera._validate_frames() is True

    @patch("birdman_putting.camera.time.sleep")
    def test_no_frames_fail(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        camera._cap = mock_cap

        assert camera._validate_frames() is False

    @patch("birdman_putting.camera.time.sleep")
    def test_no_cap_fails(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        camera._cap = None
        assert camera._validate_frames() is False
