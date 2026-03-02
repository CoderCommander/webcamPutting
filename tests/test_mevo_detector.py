"""Tests for Mevo shot detector — all OCR and capture are mocked."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np

from birdman_putting.config import MevoSettings
from birdman_putting.mevo.detector import (
    MevoDetector,
    MevoShotData,
    _compute_mse,
    _validate_metrics,
    _values_changed,
    build_rois,
)


class TestMevoShotData:
    def test_back_spin_decomposition(self) -> None:
        shot = MevoShotData(
            ball_speed=120.0, launch_angle=12.0, launch_direction=1.5,
            spin_rate=3000.0, spin_axis=20.0, club_speed=95.0,
        )
        expected_back = abs(3000.0 * math.cos(math.radians(20.0)))
        expected_side = 3000.0 * math.sin(math.radians(20.0))
        assert abs(shot.back_spin - expected_back) < 0.01
        assert abs(shot.side_spin - expected_side) < 0.01

    def test_zero_spin_axis(self) -> None:
        shot = MevoShotData(
            ball_speed=100.0, launch_angle=10.0, launch_direction=0.0,
            spin_rate=2500.0, spin_axis=0.0, club_speed=0.0,
        )
        assert abs(shot.back_spin - 2500.0) < 0.01
        assert abs(shot.side_spin) < 0.01


class TestBuildRois:
    def test_valid_rois(self) -> None:
        roi_dict = {
            "ball_speed": [10, 20, 100, 30],
            "launch_angle": [10, 60, 100, 30],
        }
        rois = build_rois(roi_dict)
        assert len(rois) == 2
        assert rois[0].name == "ball_speed"
        assert rois[0].x == 10
        assert rois[0].width == 100

    def test_invalid_coords_skipped(self) -> None:
        roi_dict = {"bad": [1, 2, 3]}  # Only 3 coords
        rois = build_rois(roi_dict)
        assert len(rois) == 0


class TestComputeMSE:
    def test_identical_frames(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _compute_mse(frame, frame) == 0.0

    def test_different_frames(self) -> None:
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mse = _compute_mse(a, b)
        assert mse > 0

    def test_different_shapes(self) -> None:
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((50, 50, 3), dtype=np.uint8)
        assert _compute_mse(a, b) == float("inf")


class TestValuesChanged:
    def test_changed(self) -> None:
        prev = {"ball_speed": 100.0}
        curr = {"ball_speed": 120.0}
        assert _values_changed(prev, curr) is True

    def test_unchanged(self) -> None:
        prev = {"ball_speed": 100.0}
        curr = {"ball_speed": 100.05}
        assert _values_changed(prev, curr) is False

    def test_new_value_appears(self) -> None:
        prev = {"ball_speed": None}
        curr = {"ball_speed": 100.0}
        assert _values_changed(prev, curr) is True


class TestValidateMetrics:
    def test_valid(self) -> None:
        metrics = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is True

    def test_missing_required(self) -> None:
        metrics = {"ball_speed": 120.0, "launch_angle": 12.0}
        assert _validate_metrics(metrics) is False

    def test_out_of_range_speed(self) -> None:
        metrics = {
            "ball_speed": 999.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is False

    def test_none_required(self) -> None:
        metrics = {
            "ball_speed": None,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is False


class TestMevoDetector:
    def _make_detector(
        self,
        mse_threshold: float = 100.0,
    ) -> tuple[MevoDetector, MagicMock, MagicMock]:
        settings = MevoSettings(enabled=True, mse_threshold=mse_threshold)
        mock_ocr = MagicMock()
        mock_capture = MagicMock()
        detector = MevoDetector(settings, mock_ocr, mock_capture)
        return detector, mock_ocr, mock_capture

    def test_returns_none_when_no_frame(self) -> None:
        detector, _ocr, capture = self._make_detector()
        capture.capture.return_value = None
        assert detector.poll() is None

    def test_returns_none_when_frame_unchanged(self) -> None:
        detector, _ocr, capture = self._make_detector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        capture.capture.return_value = frame

        # First poll: sets prev_frame
        _ocr.read_metrics.return_value = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        detector.poll()

        # Second poll: same frame → MSE below threshold
        result = detector.poll()
        assert result is None

    def test_detects_new_shot(self) -> None:
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        # First call: baseline capture — returns None (stale data suppressed)
        capture.capture.return_value = frame1
        ocr.read_metrics.return_value = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        baseline = detector.poll()
        assert baseline is None

        # Second call: different frame, different metrics → new shot
        capture.capture.return_value = frame2
        ocr.read_metrics.return_value = {
            "ball_speed": 130.0,
            "launch_angle": 15.0,
            "launch_direction": -2.0,
        }
        shot = detector.poll()
        assert shot is not None
        assert isinstance(shot, MevoShotData)
        assert shot.ball_speed == 130.0

        # Third call: different frame, different metrics → another shot
        capture.capture.return_value = frame3
        ocr.read_metrics.return_value = {
            "ball_speed": 140.0,
            "launch_angle": 18.0,
            "launch_direction": 3.0,
        }
        shot2 = detector.poll()
        assert shot2 is not None
        assert shot2.ball_speed == 140.0

    def test_same_shot_suppressed(self) -> None:
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        metrics = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }

        # First call: baseline capture
        capture.capture.return_value = frame1
        ocr.read_metrics.return_value = metrics.copy()
        assert detector.poll() is None  # baseline

        # Second call: different frame, different metrics → fires shot
        capture.capture.return_value = frame2
        ocr.read_metrics.return_value = {
            "ball_speed": 130.0,
            "launch_angle": 15.0,
            "launch_direction": -2.0,
        }
        shot = detector.poll()
        assert shot is not None

        # Third call: different frame but same metrics as second → suppressed
        capture.capture.return_value = frame3
        ocr.read_metrics.return_value = {
            "ball_speed": 130.0,
            "launch_angle": 15.0,
            "launch_direction": -2.0,
        }
        result = detector.poll()
        assert result is None

    def test_invalid_metrics_returns_none(self) -> None:
        detector, ocr, capture = self._make_detector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        capture.capture.return_value = frame
        ocr.read_metrics.return_value = {
            "ball_speed": 999.0,  # Out of range
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert detector.poll() is None
