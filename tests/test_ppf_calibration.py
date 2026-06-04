"""Tests for the calibration-marker detector used by OBS Auto Cal."""

from __future__ import annotations

import cv2
import numpy as np

from birdman_putting.ppf_calibration import (
    detect_calibration_markers,
    validate_marker_spacing,
)


def _make_frame_with_markers(
    width: int = 640,
    height: int = 360,
    band_y: int = 180,
    spacing_px: int = 55,
    n_markers: int = 6,
    marker_w: int = 4,
    marker_h: int = 14,
    first_x: int = 50,
) -> np.ndarray:
    """Build a synthetic frame: dark green background, bright vertical bars."""
    # Dark green mat
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 90, 30)  # BGR — dark green

    for i in range(n_markers):
        x = first_x + i * spacing_px
        # Bright marker (white)
        cv2.rectangle(
            frame,
            (x - marker_w // 2, band_y - marker_h // 2),
            (x + marker_w // 2, band_y + marker_h // 2),
            (255, 255, 255),
            thickness=-1,
        )
    return frame


class TestDetectCalibrationMarkers:
    def test_finds_evenly_spaced_markers(self) -> None:
        frame = _make_frame_with_markers(n_markers=6, spacing_px=55, band_y=180)
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.centers) == 6
        assert result.median_spacing_px is not None
        assert abs(result.median_spacing_px - 55.0) < 1.0

    def test_handles_uneven_spacing(self) -> None:
        # Build a frame with corks at uneven spacing — median should pick
        # the middle value of the inter-marker distances
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 30)
        xs = [50, 100, 160, 230, 310, 400]  # spacings: 50, 60, 70, 80, 90
        for x in xs:
            cv2.rectangle(frame, (x - 2, 173), (x + 2, 187), (255, 255, 255), -1)
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.centers) == 6
        # spacings: 50, 60, 70, 80, 90 -> median = 70
        assert result.median_spacing_px is not None
        assert abs(result.median_spacing_px - 70.0) < 1.0

    def test_returns_empty_when_no_markers(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 30)  # Nothing bright
        result = detect_calibration_markers(frame, band_y_center=180)
        assert result.centers == []
        assert result.median_spacing_px is None
        assert result.diffs == []

    def test_ignores_markers_outside_band(self) -> None:
        """Bright blobs above/below the band should not be picked up."""
        frame = _make_frame_with_markers(n_markers=4, band_y=180, spacing_px=55)
        # Add an extra bright blob at y=50 (way above the band)
        cv2.rectangle(frame, (200, 45), (204, 55), (255, 255, 255), -1)
        result = detect_calibration_markers(
            frame, band_y_center=180, band_half_height=40,
        )
        assert len(result.centers) == 4

    def test_centers_sorted_left_to_right(self) -> None:
        frame = _make_frame_with_markers(n_markers=5)
        result = detect_calibration_markers(frame, band_y_center=180)
        xs = [c[0] for c in result.centers]
        assert xs == sorted(xs)

    def test_diffs_match_centers(self) -> None:
        frame = _make_frame_with_markers(n_markers=4, spacing_px=55)
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.diffs) == len(result.centers) - 1
        for i, d in enumerate(result.diffs):
            expected = result.centers[i + 1][0] - result.centers[i][0]
            assert abs(d - expected) < 0.01

    def test_filters_giant_blobs(self) -> None:
        """A huge bright region should not be mistaken for a marker."""
        frame = _make_frame_with_markers(n_markers=4)
        # Add a wide bright bar
        cv2.rectangle(frame, (300, 165), (430, 195), (255, 255, 255), -1)
        result = detect_calibration_markers(
            frame, band_y_center=180, max_area=200.0,
        )
        # Only the original 4 narrow markers should be detected
        assert len(result.centers) == 4

    def test_hue_filter_excludes_orange_ball_with_cyan_markers(self) -> None:
        """When hue_range is set to cyan, orange decoys are ignored."""
        # Build a frame with cyan markers + an orange "ball" + an orange trail
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 30)  # green background

        # 5 cyan markers at 60-px spacing (BGR cyan = (255, 255, 0))
        for i in range(5):
            x = 100 + i * 60
            cv2.rectangle(frame, (x - 2, 173), (x + 2, 187), (255, 255, 0), -1)

        # Orange ball at x=50 (well before the first cyan marker)
        # BGR orange ≈ (0, 100, 255)
        cv2.circle(frame, (50, 180), 6, (0, 100, 255), -1)

        # Orange trail dot at x=500 (way past last marker)
        cv2.rectangle(frame, (498, 175), (502, 185), (0, 100, 255), -1)

        # Without hue filter — would detect markers + ball + trail
        no_filter = detect_calibration_markers(
            frame, band_y_center=180, hue_range=None,
        )
        assert len(no_filter.centers) >= 6  # 5 markers + ball/trail

        # With cyan hue filter — only the 5 cyan markers
        cyan_only = detect_calibration_markers(
            frame, band_y_center=180, hue_range=(75, 105), min_saturation=80,
        )
        assert len(cyan_only.centers) == 5
        assert cyan_only.median_spacing_px is not None
        assert abs(cyan_only.median_spacing_px - 60.0) < 1.0

    def test_hue_filter_handles_wrap_around(self) -> None:
        """Hue range that wraps around 0 (e.g., red 170-10) works correctly."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 30)
        # Red bars (BGR red ≈ (0, 0, 255), HSV hue ≈ 0)
        for i in range(4):
            x = 100 + i * 70
            cv2.rectangle(frame, (x - 2, 173), (x + 2, 187), (0, 0, 255), -1)
        # Wrap-around hue range covering red (170-179 + 0-10)
        result = detect_calibration_markers(
            frame, band_y_center=180, hue_range=(170, 10), min_saturation=80,
        )
        assert len(result.centers) == 4


class TestSpacingValidation:
    """A missing/extra marker must be detected so it can't silently shift
    the foot scale.  Exposed via MarkerDetectionResult.spacing_uniform and
    .gap_outliers (indices of the deviating gaps)."""

    def _frame_at_xs(self, xs: list[int]) -> np.ndarray:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 30)
        for x in xs:
            cv2.rectangle(frame, (x - 2, 173), (x + 2, 187), (255, 255, 255), -1)
        return frame

    def test_uniform_markers_flagged_uniform(self) -> None:
        """Evenly-spaced markers → spacing_uniform True, no gap outliers."""
        frame = self._frame_at_xs([50, 105, 160, 215, 270, 325])
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.centers) == 6
        assert result.spacing_uniform is True
        assert result.gap_outliers == []

    def test_missing_interior_marker_flagged(self) -> None:
        """One interior marker missing → a ~2x gap → flagged non-uniform
        and the oversized gap index is reported."""
        # 55px spacing, but the 4th marker (would be at 50+3*55=215) is
        # absent → gap from 160 to 270 is 110px ≈ 2x the 55px median.
        frame = self._frame_at_xs([50, 105, 160, 270, 325, 380])
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.centers) == 6
        assert result.spacing_uniform is False
        # The big gap is between centers[2] and centers[3] → index 2.
        assert 2 in result.gap_outliers

    def test_smooth_fisheye_gradient_not_flagged(self) -> None:
        """A gradual spacing gradient (legit fisheye) must NOT be flagged
        as a missing marker — only sudden ~2x jumps are."""
        # spacings 50,60,70,80,90 — each within ~35% of the median (70).
        frame = self._frame_at_xs([50, 100, 160, 230, 310, 400])
        result = detect_calibration_markers(frame, band_y_center=180)
        assert len(result.centers) == 6
        assert result.spacing_uniform is True
        assert result.gap_outliers == []

    def test_validate_marker_spacing_direct_missing(self) -> None:
        """validate_marker_spacing exposes the uniformity decision for
        callers/tests directly from a centers list."""
        centers = [(0.0, 0.0), (57.0, 0.0), (114.0, 0.0), (228.0, 0.0)]
        v = validate_marker_spacing(centers)
        assert v.uniform is False
        assert 2 in v.gap_outliers  # gap 114->228 is the bad one
        # Median spacing should be the robust ~57 (NOT pulled up by the gap).
        assert abs(v.median_spacing - 57.0) < 1.0

    def test_validate_marker_spacing_direct_uniform(self) -> None:
        centers = [(0.0, 0.0), (57.0, 0.0), (114.0, 0.0), (171.0, 0.0), (228.0, 0.0)]
        v = validate_marker_spacing(centers)
        assert v.uniform is True
        assert v.gap_outliers == []
        assert abs(v.median_spacing - 57.0) < 1e-9

    def test_validate_too_few_markers(self) -> None:
        """0 or 1 markers: nothing to validate, uniform True, no outliers."""
        assert validate_marker_spacing([]).uniform is True
        assert validate_marker_spacing([(10.0, 0.0)]).uniform is True
        assert validate_marker_spacing([]).gap_outliers == []
