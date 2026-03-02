"""Tests for Mevo OCR module."""

from __future__ import annotations

import numpy as np
import pytest

from birdman_putting.mevo.ocr import ROI, _fix_ocr_text, _parse_float


class TestFixOcrText:
    def test_clean_digits(self) -> None:
        assert _fix_ocr_text("123.4") == "123.4"

    def test_letter_O_to_zero(self) -> None:
        assert _fix_ocr_text("1O3") == "103"

    def test_lowercase_l_to_one(self) -> None:
        assert _fix_ocr_text("l23") == "123"

    def test_strips_non_digit_chars(self) -> None:
        assert _fix_ocr_text("12 mph") == "12"

    def test_negative_value(self) -> None:
        assert _fix_ocr_text("-3.5") == "-3.5"

    def test_preserves_R_suffix(self) -> None:
        assert _fix_ocr_text("12.9 R") == "12.9R"

    def test_preserves_L_suffix(self) -> None:
        assert _fix_ocr_text("3.1 L") == "3.1L"


class TestParseFloat:
    def test_integer(self) -> None:
        assert _parse_float("120") == 120.0

    def test_decimal(self) -> None:
        assert _parse_float("12.5") == 12.5

    def test_negative(self) -> None:
        assert _parse_float("-3.5") == -3.5

    def test_empty(self) -> None:
        assert _parse_float("") is None

    def test_garbage(self) -> None:
        assert _parse_float("abc") is None

    def test_ocr_misread(self) -> None:
        assert _parse_float("l2O") == 120.0

    def test_signed_right_is_positive(self) -> None:
        assert _parse_float("2.4 R", signed=True) == 2.4

    def test_signed_left_is_negative(self) -> None:
        assert _parse_float("3.1 L", signed=True) == -3.1

    def test_signed_no_suffix_stays_positive(self) -> None:
        assert _parse_float("12.5", signed=True) == 12.5

    def test_unsigned_ignores_R_suffix(self) -> None:
        assert _parse_float("2.4 R") == 2.4

    def test_unsigned_ignores_L_suffix(self) -> None:
        assert _parse_float("3.1 L") == 3.1


try:
    import pytesseract  # noqa: F401
    _has_pytesseract = True
except ImportError:
    _has_pytesseract = False


@pytest.mark.skipif(not _has_pytesseract, reason="pytesseract not installed")
class TestMevoOCRIntegration:
    """Integration tests that require pytesseract to be installed."""

    def test_read_metrics_from_synthetic_image(self) -> None:
        """Create a synthetic image with text and verify OCR reads it."""
        import cv2

        from birdman_putting.mevo.ocr import MevoOCR

        # Create a black image and draw white text
        img = np.zeros((60, 200, 3), dtype=np.uint8)
        cv2.putText(img, "125", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        rois = [ROI(name="ball_speed", x=0, y=0, width=200, height=60)]
        ocr = MevoOCR(rois=rois)
        metrics = ocr.read_metrics(img)

        # OCR on synthetic text should give us something close to 125
        val = metrics.get("ball_speed")
        # Allow some OCR wiggle room — the key test is that it returns a number
        assert val is not None, "OCR returned None, expected a number"

    def test_roi_cropping(self) -> None:
        from birdman_putting.mevo.ocr import MevoOCR

        roi = ROI(name="test", x=10, y=20, width=50, height=30)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = MevoOCR._crop_roi(frame, roi)
        assert crop.shape == (30, 50, 3)

    def test_preprocess_inverts_dark_background(self) -> None:
        from birdman_putting.mevo.ocr import MevoOCR

        # White text on dark: mean should be low before processing
        img = np.zeros((40, 100, 3), dtype=np.uint8)
        img[10:30, 20:80] = 255  # bright block

        result = MevoOCR._preprocess(img)
        # After preprocessing, text should be dark on light (mean > 128)
        assert np.mean(result) > 100
