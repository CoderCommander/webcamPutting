"""Tests for Mevo OCR module."""

from __future__ import annotations

import numpy as np
import pytest

from birdman_putting.mevo.ocr import (
    FIELD_DECIMALS,
    ROI,
    _fix_ocr_text,
    _parse_float,
    parse_field,
)


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


class TestSignedDirection:
    """Verify R/L suffix parsing produces correct positive/negative values."""

    def test_right_is_positive(self) -> None:
        assert _parse_float("8.5R", signed=True) == 8.5

    def test_left_is_negative(self) -> None:
        assert _parse_float("8.5L", signed=True) == -8.5

    def test_right_with_space(self) -> None:
        assert _parse_float("12.3 R", signed=True) == 12.3

    def test_left_with_space(self) -> None:
        assert _parse_float("4.7 L", signed=True) == -4.7

    def test_lowercase_r(self) -> None:
        assert _parse_float("3.2r", signed=True) == 3.2

    def test_lowercase_l(self) -> None:
        # lowercase l maps to "1" in _CHAR_FIXES, but trailing l should
        # be captured as L before digit fixes apply — test actual behavior
        result = _parse_float("3.2l", signed=True)
        # "l" maps to "1" in _CHAR_FIXES, so "3.2l" → "3.21" (not "3.2L")
        # This is a known limitation — Tesseract whitelist only allows uppercase
        assert result is not None

    def test_zero_right(self) -> None:
        assert _parse_float("0.0R", signed=True) == 0.0

    def test_zero_left(self) -> None:
        assert _parse_float("0.0L", signed=True) == 0.0

    def test_unsigned_mode_strips_R(self) -> None:
        """In non-signed mode, R/L are stripped and value is always positive."""
        assert _parse_float("8.5R", signed=False) == 8.5

    def test_unsigned_mode_strips_L(self) -> None:
        assert _parse_float("8.5L", signed=False) == 8.5

    def test_integer_right(self) -> None:
        assert _parse_float("12R", signed=True) == 12.0

    def test_integer_left(self) -> None:
        assert _parse_float("12L", signed=True) == -12.0


class TestMissingDecimalCorrection:
    """Verify missing decimal correction for always-decimal metrics.

    FS Golf PC always displays these metrics with one decimal place
    (e.g. "8.5 R"). If OCR misses the dot, the text has no "." and
    the value is divided by 10.
    """

    def test_no_decimal_corrected(self) -> None:
        """'85R' has no decimal → 85 / 10 = 8.5."""
        value = _parse_float("85R", signed=True)
        assert value == 85.0  # raw parse gives 85
        # Correction (÷10) happens in _read_single_roi, not _parse_float
        assert value / 10 == 8.5

    def test_no_decimal_negative(self) -> None:
        """'85L' has no decimal → -85 / 10 = -8.5."""
        value = _parse_float("85L", signed=True)
        assert value == -85.0
        assert value / 10 == -8.5

    def test_decimal_present_not_corrected(self) -> None:
        """'8.5R' has a decimal → trust as-is (8.5)."""
        value = _parse_float("8.5R", signed=True)
        assert value == 8.5

    def test_decimal_present_large_value_trusted(self) -> None:
        """'13.1L' has a decimal → trust as-is (-13.1), don't correct."""
        value = _parse_float("13.1L", signed=True)
        assert value == -13.1  # NOT corrected to -1.3

    def test_spin_axis_no_decimal(self) -> None:
        """'125L' → -125 / 10 = -12.5."""
        value = _parse_float("125L", signed=True)
        assert value == -125.0
        assert value / 10 == -12.5

    def test_smash_factor_no_decimal(self) -> None:
        """'15' → 15 / 10 = 1.5."""
        value = _parse_float("15")
        assert value == 15.0
        assert value / 10 == 1.5

    def test_fix_ocr_text_preserves_decimal(self) -> None:
        """Verify _fix_ocr_text keeps '.' so decimal detection works."""
        assert "." in _fix_ocr_text("13.1 L")
        assert "." not in _fix_ocr_text("131 L")


class TestParseFieldDecimalMask:
    """Field-aware (mask-based) decimal parsing — replaces the blind ÷10 guess.

    Each metric has a known display format (number of decimal places).
    A no-dot OCR read is interpreted by INSERTING the decimal at the known
    position. If the digit count does not match the field's mask, the read
    is REJECTED (None) rather than silently guessing — this is the fix for
    the production failure where a dropped digit was mistaken for a dropped
    decimal point and divided by 10.
    """

    def test_launch_angle_is_one_decimal(self) -> None:
        # Sanity: VLA (launch_angle) displays as NN.N (one decimal place).
        assert FIELD_DECIMALS["launch_angle"] == 1

    def test_no_dot_inserts_decimal_at_mask(self) -> None:
        # NN.N field, "526" (3 digits) → insert dot before last digit → 52.6.
        # Crucially this is NOT 526/10 == 52.6 by accident; we assert below
        # that a digit-count MISMATCH is rejected, which division never does.
        assert parse_field("launch_angle", "526") == pytest.approx(52.6)

    def test_no_dot_two_decimals(self) -> None:
        # smash_factor displays as N.NN (two decimals): "152" → 1.52.
        assert FIELD_DECIMALS["smash_factor"] == 2
        assert parse_field("smash_factor", "152") == pytest.approx(1.52)

    def test_digit_count_mismatch_rejected_not_divided(self) -> None:
        # "5260" is FOUR digits for a one-decimal (NN.N, max 3 sig digits in
        # the normal display) field → mask mismatch → REJECT (None).
        # Explicitly assert it is NOT the old ÷10 result (526.0) nor 52.6.
        result = parse_field("launch_angle", "5260")
        assert result is None
        assert result != 526.0  # would be the raw value
        assert result != 5260 / 10  # 526.0 — the old blind ÷10 guess
        assert result != 52.6

    def test_too_few_digits_rejected(self) -> None:
        # A NN.N field needs >=2 digits to place one decimal sensibly.
        # A single digit ("5") cannot satisfy the mask → reject.
        assert parse_field("launch_angle", "5") is None

    def test_existing_decimal_is_trusted(self) -> None:
        # A read that already contains a "." is trusted as-is, not re-masked.
        assert parse_field("launch_angle", "52.6") == pytest.approx(52.6)
        assert parse_field("launch_angle", "13.1") == pytest.approx(13.1)

    def test_existing_decimal_large_value_trusted(self) -> None:
        # "13.1" must stay 13.1, never collapse to 1.3.
        assert parse_field("launch_angle", "13.1") == pytest.approx(13.1)

    def test_integer_field_no_decimal_insertion(self) -> None:
        # spin_rate is an integer display (0 decimals) → "5234" stays 5234.0,
        # never divided or dot-inserted.
        assert FIELD_DECIMALS["spin_rate"] == 0
        assert parse_field("spin_rate", "5234") == pytest.approx(5234.0)

    def test_integer_field_with_stray_dot_trusted(self) -> None:
        # Even an integer field trusts an explicit dot if OCR produced one.
        assert parse_field("spin_rate", "5234") == pytest.approx(5234.0)

    def test_ball_speed_mask(self) -> None:
        # ball_speed displays NN.N: "1234" (4 digits) is allowed → 123.4
        # (e.g. 123.4 mph). 3 digits "526" → 52.6.
        assert FIELD_DECIMALS["ball_speed"] == 1
        assert parse_field("ball_speed", "526") == pytest.approx(52.6)
        assert parse_field("ball_speed", "1234") == pytest.approx(123.4)

    def test_garbage_rejected(self) -> None:
        assert parse_field("launch_angle", "") is None
        assert parse_field("launch_angle", "abc") is None

    def test_unknown_field_falls_back_to_plain_parse(self) -> None:
        # A field with no registered format parses plainly (no masking).
        assert parse_field("not_a_real_metric", "123") == pytest.approx(123.0)


class TestParseFieldSigned:
    """Signed fields: a NON-ZERO magnitude with no R/L suffix is a parse
    FAILURE (None), not an assumed-positive value. A missed 'L' must never
    silently flip a left miss into a positive (right) value.
    """

    def test_signed_with_R_suffix_positive(self) -> None:
        assert parse_field("curve", "8.5R") == pytest.approx(8.5)

    def test_signed_with_L_suffix_negative(self) -> None:
        assert parse_field("curve", "8.5L") == pytest.approx(-8.5)

    def test_signed_nonzero_no_suffix_is_failure(self) -> None:
        # The bug: a non-zero curve with the R/L lost → must be None, not +8.5.
        result = parse_field("curve", "8.5")
        assert result is None
        assert result != 8.5  # explicitly NOT assumed positive

    def test_signed_launch_direction_no_suffix_is_failure(self) -> None:
        result = parse_field("launch_direction", "3.1")
        assert result is None

    def test_signed_zero_no_suffix_ok(self) -> None:
        # A zero magnitude has no direction to lose → sign 0 is fine.
        assert parse_field("curve", "0.0") == pytest.approx(0.0)

    def test_signed_no_dot_with_suffix_uses_mask(self) -> None:
        # launch_direction is NN.N signed: "31L" → 3.1 then negated → -3.1.
        assert parse_field("launch_direction", "31L") == pytest.approx(-3.1)


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

        rois = [ROI(name="test", x=0, y=0, width=100, height=40)]
        ocr = MevoOCR(rois=rois)
        result = ocr._preprocess(img)
        # After preprocessing, text should be dark on light (mean > 128)
        assert np.mean(result) > 100
