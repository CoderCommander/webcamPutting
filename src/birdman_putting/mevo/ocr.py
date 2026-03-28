"""Tesseract OCR wrapper for reading Mevo shot metrics from screenshots."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ROI:
    """A named region of interest on the Mevo display."""

    name: str  # e.g. "ball_speed", "launch_angle"
    x: int
    y: int
    width: int
    height: int


# Common OCR misreads for digits
_CHAR_FIXES: dict[str, str] = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "D": "0",
    "G": "6",
    "q": "9",
}

# Metrics where R/L suffix indicates direction (R=positive, L=negative)
_SIGNED_METRICS: set[str] = {"launch_direction", "spin_axis"}

# Expected maximum values for each metric — values above these likely have
# a missing decimal point from OCR (e.g. "85R" should be "8.5R")
_MAX_SANE_VALUES: dict[str, float] = {
    "launch_direction": 45.0,
    "launch_angle": 50.0,
    "spin_axis": 60.0,
    "smash_factor": 2.0,
}


def _fix_ocr_text(text: str) -> str:
    """Apply common character substitutions for digit-mode OCR.

    Preserves trailing R/L characters for direction parsing.
    """
    cleaned = text.strip()
    result: list[str] = []
    for ch in cleaned:
        if ch in _CHAR_FIXES:
            result.append(_CHAR_FIXES[ch])
        elif ch.isdigit() or ch in ".-":
            result.append(ch)
        elif ch in "RrLl":
            result.append(ch.upper())
    return "".join(result)


def _parse_float(text: str, signed: bool = False) -> float | None:
    """Parse a float from OCR text, returning None on failure.

    Args:
        text: Raw OCR text.
        signed: If True, treat trailing R as positive and L as negative.
    """
    fixed = _fix_ocr_text(text)
    if not fixed:
        return None

    # Detect R/L direction suffix
    negate = False
    if signed and fixed and fixed[-1] in "RL":
        if fixed[-1] == "L":
            negate = True
        fixed = fixed[:-1]

    # Handle stray leading/trailing dots or dashes
    fixed = fixed.strip(".")
    # Strip any remaining letters (e.g. R/L in non-signed mode)
    fixed = re.sub(r"[A-Za-z]", "", fixed)
    # Allow a single leading dash for negative values
    match = re.match(r"^-?\d+\.?\d*$", fixed)
    if not match:
        return None
    try:
        value = float(match.group())
        return -value if negate else value
    except ValueError:
        return None


class MevoOCR:
    """Extracts shot metrics from a Mevo display screenshot via Tesseract OCR."""

    def __init__(
        self,
        rois: list[ROI],
        tessdata_dir: str | None = None,
    ) -> None:
        import pytesseract  # noqa: F811  # lazy import

        self._pytesseract = pytesseract
        self._rois = rois
        self._tessdata_dir = tessdata_dir

        # Build Tesseract config string (include R/L for direction suffixes)
        parts = ["--psm 7", "-c tessedit_char_whitelist=0123456789.-RL"]
        if tessdata_dir:
            parts.append(f"--tessdata-dir {tessdata_dir}")
        self._tess_config = " ".join(parts)

    def read_metrics(self, frame: np.ndarray) -> dict[str, float | None]:
        """Read all configured ROIs from the frame.

        Args:
            frame: BGR screenshot image.

        Returns:
            Dict mapping ROI name to parsed float (or None if unreadable).
        """
        results: dict[str, float | None] = {}
        for roi in self._rois:
            crop = self._crop_roi(frame, roi)
            if crop is None or crop.size == 0:
                results[roi.name] = None
                continue
            preprocessed = self._preprocess(crop)
            text = self._ocr(preprocessed)
            signed = roi.name in _SIGNED_METRICS
            value = _parse_float(text, signed=signed)

            # Sanity check: if value exceeds expected max, the decimal
            # was likely missed by OCR (e.g. "85R" should be "8.5R")
            if value is not None and roi.name in _MAX_SANE_VALUES:
                max_val = _MAX_SANE_VALUES[roi.name]
                if abs(value) > max_val and abs(value / 10) <= max_val:
                    corrected = value / 10
                    logger.info(
                        "ROI '%s': %.1f exceeds max %.0f — "
                        "correcting missing decimal to %.1f",
                        roi.name, value, max_val, corrected,
                    )
                    value = corrected

            if value is not None:
                logger.debug("ROI '%s': raw='%s' → %.2f", roi.name, text.strip(), value)
            else:
                logger.debug("ROI '%s': raw='%s' → None", roi.name, text.strip())
            results[roi.name] = value
        return results

    @staticmethod
    def _crop_roi(frame: np.ndarray, roi: ROI) -> np.ndarray | None:
        """Crop a region of interest from the frame.

        Returns None if the ROI is outside the frame bounds.
        """
        h, w = frame.shape[:2]
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(w, roi.x + roi.width)
        y2 = min(h, roi.y + roi.height)
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "ROI '%s' (%d,%d %dx%d) outside frame (%dx%d)",
                roi.name, roi.x, roi.y, roi.width, roi.height, w, h,
            )
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Preprocess a cropped ROI for better OCR accuracy."""
        # Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Upscale small crops for better OCR (decimal points need resolution)
        h, w = gray.shape[:2]
        if h < 60:
            scale = max(2, 60 // h)
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Threshold (white text on dark background → invert)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure text is dark on light background (Tesseract prefers this)
        # If most pixels are dark, the text is likely light — invert
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)

        # Light dilation to preserve decimal points that may be thin
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        return thresh

    def _ocr(self, image: np.ndarray) -> str:
        """Run Tesseract OCR on a preprocessed image."""
        return str(self._pytesseract.image_to_string(image, config=self._tess_config))
