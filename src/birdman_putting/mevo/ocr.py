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
_SIGNED_METRICS: set[str] = {
    "launch_direction", "spin_axis", "club_path", "face_to_target",
    "curve",
}

# Known display format of each metric on the FS Golf PC / Mevo readout,
# expressed as the number of decimal places shown. This drives POSITION /
# MASK based decimal parsing: when OCR loses the decimal point we re-insert
# it at the documented position instead of blindly dividing by 10.
#
# Rationale: dividing by 10 cannot tell a dropped decimal point apart from a
# dropped digit. e.g. a real "52.6" misread as "26" used to become 2.6; a
# real "526" (already an OCR error) used to become 52.6. The fix instead
# checks the DIGIT COUNT against the field's mask and REJECTS reads that
# don't fit, rather than fabricating a plausible-looking value.
#
# One decimal place ("NN.N") for the launch-monitor floats; two ("N.NN")
# for smash factor; ZERO for integer displays.
FIELD_DECIMALS: dict[str, int] = {
    # --- one decimal place (NN.N) ---
    "ball_speed": 1,
    "launch_angle": 1,
    "launch_direction": 1,
    "spin_axis": 1,
    "club_speed": 1,
    "club_path": 1,
    "face_to_target": 1,
    "aoa": 1,
    "dynamic_loft": 1,
    "descent_angle": 1,
    "flight_time": 1,
    "lateral_impact": 1,
    "vertical_impact": 1,
    "curve": 1,
    # --- two decimal places (N.NN) ---
    "smash_factor": 2,
    # --- integer displays (no decimal point shown) ---
    "spin_rate": 0,
    "carry_distance": 0,
    "total_distance": 0,
    "apex_height": 0,
    "roll_distance": 0,
}

# Maximum number of INTEGER-part digits each field's display can show. This
# pins the mask per-field so a value with too many digits is rejected rather
# than silently re-pointed. e.g. launch_angle maxes at 65.0 (2 int digits),
# so "5260" (would be 4 total digits → 3 int digits) is out of format and is
# REJECTED — it is NOT divided by 10 into 526.0. ball_speed reaches ~250 mph
# (3 int digits), so "1234" → 123.4 is allowed.
#
# Total allowed digit count for a field = int_digits + FIELD_DECIMALS[field].
# Minimum allowed = decimals + 1 (need at least one integer digit). Fields
# not listed here use _DEFAULT_INT_DIGITS.
_DEFAULT_INT_DIGITS: int = 3
FIELD_INT_DIGITS: dict[str, int] = {
    "ball_speed": 3,        # up to ~250.x mph
    "launch_angle": 2,      # VLA up to ~65 deg
    "launch_direction": 2,  # HLA up to ~45 deg
    "spin_axis": 2,         # up to ~90 deg
    "club_speed": 3,        # up to ~150 mph
    "club_path": 2,         # up to ~30 deg
    "face_to_target": 2,    # up to ~30 deg
    "aoa": 2,               # up to ~20 deg
    "dynamic_loft": 2,      # up to ~70 deg
    "descent_angle": 2,     # up to ~90 deg
    "flight_time": 2,       # up to ~15 s
    "lateral_impact": 1,    # +/- ~2 in
    "vertical_impact": 1,   # +/- ~2 in
    "curve": 3,             # up to ~100 yds
    "smash_factor": 1,      # ~1.xx
    # integer displays
    "spin_rate": 5,         # up to ~15000 rpm
    "carry_distance": 3,    # up to ~500 yds
    "total_distance": 3,    # up to ~550 yds
    "apex_height": 3,       # up to ~250 ft
    "roll_distance": 3,     # up to ~200 yds
}


def _digit_bounds(name: str, decimals: int) -> tuple[int, int]:
    """Return (min_total_digits, max_total_digits) allowed for a field."""
    int_digits = FIELD_INT_DIGITS.get(name, _DEFAULT_INT_DIGITS)
    max_total = int_digits + decimals
    min_total = decimals + 1  # at least one integer digit
    return min_total, max_total


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


def parse_field(name: str, text: str) -> float | None:
    """Parse a single metric's OCR text using its known display format.

    This is the field-aware replacement for the old blind ÷10 decimal guess.
    It handles three concerns the generic ``_parse_float`` cannot:

    1. **Mask-based decimal recovery.** If the field is documented to show
       ``D`` decimal places (see :data:`FIELD_DECIMALS`) and the OCR text has
       NO ``.``, the decimal point is re-inserted ``D`` places from the right
       (e.g. ``launch_angle`` "526" -> 52.6). If the digit count does not fit
       the field's mask, the read is REJECTED (returns ``None``) — we never
       guess between a dropped dot and a dropped digit. Reads that already
       contain a ``.`` are trusted as-is.

    2. **Signed-direction failure.** For signed metrics (R/L suffix), a
       NON-ZERO magnitude with no R/L suffix is a parse FAILURE (``None``) —
       a missed "L" must not silently become a positive value. A zero
       magnitude has no direction to lose and is accepted as 0.0.

    3. **Unknown fields** (no registered format) fall back to plain parsing.

    Args:
        name: Metric name (e.g. "launch_angle", "curve").
        text: Raw OCR text for that metric.

    Returns:
        Parsed float, or ``None`` if the read is unusable / implausible.
    """
    signed = name in _SIGNED_METRICS
    cleaned = _fix_ocr_text(text)
    if not cleaned:
        return None

    # --- Signed direction handling (before any decimal work) ---
    has_suffix = cleaned[-1] in "RL"
    negate = signed and has_suffix and cleaned[-1] == "L"
    body = cleaned[:-1] if (cleaned and cleaned[-1] in "RL") else cleaned
    # Strip any other stray letters and surrounding dots/dashes.
    body = re.sub(r"[A-Za-z]", "", body).strip()

    # Determine sign from a leading dash too (rare on this display).
    if body.startswith("-"):
        negate = True
        body = body[1:]

    if not body or not any(c.isdigit() for c in body):
        return None

    decimals = FIELD_DECIMALS.get(name)

    # --- Decimal recovery / validation ---
    if "." in body:
        # OCR captured a decimal point — trust the value as displayed.
        try:
            magnitude = float(body)
        except ValueError:
            return None
    elif decimals is None:
        # Unknown field, no dot — parse plainly.
        try:
            magnitude = float(body)
        except ValueError:
            return None
    else:
        digits = body  # only digits remain at this point
        if not digits.isdigit():
            return None
        n = len(digits)
        lo, hi = _digit_bounds(name, decimals)
        if not (lo <= n <= hi):
            logger.warning(
                "Field '%s': digit count %d in '%s' does not fit mask "
                "(%d decimals, [%d..%d] total digits) — rejecting "
                "(no ÷10 guess)",
                name, n, text.strip(), decimals, lo, hi,
            )
            return None
        if decimals == 0:
            # Integer field: validated digit count, no dot insertion.
            magnitude = float(digits)
        else:
            # Insert the decimal point `decimals` places from the right.
            int_part = digits[:-decimals]
            frac_part = digits[-decimals:]
            magnitude = float(f"{int_part}.{frac_part}")
            logger.debug(
                "Field '%s': mask-recovered '%s' -> %.3f (no decimal in OCR)",
                name, text.strip(), magnitude,
            )

    # --- Signed failure check ---
    if signed and not has_suffix and magnitude != 0.0:
        logger.warning(
            "Field '%s': signed value %.2f from '%s' has no R/L suffix — "
            "direction unknown, rejecting shot (was previously assumed +). "
            "Widen the ROI to capture the R/L character.",
            name, magnitude, text.strip(),
        )
        return None

    return -magnitude if negate else magnitude


class MevoOCR:
    """Extracts shot metrics from a Mevo display screenshot via Tesseract OCR."""

    def __init__(
        self,
        rois: list[ROI],
        tessdata_dir: str | None = None,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        import pytesseract  # noqa: F811  # lazy import

        self._pytesseract = pytesseract
        self._rois = rois
        self._tessdata_dir = tessdata_dir

        # Build Tesseract config string (include R/L for direction suffixes)
        parts = ["--psm 7", "-c tessedit_char_whitelist=0123456789.-RL"]
        if tessdata_dir:
            parts.append(f"--tessdata-dir {tessdata_dir}")
        self._tess_config = " ".join(parts)

        # Persistent thread pool (avoid create/destroy overhead per poll)
        self._pool = ThreadPoolExecutor(max_workers=min(len(rois), 4))

        # Pre-allocate reusable kernel for dilation
        self._dilate_kernel = np.ones((2, 2), np.uint8)

    def _read_single_roi(
        self, frame: np.ndarray, roi: ROI,
    ) -> tuple[str, float | None]:
        """Read a single ROI — designed to run in a thread pool."""
        crop = self._crop_roi(frame, roi)
        if crop is None or crop.size == 0:
            return roi.name, None
        preprocessed = self._preprocess(crop)
        text = self._ocr(preprocessed)
        # Field-aware parse: mask-based decimal recovery (no blind ÷10) plus
        # signed-direction failure handling. See ocr.parse_field().
        value = parse_field(roi.name, text)

        if value is not None:
            logger.debug("ROI '%s': raw='%s' → %.2f", roi.name, text.strip(), value)
        else:
            logger.debug("ROI '%s': raw='%s' → None", roi.name, text.strip())
        return roi.name, value

    def read_metrics(self, frame: np.ndarray) -> dict[str, float | None]:
        """Read all configured ROIs from the frame.

        Uses a thread pool to OCR multiple ROIs concurrently. Tesseract
        releases the GIL during its C processing, so parallel threads
        give a significant speedup (e.g. 18 ROIs in ~1s instead of ~6s).

        Args:
            frame: BGR screenshot image.

        Returns:
            Dict mapping ROI name to parsed float (or None if unreadable).
        """
        futures = [
            self._pool.submit(self._read_single_roi, frame, roi)
            for roi in self._rois
        ]
        results: dict[str, float | None] = {}
        for future in futures:
            name, value = future.result()
            results[name] = value
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

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess a cropped ROI for better OCR accuracy."""
        # Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Upscale small crops for better OCR (decimal points need resolution)
        h, w = gray.shape[:2]
        if h < 60:
            scale = max(2, 60 // h)
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

        # Threshold (white text on dark background → invert)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure text is dark on light background (Tesseract prefers this)
        # If most pixels are dark, the text is likely light — invert
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)

        # Light dilation to preserve decimal points that may be thin
        thresh = cv2.dilate(thresh, self._dilate_kernel, iterations=1)

        return thresh

    def _ocr(self, image: np.ndarray) -> str:
        """Run Tesseract OCR on a preprocessed image."""
        return str(self._pytesseract.image_to_string(image, config=self._tess_config))
