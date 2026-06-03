"""Screenshot+OCR watcher for the GSPro window's current-club display.

This is a fallback signal source for OBS scene-switching when GSPro Open
Connect v1 (port 921) is unavailable — for example, when another launch
monitor's connector is the primary client and GSPro doesn't broadcast
club-selection events to other listeners.

Re-uses the existing Mevo screenshot capture (``WindowCapture``). The OCR
is fresh because Mevo's pipeline is hard-coded to numeric digits and we
need text — pytesseract is invoked with an alphabetic whitelist.

Configuration lives under ``[gspro_watcher]`` in ``config.toml``.
Calibrate the ROI with ``python -m birdman_putting --calibrate-gspro-club``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from birdman_putting.config import GSProWatcherSettings
from birdman_putting.mevo.screenshot import WindowCapture

logger = logging.getLogger(__name__)


# Wedge qualifier -> GSPro club code. Match needs both the qualifier
# and the literal "wedge" anywhere in the normalized text.
_WEDGE_QUALIFIERS: list[tuple[str, str]] = [
    ("lob", "LW"),
    ("sand", "SW"),
    ("approach", "AW"),
    ("gap", "GW"),
    ("pitch", "PW"),  # matches "pitching" too
]

# GSPro also uses short-form wedge labels — "L Wedge", "G Wedge", etc.
# Tesseract typically collapses the space, giving "LWedge", "GWedge".
# Single-letter qualifier preceding "wedge" maps to the same codes.
_SHORT_WEDGE_RE = re.compile(r"\b([lsagp])\s*wedge\b")
_SHORT_WEDGE_CODES = {
    "l": "LW",
    "s": "SW",
    "a": "AW",
    "g": "GW",
    "p": "PW",
}

# Numbered iron / wood: digit, optional whitespace, suffix word.
# Tesseract regularly:
#   - collapses spaces ("8Iron")
#   - confuses i ↔ l ↔ 1 ("5lron", "8 1RON", "9lRON")
#   - confuses o ↔ 0 ("3w00d", "3 wo0d")
# The regex tolerates each of these. Without these, _last_club gets
# stuck on the previous club and OBS never switches.
_IRON_RE = re.compile(r"(\d)\s*[il1]ron\b")
_WOOD_RE = re.compile(r"(\d)\s*w[o0][o0]d\b")


def parse_club(text: str) -> str | None:
    """Extract a GSPro club code from raw OCR text.

    Returns the 2-character GSPro club code (e.g. "PT", "DR") if a known
    club name is recognised in the text, otherwise None.

    Tolerant of:
      - Lowercase / uppercase / mixed case
      - Tesseract collapsing spaces ("8Iron" → 8I)
      - Punctuation noise ("|. PUTTER ;")
      - Surrounding extra words

    Strict about:
      - Known anchor words: putter / driver / hybrid / rescue / wedge /
        iron / wood. No anchor → no match (returns None).
    """
    if not text:
        return None

    # Normalize: lowercase, strip non-letter/digit chars, collapse spaces.
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    # Single-word clubs first — most common cases
    if "putter" in normalized:
        return "PT"
    if "driver" in normalized:
        return "DR"
    if "hybrid" in normalized or "rescue" in normalized:
        return "HY"

    # Wedges: must contain "wedge" + a recognized qualifier
    if "wedge" in normalized:
        # Long-form first ("Lob Wedge", "Sand Wedge", "Pitching Wedge"...)
        for qualifier, code in _WEDGE_QUALIFIERS:
            if qualifier in normalized:
                return code
        # Short-form ("L Wedge", "GWedge" with collapsed space)
        m = _SHORT_WEDGE_RE.search(normalized)
        if m:
            return _SHORT_WEDGE_CODES[m.group(1)]
        # "wedge" without a qualifier we recognise — don't guess

    # Numbered iron / wood — tolerate "8Iron", "8 iron", "8-iron"
    m = _IRON_RE.search(normalized)
    if m:
        return f"{m.group(1)}I"
    m = _WOOD_RE.search(normalized)
    if m:
        return f"{m.group(1)}W"

    return None


def _ocr_text(crop: np.ndarray, tessdata_dir: str | None = None) -> str:
    """Run pytesseract on a single ROI crop, returning raw text.

    Uses a text whitelist (letters + digits + space) since the GSPro
    club display shows things like "Putter", "9 Iron", "3 Wood".
    """
    import pytesseract

    if crop is None or crop.size == 0:
        return ""

    # Grayscale + upscale + threshold — same recipe as Mevo OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    if h < 60:
        scale = max(2, 60 // h)
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 128:
        thresh = cv2.bitwise_not(thresh)

    parts = [
        "--psm 7",  # treat as a single text line
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
    ]
    if tessdata_dir:
        parts.append(f"--tessdata-dir {tessdata_dir}")
    config = " ".join(parts)

    try:
        text = pytesseract.image_to_string(thresh, config=config)
    except Exception as e:
        logger.warning("pytesseract failed: %s", e)
        return ""
    return text.strip()


def _crop_roi(frame: np.ndarray, roi: list[int]) -> np.ndarray | None:
    """Crop a [x, y, w, h] ROI from frame, clamped to frame bounds."""
    if len(roi) != 4:
        return None
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + rw), min(h, y + rh)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class GSProWatcher:
    """Polls the GSPro window, OCRs the club display, fires callbacks on change.

    Run via :py:meth:`start` (spawns a background thread) and stop with
    :py:meth:`stop`.  The callback is invoked from the polling thread —
    keep it lightweight (or schedule UI work elsewhere).
    """

    def __init__(
        self,
        settings: GSProWatcherSettings,
        on_club_change: Callable[[str], None],
    ) -> None:
        self._settings = settings
        self._on_club_change = on_club_change
        self._capture = WindowCapture(settings.window_title)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_club: str | None = None  # last code we reported

    def start(self) -> bool:
        """Start the polling thread.  Returns True if window was found."""
        if not self._capture.find_window():
            logger.error(
                "GSPro window with title containing '%s' not found",
                self._settings.window_title,
            )
            return False

        if not self._settings.club_roi:
            logger.error(
                "GSPro club ROI not configured. "
                "Run: python -m birdman_putting --calibrate-gspro-club",
            )
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gspro-watcher",
        )
        self._thread.start()
        logger.info(
            "GSPro watcher started (window='%s', poll=%.1fs, ROI=%s)",
            self._settings.window_title,
            self._settings.poll_interval,
            self._settings.club_roi,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            self._capture.close()
        except Exception:
            pass

    def poll_once(self) -> str | None:
        """Run a single poll cycle.  Returns the detected club code or None."""
        frame = self._capture.capture()
        if frame is None:
            return None
        crop = _crop_roi(frame, self._settings.club_roi)
        if crop is None:
            return None
        raw = _ocr_text(crop, self._settings.tessdata_dir or None)
        club = parse_club(raw)
        if club is None and raw:
            logger.debug("GSPro watcher: OCR='%s' (no club match)", raw)
        return club

    # ---- internal ----

    def _loop(self) -> None:
        interval = max(0.1, self._settings.poll_interval)
        # Heartbeat: log every N polls so we can tell "running, no club change"
        # apart from "frozen" in the log.  60s of polling at default 1s = 60.
        heartbeat_every_secs = 60.0
        heartbeat_every_polls = max(1, int(heartbeat_every_secs / interval))
        polls_since_beat = 0
        consecutive_capture_fail = 0
        consecutive_no_match = 0

        while not self._stop.is_set():
            try:
                frame = self._capture.capture()
                if frame is None:
                    consecutive_capture_fail += 1
                    if consecutive_capture_fail in (3, 30, 300):
                        logger.warning(
                            "GSPro watcher: window capture has failed "
                            "%d consecutive polls — is GSPro minimized "
                            "or closed?",
                            consecutive_capture_fail,
                        )
                    self._stop.wait(timeout=interval)
                    continue
                consecutive_capture_fail = 0

                crop = _crop_roi(frame, self._settings.club_roi)
                if crop is None:
                    logger.warning(
                        "GSPro watcher: ROI %s outside frame %dx%d — "
                        "GSPro window probably resized; recalibrate.",
                        self._settings.club_roi,
                        frame.shape[1], frame.shape[0],
                    )
                    self._stop.wait(timeout=interval)
                    continue

                raw = _ocr_text(crop, self._settings.tessdata_dir or None)
                club = parse_club(raw)

                if club is None:
                    consecutive_no_match += 1
                    if consecutive_no_match in (10, 60, 300):
                        logger.warning(
                            "GSPro watcher: %d consecutive polls with no "
                            "club match. Last raw OCR='%s'. ROI may be "
                            "covered by a dialog, or GSPro UI changed.",
                            consecutive_no_match, raw[:80] if raw else "",
                        )
                else:
                    consecutive_no_match = 0
                    if club != self._last_club:
                        logger.info(
                            "GSPro watcher: club changed %s -> %s",
                            self._last_club or "(none)", club,
                        )
                        self._last_club = club
                        try:
                            self._on_club_change(club)
                        except Exception:
                            logger.exception("on_club_change callback raised")
            except Exception:
                logger.exception("GSPro watcher poll failed")

            polls_since_beat += 1
            if polls_since_beat >= heartbeat_every_polls:
                logger.info(
                    "GSPro watcher heartbeat: alive, last_club=%s "
                    "(no_match streak=%d, capture_fail streak=%d)",
                    self._last_club or "(none)",
                    consecutive_no_match,
                    consecutive_capture_fail,
                )
                polls_since_beat = 0

            self._stop.wait(timeout=interval)
        logger.info("GSPro watcher thread exiting")
