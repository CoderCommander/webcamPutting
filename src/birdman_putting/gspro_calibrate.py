"""Interactive ROI calibration for the GSPro club-name display.

Captures a screenshot of the GSPro window, lets the user drag a rectangle
around the area that shows the current club, then saves the resulting
ROI to ``config.toml`` under ``[gspro_watcher].club_roi``.

Run with ``python -m birdman_putting --calibrate-gspro-club``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from birdman_putting.config import AppConfig, save_config
from birdman_putting.gspro_watcher import _ocr_text, parse_club
from birdman_putting.mevo.calibrate import (
    _RectDrawer,
    _compute_scale,
    _get_display_bounds,
)
from birdman_putting.mevo.screenshot import WindowCapture

logger = logging.getLogger(__name__)

_WINDOW_NAME = "GSPro Club ROI Calibration"


def _capture_and_scale(
    capture: WindowCapture,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Capture the GSPro window and produce a display-fit version."""
    screenshot = capture.capture()
    if screenshot is None:
        msg = "Failed to capture GSPro window — is it open and visible?"
        raise RuntimeError(msg)

    h, w = screenshot.shape[:2]
    max_w, max_h = _get_display_bounds()
    scale = _compute_scale(w, h, int(max_w * 0.92), int(max_h * 0.88))
    if scale < 1.0:
        display = cv2.resize(
            screenshot,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = screenshot.copy()
        scale = 1.0
    return screenshot, display, scale


def _draw_overlay(base: np.ndarray, drawer: _RectDrawer, hint: str) -> np.ndarray:
    """Render the live drawing overlay + status text."""
    display = base.copy()

    if drawer.drawing or drawer.rect_ready:
        x1 = min(drawer.start[0], drawer.end[0])
        y1 = min(drawer.start[1], drawer.end[1])
        x2 = max(drawer.start[0], drawer.end[0])
        y2 = max(drawer.start[1], drawer.end[1])
        overlay = display.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Status bar at top
    bar_h = 40
    cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), (40, 40, 40), -1)
    cv2.putText(
        display, "Draw a rectangle around the CLUB NAME display",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    # Bottom status — current OCR result for live feedback
    if hint:
        bot_y = display.shape[0]
        cv2.rectangle(display, (0, bot_y - 30), (display.shape[1], bot_y), (40, 40, 40), -1)
        cv2.putText(
            display, hint, (10, bot_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1,
        )

    return display


def run_gspro_club_calibration(config: AppConfig) -> None:
    """Run the interactive ROI calibrator for GSPro's club display."""
    settings = config.gspro_watcher
    capture = WindowCapture(settings.window_title)
    if not capture.find_window():
        print(
            f"[ERROR] No window found with title containing "
            f"'{settings.window_title}'. Make sure GSPro is running.",
        )
        return

    print("Capturing GSPro window… make sure it's visible and has the club shown.")
    screenshot, display, scale = _capture_and_scale(capture)
    img_h, img_w = screenshot.shape[:2]
    print(f"Captured {img_w}x{img_h} (display scale {scale:.2f})")

    drawer = _RectDrawer(scale)
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_WINDOW_NAME, drawer.mouse_callback)

    hint = "Drag to mark club display. Enter=save | r=redo | c=re-capture | ESC=abort"
    while True:
        # Live OCR preview while user is drawing
        if drawer.rect_ready:
            x, y, w, h = drawer.get_roi_original()
            crop = screenshot[y:y + h, x:x + w]
            try:
                raw = _ocr_text(crop, settings.tessdata_dir or None)
                club = parse_club(raw)
                if club is not None:
                    hint = f"OCR='{raw}' -> club={club}.  Press Enter to save."
                else:
                    hint = f"OCR='{raw}' (no club match).  Resize and try again."
            except Exception as e:
                hint = f"OCR error: {e}"

        cv2.imshow(_WINDOW_NAME, _draw_overlay(display, drawer, hint))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Aborted — no changes saved.")
            cv2.destroyWindow(_WINDOW_NAME)
            return
        if key in (ord("r"), ord("R")):
            drawer.reset()
            hint = "Reset.  Drag a new rectangle."
        elif key in (ord("c"), ord("C")):
            screenshot, display, scale = _capture_and_scale(capture)
            drawer = _RectDrawer(scale)
            cv2.setMouseCallback(_WINDOW_NAME, drawer.mouse_callback)
            hint = "Re-captured.  Drag rectangle around club name."
        elif key in (13, 10) and drawer.rect_ready:  # Enter
            x, y, w, h = drawer.get_roi_original()
            settings.club_roi = [x, y, w, h]
            settings.cal_width = img_w
            settings.cal_height = img_h
            settings.enabled = True
            save_config(config)
            print(f"Saved club_roi=[{x}, {y}, {w}, {h}] (cal {img_w}x{img_h})")
            cv2.destroyWindow(_WINDOW_NAME)
            return
