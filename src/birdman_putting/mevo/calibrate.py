"""Interactive ROI calibration tool for Mevo OCR regions."""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from birdman_putting.config import AppConfig, save_config
from birdman_putting.mevo.ocr import ROI, MevoOCR
from birdman_putting.mevo.screenshot import WindowCapture

logger = logging.getLogger(__name__)

# Metrics in calibration order: required first, then optional
_METRICS: list[tuple[str, bool]] = [
    ("ball_speed", True),
    ("launch_angle", True),
    ("launch_direction", True),
    ("spin_rate", False),
    ("spin_axis", False),
    ("club_speed", False),
]

_WINDOW_NAME = "Mevo ROI Calibration"


class _RectDrawer:
    """Mouse callback handler for drawing rectangles on an image."""

    def __init__(self) -> None:
        self.drawing = False
        self.start: tuple[int, int] = (0, 0)
        self.end: tuple[int, int] = (0, 0)
        self.rect_ready = False

    def reset(self) -> None:
        self.drawing = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.rect_ready = False

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_ready = False
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            # Only mark ready if rectangle has meaningful size
            w = abs(self.end[0] - self.start[0])
            h = abs(self.end[1] - self.start[1])
            if w > 5 and h > 5:
                self.rect_ready = True

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) with top-left normalization."""
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        return (x1, y1, x2 - x1, y2 - y1)


def _draw_overlay(
    base: np.ndarray,
    drawer: _RectDrawer,
    completed: dict[str, tuple[int, int, int, int]],
    current_metric: str,
    is_required: bool,
) -> np.ndarray:
    """Draw the calibration overlay on the screenshot."""
    display = base.copy()

    # Draw completed ROIs in blue
    for name, (x, y, w, h) in completed.items():
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 150, 0), 2)
        cv2.putText(display, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)

    # Draw current rectangle in green
    if drawer.drawing or drawer.rect_ready:
        x1 = min(drawer.start[0], drawer.end[0])
        y1 = min(drawer.start[1], drawer.end[1])
        x2 = max(drawer.start[0], drawer.end[0])
        y2 = max(drawer.start[1], drawer.end[1])
        # Semi-transparent green fill
        overlay = display.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Status bar at top
    bar_h = 40
    cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), (40, 40, 40), -1)
    req_label = "REQUIRED" if is_required else "optional - press 's' to skip"
    label = f"Select: {current_metric} ({req_label})"
    cv2.putText(display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Instructions at bottom
    bot_y = display.shape[0]
    cv2.rectangle(display, (0, bot_y - 30), (display.shape[1], bot_y), (40, 40, 40), -1)
    instructions = "Draw rectangle | Enter=confirm | r=redo | ESC=abort"
    if not is_required:
        instructions += " | s=skip"
    cv2.putText(
        display, instructions, (10, bot_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    return display


def run_calibration(config: AppConfig) -> None:
    """Run interactive ROI calibration for Mevo OCR regions.

    Captures the FS Golf window, displays it, and walks the user through
    selecting rectangular regions for each shot metric. Saves results to config.
    """
    print("=" * 60)
    print("  Mevo ROI Calibration Tool")
    print("=" * 60)
    print()
    print(f"Looking for window: '{config.mevo.window_title}'")
    print()

    # Capture the FS Golf window
    capture = WindowCapture(config.mevo.window_title)
    if not capture.find_window():
        print(f"ERROR: Could not find window '{config.mevo.window_title}'.")
        print("Make sure FS Golf is running and visible.")
        sys.exit(1)

    screenshot = capture.capture()
    if screenshot is None:
        print("ERROR: Failed to capture window screenshot.")
        sys.exit(1)

    print(f"Captured screenshot: {screenshot.shape[1]}x{screenshot.shape[0]}")
    print()
    print("Instructions:")
    print("  - Draw a rectangle around each metric value")
    print("  - Press Enter to confirm the selection")
    print("  - Press 'r' to redraw the current region")
    print("  - Press 's' to skip optional metrics")
    print("  - Press ESC to abort calibration")
    print()

    # Set up OpenCV window
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Size window to fit screen reasonably
    h, w = screenshot.shape[:2]
    if w > 1400:
        cv2.resizeWindow(_WINDOW_NAME, 1400, int(1400 * h / w))
    else:
        cv2.resizeWindow(_WINDOW_NAME, w, h)

    drawer = _RectDrawer()
    cv2.setMouseCallback(_WINDOW_NAME, drawer.mouse_callback)  # type: ignore[arg-type]

    completed: dict[str, tuple[int, int, int, int]] = {}

    for metric_name, is_required in _METRICS:
        drawer.reset()
        confirmed = False
        skipped = False

        print(f"  Select region for: {metric_name}"
              f" ({'required' if is_required else 'optional, s=skip'})")

        while not confirmed and not skipped:
            display = _draw_overlay(screenshot, drawer, completed, metric_name, is_required)
            cv2.imshow(_WINDOW_NAME, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("\nCalibration aborted.")
                cv2.destroyAllWindows()
                return

            if key == ord("r"):
                drawer.reset()

            if key == ord("s") and not is_required:
                skipped = True
                print(f"    Skipped {metric_name}")

            if key in (13, 10) and drawer.rect_ready:  # Enter
                roi = drawer.get_roi()
                completed[metric_name] = roi
                confirmed = True
                print(f"    {metric_name}: x={roi[0]}, y={roi[1]}, "
                      f"w={roi[2]}, h={roi[3]}")

    cv2.destroyAllWindows()

    if not completed:
        print("\nNo regions were selected. Calibration cancelled.")
        return

    # Test OCR on each selected region
    print()
    print("-" * 40)
    print("  OCR Test Results")
    print("-" * 40)

    rois = [ROI(name=name, x=r[0], y=r[1], width=r[2], height=r[3])
            for name, r in completed.items()]

    tessdata = config.mevo.tessdata_dir or None
    try:
        ocr = MevoOCR(rois=rois, tessdata_dir=tessdata)
        results = ocr.read_metrics(screenshot)
        for name, value in results.items():
            status = f"{value:.2f}" if value is not None else "FAILED"
            print(f"  {name:>20s}: {status}")
    except Exception as exc:
        print(f"  OCR test failed: {exc}")
        print("  (ROIs will still be saved — you can adjust and re-test)")

    # Save to config
    roi_dict: dict[str, list[int]] = {}
    for name, rect in completed.items():
        roi_dict[name] = list(rect)

    config.mevo.rois = roi_dict
    save_config(config)

    print()
    print("ROIs saved to config.toml")
    print("Run with --mevo to use these regions for shot detection.")
