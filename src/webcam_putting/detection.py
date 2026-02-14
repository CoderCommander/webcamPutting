"""Ball detection using HSV color filtering and contour analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from webcam_putting.color_presets import HSVRange

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """Result of detecting a ball in a single frame."""

    x: int
    y: int
    radius: int
    contour_area: float
    timestamp: float  # time.perf_counter() value


class BallDetector:
    """Detects a golf ball in video frames using HSV color filtering.

    Fixes the double HSV conversion bug from the original code:
    the original converted BGR->HSV, then passed the HSV image to
    ColorModuleExtended which converted it again. This class converts once.
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        blur_kernel: tuple[int, int] = (11, 11),
        min_radius: int = 5,
    ):
        self.hsv_range = hsv_range
        self.blur_kernel = blur_kernel
        self.min_radius = min_radius

    def update_hsv(self, hsv_range: HSVRange) -> None:
        """Update the HSV range for detection."""
        self.hsv_range = hsv_range

    def detect(
        self,
        frame: np.ndarray,
        zone_x1: int,
        zone_x2_limit: int,
        zone_y1: int,
        zone_y2: int,
        timestamp: float,
        expected_radius: int | None = None,
        radius_tolerance: int = 50,
    ) -> BallDetection | None:
        """Find the golf ball in the frame within the detection zone.

        Args:
            frame: BGR image from camera.
            zone_x1: Left edge of detection zone.
            zone_x2_limit: Right edge limit for masking (typically frame width).
            zone_y1: Top edge of detection zone.
            zone_y2: Bottom edge of detection zone.
            timestamp: time.perf_counter() value for this frame.
            expected_radius: If set, filter contours to match this radius.
            radius_tolerance: Pixel tolerance for radius matching.

        Returns:
            BallDetection if ball found, None otherwise.
        """
        # Blur and convert to HSV (single conversion - fixing original bug)
        blurred = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Create color mask
        lower = np.array([
            self.hsv_range.hmin, self.hsv_range.smin, self.hsv_range.vmin
        ])
        upper = np.array([
            self.hsv_range.hmax, self.hsv_range.smax, self.hsv_range.vmax
        ])
        mask = cv2.inRange(hsv, lower, upper)

        # Crop mask to detection zone
        zone_mask = mask[zone_y1:zone_y2, zone_x1:zone_x2_limit]

        # Find contours sorted by area (largest first)
        contours, _ = cv2.findContours(
            zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            ((cx, cy), r) = cv2.minEnclosingCircle(contour)

            # Offset coordinates back to full frame
            cx += zone_x1
            cy += zone_y1
            r_int = int(r)

            # Check Y bounds
            if not (zone_y1 <= cy <= zone_y2):
                break  # Contours are sorted by area; if largest is out of bounds, stop

            # Filter by minimum radius
            if r_int < self.min_radius:
                continue

            # Filter by expected radius if provided
            if expected_radius is not None and not (
                expected_radius - radius_tolerance < r_int < expected_radius + radius_tolerance
            ):
                continue

            return BallDetection(
                x=int(cx),
                y=int(cy),
                radius=r_int,
                contour_area=cv2.contourArea(contour),
                timestamp=timestamp,
            )

        return None

    def get_mask(
        self,
        frame: np.ndarray,
        zone_x1: int,
        zone_x2_limit: int,
        zone_y1: int,
        zone_y2: int,
    ) -> np.ndarray:
        """Get the color detection mask for debug visualization."""
        blurred = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array([
            self.hsv_range.hmin, self.hsv_range.smin, self.hsv_range.vmin
        ])
        upper = np.array([
            self.hsv_range.hmax, self.hsv_range.smax, self.hsv_range.vmax
        ])
        mask = cv2.inRange(hsv, lower, upper)
        return mask[zone_y1:zone_y2, zone_x1:zone_x2_limit]


def resize_with_aspect_ratio(
    image: np.ndarray,  # type: ignore[type-arg]
    width: int | None = None,
    height: int | None = None,
    inter: int = cv2.INTER_AREA,
) -> np.ndarray:  # type: ignore[type-arg]
    """Resize image maintaining aspect ratio (replaces imutils.resize)."""
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        assert height is not None
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv2.resize(image, dim, interpolation=inter)
