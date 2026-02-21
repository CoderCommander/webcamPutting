"""Ball detection using HSV color filtering and contour analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from birdman_putting.color_presets import HSVRange

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

    Uses the same double BGR→HSV conversion as the original code:
    the original converted BGR→HSV, then passed the HSV image to
    ColorModuleExtended which converted it *again* (treating HSV bytes
    as BGR). All 12 built-in presets were tuned against this
    double-converted color space, so we must replicate it here.
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        blur_kernel: tuple[int, int] = (11, 11),
        min_radius: int = 5,
        min_circularity: float = 0.5,
        morph_iterations: int = 5,
    ):
        self.hsv_range = hsv_range
        self.blur_kernel = blur_kernel
        self.min_radius = min_radius
        self.min_circularity = min_circularity
        self.morph_iterations = morph_iterations

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
        # Blur and double-convert to match the original calibrated color space.
        # The presets were tuned against HSV bytes re-interpreted as BGR then
        # converted a second time, so we replicate that transform here.
        blurred = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

        # Create color mask
        lower = np.array([
            self.hsv_range.hmin, self.hsv_range.smin, self.hsv_range.vmin
        ])
        upper = np.array([
            self.hsv_range.hmax, self.hsv_range.smax, self.hsv_range.vmax
        ])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological close: fill gaps in the ball's mask so fragmented
        # pixels merge into a solid contour. erode(1) removes noise,
        # dilate(5) fills gaps and connects nearby blobs.
        if self.morph_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=self.morph_iterations)

        # Crop mask to detection zone
        zone_mask = mask[zone_y1:zone_y2, zone_x1:zone_x2_limit]

        # Find contours sorted by area (largest first)
        contours, _ = cv2.findContours(
            zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            logger.info("Contours found: %d", len(contours))

        for contour in contours:
            ((cx, cy), r) = cv2.minEnclosingCircle(contour)

            # Offset coordinates back to full frame
            cx += zone_x1
            cy += zone_y1
            r_int = int(r)

            # Check Y bounds
            if not (zone_y1 <= cy <= zone_y2):
                logger.info("  reject y-bounds: (%d,%d) r=%d", int(cx), int(cy), r_int)
                continue

            # Filter by minimum radius
            if r_int < self.min_radius:
                logger.info(
                    "  reject min-radius: (%d,%d) r=%d < %d",
                    int(cx), int(cy), r_int, self.min_radius,
                )
                continue

            # Filter by circularity: area / (π * r²). A ball ≈ 0.7-0.85; a hand ≈ 0.3-0.5
            if r > 0 and self.min_circularity > 0:
                area = cv2.contourArea(contour)
                circularity = area / (np.pi * r * r)
                logger.info(
                    "  contour (%d,%d) r=%d circ=%.3f (min=%.2f)",
                    int(cx), int(cy), r_int, circularity, self.min_circularity,
                )
                if circularity < self.min_circularity:
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

    def detect_full_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        expected_radius: int | None = None,
        radius_tolerance: int = 50,
    ) -> BallDetection | None:
        """Detect ball anywhere in the full frame (no zone cropping).

        Convenience wrapper for calibration use.
        """
        h, w = frame.shape[:2]
        return self.detect(
            frame=frame,
            zone_x1=0,
            zone_x2_limit=w,
            zone_y1=0,
            zone_y2=h,
            timestamp=timestamp,
            expected_radius=expected_radius,
            radius_tolerance=radius_tolerance,
        )

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
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        lower = np.array([
            self.hsv_range.hmin, self.hsv_range.smin, self.hsv_range.vmin
        ])
        upper = np.array([
            self.hsv_range.hmax, self.hsv_range.smax, self.hsv_range.vmax
        ])
        mask = cv2.inRange(hsv, lower, upper)
        if self.morph_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=self.morph_iterations)
        return mask[zone_y1:zone_y2, zone_x1:zone_x2_limit]


def generate_hsv_from_patch(
    frame: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 21,
) -> HSVRange:
    """Generate an HSV range by sampling a patch around (x, y) in a BGR frame.

    Computes mean ± 2*stddev per channel from the patch, clamped to valid ranges.
    Handles hue wraparound near the 0/180 boundary (relevant for red/orange).

    Args:
        frame: BGR image.
        x: Center X of the sample point.
        y: Center Y of the sample point.
        patch_size: Side length of the square patch (should be odd).

    Returns:
        HSVRange covering the sampled color.
    """
    h, w = frame.shape[:2]
    half = patch_size // 2

    # Clamp patch to frame bounds
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half + 1)
    y2 = min(h, y + half + 1)

    patch_bgr = frame[y1:y2, x1:x2]
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    # Split channels
    h_ch = patch_hsv[:, :, 0].astype(np.float64)
    s_ch = patch_hsv[:, :, 1].astype(np.float64)
    v_ch = patch_hsv[:, :, 2].astype(np.float64)

    # Hue wraparound detection: if hue values span near 0 and near 179,
    # shift them to avoid averaging across the boundary
    hue_shifted = False
    if np.any(h_ch < 30) and np.any(h_ch > 150):
        h_ch = (h_ch + 90) % 180
        hue_shifted = True

    h_mean, h_std = float(np.mean(h_ch)), float(np.std(h_ch))
    s_mean, s_std = float(np.mean(s_ch)), float(np.std(s_ch))
    v_mean, v_std = float(np.mean(v_ch)), float(np.std(v_ch))

    # Use at least a minimum margin so single-color patches still work
    margin = 2.0
    h_lo = h_mean - max(margin * h_std, 10)
    h_hi = h_mean + max(margin * h_std, 10)
    s_lo = s_mean - max(margin * s_std, 40)
    s_hi = s_mean + max(margin * s_std, 40)
    v_lo = v_mean - max(margin * v_std, 40)
    v_hi = v_mean + max(margin * v_std, 40)

    if hue_shifted:
        h_lo = (h_lo - 90) % 180
        h_hi = (h_hi - 90) % 180
        if h_lo > h_hi:
            h_lo, h_hi = h_hi, h_lo

    return HSVRange(
        hmin=int(max(0, h_lo)),
        smin=int(max(0, s_lo)),
        vmin=int(max(0, v_lo)),
        hmax=int(min(179, h_hi)),
        smax=int(min(255, s_hi)),
        vmax=int(min(255, v_hi)),
    )


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
