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

    Uses a single BGR→HSV conversion.  The original cam-putting-py code
    accidentally converted twice (BGR→HSV, then treated the HSV bytes as
    BGR and converted again).  While some legacy presets were tuned
    against that double-converted space, the double conversion produces
    near-100% mask coverage with common camera settings, making detection
    unreliable.  A single conversion gives correct HSV values and stable
    detections.
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        blur_kernel: tuple[int, int] = (7, 7),
        min_radius: int = 5,
        min_circularity: float = 0.5,
        morph_iterations: int = 5,
    ):
        self.hsv_range = hsv_range
        self.blur_kernel = blur_kernel
        self.min_radius = min_radius
        self.min_circularity = min_circularity
        self.morph_iterations = morph_iterations

        # Pre-allocate reusable objects (avoid per-frame allocation)
        self._morph_kernel = np.ones((3, 3), np.uint8)
        self._update_hsv_bounds(hsv_range)

    def _update_hsv_bounds(self, hsv_range: HSVRange) -> None:
        """Pre-compute cached HSV bound arrays."""
        self._lower = np.array([hsv_range.hmin, hsv_range.smin, hsv_range.vmin])
        self._upper = np.array([hsv_range.hmax, hsv_range.smax, hsv_range.vmax])

    def update_hsv(self, hsv_range: HSVRange) -> None:
        """Update the HSV range for detection."""
        self.hsv_range = hsv_range
        self._update_hsv_bounds(hsv_range)

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
        # Crop to detection zone + margin BEFORE expensive operations.
        # This processes ~200x250 pixels instead of ~640x360 (~4x fewer).
        h, w = frame.shape[:2]
        margin = 15  # Extra pixels for blur edge effects
        crop_y1 = max(0, zone_y1 - margin)
        crop_y2 = min(h, zone_y2 + margin)
        crop_x1 = max(0, zone_x1 - margin)
        crop_x2 = min(w, zone_x2_limit + margin)
        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Blur and convert to HSV for color-based detection.
        blurred = cv2.GaussianBlur(roi, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Create color mask (using cached bounds)
        mask = cv2.inRange(hsv, self._lower, self._upper)

        # Morphological close: fill gaps in the ball's mask
        if self.morph_iterations > 0 and cv2.countNonZero(mask) > 0:
            mask = cv2.erode(mask, self._morph_kernel, iterations=1)
            mask = cv2.dilate(mask, self._morph_kernel, iterations=self.morph_iterations)

        # Extract the detection zone from the cropped mask
        # (offset by the margin we added)
        inner_y1 = zone_y1 - crop_y1
        inner_y2 = inner_y1 + (zone_y2 - zone_y1)
        inner_x1 = zone_x1 - crop_x1
        inner_x2 = inner_x1 + (zone_x2_limit - zone_x1)
        zone_mask = mask[inner_y1:inner_y2, inner_x1:inner_x2]

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
                continue

            # Filter by minimum radius
            if r_int < self.min_radius:
                continue

            # Compute area once for both circularity check and return value
            area = cv2.contourArea(contour)

            # Filter by circularity: area / (π * r²). A ball ≈ 0.7-0.85; a hand ≈ 0.3-0.5
            if r > 0 and self.min_circularity > 0:
                circularity = area / (np.pi * r * r)
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
                contour_area=area,
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
        h, w = frame.shape[:2]
        margin = 15
        cy1 = max(0, zone_y1 - margin)
        cy2 = min(h, zone_y2 + margin)
        cx1 = max(0, zone_x1 - margin)
        cx2 = min(w, zone_x2_limit + margin)
        roi = frame[cy1:cy2, cx1:cx2]
        blurred = cv2.GaussianBlur(roi, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        if self.morph_iterations > 0 and cv2.countNonZero(mask) > 0:
            mask = cv2.erode(mask, self._morph_kernel, iterations=1)
            mask = cv2.dilate(mask, self._morph_kernel, iterations=self.morph_iterations)
        iy1 = zone_y1 - cy1
        ix1 = zone_x1 - cx1
        return mask[iy1:iy1 + (zone_y2 - zone_y1), ix1:ix1 + (zone_x2_limit - zone_x1)]


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
