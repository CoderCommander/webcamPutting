"""Ball detection using HSV color filtering and contour analysis."""

from __future__ import annotations

import logging
import math
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
        radius_tolerance: int | None = None,
        expected_pos: tuple[int, int] | None = None,
    ) -> BallDetection | None:
        """Find the golf ball in the frame within the detection zone.

        Args:
            frame: BGR image from camera.
            zone_x1: Left edge of detection zone.
            zone_x2_limit: Right edge limit for masking (typically frame width).
            zone_y1: Top edge of detection zone.
            zone_y2: Bottom edge of detection zone.
            timestamp: time.perf_counter() value for this frame.
            expected_radius: If set, prefer contours matching this radius and
                filter out contours whose radius is outside the tolerance.
            radius_tolerance: Pixel tolerance for radius matching.  If left as
                ``None`` AND ``expected_radius`` is given, a *proportional*
                tolerance of ±40% of ``expected_radius`` (min ±6px) is used —
                much tighter than the legacy ±50px.  If an explicit value is
                passed it is always respected.  When ``expected_radius`` is
                ``None`` the tolerance is unused.
            expected_pos: If set (with or without ``expected_radius``), every
                passing candidate is scored by a blend of distance-to-this
                point, radius match, and circularity, and the BEST-scoring
                candidate is returned (inter-frame continuity) instead of the
                largest contour.  This prevents the tracer from jumping to a
                larger hand/shadow blob when the real ball is near its last
                known position.

        Behavior contract:
            When BOTH ``expected_pos`` and ``expected_radius`` are ``None`` the
            method preserves the original behavior EXACTLY — it returns the
            FIRST passing contour in largest-area order.

        Returns:
            BallDetection if ball found, None otherwise.
        """
        # Resolve the effective radius tolerance.  Only apply the proportional
        # default when expected_radius is given and the caller did not pass an
        # explicit tolerance.  This keeps the no-expectation path untouched and
        # honors callers (and tests) that pass a specific tolerance.
        if radius_tolerance is None:
            # Loose gate BY DESIGN: a motion-blurred rolling ball's
            # minEnclosingCircle radius balloons well beyond its rest radius,
            # so a tight tolerance drops nearly every moving frame (this
            # regressed live putt capture from ~40-80 motion frames to 1-2).
            # Keep the gate loose so blurred balls survive; _score_candidate's
            # radius term still gently prefers the rest-sized ball among them.
            eff_radius_tol = 50
        else:
            eff_radius_tol = radius_tolerance
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

        # Find contours sorted by area (largest first).  Largest-first order
        # is what the no-expectation path returns (first passing = largest);
        # the scoring path re-ranks the survivors by score instead.
        contours, _ = cv2.findContours(
            zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        use_scoring = expected_pos is not None or expected_radius is not None
        best_detection: BallDetection | None = None
        best_score = float("-inf")

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
            # (computed unconditionally — the scoring path below also needs it).
            circularity = area / (np.pi * r * r) if r > 0 else 0.0
            if r > 0 and self.min_circularity > 0 and circularity < self.min_circularity:
                continue

            # Filter by expected radius if provided
            if expected_radius is not None and not (
                expected_radius - eff_radius_tol < r_int < expected_radius + eff_radius_tol
            ):
                continue

            candidate = BallDetection(
                x=int(cx),
                y=int(cy),
                radius=r_int,
                contour_area=area,
                timestamp=timestamp,
            )

            if not use_scoring:
                # Preserve EXACT legacy behavior: contours are largest-first,
                # so the first passing candidate is the largest-area one.
                return candidate

            score = self._score_candidate(
                cx=float(cx),
                cy=float(cy),
                r=float(r),
                circularity=circularity,
                expected_pos=expected_pos,
                expected_radius=expected_radius,
            )
            if score > best_score:
                best_score = score
                best_detection = candidate

        return best_detection

    def _score_candidate(
        self,
        cx: float,
        cy: float,
        r: float,
        circularity: float,
        expected_pos: tuple[int, int] | None,
        expected_radius: int | None,
    ) -> float:
        """Score a passing contour for best-match selection (higher = better).

        Three additive, bounded-in-[0,1] terms, weighted so that *proximity*
        to the last known ball position dominates, roundness is the next
        tie-breaker, and radius match is a lighter nudge:

            score = 0.55 * proximity + 0.30 * circularity + 0.15 * radius_match

        Rationale:
        - Proximity (inter-frame continuity) is the strongest cue: the real
          ball is near where it was last frame; a hand/shadow that enters the
          zone is typically offset.  Uses a soft 1/(1+d/scale) falloff so a
          far, larger blob can never beat a near ball.
        - Circularity rewards a rounder blob so the ball outscores a less-round
          hand at the same distance — this is what fixes the hand-blob problem
          EVEN when the circularity gate is relaxed to 0.0 upstream.
        - Radius match keeps the chosen blob consistent in size with the
          known ball, but is intentionally the weakest term (motion blur and
          partial occlusion shift the measured radius frame-to-frame).

        When ``expected_pos`` is absent, the proximity term is neutralized
        (set to its weight) so radius+circularity decide; when
        ``expected_radius`` is absent, the radius term is likewise neutral.
        """
        # Proximity term in [0, 1]: 1.0 at the expected point, decaying with
        # distance.  ``scale`` is anchored to the ball size so the falloff is
        # resolution-independent: a blob one ball-diameter away scores ~0.5.
        prox_w, circ_w, rad_w = 0.55, 0.30, 0.15
        if expected_pos is not None:
            dist = math.hypot(cx - expected_pos[0], cy - expected_pos[1])
            scale = max(float(expected_radius or r) * 2.0, 20.0)
            proximity = 1.0 / (1.0 + dist / scale)
        else:
            proximity = 1.0  # neutral — no positional expectation

        # Radius-match term in [0, 1]: 1.0 at an exact match, decaying as the
        # measured radius diverges from the expected radius.
        if expected_radius is not None and expected_radius > 0:
            radius_match = 1.0 / (1.0 + abs(r - expected_radius) / float(expected_radius))
        else:
            radius_match = 1.0  # neutral — no size expectation

        # Circularity is already ~[0, 1] for blobs (can slightly exceed 1 for
        # tiny/quantized contours); clamp for a well-behaved score.
        circ = max(0.0, min(1.0, circularity))

        return prox_w * proximity + circ_w * circ + rad_w * radius_match

    def detect_full_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        expected_radius: int | None = None,
        radius_tolerance: int | None = None,
        expected_pos: tuple[int, int] | None = None,
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
            expected_pos=expected_pos,
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
