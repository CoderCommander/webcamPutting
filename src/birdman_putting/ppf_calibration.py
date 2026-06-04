"""Pixel-to-foot calibration via evenly-spaced markers in the camera frame.

Detects bright vertical "cork-like" markers in a horizontal band around
the detection zone, sorted left-to-right.  With markers placed (or
projected) at known 1-foot intervals, the median inter-marker pixel
distance is the pixels_per_foot value the tracker uses.

This is the building block for both the manual cork-on-mat measurement
(``measure_ppf.py``) and the OBS-projected Auto Cal flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpacingValidation:
    """Uniformity assessment of a sorted marker list.

    A *missing* marker shows up as a gap roughly 2x the robust median
    spacing; a *spurious* marker shows up as a gap roughly half the
    median.  Either one silently corrupts an array-index→foot mapping, so
    we surface them explicitly.
    """

    uniform: bool                 # True if every gap is within tolerance
    median_spacing: float         # robust median of adjacent gaps (0 if <2)
    gap_outliers: list[int]       # indices i where diffs[i] deviates badly
    diffs: list[float]            # the adjacent gaps that were assessed


# Fraction a single gap may deviate from the robust median before it is
# flagged.  A missing marker doubles a gap (+100%); a spurious one halves
# it (-50%).  35% sits comfortably below both while tolerating realistic
# fisheye gradients (Greg's corks span ~45-57 px/ft → ±~13% about median).
_SPACING_TOLERANCE = 0.35


def validate_marker_spacing(
    centers: list[tuple[float, float]],
    tolerance: float = _SPACING_TOLERANCE,
) -> SpacingValidation:
    """Check that adjacent marker gaps are uniform to within ``tolerance``.

    Computes the robust (median) adjacent-gap spacing and flags any gap
    whose deviation from that median exceeds ``tolerance`` (fractional).
    A ~2x gap (missing marker) or ~0.5x gap (spurious marker) is reported
    in ``gap_outliers`` as the index into the diffs list.

    With fewer than 2 markers there is nothing to validate, so the result
    is trivially uniform.
    """
    if len(centers) < 2:
        return SpacingValidation(
            uniform=True, median_spacing=0.0, gap_outliers=[], diffs=[],
        )

    xs = [c[0] for c in centers]
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

    # Robust median spacing: exclude gaps that are obviously not a single
    # marker interval (a 2x+ missing-marker gap) so the median reflects
    # the true 1-ft spacing rather than being dragged upward by the gap.
    sorted_diffs = sorted(diffs)
    p25 = sorted_diffs[len(sorted_diffs) // 4]
    plausible = [d for d in diffs if d <= p25 * 1.5] if p25 > 0 else diffs
    base = plausible if plausible else diffs
    median = float(np.median(base))

    gap_outliers: list[int] = []
    if median > 0:
        for i, d in enumerate(diffs):
            if abs(d - median) / median > tolerance:
                gap_outliers.append(i)

    return SpacingValidation(
        uniform=(len(gap_outliers) == 0),
        median_spacing=median,
        gap_outliers=gap_outliers,
        diffs=diffs,
    )


@dataclass
class MarkerDetectionResult:
    """Result of detecting calibration markers in a frame."""

    centers: list[tuple[float, float]]  # sorted left-to-right (x, y)
    median_spacing_px: float | None     # px/ft if 2+ markers, else None
    mean_spacing_px: float | None
    diffs: list[float]                  # adjacent inter-marker distances
    # Spacing-validation outputs (defaulted so existing keyword construction
    # in callers keeps working).  spacing_uniform is False when a gap looks
    # like a missing/extra marker; gap_outliers lists the offending indices.
    spacing_uniform: bool = True
    gap_outliers: list[int] = field(default_factory=list)


def detect_calibration_markers(
    frame: np.ndarray,
    band_y_center: int,
    band_half_height: int = 60,
    luma_threshold: int = 180,  # legacy fallback (white-on-green corks)
    v_threshold: int = 160,  # HSV value threshold — bright pixels
    hue_range: tuple[int, int] | None = None,  # OBS Auto Cal passes (75, 105) for cyan
    min_saturation: int = 80,  # exclude desaturated content
    min_area: float = 4.0,
    max_area: float = 400.0,
    aspect_min: float = 0.1,
    aspect_max: float = 1.5,
) -> MarkerDetectionResult:
    """Find vertically-oriented bright markers in a horizontal band.

    Detection uses the HSV channels:
      - V (Value/brightness) ≥ v_threshold: marker must be bright
      - H (Hue) within hue_range: filters by color (cyan default)
      - S (Saturation) ≥ min_saturation: excludes desaturated noise

    Default hue_range=(75, 105) keeps only cyan markers — useful for OBS
    projections so the orange ball and orange shot-trails aren't picked
    up as extra calibration points. Set hue_range=None to accept any hue
    (e.g., for white corks, where the legacy luma fallback is also OR'd in).

    Args:
        frame: BGR image at the tracker's processing resolution.
        band_y_center: Y coordinate (post-rotation) of the markers.
        band_half_height: Half-height of the horizontal search band.
        luma_threshold: Grayscale threshold for legacy white-marker mode.
        v_threshold: HSV V-channel threshold for "bright" pixels.
        hue_range: (h_min, h_max) inclusive in OpenCV's 0-179 hue range.
        min_saturation: Reject pixels below this S to filter near-white noise.
        min_area / max_area: Contour area bounds in pixels.
        aspect_min / aspect_max: width/height aspect ratio bounds.

    Returns:
        MarkerDetectionResult.
    """
    h, w = frame.shape[:2]
    band_y1 = max(0, band_y_center - band_half_height)
    band_y2 = min(h, band_y_center + band_half_height)
    band = frame[band_y1:band_y2]

    # Build the bright mask. By default (cyan-projection mode), we want
    # ONLY cyan-saturated bright pixels — ignore the orange ball and
    # orange shot trails OBS may overlay on the scene.
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    if hue_range is not None:
        h_lo, h_hi = hue_range
        if h_lo <= h_hi:
            hue_mask = (h_ch >= h_lo) & (h_ch <= h_hi)
        else:
            # wrap-around (e.g. red near 0/179)
            hue_mask = (h_ch >= h_lo) | (h_ch <= h_hi)
        bright_mask = (v_ch >= v_threshold) & (s_ch >= min_saturation) & hue_mask
        mask = bright_mask.astype(np.uint8) * 255
    else:
        # Hue-agnostic mode: accept bright saturated colors OR bright near-white
        bright_color = (v_ch >= v_threshold) & (s_ch >= 50)
        bright_white = (v_ch >= v_threshold) & (s_ch < 50)
        v_mask = (bright_color | bright_white).astype(np.uint8) * 255
        # Also OR the legacy grayscale luma mask for white-cork compat
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, gray_mask = cv2.threshold(gray, luma_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(v_mask, gray_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_centers: list[tuple[float, float]] = []
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area or a > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / max(ch, 1)
        if not (aspect_min <= ar <= aspect_max):
            continue
        cx = x + cw / 2.0
        cy = y + ch / 2.0 + band_y1
        raw_centers.append((cx, cy))

    raw_centers.sort(key=lambda p: p[0])

    # Deduplicate: merge centers that are very close in X — a single
    # projected bar can fragment into 2-3 contours due to mask edges or
    # the bar having a thin vertical gap.  Cluster anything within
    # `dedup_px` and use mean position.
    dedup_px = 12.0
    centers: list[tuple[float, float]] = []
    if raw_centers:
        cluster_xs = [raw_centers[0][0]]
        cluster_ys = [raw_centers[0][1]]
        for cx, cy in raw_centers[1:]:
            if cx - cluster_xs[-1] <= dedup_px:
                cluster_xs.append(cx)
                cluster_ys.append(cy)
            else:
                centers.append(
                    (sum(cluster_xs) / len(cluster_xs),
                     sum(cluster_ys) / len(cluster_ys)),
                )
                cluster_xs = [cx]
                cluster_ys = [cy]
        centers.append(
            (sum(cluster_xs) / len(cluster_xs),
             sum(cluster_ys) / len(cluster_ys)),
        )

    diffs: list[float] = [
        centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)
    ]

    # Robust median: drop spacings that are clearly not adjacent-marker
    # gaps.  If the bulk has tight spacings around X but one outlier is
    # 3X+ larger, that's a missing-marker gap (e.g., start-zone exclusion
    # removed an interior marker), not 1 ft.  Keep only spacings within
    # 1.5x of the smallest "plausible" spacing (the 25th percentile).
    if diffs:
        sorted_diffs = sorted(diffs)
        p25 = sorted_diffs[len(sorted_diffs) // 4]
        # If everything is uniform, keep all.  Otherwise reject huge gaps.
        kept = [d for d in diffs if d <= p25 * 1.5] if p25 > 0 else diffs
        if kept:
            median = float(np.median(kept))
            mean = float(np.mean(kept))
        else:
            median = float(np.median(diffs))
            mean = float(np.mean(diffs))
    else:
        median = None
        mean = None

    # Validate that the gaps are uniform.  A missing interior marker (e.g.
    # one filtered out because it fell in the start zone) leaves a ~2x gap
    # that would silently shift every downstream foot reading; surface it
    # so callers can repair or reject rather than trust it blindly.
    validation = validate_marker_spacing(centers)
    if not validation.uniform:
        logger.warning(
            "Calibration spacing non-uniform: median=%.1fpx, suspect gaps at "
            "indices %s (diffs=%s) — a missing/extra marker may be present",
            validation.median_spacing,
            validation.gap_outliers,
            [round(d, 1) for d in validation.diffs],
        )

    return MarkerDetectionResult(
        centers=centers, median_spacing_px=median,
        mean_spacing_px=mean, diffs=diffs,
        spacing_uniform=validation.uniform,
        gap_outliers=validation.gap_outliers,
    )
