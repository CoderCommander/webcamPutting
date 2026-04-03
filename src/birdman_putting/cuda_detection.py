"""GPU-accelerated ball detection using CuPy (NVIDIA CUDA)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from birdman_putting.color_presets import HSVRange
from birdman_putting.detection import BallDetection, BallDetector

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _bgr_to_hsv_gpu(bgr: cp.ndarray) -> cp.ndarray:
    """Convert BGR uint8 to HSV uint8 on GPU (OpenCV convention).

    H: 0-179, S: 0-255, V: 0-255.
    """
    img = bgr.astype(cp.float32) / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    max_c = cp.maximum(cp.maximum(r, g), b)
    min_c = cp.minimum(cp.minimum(r, g), b)
    delta = max_c - min_c

    # Value
    v = max_c

    # Saturation
    s = cp.where(max_c > 0, delta / max_c, 0.0)

    # Hue
    h = cp.zeros_like(v)
    nonzero = delta > 1e-6

    mask_r = nonzero & (max_c == r)
    h = cp.where(mask_r, 60.0 * (((g - b) / (delta + 1e-6)) % 6.0), h)

    mask_g = nonzero & (max_c == g)
    h = cp.where(mask_g, 60.0 * ((b - r) / (delta + 1e-6) + 2.0), h)

    mask_b = nonzero & (max_c == b)
    h = cp.where(mask_b, 60.0 * ((r - g) / (delta + 1e-6) + 4.0), h)

    # OpenCV convention: H in [0, 179]
    h = (h % 360.0) / 2.0

    hsv = cp.stack([h, s * 255.0, v * 255.0], axis=2)
    return cp.clip(hsv, 0, 255).astype(cp.uint8)


class CupyBallDetector(BallDetector):
    """Ball detector that offloads image processing to GPU via CuPy.

    Subclasses :class:`BallDetector` and overrides :meth:`detect` and
    :meth:`get_mask`. The GPU pipeline handles blur, color conversion,
    thresholding, and morphological operations. Contour analysis stays
    on CPU (no GPU equivalent).

    Requires: ``pip install cupy-cuda12x`` (match your CUDA version).
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        blur_kernel: tuple[int, int] = (7, 7),
        min_radius: int = 5,
        min_circularity: float = 0.5,
        morph_iterations: int = 5,
    ):
        if not _HAS_CUPY:
            raise ImportError("CuPy not installed — pip install cupy-cuda12x")

        super().__init__(
            hsv_range=hsv_range,
            blur_kernel=blur_kernel,
            min_radius=min_radius,
            min_circularity=min_circularity,
            morph_iterations=morph_iterations,
        )

        # Blur sigma: OpenCV (7,7) with sigma=0 → sigma = 0.3*((7-1)*0.5 - 1) + 0.8 = 1.5
        self._blur_sigma = 0.3 * ((blur_kernel[0] - 1) * 0.5 - 1) + 0.8

        # Pre-cache HSV bounds as CuPy arrays
        self._lower_gpu = cp.array(
            [hsv_range.hmin, hsv_range.smin, hsv_range.vmin], dtype=cp.uint8,
        )
        self._upper_gpu = cp.array(
            [hsv_range.hmax, hsv_range.smax, hsv_range.vmax], dtype=cp.uint8,
        )

        device = cp.cuda.Device()
        dev_name = f"device {device.id}"
        logger.info(
            "CupyBallDetector initialized (GPU: %s, CuPy %s)",
            dev_name, cp.__version__,
        )

    def _update_hsv_bounds(self, hsv_range: HSVRange) -> None:
        """Update cached HSV bounds on both CPU and GPU."""
        super()._update_hsv_bounds(hsv_range)
        if _HAS_CUPY:
            self._lower_gpu = cp.array(
                [hsv_range.hmin, hsv_range.smin, hsv_range.vmin], dtype=cp.uint8,
            )
            self._upper_gpu = cp.array(
                [hsv_range.hmax, hsv_range.smax, hsv_range.vmax], dtype=cp.uint8,
            )

    def _gpu_pipeline(self, roi: np.ndarray) -> np.ndarray:
        """Run the GPU processing pipeline, return a uint8 mask on CPU.

        Pipeline: upload → blur → cvtColor x2 → inRange → erode → dilate → download
        """
        # Upload to GPU
        gpu = cp.asarray(roi)

        # Gaussian blur (per-channel)
        blurred = gaussian_filter(
            gpu.astype(cp.float32), sigma=(self._blur_sigma, self._blur_sigma, 0),
        ).astype(cp.uint8)

        # Double BGR→HSV conversion (matches calibrated color space)
        hsv = _bgr_to_hsv_gpu(blurred)
        hsv = _bgr_to_hsv_gpu(hsv)

        # inRange: element-wise threshold (no cv2.cuda.inRange equivalent)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        lo, hi = self._lower_gpu, self._upper_gpu
        mask = (
            (h >= lo[0]) & (h <= hi[0])
            & (s >= lo[1]) & (s <= hi[1])
            & (v >= lo[2]) & (v <= hi[2])
        )

        # Morphological operations
        if self.morph_iterations > 0 and cp.any(mask):
            mask = binary_erosion(mask, iterations=1)
            mask = binary_dilation(mask, iterations=self.morph_iterations)

        # Download to CPU as uint8 (0/255)
        return (mask.astype(cp.uint8) * 255).get()

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
        """Detect ball using CuPy GPU-accelerated pipeline."""
        # Crop to detection zone + margin
        h, w = frame.shape[:2]
        margin = 15
        crop_y1 = max(0, zone_y1 - margin)
        crop_y2 = min(h, zone_y2 + margin)
        crop_x1 = max(0, zone_x1 - margin)
        crop_x2 = min(w, zone_x2_limit + margin)
        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if not roi.flags["C_CONTIGUOUS"]:
            roi = np.ascontiguousarray(roi)

        # GPU pipeline → CPU mask
        mask = self._gpu_pipeline(roi)

        # Skip morph countNonZero check (already in _gpu_pipeline)

        # Extract detection zone from cropped mask
        inner_y1 = zone_y1 - crop_y1
        inner_y2 = inner_y1 + (zone_y2 - zone_y1)
        inner_x1 = zone_x1 - crop_x1
        inner_x2 = inner_x1 + (zone_x2_limit - zone_x1)
        zone_mask = mask[inner_y1:inner_y2, inner_x1:inner_x2]

        # CPU contour analysis (same as parent)
        contours, _ = cv2.findContours(
            zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            ((cx, cy), r) = cv2.minEnclosingCircle(contour)
            cx += zone_x1
            cy += zone_y1
            r_int = int(r)

            if not (zone_y1 <= cy <= zone_y2):
                continue
            if r_int < self.min_radius:
                continue

            area = cv2.contourArea(contour)
            if r > 0 and self.min_circularity > 0:
                circularity = area / (np.pi * r * r)
                if circularity < self.min_circularity:
                    continue

            if expected_radius is not None and not (
                expected_radius - radius_tolerance < r_int < expected_radius + radius_tolerance
            ):
                continue

            return BallDetection(
                x=int(cx), y=int(cy), radius=r_int,
                contour_area=area, timestamp=timestamp,
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
        """Get color detection mask using GPU acceleration."""
        h, w = frame.shape[:2]
        margin = 15
        cy1 = max(0, zone_y1 - margin)
        cy2 = min(h, zone_y2 + margin)
        cx1 = max(0, zone_x1 - margin)
        cx2 = min(w, zone_x2_limit + margin)
        roi = frame[cy1:cy2, cx1:cx2]

        if not roi.flags["C_CONTIGUOUS"]:
            roi = np.ascontiguousarray(roi)

        mask = self._gpu_pipeline(roi)
        iy1 = zone_y1 - cy1
        ix1 = zone_x1 - cx1
        return mask[iy1:iy1 + (zone_y2 - zone_y1), ix1:ix1 + (zone_x2_limit - zone_x1)]
