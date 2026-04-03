"""GPU-accelerated ball detection using OpenCV CUDA."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from birdman_putting.color_presets import HSVRange
from birdman_putting.detection import BallDetection, BallDetector

logger = logging.getLogger(__name__)


class CudaBallDetector(BallDetector):
    """Ball detector that offloads image processing to a CUDA GPU.

    Subclasses :class:`BallDetector` and overrides :meth:`detect` and
    :meth:`get_mask` to run blur, color conversion, and morphological
    operations on the GPU. Contour analysis remains on the CPU (no CUDA
    equivalent in OpenCV).

    The hybrid pipeline:
        1. Crop ROI on CPU (numpy slice)
        2. Upload ROI to GPU
        3. Gaussian blur on GPU
        4. BGR→HSV double conversion on GPU
        5. Download HSV to CPU (cv2.cuda.inRange doesn't exist)
        6. inRange on CPU
        7. Upload mask to GPU
        8. Erode + dilate on GPU
        9. Download mask to CPU
        10. findContours + contour filtering on CPU
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        blur_kernel: tuple[int, int] = (7, 7),
        min_radius: int = 5,
        min_circularity: float = 0.5,
        morph_iterations: int = 5,
    ):
        super().__init__(
            hsv_range=hsv_range,
            blur_kernel=blur_kernel,
            min_radius=min_radius,
            min_circularity=min_circularity,
            morph_iterations=morph_iterations,
        )

        # Pre-allocate GpuMat objects for reuse (avoid per-frame allocation)
        self._gpu_roi = cv2.cuda.GpuMat()
        self._gpu_blurred = cv2.cuda.GpuMat()
        self._gpu_hsv1 = cv2.cuda.GpuMat()
        self._gpu_hsv2 = cv2.cuda.GpuMat()
        self._gpu_mask = cv2.cuda.GpuMat()

        # Create CUDA filter objects once
        self._cuda_blur = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, blur_kernel, 0,
        )
        self._cuda_erode = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_ERODE, cv2.CV_8U, self._morph_kernel,
        )
        self._cuda_dilate = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8U, self._morph_kernel,
        )

        logger.info("CudaBallDetector initialized (GPU-accelerated)")

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
        """Detect ball using GPU-accelerated image processing.

        Same interface as :meth:`BallDetector.detect`.
        """
        # Crop to detection zone + margin BEFORE GPU upload
        h, w = frame.shape[:2]
        margin = 15
        crop_y1 = max(0, zone_y1 - margin)
        crop_y2 = min(h, zone_y2 + margin)
        crop_x1 = max(0, zone_x1 - margin)
        crop_x2 = min(w, zone_x2_limit + margin)
        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Ensure contiguous for GPU upload
        if not roi.flags["C_CONTIGUOUS"]:
            roi = np.ascontiguousarray(roi)

        # === GPU pipeline ===
        # Upload ROI to GPU
        self._gpu_roi.upload(roi)

        # Gaussian blur on GPU
        self._cuda_blur.apply(self._gpu_roi, self._gpu_blurred)

        # Double BGR→HSV conversion on GPU (matches calibrated color space)
        cv2.cuda.cvtColor(self._gpu_blurred, cv2.COLOR_BGR2HSV, self._gpu_hsv1)
        cv2.cuda.cvtColor(self._gpu_hsv1, cv2.COLOR_BGR2HSV, self._gpu_hsv2)

        # Download HSV to CPU for inRange (no CUDA equivalent)
        hsv = self._gpu_hsv2.download()

        # inRange on CPU (using cached bounds from parent class)
        mask = cv2.inRange(hsv, self._lower, self._upper)

        # Upload mask to GPU for morphological operations
        if self.morph_iterations > 0 and cv2.countNonZero(mask) > 0:
            self._gpu_mask.upload(mask)
            # Erode once
            self._cuda_erode.apply(self._gpu_mask, self._gpu_mask)
            # Dilate N times
            for _ in range(self.morph_iterations):
                self._cuda_dilate.apply(self._gpu_mask, self._gpu_mask)
            # Download mask back to CPU
            mask = self._gpu_mask.download()

        # === CPU contour analysis (no CUDA equivalent) ===
        # Extract the detection zone from the cropped mask
        inner_y1 = zone_y1 - crop_y1
        inner_y2 = inner_y1 + (zone_y2 - zone_y1)
        inner_x1 = zone_x1 - crop_x1
        inner_x2 = inner_x1 + (zone_x2_limit - zone_x1)
        zone_mask = mask[inner_y1:inner_y2, inner_x1:inner_x2]

        # Find contours sorted by area (largest first)
        contours, _ = cv2.findContours(
            zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            ((cx, cy), r) = cv2.minEnclosingCircle(contour)

            # Offset coordinates back to full frame
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
                x=int(cx),
                y=int(cy),
                radius=r_int,
                contour_area=area,
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
        """Get the color detection mask using GPU acceleration."""
        h, w = frame.shape[:2]
        margin = 15
        cy1 = max(0, zone_y1 - margin)
        cy2 = min(h, zone_y2 + margin)
        cx1 = max(0, zone_x1 - margin)
        cx2 = min(w, zone_x2_limit + margin)
        roi = frame[cy1:cy2, cx1:cx2]

        if not roi.flags["C_CONTIGUOUS"]:
            roi = np.ascontiguousarray(roi)

        # GPU pipeline
        self._gpu_roi.upload(roi)
        self._cuda_blur.apply(self._gpu_roi, self._gpu_blurred)
        cv2.cuda.cvtColor(self._gpu_blurred, cv2.COLOR_BGR2HSV, self._gpu_hsv1)
        cv2.cuda.cvtColor(self._gpu_hsv1, cv2.COLOR_BGR2HSV, self._gpu_hsv2)
        hsv = self._gpu_hsv2.download()

        mask = cv2.inRange(hsv, self._lower, self._upper)

        if self.morph_iterations > 0 and cv2.countNonZero(mask) > 0:
            self._gpu_mask.upload(mask)
            self._cuda_erode.apply(self._gpu_mask, self._gpu_mask)
            for _ in range(self.morph_iterations):
                self._cuda_dilate.apply(self._gpu_mask, self._gpu_mask)
            mask = self._gpu_mask.download()

        iy1 = zone_y1 - cy1
        ix1 = zone_x1 - cx1
        return mask[iy1:iy1 + (zone_y2 - zone_y1), ix1:ix1 + (zone_x2_limit - zone_x1)]
