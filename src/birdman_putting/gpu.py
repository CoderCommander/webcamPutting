"""CUDA GPU initialization and device selection for OpenCV acceleration."""

from __future__ import annotations

import logging

import cv2

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if OpenCV was built with CUDA and a GPU is available."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except (cv2.error, AttributeError):
        return False


def get_device_count() -> int:
    """Return the number of CUDA-enabled devices."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount()
    except (cv2.error, AttributeError):
        return 0


def get_device_name(index: int) -> str:
    """Get the name of a CUDA device by index."""
    try:
        cv2.cuda.setDevice(index)
        # cv2.cuda.DeviceInfo is not always available; fall back to printShortCudaDeviceInfo
        info = cv2.cuda.DeviceInfo(index)
        return info.name()
    except Exception:
        return f"device_{index}"


def init_cuda(preferred_device: str = "A3000") -> bool:
    """Initialize CUDA on the preferred GPU device.

    Enumerates all CUDA devices and selects one whose name contains
    ``preferred_device`` (case-insensitive substring match). This allows
    targeting a specific GPU (e.g. the laptop's A3000) while skipping
    others (e.g. an eGPU A4500 used by GSPro).

    Args:
        preferred_device: Substring to match in the device name.

    Returns:
        True if a matching device was found and selected.
    """
    if not is_cuda_available():
        logger.warning("CUDA not available (OpenCV built without CUDA?)")
        return False

    count = get_device_count()
    logger.info("Found %d CUDA device(s)", count)

    target = preferred_device.lower()
    selected_index: int | None = None

    for i in range(count):
        name = get_device_name(i)
        logger.info("  CUDA device %d: %s", i, name)
        if target in name.lower() and selected_index is None:
            selected_index = i

    if selected_index is not None:
        cv2.cuda.setDevice(selected_index)
        name = get_device_name(selected_index)
        logger.info("Selected CUDA device %d: %s", selected_index, name)
        return True

    # No preferred match — use the first device as fallback
    if count > 0:
        cv2.cuda.setDevice(0)
        name = get_device_name(0)
        logger.warning(
            "No device matching '%s' found, using device 0: %s",
            preferred_device, name,
        )
        return True

    return False
