"""CUDA GPU initialization and device selection via CuPy."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CuPy is installed and a CUDA GPU is available."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_device_count() -> int:
    """Return the number of CUDA-enabled devices."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def get_device_name(index: int) -> str:
    """Get the name of a CUDA device by index."""
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(index)
        return props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    except Exception:
        return f"device_{index}"


def init_cuda(preferred_device: str = "A3000") -> bool:
    """Initialize CuPy on the preferred GPU device.

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
        logger.warning("CUDA not available (CuPy not installed or no GPU?)")
        return False

    try:
        import cupy as cp
    except ImportError:
        logger.warning("CuPy not installed — pip install cupy-cuda12x")
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
        cp.cuda.Device(selected_index).use()
        name = get_device_name(selected_index)
        logger.info("Selected CUDA device %d: %s", selected_index, name)
        return True

    # No preferred match — use the first device as fallback
    if count > 0:
        cp.cuda.Device(0).use()
        name = get_device_name(0)
        logger.warning(
            "No device matching '%s' found, using device 0: %s",
            preferred_device, name,
        )
        return True

    return False
