"""Windows GDI window capture via ctypes."""

from __future__ import annotations

import logging
import platform
import sys

import numpy as np

logger = logging.getLogger(__name__)

if sys.platform != "win32":

    class WindowCapture:
        """Stub for non-Windows platforms — raises on construction."""

        def __init__(self, window_title: str) -> None:
            raise OSError(
                f"WindowCapture requires Windows (current platform: {platform.system()})"
            )

        def find_window(self) -> bool:  # pragma: no cover
            return False

        def capture(self) -> np.ndarray | None:  # pragma: no cover
            return None

        def close(self) -> None:  # pragma: no cover
            pass

else:
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]

    # GDI constants
    SRCCOPY = 0x00CC0020
    DIB_RGB_COLORS = 0
    BI_RGB = 0

    # Set argtypes for GDI functions that take pointers (fixes 64-bit overflow)
    gdi32.GetDIBits.argtypes = [
        wt.HDC, wt.HBITMAP, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wt.DWORD),
            ("biWidth", wt.LONG),
            ("biHeight", wt.LONG),
            ("biPlanes", wt.WORD),
            ("biBitCount", wt.WORD),
            ("biCompression", wt.DWORD),
            ("biSizeImage", wt.DWORD),
            ("biXPelsPerMeter", wt.LONG),
            ("biYPelsPerMeter", wt.LONG),
            ("biClrUsed", wt.DWORD),
            ("biClrImportant", wt.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", wt.DWORD * 3),
        ]

    class WindowCapture:
        """Captures a Windows application window via GDI API."""

        def __init__(self, window_title: str) -> None:
            self._title = window_title
            self._hwnd: int = 0

        def find_window(self) -> bool:
            """Locate the target window by title.

            Returns True if the window was found.
            """
            hwnd = user32.FindWindowW(None, self._title)
            if hwnd:
                self._hwnd = hwnd
                logger.info("Found window '%s' (hwnd=%d)", self._title, hwnd)
                return True
            logger.warning("Window '%s' not found", self._title)
            return False

        def capture(self) -> np.ndarray | None:
            """Capture the window contents as a BGR numpy array.

            Returns None if the window is not found or cannot be captured.
            """
            if not self._hwnd and not self.find_window():
                return None

            # Get window dimensions
            rect = wt.RECT()
            if not user32.GetClientRect(self._hwnd, ctypes.byref(rect)):
                logger.debug("GetClientRect failed")
                return None

            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width <= 0 or height <= 0:
                return None

            # Get device contexts
            hwnd_dc = user32.GetDC(self._hwnd)
            if not hwnd_dc:
                return None

            mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
            bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
            gdi32.SelectObject(mem_dc, bitmap)

            # BitBlt copy
            result = user32.PrintWindow(self._hwnd, mem_dc, 1)
            if not result:
                # Fall back to BitBlt
                gdi32.BitBlt(
                    mem_dc, 0, 0, width, height,
                    hwnd_dc, 0, 0, SRCCOPY,
                )

            # Read pixel data
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = -height  # top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB

            buffer = np.empty((height, width, 4), dtype=np.uint8)
            gdi32.GetDIBits(
                mem_dc, bitmap, 0, height,
                buffer.ctypes.data,
                ctypes.byref(bmi),
                DIB_RGB_COLORS,
            )

            # Cleanup GDI objects
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(self._hwnd, hwnd_dc)

            # BGRA → BGR
            return np.ascontiguousarray(buffer[:, :, :3])

        def close(self) -> None:
            """Release any held resources."""
            self._hwnd = 0
