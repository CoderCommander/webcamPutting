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

        def widen(self, extra_pct: float = 0.3) -> bool:  # pragma: no cover
            return False

        def restore(self) -> None:  # pragma: no cover
            pass

        def close(self) -> None:  # pragma: no cover
            pass

else:
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]

    # Enable per-monitor DPI awareness so GetClientRect/PrintWindow return
    # physical pixels, not logical.  Without this, 125% DPI scaling causes
    # the capture to miss ~20% of the window content on the right side.
    import contextlib

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:
        # Windows 7 or already set — fall back to older API
        with contextlib.suppress(Exception):
            user32.SetProcessDPIAware()

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
            """Locate the target window by title (exact or partial match).

            Tries exact match first via FindWindowW, then falls back to
            EnumWindows with case-insensitive substring match.

            Returns True if the window was found.
            """
            # Try exact match first
            hwnd = user32.FindWindowW(None, self._title)
            if hwnd:
                self._hwnd = hwnd
                logger.info("Found window '%s' (hwnd=%d)", self._title, hwnd)
                return True

            # Fall back to partial title match via EnumWindows
            target = self._title.lower()
            found_hwnd = ctypes.c_int(0)

            @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
            def _enum_callback(hwnd: int, _lparam: int) -> bool:
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if target in title.lower():
                        found_hwnd.value = hwnd
                        logger.info(
                            "Found window by partial match: '%s' (hwnd=%d)",
                            title, hwnd,
                        )
                        return False  # Stop enumerating
                return True  # Continue

            user32.EnumWindows(_enum_callback, 0)

            if found_hwnd.value:
                self._hwnd = found_hwnd.value
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

        def widen(self, extra_pct: float = 0.3) -> bool:
            """Temporarily widen the window to reveal overflowing content.

            FS Golf's web-based layout may have columns that overflow the
            viewport.  Making the window wider forces a reflow so all
            columns render and can be captured.

            Returns True if the window was resized.
            """
            if not self._hwnd:
                return False

            SW_RESTORE = 9
            SWP_NOZORDER = 0x0004

            self._was_maximized = bool(user32.IsZoomed(self._hwnd))
            if self._was_maximized:
                user32.ShowWindow(self._hwnd, SW_RESTORE)
                import time
                time.sleep(0.3)

            rect = wt.RECT()
            user32.GetWindowRect(self._hwnd, ctypes.byref(rect))
            self._orig_rect = (
                rect.left, rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            )
            # Use full screen dimensions so the web-based FS Golf layout
            # reflows all columns on-screen.  Position at (0, 0) and use
            # full screen size — off-screen content won't render.
            screen_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            screen_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            wider_w = min(
                max(int(self._orig_rect[2] * (1.0 + extra_pct)), screen_w),
                screen_w,
            )
            # Use full screen height to prevent vertical compression
            taller_h = max(self._orig_rect[3], screen_h)
            user32.SetWindowPos(
                self._hwnd, 0,
                0, 0, wider_w, taller_h,
                SWP_NOZORDER,
            )
            import time
            time.sleep(0.5)
            logger.info(
                "Widened window '%s' from %d to %d px (screen=%d)",
                self._title, self._orig_rect[2], wider_w, screen_w,
            )
            return True

        def restore(self) -> None:
            """Restore the original window size after widen()."""
            if not self._hwnd:
                return

            SWP_NOZORDER = 0x0004
            SW_MAXIMIZE = 3

            orig = getattr(self, "_orig_rect", None)
            if orig is not None:
                user32.SetWindowPos(
                    self._hwnd, 0,
                    orig[0], orig[1], orig[2], orig[3],
                    SWP_NOZORDER,
                )
                if getattr(self, "_was_maximized", False):
                    user32.ShowWindow(self._hwnd, SW_MAXIMIZE)
                logger.info("Restored window '%s' to original size", self._title)

        def send_key(self, char: str) -> bool:
            """Send a single key press to the window.

            Args:
                char: Single character to send (e.g. 'c' for chipping, 'f' for full swing).

            Returns True if the message was posted.
            """
            if not self._hwnd:
                return False
            WM_CHAR = 0x0102
            result = user32.PostMessageW(self._hwnd, WM_CHAR, ord(char), 0)
            if result:
                logger.info("Sent key '%s' to window '%s'", char, self._title)
            else:
                logger.warning("Failed to send key '%s' to window '%s'", char, self._title)
            return bool(result)

        def close(self) -> None:
            """Release any held resources."""
            self._hwnd = 0
