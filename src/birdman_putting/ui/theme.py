"""Birdman Putting UI theme — colors, fonts, and styling constants.

Design derived from the Birdman Stitch project:
  - Dark navy/black backgrounds
  - Neon green accent for status and highlights
  - Blue (#1173d4) primary accent for buttons
  - Space Grotesk font family
  - 8px rounded corners
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Backgrounds
BG_ROOT = "#0a0e14"        # Deepest background (root window)
BG_PANEL = "#111820"       # Side panel / card container
BG_CARD = "#141c26"        # Individual cards
BG_CARD_HOVER = "#1a2432"  # Card hover state
BG_INPUT = "#0d1219"       # Entry/textbox fields
BG_VIDEO = "#080c10"       # Video panel placeholder

# Accent colors
ACCENT_GREEN = "#00e85a"       # Neon green (status, active states, connected)
ACCENT_GREEN_DIM = "#00a040"   # Dimmer green for hover/secondary
ACCENT_BLUE = "#1173d4"        # Primary blue (buttons, links)
ACCENT_BLUE_HOVER = "#0d5fb3"  # Blue hover state
ACCENT_ORANGE = "#e88a00"      # Warning / edit mode
ACCENT_ORANGE_HOVER = "#c47400"

# Status colors
STATUS_OK = "#00e85a"
STATUS_ERROR = "#ff4444"
STATUS_WARNING = "#e8aa00"
STATUS_IDLE = "#4a5568"

# Text colors
TEXT_PRIMARY = "#e6edf3"     # Main text (off-white, easier on eyes than pure white)
TEXT_SECONDARY = "#8b949e"   # Labels, descriptions
TEXT_MUTED = "#484f58"       # Disabled / hint text
TEXT_HEADER = "#00e85a"      # Section headers (neon green)

# Borders
BORDER_SUBTLE = "#1e2832"    # Card/panel borders
BORDER_ACTIVE = "#1173d4"    # Active/focused element borders

# Button styles (fg_color, hover_color)
BTN_PRIMARY = (ACCENT_BLUE, ACCENT_BLUE_HOVER)
BTN_SUCCESS = ("#1a8a3e", "#157a34")
BTN_DANGER = ("#cc3333", "#aa2222")
BTN_SECONDARY = ("#1e2832", "#252f3c")
BTN_WARNING = (ACCENT_ORANGE, ACCENT_ORANGE_HOVER)

# Slider
SLIDER_FG = ACCENT_BLUE
SLIDER_PROGRESS = ACCENT_GREEN_DIM
SLIDER_BG = "#1e2832"

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
CORNER_RADIUS = 8
CORNER_RADIUS_SM = 6
CARD_PAD_X = 8
CARD_PAD_Y = 6
SECTION_PAD_TOP = 10

# ---------------------------------------------------------------------------
# OpenCV overlay colors (BGR format for cv2)
# ---------------------------------------------------------------------------
CV_GREEN = (90, 232, 0)        # ACCENT_GREEN in BGR
CV_GREEN_DIM = (64, 160, 0)    # Dimmer green
CV_BLUE = (212, 115, 17)       # ACCENT_BLUE in BGR
CV_WHITE = (227, 237, 230)     # TEXT_PRIMARY in BGR
CV_GRAY = (88, 79, 72)         # TEXT_MUTED in BGR
CV_RED = (68, 68, 255)         # STATUS_ERROR in BGR
CV_CYAN = (218, 170, 0)        # Teal/cyan for trails

# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "fonts"
FONT_FILE = FONT_DIR / "SpaceGrotesk-Variable.ttf"
FONT_FAMILY = "Space Grotesk"

_font_loaded = False


def load_font() -> str:
    """Register Space Grotesk with the OS so tkinter can use it.

    Returns the font family name to use. Falls back to a system sans-serif
    if loading fails.
    """
    global _font_loaded  # noqa: PLW0603
    if _font_loaded:
        return FONT_FAMILY

    if not FONT_FILE.exists():
        logger.warning("Font file not found: %s — using system font", FONT_FILE)
        return _fallback_font()

    system = platform.system()
    try:
        if system == "Windows":
            _load_font_windows()
        elif system == "Darwin":
            _load_font_macos()
        else:
            _load_font_linux()
        _font_loaded = True
        logger.info("Loaded font: %s", FONT_FAMILY)
        return FONT_FAMILY
    except Exception:
        logger.warning("Failed to load Space Grotesk font", exc_info=True)
        return _fallback_font()


def _fallback_font() -> str:
    """Return a suitable fallback font for the current platform."""
    system = platform.system()
    if system == "Windows":
        return "Segoe UI"
    if system == "Darwin":
        return "SF Pro"
    return "sans-serif"


def _load_font_windows() -> None:
    """Load font on Windows using GDI."""
    import ctypes

    FR_PRIVATE = 0x10
    result = ctypes.windll.gdi32.AddFontResourceExW(  # type: ignore[attr-defined]
        str(FONT_FILE), FR_PRIVATE, 0,
    )
    if result == 0:
        msg = "AddFontResourceExW returned 0"
        raise RuntimeError(msg)


def _load_font_macos() -> None:
    """Load font on macOS using CoreText."""
    import ctypes
    import ctypes.util

    ct_path = ctypes.util.find_library("CoreText")
    if not ct_path:
        msg = "CoreText framework not found"
        raise RuntimeError(msg)

    ct = ctypes.cdll.LoadLibrary(ct_path)

    cf_path = ctypes.util.find_library("CoreFoundation")
    if not cf_path:
        msg = "CoreFoundation framework not found"
        raise RuntimeError(msg)
    cf = ctypes.cdll.LoadLibrary(cf_path)

    # Create CFString from font path
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
    ]
    cf_str = cf.CFStringCreateWithCString(None, str(FONT_FILE).encode(), 0x08000100)

    # Create CFURL
    cf.CFURLCreateWithFileSystemPath.restype = ctypes.c_void_p
    cf.CFURLCreateWithFileSystemPath.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool,
    ]
    cf_url = cf.CFURLCreateWithFileSystemPath(None, cf_str, 0, False)

    # Register font
    ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
    ct.CTFontManagerRegisterFontsForURL.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    success = ct.CTFontManagerRegisterFontsForURL(cf_url, 1, None)  # kCTFontManagerScopeProcess

    # Release
    cf.CFRelease(cf_str)
    cf.CFRelease(cf_url)

    if not success:
        msg = "CTFontManagerRegisterFontsForURL failed"
        raise RuntimeError(msg)


def _load_font_linux() -> None:
    """Load font on Linux by symlinking into user font directory."""
    user_fonts = Path.home() / ".local" / "share" / "fonts"
    user_fonts.mkdir(parents=True, exist_ok=True)
    dest = user_fonts / FONT_FILE.name
    if not dest.exists():
        import shutil
        shutil.copy2(FONT_FILE, dest)


# ---------------------------------------------------------------------------
# Font helpers for CTk widgets
# ---------------------------------------------------------------------------

def font(size: int = 12, weight: str = "normal") -> tuple[str, int] | tuple[str, int, str]:
    """Return a font tuple for CustomTkinter widgets.

    Args:
        size: Font size in points.
        weight: "normal" or "bold".
    """
    family = FONT_FAMILY if _font_loaded else _fallback_font()
    if weight == "bold":
        return (family, size, "bold")
    return (family, size)
