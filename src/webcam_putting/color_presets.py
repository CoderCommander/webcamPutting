"""HSV color presets for different ball colors and lighting conditions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HSVRange:
    """HSV color range for ball detection."""

    hmin: int
    smin: int
    vmin: int
    hmax: int
    smax: int
    vmax: int

    def to_dict(self) -> dict[str, int]:
        return {
            "hmin": self.hmin, "smin": self.smin, "vmin": self.vmin,
            "hmax": self.hmax, "smax": self.smax, "vmax": self.vmax,
        }

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> HSVRange:
        return cls(
            hmin=d["hmin"], smin=d["smin"], vmin=d["vmin"],
            hmax=d["hmax"], smax=d["smax"], vmax=d["vmax"],
        )


# All presets from the original cam-putting-py ball_tracking.py
# Format: "name" -> HSVRange, organized by base color and lighting variant

PRESETS: dict[str, HSVRange] = {
    # Red
    "red": HSVRange(hmin=1, smin=208, vmin=0, hmax=50, smax=255, vmax=249),       # bright
    "red2": HSVRange(hmin=1, smin=240, vmin=61, hmax=50, smax=255, vmax=249),      # dark

    # White
    "white": HSVRange(hmin=168, smin=218, vmin=118, hmax=179, smax=247, vmax=216), # very bright
    "white2": HSVRange(hmin=159, smin=217, vmin=152, hmax=179, smax=255, vmax=255),# bright
    "white3": HSVRange(hmin=0, smin=181, vmin=0, hmax=42, smax=255, vmax=255),     # test

    # Yellow
    "yellow": HSVRange(hmin=0, smin=210, vmin=0, hmax=15, smax=255, vmax=255),     # bright
    "yellow2": HSVRange(hmin=0, smin=150, vmin=100, hmax=46, smax=255, vmax=206),  # dark

    # Green
    "green": HSVRange(hmin=0, smin=169, vmin=161, hmax=177, smax=204, vmax=255),   # bright
    "green2": HSVRange(hmin=0, smin=109, vmin=74, hmax=81, smax=193, vmax=117),    # dark

    # Orange
    "orange": HSVRange(hmin=0, smin=219, vmin=147, hmax=19, smax=255, vmax=255),   # bright
    "orange2": HSVRange(hmin=3, smin=181, vmin=134, hmax=40, smax=255, vmax=255),  # dark
    "orange3": HSVRange(hmin=0, smin=73, vmin=150, hmax=40, smax=255, vmax=255),   # test
    "orange4": HSVRange(hmin=3, smin=181, vmin=216, hmax=40, smax=255, vmax=255),  # ps3eye
}

# Labels for UI display
PRESET_DESCRIPTIONS: dict[str, str] = {
    "red": "Red (bright)",
    "red2": "Red (dark)",
    "white": "White (very bright)",
    "white2": "White (bright)",
    "white3": "White (test)",
    "yellow": "Yellow (bright)",
    "yellow2": "Yellow (dark)",
    "green": "Green (bright)",
    "green2": "Green (dark)",
    "orange": "Orange (bright)",
    "orange2": "Orange (dark)",
    "orange3": "Orange (test)",
    "orange4": "Orange (PS3 Eye)",
}


def get_preset(name: str) -> HSVRange:
    """Get HSV preset by name, defaulting to yellow."""
    return PRESETS.get(name, PRESETS["yellow"])
