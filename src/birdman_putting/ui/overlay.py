"""HUD overlay renderer — draws detection zones, ball markers, and status onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from birdman_putting.config import DetectionZone
from birdman_putting.detection import BallDetection
from birdman_putting.tracking import ShotState
from birdman_putting.ui.theme import (
    CV_BLUE,
    CV_CYAN,
    CV_GRAY,
    CV_GREEN,
    CV_GREEN_DIM,
    CV_RED,
    CV_WHITE,
)

# Colors (BGR) — theme-consistent, sourced from theme.py
COLOR_YELLOW = (0, 210, 255)
COLOR_RED = CV_RED
COLOR_GREEN = CV_GREEN
COLOR_WHITE = CV_WHITE
COLOR_CYAN = CV_CYAN
COLOR_GRAY = CV_GRAY
COLOR_DARK_GREEN = CV_GREEN_DIM
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = CV_BLUE

_HANDLE_SIZE = 8  # Half-size of handle square in pixels

# Named color palette for zone/gateway customization (BGR)
ZONE_COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "yellow": COLOR_YELLOW,
    "red": COLOR_RED,
    "green": COLOR_GREEN,
    "cyan": COLOR_CYAN,
    "white": COLOR_WHITE,
    "orange": (0, 140, 255),
    "magenta": (255, 0, 255),
    "blue": (255, 128, 0),
    "gray": COLOR_GRAY,
}


def _resolve_zone_color(
    name: str, fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Look up a named zone color, falling back to a default."""
    return ZONE_COLOR_PALETTE.get(name, fallback)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1


def draw_detection_zones(
    frame: np.ndarray,
    zone: DetectionZone,
    state: ShotState,
    edit_mode: bool = False,
) -> None:
    """Draw start zone and detection gateway rectangles."""
    zone_color = _resolve_zone_color(zone.zone_color, COLOR_YELLOW)
    gw_idle_color = _resolve_zone_color(zone.gateway_color, COLOR_RED)

    if zone.direction == "right_to_left":
        gateway_x2 = zone.start_x1 - zone.gateway_width
        gateway_x1 = gateway_x2 - zone.gateway_width
    else:
        gateway_x1 = zone.start_x2 + zone.gateway_width
        gateway_x2 = gateway_x1 + zone.gateway_width

    # Semi-transparent fill when editing
    if edit_mode:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (zone.start_x1, zone.y1),
            (zone.start_x2, zone.y2),
            zone_color, -1,
        )
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Start zone outline
    cv2.rectangle(
        frame,
        (zone.start_x1, zone.y1),
        (zone.start_x2, zone.y2),
        zone_color, 2,
    )

    # Detection gateway — green when ball has crossed, user color otherwise
    gw_color = COLOR_GREEN if state in (ShotState.ENTERED, ShotState.LEFT) else gw_idle_color
    cv2.rectangle(
        frame,
        (gateway_x1, zone.y1),
        (gateway_x2, zone.y2),
        gw_color, 2,
    )

    # Draw drag handles when editing
    if edit_mode:
        _draw_zone_handles(frame, zone)


def _draw_zone_handles(frame: np.ndarray, zone: DetectionZone) -> None:
    """Draw drag handles at corners and edge midpoints of the start zone."""
    x1, x2 = zone.start_x1, zone.start_x2
    y1, y2 = zone.y1, zone.y2
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    s = _HANDLE_SIZE

    handles = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        (mid_x, y1), (mid_x, y2), (x1, mid_y), (x2, mid_y),
    ]
    for hx, hy in handles:
        cv2.rectangle(frame, (hx - s, hy - s), (hx + s, hy + s), COLOR_BLACK, -1)
        cv2.rectangle(
            frame, (hx - s + 1, hy - s + 1), (hx + s - 1, hy + s - 1),
            COLOR_WHITE, -1,
        )


def draw_ball_marker(
    frame: np.ndarray,
    detection: BallDetection | None,
) -> None:
    """Draw circle around detected ball."""
    if detection is None:
        return
    cv2.circle(frame, (detection.x, detection.y), detection.radius, COLOR_RED, 2)
    cv2.circle(frame, (detection.x, detection.y), 2, COLOR_GREEN, -1)


def draw_ball_trail(
    frame: np.ndarray,
    positions: list[tuple[int, int]],
    color: tuple[int, int, int] = COLOR_CYAN,
    max_points: int = 50,
) -> None:
    """Draw a smooth streaking trail with taper and fade.

    Renders connected line segments that taper from thin (oldest) to thick
    (newest) with brightness fade and a subtle glow effect.
    """
    if len(positions) < 2:
        return

    # Subsample to max_points evenly-spaced points
    if len(positions) > max_points:
        step = len(positions) / max_points
        indices = [int(i * step) for i in range(max_points)]
        if indices[-1] != len(positions) - 1:
            indices.append(len(positions) - 1)
        sampled = [positions[i] for i in indices]
    else:
        sampled = positions

    n = len(sampled)
    if n < 2:
        return

    # Glow pass: wider, semi-transparent lines on an overlay
    glow = frame.copy()
    for i in range(1, n):
        t = i / (n - 1)  # 0.0 (oldest) → 1.0 (newest)
        alpha = 0.2 + 0.8 * t
        thickness = max(1, int(2 + 10 * t))
        c = (int(color[0] * alpha * 0.4), int(color[1] * alpha * 0.4),
             int(color[2] * alpha * 0.4))
        cv2.line(glow, sampled[i - 1], sampled[i], c, thickness, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.6, frame, 0.4, 0, frame)

    # Main trail: tapering, bright line segments
    for i in range(1, n):
        t = i / (n - 1)
        alpha = 0.2 + 0.8 * t
        thickness = max(1, int(1 + 5 * t))
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.line(frame, sampled[i - 1], sampled[i], c, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    zone: DetectionZone,
    state: ShotState,
    detection: BallDetection | None,
    fps: float,
    connected: bool,
    connection_mode: str,
    last_speed: float,
    last_hla: float,
    last_start: tuple[int, int],
    last_end: tuple[int, int],
    shot_count: int,
    edit_mode: bool = False,
    active_trail: list[tuple[int, int]] | None = None,
    last_shot_trail: list[tuple[int, int]] | None = None,
) -> None:
    """Draw the complete HUD overlay onto a frame.

    This is the single entry point — call this from the processing loop
    after detection/tracking but before handing the frame to the UI.
    """
    draw_detection_zones(frame, zone, state, edit_mode=edit_mode)
    draw_ball_marker(frame, detection)

    # Ball trails (golf simulator stripe style)
    if last_shot_trail:
        draw_ball_trail(frame, last_shot_trail, color=COLOR_CYAN)
    if active_trail:
        draw_ball_trail(frame, active_trail, color=COLOR_GREEN)

    # --- Status text (top-left) with semi-transparent backdrop ---
    y = 18
    line_h = 18
    num_lines = 4 + (1 if last_speed > 0 else 0)
    _pad = 6

    # Draw dark backdrop behind status text
    overlay_bg = frame.copy()
    cv2.rectangle(overlay_bg, (0, 0), (180, _pad + num_lines * line_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay_bg, 0.55, frame, 0.45, 0, frame)

    def _text(text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, pos, FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    # FPS
    _text(f"FPS: {fps:.1f}", (8, y), COLOR_GREEN)
    y += line_h

    # State
    state_color = COLOR_GREEN if state in (ShotState.STARTED, ShotState.ENTERED) else COLOR_GRAY
    _text(f"State: {state.value}", (8, y), state_color)
    y += line_h

    # Connection
    conn_color = COLOR_GREEN if connected else COLOR_RED
    conn_label = "Connected" if connected else "Disconnected"
    _text(f"GSPro: {conn_label}", (8, y), conn_color)
    y += line_h

    # Shot count
    _text(f"Shots: {shot_count}", (8, y), COLOR_WHITE)

    # --- Last shot (bottom-left) ---
    if last_speed > 0:
        h = frame.shape[0]
        # Backdrop for last shot
        shot_bg = frame.copy()
        cv2.rectangle(shot_bg, (0, h - 28), (280, h), COLOR_BLACK, -1)
        cv2.addWeighted(shot_bg, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            f"Last: {last_speed:.1f} MPH  /  {last_hla:+.1f} HLA",
            (8, h - 10),
            FONT, 0.55, COLOR_GREEN, FONT_THICKNESS, cv2.LINE_AA,
        )


def draw_calibration_overlay(
    frame: np.ndarray,
    state_text: str,
    ball_pos: tuple[int, int] | None = None,
) -> None:
    """Draw calibration mode overlay: green border + status + crosshair."""
    h, w = frame.shape[:2]

    # Green border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_GREEN, 3)

    # Status text (top-center)
    text_size = cv2.getTextSize(state_text, FONT, 0.65, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, state_text, (text_x, 28), FONT, 0.65, COLOR_GREEN, 2)

    # Crosshair on detected ball
    if ball_pos is not None:
        bx, by = ball_pos
        arm = 20
        cv2.line(frame, (bx - arm, by), (bx + arm, by), COLOR_GREEN, 1)
        cv2.line(frame, (bx, by - arm), (bx, by + arm), COLOR_GREEN, 1)
        cv2.circle(frame, (bx, by), 6, COLOR_GREEN, 2)
