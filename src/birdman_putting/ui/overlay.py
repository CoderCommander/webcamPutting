"""HUD overlay renderer — draws detection zones, ball markers, and status onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from birdman_putting.config import DetectionZone
from birdman_putting.detection import BallDetection
from birdman_putting.tracking import ShotState

# Colors (BGR)
COLOR_YELLOW = (0, 210, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK_GREEN = (0, 180, 0)
COLOR_BLACK = (0, 0, 0)

_HANDLE_SIZE = 5  # Half-size of handle square in pixels

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
    if zone.direction == "right_to_left":
        gateway_x2 = zone.start_x1 - zone.gateway_width
        gateway_x1 = gateway_x2 - zone.gateway_width
    else:
        gateway_x1 = zone.start_x2 + zone.gateway_width
        gateway_x2 = gateway_x1 + zone.gateway_width

    # Semi-transparent yellow fill when editing
    if edit_mode:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (zone.start_x1, zone.y1),
            (zone.start_x2, zone.y2),
            COLOR_YELLOW, -1,
        )
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Start zone (yellow)
    cv2.rectangle(
        frame,
        (zone.start_x1, zone.y1),
        (zone.start_x2, zone.y2),
        COLOR_YELLOW, 2,
    )

    # Detection gateway — green when ball has crossed, red otherwise
    gw_color = COLOR_GREEN if state in (ShotState.ENTERED, ShotState.LEFT) else COLOR_RED
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
    max_dots: int = 30,
    dot_radius: int = 3,
) -> None:
    """Draw golf-simulator-style dotted trail along ball path.

    Renders evenly-spaced marker dots with a brightness fade
    from dim (oldest) to bright (newest), each with a thin white outline.
    """
    if len(positions) < 2:
        return

    # Subsample to max_dots evenly-spaced points
    if len(positions) > max_dots:
        step = len(positions) / max_dots
        indices = [int(i * step) for i in range(max_dots)]
        if indices[-1] != len(positions) - 1:
            indices.append(len(positions) - 1)
        sampled = [positions[i] for i in indices]
    else:
        sampled = positions

    n = len(sampled)
    for i, (x, y) in enumerate(sampled):
        # Brightness ramp: oldest ~30%, newest 100%
        alpha = 0.3 + 0.7 * (i / max(1, n - 1))
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.circle(frame, (x, y), dot_radius, c, -1)
        cv2.circle(frame, (x, y), dot_radius, COLOR_WHITE, 1)


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
        draw_ball_trail(frame, active_trail, color=COLOR_GREEN, dot_radius=4)

    # --- Status text (top-left) ---
    y = 18
    line_h = 18

    def _text(text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, pos, FONT, FONT_SCALE, color, FONT_THICKNESS)

    # FPS
    _text(f"FPS: {fps:.1f}", (8, y), COLOR_GREEN)
    y += line_h

    # State
    state_color = COLOR_GREEN if state in (ShotState.STARTED, ShotState.ENTERED) else COLOR_GRAY
    _text(f"State: {state.value}", (8, y), state_color)
    y += line_h

    # Connection
    conn_color = COLOR_DARK_GREEN if connected else COLOR_RED
    conn_label = "Connected" if connected else "Disconnected"
    _text(f"GSPro: {conn_label}", (8, y), conn_color)
    y += line_h

    # Shot count
    _text(f"Shots: {shot_count}", (8, y), COLOR_WHITE)

    # --- Last shot (bottom-left) ---
    if last_speed > 0:
        h = frame.shape[0]
        cv2.putText(
            frame,
            f"Last: {last_speed:.1f} MPH  /  {last_hla:+.1f} HLA",
            (8, h - 12),
            FONT, 0.55, COLOR_WHITE, FONT_THICKNESS,
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
