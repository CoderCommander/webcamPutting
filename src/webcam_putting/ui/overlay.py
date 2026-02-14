"""HUD overlay renderer — draws detection zones, ball markers, and status onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from webcam_putting.config import DetectionZone
from webcam_putting.detection import BallDetection
from webcam_putting.tracking import ShotState

# Colors (BGR)
COLOR_YELLOW = (0, 210, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_GRAY = (140, 140, 140)
COLOR_DARK_GREEN = (0, 180, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1


def draw_detection_zones(
    frame: np.ndarray,
    zone: DetectionZone,
    state: ShotState,
) -> None:
    """Draw start zone and detection gateway rectangles."""
    gateway_x1 = zone.start_x2 + zone.gateway_width
    gateway_x2 = gateway_x1 + zone.gateway_width

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


def draw_ball_marker(
    frame: np.ndarray,
    detection: BallDetection | None,
) -> None:
    """Draw circle around detected ball."""
    if detection is None:
        return
    cv2.circle(frame, (detection.x, detection.y), detection.radius, COLOR_RED, 2)
    cv2.circle(frame, (detection.x, detection.y), 2, COLOR_GREEN, -1)


def draw_last_shot_trajectory(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """Draw trajectory line from last shot."""
    if start == (0, 0) and end == (0, 0):
        return
    cv2.line(frame, start, end, COLOR_CYAN, 2)


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
) -> None:
    """Draw the complete HUD overlay onto a frame.

    This is the single entry point — call this from the processing loop
    after detection/tracking but before handing the frame to the UI.
    """
    draw_detection_zones(frame, zone, state)
    draw_ball_marker(frame, detection)
    draw_last_shot_trajectory(frame, last_start, last_end)

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
