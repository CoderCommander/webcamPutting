"""Speed and HLA calculation with multi-point trajectory fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Standard golf ball diameter in mm
GOLF_BALL_DIAMETER_MM = 42.67
GOLF_BALL_RADIUS_MM = GOLF_BALL_DIAMETER_MM / 2  # 21.335 mm


@dataclass
class ShotData:
    """Calculated shot metrics."""

    speed_mph: float
    hla_degrees: float
    distance_mm: float
    elapsed_seconds: float


def pixel_to_mm_ratio(ball_radius_px: int) -> float:
    """Calculate pixels-per-mm ratio from detected ball radius.

    The ratio is: detected_radius_pixels / known_ball_radius_mm.
    To convert pixel distances to mm, divide by this ratio.
    """
    if ball_radius_px <= 0:
        return 0.0
    return ball_radius_px / GOLF_BALL_RADIUS_MM


def calculate_angle(p1: tuple[int, int], p2: tuple[int, int], flip: bool = False) -> float:
    """Calculate angle in degrees from p1 to p2.

    Returns angle in degrees using atan2, with golf convention
    (negative = left, positive = right when looking down the line).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(-dy, dx)
    if flip:
        rads = -rads
    return math.degrees(rads)


def fit_trajectory(
    positions: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a line through tracked positions using least-squares.

    Args:
        positions: List of (x, y, timestamp) tuples.

    Returns:
        (x_arr, y_arr, t_arr) numpy arrays with outliers removed,
        or None if insufficient data.
    """
    if len(positions) < 2:
        return None

    x = np.array([p[0] for p in positions], dtype=np.float64)
    y = np.array([p[1] for p in positions], dtype=np.float64)
    t = np.array([p[2] for p in positions], dtype=np.float64)

    if len(positions) < 4:
        # Not enough points for outlier rejection, return as-is
        return x, y, t

    # Fit line y = mx + b
    coeffs = np.polyfit(x, y, 1)
    predicted_y = np.polyval(coeffs, x)
    residuals = y - predicted_y

    # Reject outliers beyond 2 standard deviations
    std = np.std(residuals)
    if std > 0:
        mask = np.abs(residuals) < 2 * std
        if np.sum(mask) >= 2:
            return x[mask], y[mask], t[mask]

    return x, y, t


def calculate_shot(
    start_pos: tuple[int, int],
    end_pos: tuple[int, int],
    entry_time: float,
    exit_time: float,
    px_mm_ratio: float,
    positions: list[tuple[int, int, float]] | None = None,
    flip: bool = False,
) -> ShotData | None:
    """Calculate ball speed and HLA from tracked positions.

    Uses multi-point trajectory fitting when positions are available,
    falls back to simple start/end calculation otherwise.

    Args:
        start_pos: Ball position when entering detection gateway.
        end_pos: Ball position when exiting detection gateway.
        entry_time: perf_counter timestamp at entry.
        exit_time: perf_counter timestamp at exit.
        px_mm_ratio: Pixels per mm conversion ratio.
        positions: Optional list of (x, y, timestamp) for trajectory fitting.
        flip: Whether image is flipped (left-handed setup).

    Returns:
        ShotData with speed and HLA, or None if calculation fails.
    """
    if px_mm_ratio <= 0:
        return None

    elapsed = exit_time - entry_time
    if elapsed <= 0:
        return None

    # Try trajectory fitting with multiple points
    hla_degrees: float
    distance_mm: float

    if positions and len(positions) >= 3:
        result = fit_trajectory(positions)
        if result is not None:
            x_arr, y_arr, t_arr = result
            # HLA from fitted line slope
            if len(x_arr) >= 2:
                coeffs = np.polyfit(x_arr, y_arr, 1)
                # Slope gives angle: negative slope = ball going up = positive HLA in golf terms
                hla_degrees = -math.degrees(math.atan(coeffs[0]))
                if flip:
                    hla_degrees = -hla_degrees

            # Distance from first to last fitted point
            dx = float(x_arr[-1] - x_arr[0])
            dy = float(y_arr[-1] - y_arr[0])
            distance_px = math.sqrt(dx * dx + dy * dy)
            distance_mm = distance_px / px_mm_ratio
        else:
            # Fallback to simple calculation
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance_px = math.sqrt(dx * dx + dy * dy)
            distance_mm = distance_px / px_mm_ratio
            hla_degrees = calculate_angle(start_pos, end_pos, flip=flip)
    else:
        # Simple two-point calculation (original algorithm)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance_px = math.sqrt(dx * dx + dy * dy)
        distance_mm = distance_px / px_mm_ratio
        hla_degrees = calculate_angle(start_pos, end_pos, flip=flip)

    # Convert mm distance and seconds to MPH
    # mm -> m -> km, then km/s -> km/h -> mph
    distance_km = distance_mm / 1_000_000
    speed_kmh = (distance_km / elapsed) * 3600
    speed_mph = speed_kmh * 0.621371

    return ShotData(
        speed_mph=round(speed_mph, 2),
        hla_degrees=round(hla_degrees, 2),
        distance_mm=round(distance_mm, 2),
        elapsed_seconds=round(elapsed, 4),
    )
