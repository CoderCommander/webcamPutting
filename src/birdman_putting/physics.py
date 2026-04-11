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


# Stimpmeter ramp launch speed in ft/s (USGA standard)
_STIMP_RAMP_SPEED_FPS = 6.08
_MPH_TO_FPS = 5280.0 / 3600.0  # 1.4667


def estimate_putt_distance_feet(speed_mph: float, stimpmeter: float = 10.0) -> float:
    """Estimate putt distance from ball speed using stimpmeter-based deceleration.

    Physics: deceleration a = v_ramp² / (2 × stimp), then d = v² / (2a).
    Simplifies to: d = v² × stimp / v_ramp².

    Args:
        speed_mph: Ball speed in MPH.
        stimpmeter: Green speed rating (typical range 7-13).

    Returns:
        Estimated roll distance in feet.
    """
    v_fps = speed_mph * _MPH_TO_FPS
    return v_fps ** 2 * stimpmeter / _STIMP_RAMP_SPEED_FPS ** 2


def target_speed_for_distance(distance_ft: float, stimpmeter: float = 10.0) -> float:
    """Compute the ball speed (MPH) required to roll a given distance.

    Inverse of estimate_putt_distance_feet.

    Args:
        distance_ft: Target roll distance in feet.
        stimpmeter: Green speed rating (typical range 7-13).

    Returns:
        Required ball speed in MPH.
    """
    v_fps = math.sqrt(distance_ft * _STIMP_RAMP_SPEED_FPS ** 2 / stimpmeter)
    return v_fps / _MPH_TO_FPS


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
    reverse_x: bool = False,
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
        reverse_x: Mirror x-coordinates (for right-to-left ball roll).

    Returns:
        ShotData with speed and HLA, or None if calculation fails.
    """
    # Mirror x-coordinates for RtL so all existing math works as LtR
    if reverse_x:
        start_pos = (-start_pos[0], start_pos[1])
        end_pos = (-end_pos[0], end_pos[1])
        if positions:
            positions = [(-x, y, t) for x, y, t in positions]
    if px_mm_ratio <= 0:
        return None

    elapsed = exit_time - entry_time
    if elapsed <= 0:
        return None

    # Always compute start→end as the baseline measurement.
    # This is the most reliable for fast putts that skip the gateway
    # (only 2-3 position samples).
    dx_base = end_pos[0] - start_pos[0]
    dy_base = end_pos[1] - start_pos[1]
    distance_px = math.sqrt(dx_base * dx_base + dy_base * dy_base)
    distance_mm = distance_px / px_mm_ratio
    hla_degrees = calculate_angle(start_pos, end_pos, flip=flip)

    # Try movement-onset refinement when we have enough position samples.
    # With many samples (10+), the movement-onset window gives better
    # timing than the raw entry/exit times.  With few samples (fast putts
    # that jump past the gateway), the baseline above is more reliable.
    if positions and len(positions) >= 6:
        rest_x, rest_y = float(positions[0][0]), float(positions[0][1])
        _MOVE_THRESHOLD_PX = 8

        move_idx = 0
        for i, (px, py, _pt) in enumerate(positions):
            if math.sqrt((px - rest_x) ** 2 + (py - rest_y) ** 2) > _MOVE_THRESHOLD_PX:
                move_idx = i
                break

        moving = positions[move_idx:]
        if len(moving) >= 4:
            result = fit_trajectory(moving)
            if result is not None:
                x_arr, y_arr, t_arr = result
                if len(x_arr) >= 2:
                    coeffs = np.polyfit(x_arr, y_arr, 1)
                    hla_degrees = -math.degrees(math.atan(coeffs[0]))
                    if flip:
                        hla_degrees = -hla_degrees

                    mdx = float(x_arr[-1] - x_arr[0])
                    mdy = float(y_arr[-1] - y_arr[0])
                    move_dist = math.sqrt(mdx * mdx + mdy * mdy)
                    move_elapsed = float(t_arr[-1] - t_arr[0])

                    # Only use movement-onset if it gives a reasonable result
                    # (distance > 50% of baseline, elapsed > 20ms)
                    if move_dist > distance_px * 0.5 and move_elapsed > 0.02:
                        distance_mm = move_dist / px_mm_ratio
                        elapsed = move_elapsed

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
