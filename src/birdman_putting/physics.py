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


def speed_from_visible_distance(
    distance_px: float,
    pixels_per_foot: float,
    stimpmeter: float = 10.0,
) -> float:
    """Compute ball speed from visible pixel distance (PutTrak-style).

    Instead of measuring speed from time (unreliable when the ball is only
    visible for 1-2 frames), compute the visible on-camera distance in feet
    and derive speed from stimpmeter physics.

    The visible distance is how far the ball traveled on camera from its
    rest position.  This correlates with launch speed because faster putts
    cover more camera pixels before exiting the frame.

    Args:
        distance_px: Pixel distance from ball rest to last-seen position.
        pixels_per_foot: Calibrated pixels-per-foot ratio.
        stimpmeter: Green speed rating.

    Returns:
        Ball speed in MPH.
    """
    if pixels_per_foot <= 0 or distance_px <= 0:
        return 0.0
    visible_ft = distance_px / pixels_per_foot
    return target_speed_for_distance(visible_ft, stimpmeter)


# Launch-velocity measurement tunables
_LAUNCH_MOVE_THRESHOLD_PX = 8    # Distance from rest to declare "moving"
_LAUNCH_WINDOW_FRAMES = 6        # Max frames after motion onset for the window
_LAUNCH_WINDOW_MAX_SECONDS = 0.15  # Cap window duration regardless of frame count
                                 # — at 60fps, 6 frames = 100ms (this allows it).
                                 # At 13fps, 6 frames = 460ms (this clips to 150ms).
                                 # Without this clip, low processing FPS averages
                                 # in deceleration and underestimates launch speed.
_LAUNCH_MIN_WINDOW_SECONDS = 0.005  # 5ms — guards only against frame-dup/jitter
_LAUNCH_MAX_MPH = 20.0           # Any reading above this is almost certainly
                                 # a dropped-frame artifact (~60ft putt)


def speed_from_launch_velocity(
    positions: list[tuple[int, int, float]],
    pixels_per_foot: float,
) -> tuple[float, dict]:
    """Compute launch speed from the first frames of ball motion.

    Unlike speed_from_visible_distance (which saturates once the ball
    exits the frame), this measures true velocity during the early
    motion window. A 30-ft putt and a 15-ft putt both cover the full
    visible frame — but the 30-ft putt covers it faster. Measuring
    pixels-per-second across the launch transition reads that velocity
    directly, independent of where the ball exits.

    Algorithm:
      1. Find motion onset: first frame > LAUNCH_MOVE_THRESHOLD_PX from rest.
      2. Anchor at the LAST REST frame immediately before motion onset
         (critical for hard putts that jump the gateway in one frame —
         we always get at least a 2-point velocity measurement).
      3. Include up to LAUNCH_WINDOW_FRAMES additional moving samples.
      4. Compute velocity = (pixels / pixels_per_foot) / dt.
      5. Reject readings above LAUNCH_MAX_MPH (dropped-frame artifacts).

    Args:
        positions: List of (x, y, timestamp) tuples, earliest first.
        pixels_per_foot: Calibrated pixels-per-foot ratio.

    Returns:
        (speed_mph, debug_info). speed_mph is 0.0 if the measurement is
        unreliable. debug_info is a dict with 'num_pos', 'num_moving',
        'window_px', 'window_dt' so callers can diagnose.
    """
    debug: dict = {
        "num_pos": len(positions) if positions else 0,
        "num_moving": 0,
        "window_px": 0.0,
        "window_dt": 0.0,
        "reason": "",
    }

    if pixels_per_foot <= 0:
        debug["reason"] = "ppf<=0"
        return 0.0, debug
    if not positions or len(positions) < 2:
        debug["reason"] = "need 2+ positions"
        return 0.0, debug

    # Find motion onset: first frame > threshold from rest (positions[0])
    rest_x, rest_y = float(positions[0][0]), float(positions[0][1])
    move_start = None
    for i, (px, py, _t) in enumerate(positions):
        if math.hypot(px - rest_x, py - rest_y) > _LAUNCH_MOVE_THRESHOLD_PX:
            move_start = i
            break

    if move_start is None:
        debug["reason"] = "no motion detected"
        return 0.0, debug

    debug["num_moving"] = len(positions) - move_start

    # Window selection.  Prefer motion-only frames — they give the
    # cleanest launch velocity reading.  Only fall back to including the
    # rest frame as anchor when we don't have 2 motion frames (rare hard
    # putts that jump the gateway in a single detection).
    #
    # Why the rest frame is bad as a normal anchor: positions[0]'s
    # timestamp comes from the most recent re-start, which can be up to
    # 0.5 s before actual motion begins.  Furthermore, ball motion
    # through the start zone is not appended (only frames *outside* the
    # start zone are), so positions[1] is already 0.1-0.2 s after
    # motion onset.  Using the rest frame as anchor adds that pre-motion
    # gap to window_dt without contributing to dx → velocity underread.
    motion_window = positions[move_start:move_start + _LAUNCH_WINDOW_FRAMES]
    if len(motion_window) >= 2:
        window = list(motion_window)
    else:
        # Sparse detection: include rest as 2nd point so we have a
        # velocity measurement at all.
        anchor_idx = max(0, move_start - 1)
        window = list(positions[anchor_idx:anchor_idx + _LAUNCH_WINDOW_FRAMES + 1])

    if len(window) < 2:
        debug["reason"] = "window too small"
        return 0.0, debug

    # Time-cap the window so low processing FPS doesn't dilute the
    # launch reading with deceleration. Trim trailing frames whose
    # timestamp exceeds anchor_t + LAUNCH_WINDOW_MAX_SECONDS, but
    # always keep at least 2 points for a velocity calculation.
    anchor_t = window[0][2]
    capped: list[tuple[int, int, float]] = [window[0]]
    for entry in window[1:]:
        if entry[2] - anchor_t <= _LAUNCH_WINDOW_MAX_SECONDS:
            capped.append(entry)
        else:
            # Always keep one frame past the cap so we have 2 points
            # even if the cap fires immediately.
            if len(capped) < 2:
                capped.append(entry)
            break
    window = capped

    x0, y0, t0 = window[0]
    x1, y1, t1 = window[-1]
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    dt = float(t1 - t0)

    debug["window_px"] = math.hypot(dx, dy)
    debug["window_dt"] = dt

    if dt < _LAUNCH_MIN_WINDOW_SECONDS:
        debug["reason"] = "dt too small"
        return 0.0, debug

    distance_px = math.hypot(dx, dy)
    if distance_px <= 0:
        debug["reason"] = "no pixel travel"
        return 0.0, debug

    launch_ft_per_sec = (distance_px / pixels_per_foot) / dt
    mph = launch_ft_per_sec / _MPH_TO_FPS

    if mph > _LAUNCH_MAX_MPH:
        # Likely a dropped-frame artifact: big dx with small dt because
        # intermediate detections were lost during motion blur.
        debug["reason"] = f"above {_LAUNCH_MAX_MPH} MPH cap"
        return 0.0, debug

    debug["reason"] = "ok"
    return mph, debug


# Trajectory-fit launch-velocity tunables
_FIT_MOTION_THRESHOLD_PX = 5.0   # px from rest to count as "moving"
_FIT_MIN_FRAMES = 4              # need at least this many motion samples
_FIT_MIN_TAU_SECONDS = 0.05      # smallest time-span over which to fit
_FIT_MAX_MPH = 25.0              # sanity cap (above any real putt)

# Outlier-rejection tunables for the speed fit.
_FIT_RESIDUAL_SIGMA = 2.0        # reject samples beyond this many σ of residual
# Floor on the residual magnitude (ft) below which NO sample is rejected.
# This guarantees the rejection is a strict no-op on clean trajectories
# (sub-foot, unbiased residuals from pixel quantization never trip it) while
# still catching a noise re-lock that lands feet away from the fitted curve.
_FIT_RESIDUAL_FLOOR_FT = 0.5
_FIT_REJECT_ITERS = 2            # refit passes after the initial fit
# If more than this fraction of motion samples are rejected, the trajectory
# is too corrupted to trust → return the low/zero-confidence signal.
_FIT_MAX_REJECT_FRACTION = 0.30
_FIT_MIN_FIT_SAMPLES = 3         # need at least this many survivors to fit


def pixel_x_to_feet(
    x: float,
    markers: list[float] | None,
    fallback_ppf: float,
) -> float:
    """Convert a pixel X-coordinate to feet using per-x calibration.

    When `markers` is a list of N pixel X-positions at known 1-foot
    spacing (markers[0]=0ft, markers[1]=1ft, ..., markers[N-1]=(N-1)ft),
    this returns the position in feet along the putt line via
    piecewise-linear interpolation.  Outside the calibrated range, the
    nearest segment's local ppf is used to extrapolate.

    With markers=None or len<2, falls back to a flat `fallback_ppf`
    scale.  Returns ft = x / fallback_ppf in that case.

    This corrects for fisheye / off-axis distortion where pixels-per-foot
    varies across the frame.  Greg's Kiyo Pro setup, for example, has
    ~57 px/ft near the start zone but only ~46 px/ft at the right edge.

    Robustness to a *missing* marker: the naive "array index == foot
    count" assumption breaks if an interior marker is absent (e.g. it fell
    in the start zone and was filtered).  Instead of trusting the index,
    we infer each marker's true foot number from its cumulative spacing
    relative to the robust median gap (``_marker_foot_indices``).  For
    genuinely uniform markers — and for smooth fisheye gradients — this
    yields exactly [0, 1, …, N-1], so behavior is unchanged.  When a ~2x
    gap is present the foot indices step (…, 2, 4, …), so positions past
    the gap read their true feet rather than coming up short.
    """
    if markers and len(markers) >= 2:
        m_sorted = sorted(markers)
        feet = _marker_foot_indices(m_sorted)
        if x <= m_sorted[0]:
            # Before the first marker — extrapolate using the leftmost
            # segment's per-foot pixel rate.
            span_ft = feet[1] - feet[0]
            local_ppf = (m_sorted[1] - m_sorted[0]) / span_ft if span_ft > 0 else 0.0
            if local_ppf > 0:
                return feet[0] - (m_sorted[0] - x) / local_ppf
            return float(feet[0])
        if x >= m_sorted[-1]:
            # After the last marker — extrapolate with the rightmost segment.
            span_ft = feet[-1] - feet[-2]
            local_ppf = (m_sorted[-1] - m_sorted[-2]) / span_ft if span_ft > 0 else 0.0
            if local_ppf > 0:
                return feet[-1] + (x - m_sorted[-1]) / local_ppf
            return float(feet[-1])
        # Within range — find the bracketing segment.
        for i in range(len(m_sorted) - 1):
            if m_sorted[i] <= x <= m_sorted[i + 1]:
                seg_px = m_sorted[i + 1] - m_sorted[i]
                seg_ft = feet[i + 1] - feet[i]
                if seg_px > 0 and seg_ft > 0:
                    # Interpolate in feet across the (possibly multi-foot)
                    # segment so a missing-marker gap spans its true feet.
                    return feet[i] + (x - m_sorted[i]) / (seg_px / seg_ft)
                return float(feet[i])
    # No markers — flat ppf scale
    if fallback_ppf > 0:
        return x / fallback_ppf
    return 0.0


def _marker_foot_indices(m_sorted: list[float]) -> list[float]:
    """Infer the true foot number of each sorted marker.

    Returns a list ``feet`` where ``feet[i]`` is how many feet marker
    ``i`` sits from the first marker.  Normally this is simply the array
    index ``i``; but if an interior marker is missing, the gap is ~2x the
    typical spacing and the index undercounts the feet past the gap.

    We compute it from cumulative pixel distance divided by the robust
    median single-marker spacing, rounded to the nearest integer foot and
    forced strictly increasing.  For uniform spacing (and smooth fisheye
    gradients) the rounding reproduces [0, 1, …, N-1] exactly, so callers
    see no change.
    """
    n = len(m_sorted)
    if n < 2:
        return [float(i) for i in range(n)]

    diffs = [m_sorted[i + 1] - m_sorted[i] for i in range(n - 1)]

    # Robust median spacing: exclude obvious missing-marker gaps (>1.5x the
    # 25th-percentile gap) so the median reflects a true 1-ft spacing.
    sorted_diffs = sorted(diffs)
    p25 = sorted_diffs[len(sorted_diffs) // 4]
    plausible = [d for d in diffs if d <= p25 * 1.5] if p25 > 0 else diffs
    base = plausible if plausible else diffs
    median = float(np.median(base))
    if median <= 0:
        return [float(i) for i in range(n)]

    feet: list[float] = [0.0]
    cum = 0.0
    for d in diffs:
        cum += d
        # Number of foot-intervals this gap represents (≈1 for normal gaps,
        # ≈2 for a missing-marker gap).  At least 1 so feet strictly rise.
        step = max(1, int(round(d / median)))
        feet.append(feet[-1] + step)

    # Sanity: if the inferred span is wildly larger than the marker count
    # (median misestimated), fall back to plain indices to avoid garbage.
    if feet[-1] > 4 * (n - 1):
        return [float(i) for i in range(n)]
    return feet


def speed_from_trajectory_fit(
    positions: list[tuple[int, int, float]],
    pixels_per_foot: float,
    stimpmeter: float = 11.0,
    calibration_markers: list[float] | None = None,
) -> tuple[float, dict]:
    """Estimate launch velocity by fitting all motion samples to the
    decelerating trajectory model.

    Model: travel(τ) = v0·τ − ½·a·τ²

    where τ is time since the first observed motion frame, travel is
    Euclidean distance from that first motion position (in feet), and
    `a = v_ramp² / (2·stimp)` is the constant deceleration on a level
    green at the configured stimp.

    When `calibration_markers` is provided (a list of pixel X-positions
    at known 1-ft spacing along the putt line), travel is computed via
    piecewise-linear interpolation between markers — this corrects for
    fisheye / off-axis distortion where pixels-per-foot varies across
    the frame.  Without markers, falls back to flat `pixels_per_foot`
    scaling.

    Reformulation as a linear least-squares problem in v0:
        y_i := travel_i + ½·a·τ_i²
        y_i = v0 · τ_i  (best-fit v0 = Σ(y·τ) / Σ(τ²))

    Returns:
        (speed_mph, debug_info).  speed_mph is 0.0 if the fit fails.
    """
    debug: dict = {
        "n_motion": 0,
        "tau_max": 0.0,
        "travel_max_ft": 0.0,
        "uses_markers": bool(calibration_markers and len(calibration_markers) >= 2),
        "reason": "",
    }

    use_markers = bool(calibration_markers and len(calibration_markers) >= 2)
    if not use_markers and pixels_per_foot <= 0:
        debug["reason"] = "ppf<=0 and no markers"
        return 0.0, debug
    if not positions or len(positions) < 2:
        debug["reason"] = "need 2+ positions"
        return 0.0, debug

    rest_x, rest_y = float(positions[0][0]), float(positions[0][1])
    motion_start = None
    for i, (px, py, _t) in enumerate(positions):
        if math.hypot(px - rest_x, py - rest_y) > _FIT_MOTION_THRESHOLD_PX:
            motion_start = i
            break

    if motion_start is None:
        debug["reason"] = "no motion detected"
        return 0.0, debug

    motion = list(positions[motion_start:])
    if len(motion) < _FIT_MIN_FRAMES:
        debug["reason"] = f"only {len(motion)} motion frames (need {_FIT_MIN_FRAMES}+)"
        return 0.0, debug

    debug["n_motion"] = len(motion)
    x0, y0, t0 = float(motion[0][0]), float(motion[0][1]), motion[0][2]

    a_fps2 = _STIMP_RAMP_SPEED_FPS ** 2 / (2.0 * stimpmeter)

    # Anchor X position in feet (ground plane).  For per-x calibration
    # we use the markers; otherwise just zero (anchor everything to
    # the first motion frame).
    if use_markers:
        x0_ft = pixel_x_to_feet(x0, calibration_markers, pixels_per_foot)
    else:
        x0_ft = 0.0  # not used in flat-ppf branch (we compute Euclidean directly)

    # ---- Build (tau, travel_ft) samples ----------------------------------
    # Travel is measured forward from the first motion frame.  With per-x
    # markers we take the *signed* travel along the putt line (a putt only
    # advances through the gateway, so a noise re-lock *behind* the anchor
    # reads negative and produces a large fit residual → rejected below —
    # rather than abs() turning it into bogus positive travel).  Without
    # markers we use Euclidean pixel distance, exactly as before.  On a
    # clean forward putt signed travel == abs travel, so this is a no-op.
    samples: list[tuple[float, float]] = []  # (tau, travel_ft)
    last_tau = 0.0
    travel_max_ft = 0.0
    for px, py, t in motion:
        tau = float(t - t0)
        if tau < 0:
            continue
        last_tau = tau

        if use_markers:
            x_ft = pixel_x_to_feet(float(px), calibration_markers, pixels_per_foot)
            travel_ft = x_ft - x0_ft  # signed forward travel (Y ignored)
        else:
            travel_px = math.hypot(float(px) - x0, float(py) - y0)
            travel_ft = travel_px / pixels_per_foot

        if travel_ft > travel_max_ft:
            travel_max_ft = travel_ft

        samples.append((tau, travel_ft))

    debug["tau_max"] = last_tau
    debug["travel_max_ft"] = travel_max_ft

    if last_tau < _FIT_MIN_TAU_SECONDS:
        debug["reason"] = "tau range too small"
        return 0.0, debug

    n_total = len(samples)

    # ---- Iteratively-reweighted v0 fit -----------------------------------
    # y_i = travel_i + ½·a·τ_i²;  fit y = v0·τ  (v0 = Σ(y·τ) / Σ(τ²)).
    # After the initial pass, drop samples whose residual to the fitted
    # curve exceeds max(2σ, floor) and refit.  The fixed floor makes this a
    # guaranteed no-op on clean data (tiny sub-foot residuals never exceed
    # it), so existing accuracy tests are unaffected.
    def _fit_v0(pts: list[tuple[float, float]]) -> float | None:
        s_ytau = 0.0
        s_tausq = 0.0
        for tau, travel in pts:
            y = travel + 0.5 * a_fps2 * tau * tau
            s_ytau += y * tau
            s_tausq += tau * tau
        if s_tausq <= 0:
            return None
        return s_ytau / s_tausq

    kept = list(samples)
    v0_fps = _fit_v0(kept)
    if v0_fps is None:
        debug["reason"] = "zero tau²"
        return 0.0, debug

    for _ in range(_FIT_REJECT_ITERS):
        if len(kept) <= _FIT_MIN_FIT_SAMPLES:
            break
        residuals = [
            (travel + 0.5 * a_fps2 * tau * tau) - v0_fps * tau
            for tau, travel in kept
        ]
        sigma = float(np.std(residuals))
        threshold = max(_FIT_RESIDUAL_SIGMA * sigma, _FIT_RESIDUAL_FLOOR_FT)
        survivors = [
            pt for pt, r in zip(kept, residuals, strict=True) if abs(r) <= threshold
        ]
        # Stop once nothing new is rejected, or rejecting more would drop
        # below the minimum fit support.
        if len(survivors) == len(kept) or len(survivors) < _FIT_MIN_FIT_SAMPLES:
            break
        kept = survivors
        new_v0 = _fit_v0(kept)
        if new_v0 is None:
            break
        v0_fps = new_v0

    n_rejected = n_total - len(kept)
    debug["n_rejected"] = n_rejected
    debug["n_used"] = len(kept)

    # If we threw away too much of the trajectory, the fit is untrustworthy.
    if n_total > 0 and n_rejected > n_total * _FIT_MAX_REJECT_FRACTION:
        debug["reason"] = "too many outliers rejected"
        return 0.0, debug
    if len(kept) < _FIT_MIN_FIT_SAMPLES:
        debug["reason"] = "too few samples after rejection"
        return 0.0, debug

    v0_mph = v0_fps / _MPH_TO_FPS

    if v0_mph > _FIT_MAX_MPH:
        debug["reason"] = f"above {_FIT_MAX_MPH} MPH cap"
        return 0.0, debug
    if v0_mph <= 0:
        debug["reason"] = "non-positive velocity"
        return 0.0, debug

    debug["reason"] = "ok"
    return v0_mph, debug


def pixel_to_mm_ratio(ball_radius_px: int) -> float:
    """Calculate pixels-per-mm ratio from detected ball radius.

    The ratio is: detected_radius_pixels / known_ball_radius_mm.
    To convert pixel distances to mm, divide by this ratio.
    """
    if ball_radius_px <= 0:
        return 0.0
    return ball_radius_px / GOLF_BALL_RADIUS_MM


# 304.8 mm per foot / 21.335 mm per ball-radius ≈ 14.286 ft-scale per radius-px.
# i.e. ppf_from_radius(r_px) = r_px × 14.286.
_MM_PER_FOOT = 304.8
_PPF_PER_RADIUS_PX = _MM_PER_FOOT / GOLF_BALL_RADIUS_MM  # ≈ 14.286


def ppf_from_ball_radius(ball_radius_px: float) -> float:
    """Derive pixels_per_foot from the detected ball radius.

    Uses the known 21.335mm golf ball radius as an in-frame reference
    object. For a perpendicular camera, this yields the true ground-plane
    scale directly. For a tilted camera, it is off by cos(tilt_angle) —
    a small error for shallow tilts (cos(10°) = 0.98, cos(20°) = 0.94).

    Critically, this is independent of the user's putting distance
    estimates and independent of the shot-completion pixel saturation
    that breaks the roll-based Dist Cal wizard.

    Args:
        ball_radius_px: Detected ball radius in pixels.

    Returns:
        pixels_per_foot, or 0.0 if input is invalid.
    """
    if ball_radius_px <= 0:
        return 0.0
    return ball_radius_px * _PPF_PER_RADIUS_PX


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
