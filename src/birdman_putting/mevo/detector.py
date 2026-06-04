"""Mevo shot detection via screenshot comparison + OCR."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from birdman_putting.config import MevoSettings
from birdman_putting.mevo.ocr import ROI, MevoOCR
from birdman_putting.mevo.screenshot import WindowCapture

logger = logging.getLogger(__name__)

# Number of consecutive agreeing OCR reads required before a shot is emitted.
# The Mevo display animates the digits into place when a shot lands; the first
# changed frame catches it mid-animation. Requiring K stable reads ensures we
# emit the SETTLED values, not a transitional misread. Module-level so the app
# / tests can reference it.
STABILITY_K: int = 3

# Max polls to spend confirming a single display change before giving up. If
# the display never settles to K agreeing reads within this many polls, we
# abandon the candidate (return None, reset) rather than latch garbage.
SETTLE_MAX_POLLS: int = 12

# Per-field tolerance for "agrees with previous read" during stability gating.
# Generous enough to absorb +/-1 OCR jitter on the last digit, tight enough
# that a real value change resets the streak.
_STABILITY_TOL: dict[str, float] = {
    "ball_speed": 0.5,
    "launch_angle": 0.5,
    "launch_direction": 0.5,
    "spin_rate": 75.0,
    "spin_axis": 1.0,
    "club_speed": 0.5,
    "carry_distance": 2.0,
    "total_distance": 2.0,
}
_STABILITY_TOL_DEFAULT: float = 0.5

# Valid ranges for shot metrics. Tightened from the original wide bands so an
# implausible shot is rejected rather than forwarded. In particular the VLA
# lower bound and the spin lower bound were widened-too-far before (VLA down to
# -15, spin down to 0), which let tonight's misread (VLA=26.5, spin=2013) pass.
_VALID_RANGES: dict[str, tuple[float, float]] = {
    "ball_speed": (5.0, 250.0),        # mph
    "launch_angle": (-10.0, 60.0),     # degrees VLA (negative only for putts/chunks)
    "launch_direction": (-45.0, 45.0),  # degrees HLA
    "spin_rate": (0.0, 15000.0),       # rpm
    "spin_axis": (-90.0, 90.0),        # degrees
    "club_speed": (0.0, 160.0),        # mph
    "smash_factor": (0.5, 1.7),
    "carry_distance": (0.0, 400.0),    # yards
    "total_distance": (0.0, 420.0),    # yards
    "apex_height": (0.0, 250.0),       # yards/feet
    "flight_time": (0.0, 12.0),        # seconds
    "descent_angle": (0.0, 90.0),      # degrees
    "curve": (-100.0, 100.0),          # yards
    "roll_distance": (0.0, 200.0),     # yards
    "aoa": (-20.0, 20.0),             # degrees (angle of attack)
    "club_path": (-30.0, 30.0),       # degrees
    "dynamic_loft": (0.0, 70.0),      # degrees
    "face_to_target": (-30.0, 30.0),  # degrees
    "lateral_impact": (-2.0, 2.0),    # inches
    "vertical_impact": (-2.0, 2.0),   # inches
}

# --- Cross-field coherence (club-agnostic) ------------------------------
# A shot that is simultaneously low-speed, very low VLA, AND very low spin is
# physically incoherent for a STRUCK shot: a slow ball with almost no launch
# and almost no spin is either a putt (which has ~0 VLA AND ~0 spin together,
# and is handled separately) or — as happened tonight — a wedge chip misread
# mid-animation. We reject the "in-between" case: enough speed/VLA/spin to not
# be a putt, but too little to be a real lofted strike.
_COH_SPEED_MAX: float = 60.0     # mph — only scrutinise sub-60 mph shots
_COH_VLA_MIN: float = 35.0       # deg — a struck sub-60mph shot launching <35
_COH_SPIN_MIN: float = 3000.0    # rpm — ...and spinning <3000 is incoherent
_COH_PUTT_VLA_MAX: float = 8.0   # deg — below this AND
_COH_PUTT_SPIN_MAX: float = 600.0  # rpm — below this → treat as a putt (OK)

# --- Club-conditioned floors --------------------------------------------
# Lofted clubs cannot produce low launch + low spin. When the app supplies the
# selected club (or its loft), we impose a floor on VLA and spin. Keyed by the
# GSPro club codes the app already uses (see app.py).
_WEDGE_CODES: frozenset[str] = frozenset({"LW", "SW", "AW", "GW", "PW"})
# (min_spin_rpm, min_vla_deg) floors per club class.
_CLUB_FLOORS: dict[str, tuple[float, float]] = {
    "LW": (3000.0, 30.0),
    "SW": (3000.0, 28.0),
    "AW": (2800.0, 24.0),
    "GW": (2800.0, 22.0),
    "PW": (2500.0, 18.0),
}
_CLUB_FLOOR_DEFAULT: tuple[float, float] = (0.0, -90.0)  # no floor (e.g. driver)


def club_min_spin_vla(club: str | None) -> tuple[float, float]:
    """Return (min_spin_rpm, min_vla_deg) floors for a club code.

    Non-wedge / unknown clubs impose no lofted-shot floor. Used by the
    club-conditioned plausibility gate.
    """
    if not club:
        return _CLUB_FLOOR_DEFAULT
    return _CLUB_FLOORS.get(club.strip().upper(), _CLUB_FLOOR_DEFAULT)


def _loft_floor(loft: float) -> tuple[float, float]:
    """Derive (min_spin_rpm, min_vla_deg) floors from a club loft in degrees.

    A higher-lofted club must launch higher and spin more. Conservative,
    linear-ish floors calibrated so a 60 deg wedge requires VLA >= ~35 and
    spin >= ~5000, while a low-loft iron imposes little.
    """
    if loft <= 30.0:
        return (0.0, -90.0)
    # Scale floors with loft above 30 degrees.
    min_vla = max(0.0, (loft - 30.0) * 0.85)       # 60 deg → ~25.5 deg floor
    min_spin = max(0.0, (loft - 30.0) * 130.0)     # 60 deg → ~3900 rpm floor
    return (min_spin, min_vla)


def _coherence_ok(metrics: dict[str, float | None]) -> bool:
    """Club-agnostic cross-field sanity check.

    Returns False for the incoherent low-speed / very-low-VLA / very-low-spin
    combination (tonight's failure). Putts (near-zero VLA AND near-zero spin)
    are explicitly allowed.
    """
    speed = metrics.get("ball_speed")
    vla = metrics.get("launch_angle")
    spin = metrics.get("spin_rate")
    if speed is None or vla is None:
        return True  # not enough info to judge — leave to range checks
    if speed >= _COH_SPEED_MAX:
        return True  # fast shots are out of scope for this check

    # Putt exemption: essentially no launch AND essentially no spin.
    if spin is not None and vla <= _COH_PUTT_VLA_MAX and spin <= _COH_PUTT_SPIN_MAX:
        return True

    # Incoherent struck shot: meaningful speed but launch < floor and
    # spin < floor (a real lofted strike has high spin even when slow).
    low_vla = vla < _COH_VLA_MIN
    low_spin = spin is None or spin < _COH_SPIN_MIN
    return not (low_vla and low_spin)


def _club_gate_ok(
    metrics: dict[str, float | None],
    expected_club: str | None,
    loft: float | None,
) -> bool:
    """Club-conditioned plausibility floor on VLA and spin.

    When a club code or loft is supplied, a lofted club with too-low launch or
    too-low spin is rejected. Returns True when no club/loft is given.
    """
    if expected_club is None and loft is None:
        return True
    if loft is not None:
        min_spin, min_vla = _loft_floor(loft)
    else:
        min_spin, min_vla = club_min_spin_vla(expected_club)
    if min_spin <= 0.0 and min_vla <= -90.0:
        return True  # club imposes no floor

    vla = metrics.get("launch_angle")
    spin = metrics.get("spin_rate")
    if vla is not None and vla < min_vla:
        logger.warning(
            "Club gate (%s/loft=%s): VLA %.1f below floor %.1f — rejecting",
            expected_club, loft, vla, min_vla,
        )
        return False
    if spin is not None and spin < min_spin:
        logger.warning(
            "Club gate (%s/loft=%s): spin %.0f below floor %.0f — rejecting",
            expected_club, loft, spin, min_spin,
        )
        return False
    return True


@dataclass
class MevoShotData:
    """Full shot data from Mevo OCR."""

    # Core (required)
    ball_speed: float      # mph
    launch_angle: float    # degrees (VLA)
    launch_direction: float  # degrees (HLA)

    # Standard optional
    spin_rate: float = 0.0       # rpm (total spin)
    spin_axis: float = 0.0       # degrees
    club_speed: float = 0.0      # mph (0 if not available)

    # Premium optional metrics
    smash_factor: float = 0.0
    carry_distance: float = 0.0  # yards
    total_distance: float = 0.0  # yards
    apex_height: float = 0.0     # yards (or feet, depending on FS Golf setting)
    flight_time: float = 0.0     # seconds
    descent_angle: float = 0.0   # degrees
    curve: float = 0.0           # yards
    roll_distance: float = 0.0   # yards

    # Club data metrics
    aoa: float = 0.0             # degrees (angle of attack)
    club_path: float = 0.0       # degrees
    dynamic_loft: float = 0.0    # degrees
    face_to_target: float = 0.0  # degrees
    lateral_impact: float = 0.0  # inches
    vertical_impact: float = 0.0  # inches

    @property
    def back_spin(self) -> float:
        """Decompose total spin into back spin component."""
        return abs(self.spin_rate * math.cos(math.radians(self.spin_axis)))

    @property
    def side_spin(self) -> float:
        """Decompose total spin into side spin component."""
        return self.spin_rate * math.sin(math.radians(self.spin_axis))


def build_rois(roi_dict: dict[str, list[int]]) -> list[ROI]:
    """Convert config ROI dict to list of ROI objects.

    Args:
        roi_dict: Mapping of metric name to [x, y, width, height].
    """
    rois: list[ROI] = []
    for name, coords in roi_dict.items():
        if len(coords) != 4:
            logger.warning("ROI '%s' has %d coords (expected 4), skipping", name, len(coords))
            continue
        rois.append(ROI(name=name, x=coords[0], y=coords[1], width=coords[2], height=coords[3]))
    return rois


def _compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if a.shape != b.shape:
        return float("inf")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def _values_changed(
    prev: dict[str, float | None],
    curr: dict[str, float | None],
    tolerance: float = 0.1,
) -> bool:
    """Check if OCR values have changed beyond tolerance."""
    for key in curr:
        pv = prev.get(key)
        cv = curr.get(key)
        if pv is None and cv is not None:
            return True
        if pv is not None and cv is None:
            continue  # Don't trigger on lost readings
        if pv is not None and cv is not None and abs(pv - cv) > tolerance:
            return True
    return False


def _validate_metrics(
    metrics: dict[str, float | None],
    expected_club: str | None = None,
    loft: float | None = None,
) -> bool:
    """Check required metrics are present, in range, and plausible.

    Args:
        metrics: Parsed metric values.
        expected_club: Optional GSPro club code (e.g. "LW"). When supplied,
            club-conditioned floors on VLA/spin are applied — a lofted club
            with too-low launch or spin is rejected.
        loft: Optional club loft in degrees, an alternative to ``expected_club``
            for deriving the same floors.

    Returns:
        True only if every required metric is present and within range, all
        present optional metrics are in range, the cross-field coherence check
        passes, and (if a club/loft is given) the club gate passes.
    """
    required = ["ball_speed", "launch_angle", "launch_direction"]
    for key in required:
        value = metrics.get(key)
        if value is None:
            return False
        lo, hi = _VALID_RANGES.get(key, (float("-inf"), float("inf")))
        if not (lo <= value <= hi):
            logger.debug("Metric '%s' = %.2f out of range [%.1f, %.1f]", key, value, lo, hi)
            return False

    # Validate optional metrics if present
    for key in _VALID_RANGES:
        if key in required:
            continue
        value = metrics.get(key)
        if value is not None:
            lo, hi = _VALID_RANGES[key]
            if not (lo <= value <= hi):
                logger.debug("Metric '%s' = %.2f out of range", key, value)
                return False

    # Cross-field coherence (club-agnostic) — rejects tonight's signature.
    if not _coherence_ok(metrics):
        logger.warning(
            "Incoherent shot rejected: speed=%s VLA=%s spin=%s "
            "(low-speed + very-low-VLA + very-low-spin)",
            metrics.get("ball_speed"), metrics.get("launch_angle"),
            metrics.get("spin_rate"),
        )
        return False

    # Club-conditioned floors (only when a club / loft is supplied).
    return _club_gate_ok(metrics, expected_club, loft)


class MevoDetector:
    """Detects new Mevo shots via screenshot + OCR.

    Call ``poll()`` in a loop; it returns ``MevoShotData`` when a new shot
    is detected, or ``None`` otherwise.
    """

    def __init__(
        self,
        settings: MevoSettings,
        ocr: MevoOCR,
        capture: WindowCapture,
    ) -> None:
        self._settings = settings
        self._ocr = ocr
        self._capture = capture
        self._prev_frame: np.ndarray | None = None
        self._prev_crops: dict[str, np.ndarray] | None = None
        self._prev_metrics: dict[str, float | None] = {}
        self._baseline_captured: bool = False

        # --- Stability gating state ---
        # Once a display change is detected we enter a "confirming" window and
        # OCR every poll (regardless of MSE) until STABILITY_K consecutive
        # reads agree, or we exhaust SETTLE_MAX_POLLS.
        self._confirming: bool = False
        self._settle_polls: int = 0
        # Buffer of recent agreeing reads (the candidate shot, growing toward K).
        self._stable_buffer: list[dict[str, float | None]] = []

        # Optional club / loft for the plausibility gate. Set by the app via
        # set_expected_club() / set_expected_loft(), or passed to poll().
        self._expected_club: str | None = None
        self._expected_loft: float | None = None

    def set_expected_club(self, club: str | None) -> None:
        """Set the GSPro club code used by the plausibility gate.

        The app calls this on club change (code 201) so lofted-club shots with
        implausibly low VLA/spin are rejected. Pass None to clear.
        """
        self._expected_club = club

    def set_expected_loft(self, loft: float | None) -> None:
        """Set the club loft (degrees) used by the plausibility gate.

        Alternative to set_expected_club() when only the loft is known.
        """
        self._expected_loft = loft

    @property
    def _configured_names(self) -> set[str]:
        """Names of metrics that have a configured ROI (i.e. expected to OCR)."""
        return {roi.name for roi in self._ocr._rois}

    def _read_is_candidate(self, metrics: dict[str, float | None]) -> bool:
        """A read is a viable shot candidate only if the required metrics are
        present AND every CONFIGURED ROI produced a value.

        This is the no-fabricated-zeros guard: a configured-but-failed field
        (None) means OCR failed on a field the user calibrated, so the read is
        not trustworthy — never silently substitute 0.0.
        """
        for key in ("ball_speed", "launch_angle", "launch_direction"):
            if metrics.get(key) is None:
                return False
        for name in self._configured_names:
            if metrics.get(name) is None:
                logger.debug(
                    "Configured field '%s' failed OCR — holding shot "
                    "(no fabricated 0.0)", name,
                )
                return False
        return True

    @staticmethod
    def _reads_agree(
        a: dict[str, float | None], b: dict[str, float | None],
    ) -> bool:
        """Whether two reads agree within per-field stability tolerance."""
        keys = set(a) | set(b)
        for key in keys:
            av = a.get(key)
            bv = b.get(key)
            if (av is None) != (bv is None):
                return False
            if av is not None and bv is not None:
                tol = _STABILITY_TOL.get(key, _STABILITY_TOL_DEFAULT)
                if abs(av - bv) > tol:
                    return False
        return True

    def _reset_confirm(self) -> None:
        """Exit the confirming window and clear the candidate buffer."""
        self._confirming = False
        self._settle_polls = 0
        self._stable_buffer = []

    def _compute_roi_mse(self, frame: np.ndarray) -> float:
        """Compute MSE over ROI regions only (faster than full-frame)."""
        if self._prev_crops is None:
            return float("inf")
        total_mse = 0.0
        total_pixels = 0
        for roi in self._ocr._rois:
            crop = MevoOCR._crop_roi(frame, roi)
            if crop is None:
                continue
            prev = self._prev_crops.get(roi.name)
            if prev is None or crop.shape != prev.shape:
                return float("inf")
            diff = crop.astype(np.float64) - prev.astype(np.float64)
            total_mse += float(np.sum(diff * diff))
            total_pixels += crop.size
        return total_mse / total_pixels if total_pixels > 0 else float("inf")

    def _store_crops(self, frame: np.ndarray) -> None:
        """Store cropped ROI regions for next comparison."""
        crops: dict[str, np.ndarray] = {}
        for roi in self._ocr._rois:
            crop = MevoOCR._crop_roi(frame, roi)
            if crop is not None:
                crops[roi.name] = crop.copy()
        self._prev_crops = crops

    def poll(
        self,
        expected_club: str | None = None,
        loft: float | None = None,
    ) -> MevoShotData | None:
        """Check for a new shot.

        A shot is emitted only after its full metric set has been STABLE across
        :data:`STABILITY_K` consecutive polls (so the settled display values are
        captured, not a mid-animation misread) AND it passes range, coherence,
        and (if a club/loft is supplied) club-conditioned plausibility checks.

        Args:
            expected_club: Optional GSPro club code (e.g. "LW"). Falls back to a
                value set via :meth:`set_expected_club`. Enables club-conditioned
                rejection of lofted-club shots with implausibly low VLA/spin.
            loft: Optional club loft in degrees (alternative to ``expected_club``).
                Falls back to :meth:`set_expected_loft`.

        Returns:
            MevoShotData when a new, stable, plausible shot is detected, else None.
        """
        club = expected_club if expected_club is not None else self._expected_club
        use_loft = loft if loft is not None else self._expected_loft

        frame = self._capture.capture()
        if frame is None:
            return None

        # Decide whether to OCR this poll. Outside the confirming window we
        # short-circuit on an unchanged display (MSE below threshold). Inside
        # the window we OCR every poll so a settled display (MSE ~ 0) is still
        # re-read until it accumulates K agreeing reads.
        changed = self._display_changed(frame)
        if not self._confirming and not changed:
            return None

        # Track the frame for the next MSE comparison whenever we OCR.
        if self._prev_crops is None:
            self._prev_frame = frame.copy()
        else:
            self._prev_frame = None  # ROI crops are sufficient — free memory
        self._store_crops(frame)

        metrics = self._ocr.read_metrics(frame)

        # First poll captures baseline — don't fire stale data as a shot.
        if not self._baseline_captured:
            self._baseline_captured = True
            if _validate_metrics(metrics):
                self._prev_metrics = metrics
                logger.info(
                    "Mevo baseline captured (stale): %.1f mph — waiting for new shot",
                    metrics.get("ball_speed", 0) or 0,
                )
            return None

        # A display change (re)opens / extends the confirming window.
        if changed and not self._confirming:
            self._confirming = True
            self._settle_polls = 0
            self._stable_buffer = []

        if not self._confirming:
            return None

        self._settle_polls += 1

        # A read with a missing required/configured field is not a viable
        # candidate — reset the streak (no fabricated zeros, no partial shot).
        if not self._read_is_candidate(metrics):
            self._stable_buffer = []
            if self._settle_polls >= SETTLE_MAX_POLLS:
                self._reset_confirm()
            return None

        # Stability streak: keep the buffer only while consecutive reads agree.
        if self._stable_buffer and self._reads_agree(self._stable_buffer[-1], metrics):
            self._stable_buffer.append(metrics)
        else:
            self._stable_buffer = [metrics]

        if len(self._stable_buffer) < STABILITY_K:
            if self._settle_polls >= SETTLE_MAX_POLLS:
                logger.warning(
                    "Mevo display never settled within %d polls — discarding "
                    "candidate (last read: %s)", SETTLE_MAX_POLLS, metrics,
                )
                self._reset_confirm()
            return None

        # We have K stable reads. This is the committed candidate.
        committed = self._stable_buffer[-1]
        self._reset_confirm()

        # Suppress if identical to the previously emitted shot.
        if self._prev_metrics and not _values_changed(self._prev_metrics, committed):
            return None

        # Final plausibility gate (range + coherence + club).
        if not _validate_metrics(committed, expected_club=club, loft=use_loft):
            logger.warning(
                "Stable shot rejected by plausibility gate "
                "(speed=%s VLA=%s HLA=%s spin=%s club=%s loft=%s) — not emitting. "
                "Raw reads: %s",
                committed.get("ball_speed"), committed.get("launch_angle"),
                committed.get("launch_direction"), committed.get("spin_rate"),
                club, use_loft, committed,
            )
            return None

        # Commit and emit. Latch ONLY a real, accepted shot as _prev_metrics.
        self._prev_metrics = committed
        shot = self._build_shot(committed)
        logger.info(
            "New Mevo shot: %.1f mph, VLA=%.1f, HLA=%.1f, Spin=%d",
            shot.ball_speed, shot.launch_angle, shot.launch_direction, shot.spin_rate,
        )
        return shot

    def _display_changed(self, frame: np.ndarray) -> bool:
        """Whether the display changed beyond the MSE threshold vs the last
        captured frame/crops."""
        if self._prev_crops is not None:
            mse = self._compute_roi_mse(frame)
        elif self._prev_frame is not None:
            mse = _compute_mse(frame, self._prev_frame)
        else:
            return True  # no baseline yet → treat as changed
        if mse < self._settings.mse_threshold:
            return False
        logger.debug("Display change detected (MSE=%.1f)", mse)
        return True

    def _build_shot(self, metrics: dict[str, float | None]) -> MevoShotData:
        """Build MevoShotData from a validated, stable metric set.

        Optional fields absent from the metric set (i.e. NOT configured) default
        to 0.0. A configured-but-failed field never reaches here — it is held
        upstream by :meth:`_read_is_candidate`.
        """
        return MevoShotData(
            ball_speed=metrics["ball_speed"],  # type: ignore[arg-type]
            launch_angle=metrics["launch_angle"],  # type: ignore[arg-type]
            launch_direction=metrics["launch_direction"],  # type: ignore[arg-type]
            spin_rate=metrics.get("spin_rate") or 0.0,
            spin_axis=metrics.get("spin_axis") or 0.0,
            club_speed=metrics.get("club_speed") or 0.0,
            smash_factor=metrics.get("smash_factor") or 0.0,
            carry_distance=metrics.get("carry_distance") or 0.0,
            total_distance=metrics.get("total_distance") or 0.0,
            apex_height=metrics.get("apex_height") or 0.0,
            flight_time=metrics.get("flight_time") or 0.0,
            descent_angle=metrics.get("descent_angle") or 0.0,
            curve=metrics.get("curve") or 0.0,
            roll_distance=metrics.get("roll_distance") or 0.0,
            aoa=metrics.get("aoa") or 0.0,
            club_path=metrics.get("club_path") or 0.0,
            dynamic_loft=metrics.get("dynamic_loft") or 0.0,
            face_to_target=metrics.get("face_to_target") or 0.0,
            lateral_impact=metrics.get("lateral_impact") or 0.0,
            vertical_impact=metrics.get("vertical_impact") or 0.0,
        )
