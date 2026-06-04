"""Tests for Mevo shot detector — all OCR and capture are mocked."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np

from birdman_putting.config import MevoSettings
from birdman_putting.mevo.detector import (
    STABILITY_K,
    MevoDetector,
    MevoShotData,
    _coherence_ok,
    _compute_mse,
    _validate_metrics,
    _values_changed,
    build_rois,
    club_min_spin_vla,
)
from birdman_putting.mevo.ocr import ROI


class TestMevoShotData:
    def test_back_spin_decomposition(self) -> None:
        shot = MevoShotData(
            ball_speed=120.0, launch_angle=12.0, launch_direction=1.5,
            spin_rate=3000.0, spin_axis=20.0, club_speed=95.0,
        )
        expected_back = abs(3000.0 * math.cos(math.radians(20.0)))
        expected_side = 3000.0 * math.sin(math.radians(20.0))
        assert abs(shot.back_spin - expected_back) < 0.01
        assert abs(shot.side_spin - expected_side) < 0.01

    def test_zero_spin_axis(self) -> None:
        shot = MevoShotData(
            ball_speed=100.0, launch_angle=10.0, launch_direction=0.0,
            spin_rate=2500.0, spin_axis=0.0, club_speed=0.0,
        )
        assert abs(shot.back_spin - 2500.0) < 0.01
        assert abs(shot.side_spin) < 0.01


class TestBuildRois:
    def test_valid_rois(self) -> None:
        roi_dict = {
            "ball_speed": [10, 20, 100, 30],
            "launch_angle": [10, 60, 100, 30],
        }
        rois = build_rois(roi_dict)
        assert len(rois) == 2
        assert rois[0].name == "ball_speed"
        assert rois[0].x == 10
        assert rois[0].width == 100

    def test_invalid_coords_skipped(self) -> None:
        roi_dict = {"bad": [1, 2, 3]}  # Only 3 coords
        rois = build_rois(roi_dict)
        assert len(rois) == 0


class TestComputeMSE:
    def test_identical_frames(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _compute_mse(frame, frame) == 0.0

    def test_different_frames(self) -> None:
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mse = _compute_mse(a, b)
        assert mse > 0

    def test_different_shapes(self) -> None:
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((50, 50, 3), dtype=np.uint8)
        assert _compute_mse(a, b) == float("inf")


class TestValuesChanged:
    def test_changed(self) -> None:
        prev = {"ball_speed": 100.0}
        curr = {"ball_speed": 120.0}
        assert _values_changed(prev, curr) is True

    def test_unchanged(self) -> None:
        prev = {"ball_speed": 100.0}
        curr = {"ball_speed": 100.05}
        assert _values_changed(prev, curr) is False

    def test_new_value_appears(self) -> None:
        prev = {"ball_speed": None}
        curr = {"ball_speed": 100.0}
        assert _values_changed(prev, curr) is True


class TestValidateMetrics:
    def test_valid(self) -> None:
        metrics = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is True

    def test_missing_required(self) -> None:
        metrics = {"ball_speed": 120.0, "launch_angle": 12.0}
        assert _validate_metrics(metrics) is False

    def test_out_of_range_speed(self) -> None:
        metrics = {
            "ball_speed": 999.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is False

    def test_none_required(self) -> None:
        metrics = {
            "ball_speed": None,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert _validate_metrics(metrics) is False


class TestMevoDetector:
    def _make_detector(
        self,
        mse_threshold: float = 100.0,
        roi_names: tuple[str, ...] = ("ball_speed", "launch_angle", "launch_direction"),
    ) -> tuple[MevoDetector, MagicMock, MagicMock]:
        settings = MevoSettings(enabled=True, mse_threshold=mse_threshold)
        mock_ocr = MagicMock()
        # The detector iterates ocr._rois for MSE / configured-field checks,
        # so the mock must expose a realistic ROI list.
        mock_ocr._rois = [
            ROI(name=n, x=0, y=0, width=10, height=10) for n in roi_names
        ]
        mock_capture = MagicMock()
        detector = MevoDetector(settings, mock_ocr, mock_capture)
        return detector, mock_ocr, mock_capture

    @staticmethod
    def _emit_stable(
        detector: MevoDetector,
        capture: MagicMock,
        ocr: MagicMock,
        frame: np.ndarray,
        metrics: dict[str, float | None],
        polls: int = STABILITY_K,
    ) -> MevoShotData | None:
        """Drive `polls` consecutive identical reads and return the result.

        Returns the first non-None shot emitted, else None. Used to simulate
        a display that has settled to stable values across STABILITY_K polls.
        """
        capture.capture.return_value = frame
        ocr.read_metrics.return_value = dict(metrics)
        result: MevoShotData | None = None
        for _ in range(polls):
            r = detector.poll()
            if r is not None:
                result = r
        return result

    def test_returns_none_when_no_frame(self) -> None:
        detector, _ocr, capture = self._make_detector()
        capture.capture.return_value = None
        assert detector.poll() is None

    def test_returns_none_when_frame_unchanged(self) -> None:
        detector, _ocr, capture = self._make_detector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        capture.capture.return_value = frame

        # First poll: sets prev_frame
        _ocr.read_metrics.return_value = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        detector.poll()

        # Second poll: same frame → MSE below threshold
        result = detector.poll()
        assert result is None

    # NOTE: the next two tests were rewritten from the original single-frame
    # behavior. The old detector emitted a shot on the FIRST changed frame,
    # which is exactly the production bug (it catches the display mid-
    # animation). The detector now requires STABILITY_K consecutive agreeing
    # reads before emitting, so these tests drive K stable polls via the
    # _emit_stable helper. The baseline-capture step is preserved.
    def test_detects_new_shot(self) -> None:
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        # First call: baseline capture — returns None (stale data suppressed)
        capture.capture.return_value = frame1
        ocr.read_metrics.return_value = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        baseline = detector.poll()
        assert baseline is None

        # New shot: drive K stable reads of the settled values.
        shot = self._emit_stable(
            detector, capture, ocr, frame2,
            {"ball_speed": 130.0, "launch_angle": 15.0, "launch_direction": -2.0},
        )
        assert shot is not None
        assert isinstance(shot, MevoShotData)
        assert shot.ball_speed == 130.0

        # Another, different stable shot.
        shot2 = self._emit_stable(
            detector, capture, ocr, frame3,
            {"ball_speed": 140.0, "launch_angle": 18.0, "launch_direction": 3.0},
        )
        assert shot2 is not None
        assert shot2.ball_speed == 140.0

    def test_is_confirming_tracks_settle_window(self) -> None:
        """is_confirming flags the settle window so the app loop polls FAST
        only while a shot's values are settling, then returns to the slow
        cadence. This is what cuts the Mevo full-swing latency."""
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Baseline poll — not confirming yet.
        capture.capture.return_value = frame1
        ocr.read_metrics.return_value = {
            "ball_speed": 120.0, "launch_angle": 12.0, "launch_direction": 1.5,
        }
        detector.poll()
        assert detector.is_confirming is False

        # A display change opens the confirming window (1 read so far, < K).
        capture.capture.return_value = frame2
        ocr.read_metrics.return_value = {
            "ball_speed": 130.0, "launch_angle": 15.0, "launch_direction": -2.0,
        }
        detector.poll()
        assert detector.is_confirming is True

        # Further identical reads accumulate to K and commit the shot; the
        # window then closes so the loop drops back to the slow cadence.
        committed = None
        for _ in range(STABILITY_K + 1):
            r = detector.poll()
            if r is not None:
                committed = r
        assert committed is not None
        assert detector.is_confirming is False

    def test_same_shot_suppressed(self) -> None:
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        # Baseline capture.
        capture.capture.return_value = frame1
        ocr.read_metrics.return_value = {
            "ball_speed": 120.0,
            "launch_angle": 12.0,
            "launch_direction": 1.5,
        }
        assert detector.poll() is None  # baseline

        # First stable shot fires.
        shot = self._emit_stable(
            detector, capture, ocr, frame2,
            {"ball_speed": 130.0, "launch_angle": 15.0, "launch_direction": -2.0},
        )
        assert shot is not None

        # Same metrics again (even across more polls) → suppressed, no re-emit.
        result = self._emit_stable(
            detector, capture, ocr, frame3,
            {"ball_speed": 130.0, "launch_angle": 15.0, "launch_direction": -2.0},
            polls=STABILITY_K + 2,
        )
        assert result is None

    def test_invalid_metrics_returns_none(self) -> None:
        # Out-of-range required metric must never emit, even when stable.
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 64
        result = self._emit_stable(
            detector, capture, ocr, frame,
            {"ball_speed": 999.0, "launch_angle": 12.0, "launch_direction": 1.5},
            polls=STABILITY_K + 2,
        )
        assert result is None

    def test_transitional_frame_not_emitted_or_latched(self) -> None:
        """A mid-animation garbage frame followed by stable values:

        the garbage must NEVER be emitted, and must not latch as the
        committed shot (which would suppress the correct final values).
        Exactly one shot is emitted, equal to the stable values.
        """
        detector, ocr, capture = self._make_detector(mse_threshold=1.0)

        # Baseline.
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        capture.capture.return_value = base
        ocr.read_metrics.return_value = {
            "ball_speed": 100.0, "launch_angle": 10.0, "launch_direction": 0.0,
        }
        assert detector.poll() is None

        emitted: list[MevoShotData] = []

        # Frame A: transitional / garbage VLA caught mid-animation.
        capture.capture.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 50
        ocr.read_metrics.return_value = {
            "ball_speed": 88.0, "launch_angle": 3.0, "launch_direction": 0.0,
        }
        r = detector.poll()
        if r is not None:
            emitted.append(r)

        # Frames B,C,D: settled, stable, correct values.
        stable = {"ball_speed": 95.0, "launch_angle": 52.6, "launch_direction": 1.2}
        capture.capture.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 150
        ocr.read_metrics.return_value = dict(stable)
        for _ in range(STABILITY_K):
            r = detector.poll()
            if r is not None:
                emitted.append(r)

        assert len(emitted) == 1, f"expected exactly one shot, got {len(emitted)}"
        assert emitted[0].launch_angle == 52.6
        assert emitted[0].ball_speed == 95.0
        # The garbage VLA=3.0 was never emitted.
        assert all(s.launch_angle != 3.0 for s in emitted)


class TestCoherenceGate:
    """Cross-field plausibility: a lofted-club-style low-speed + very-low-VLA
    + very-low-spin combination is physically incoherent and rejected, even
    without club info. This is the exact tonight-failure signature.
    """

    def test_tonight_failure_signature_rejected(self) -> None:
        # The real bad shot: a chip read as a near-flat, barely-spinning,
        # low-speed shot. Incoherent → reject.
        metrics = {
            "ball_speed": 18.0,
            "launch_angle": 26.5,
            "launch_direction": 0.5,
            "spin_rate": 2013.0,
        }
        assert _coherence_ok(metrics) is False
        assert _validate_metrics(metrics) is False

    def test_normal_full_shot_passes(self) -> None:
        metrics = {
            "ball_speed": 130.0,
            "launch_angle": 14.0,
            "launch_direction": 1.0,
            "spin_rate": 4200.0,
        }
        assert _coherence_ok(metrics) is True
        assert _validate_metrics(metrics) is True

    def test_legit_chip_high_spin_passes(self) -> None:
        # A real wedge chip: low speed but proper loft + high spin → coherent.
        metrics = {
            "ball_speed": 22.0,
            "launch_angle": 48.0,
            "launch_direction": 0.0,
            "spin_rate": 6500.0,
        }
        assert _coherence_ok(metrics) is True

    def test_putt_low_everything_passes(self) -> None:
        # A putt (very low speed, near-zero VLA, near-zero spin) is coherent
        # for a putt — the coherence gate must not nuke putts. Putts have
        # essentially zero launch and zero spin, distinct from a misread chip.
        metrics = {
            "ball_speed": 4.5,
            "launch_angle": 1.0,
            "launch_direction": 0.0,
            "spin_rate": 0.0,
        }
        assert _coherence_ok(metrics) is True


class TestClubConditionedGate:
    """When a club / loft is supplied, club-aware bounds reject implausible
    lofted-club shots (e.g. a 60° lob wedge with VLA < ~35 or spin < ~3000).
    """

    def test_lw_rejects_tonight_failure(self) -> None:
        metrics = {
            "ball_speed": 18.0,
            "launch_angle": 26.5,
            "launch_direction": 0.5,
            "spin_rate": 2013.0,
        }
        # Club code path.
        assert _validate_metrics(metrics, expected_club="LW") is False
        # Loft path (60-degree wedge).
        assert _validate_metrics(metrics, loft=60.0) is False

    def test_lw_accepts_real_wedge_shot(self) -> None:
        metrics = {
            "ball_speed": 30.0,
            "launch_angle": 42.0,
            "launch_direction": 0.0,
            "spin_rate": 7000.0,
        }
        assert _validate_metrics(metrics, expected_club="LW") is True

    def test_club_min_spin_vla_lookup(self) -> None:
        min_spin, min_vla = club_min_spin_vla("LW")
        assert min_spin >= 3000.0
        assert min_vla >= 30.0
        # A driver imposes no lofted-wedge floor.
        dr_spin, dr_vla = club_min_spin_vla("DR")
        assert dr_spin < min_spin
        assert dr_vla < min_vla

    def test_poll_with_club_rejects_low_shot(self) -> None:
        """End-to-end: a stable but club-incoherent shot with club=LW
        supplied to poll() is held (returns None)."""
        det, ocr, capture = _detector_with_rois(mse_threshold=1.0)
        # Baseline.
        capture.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr.read_metrics.return_value = {
            "ball_speed": 100.0, "launch_angle": 10.0,
            "launch_direction": 0.0, "spin_rate": 4000.0,
        }
        det.poll()

        capture.capture.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 120
        ocr.read_metrics.return_value = {
            "ball_speed": 18.0, "launch_angle": 26.5,
            "launch_direction": 0.5, "spin_rate": 2013.0,
        }
        result = None
        for _ in range(STABILITY_K + 2):
            r = det.poll(expected_club="LW")
            if r is not None:
                result = r
        assert result is None


class TestNoFabricatedZeros:
    """A configured-but-failed optional field must not be silently sent as
    0.0; the shot is held/discarded instead (same path as a required-field
    failure)."""

    def test_configured_field_failed_holds_shot(self) -> None:
        # spin_rate is a CONFIGURED ROI but OCR returns None for it →
        # discard the shot rather than fabricate spin=0.0.
        det, ocr, capture = _detector_with_rois(
            mse_threshold=1.0,
            roi_names=("ball_speed", "launch_angle", "launch_direction", "spin_rate"),
        )
        capture.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr.read_metrics.return_value = {
            "ball_speed": 100.0, "launch_angle": 10.0,
            "launch_direction": 0.0, "spin_rate": 4000.0,
        }
        det.poll()  # baseline

        capture.capture.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 120
        ocr.read_metrics.return_value = {
            "ball_speed": 120.0, "launch_angle": 14.0,
            "launch_direction": 1.0, "spin_rate": None,  # configured but failed
        }
        result = None
        for _ in range(STABILITY_K + 2):
            r = det.poll()
            if r is not None:
                result = r
        assert result is None

    def test_unconfigured_field_defaults_zero(self) -> None:
        # spin_rate is NOT a configured ROI → its absence is fine, default 0.0.
        det, ocr, capture = _detector_with_rois(
            mse_threshold=1.0,
            roi_names=("ball_speed", "launch_angle", "launch_direction"),
        )
        capture.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr.read_metrics.return_value = {
            "ball_speed": 100.0, "launch_angle": 10.0, "launch_direction": 0.0,
        }
        det.poll()  # baseline

        capture.capture.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 120
        stable = {"ball_speed": 120.0, "launch_angle": 14.0, "launch_direction": 1.0}
        ocr.read_metrics.return_value = dict(stable)
        result = None
        for _ in range(STABILITY_K):
            r = det.poll()
            if r is not None:
                result = r
        assert result is not None
        assert result.spin_rate == 0.0


def _detector_with_rois(
    mse_threshold: float = 1.0,
    roi_names: tuple[str, ...] = ("ball_speed", "launch_angle", "launch_direction"),
) -> tuple[MevoDetector, MagicMock, MagicMock]:
    """Build a detector whose mock OCR exposes a realistic ROI list."""
    settings = MevoSettings(enabled=True, mse_threshold=mse_threshold)
    ocr = MagicMock()
    ocr._rois = [ROI(name=n, x=0, y=0, width=10, height=10) for n in roi_names]
    capture = MagicMock()
    return MevoDetector(settings, ocr, capture), ocr, capture


class TestPuttFallbackConfig:
    """Sanity-check the putt-fallback config flags the detector depends on."""

    def test_defaults(self) -> None:
        settings = MevoSettings()
        assert settings.putt_fallback is False
        assert settings.putt_fallback_max_age_s == 3.0

    def test_enable(self) -> None:
        settings = MevoSettings(putt_fallback=True, putt_fallback_max_age_s=5.0)
        assert settings.putt_fallback is True
        assert settings.putt_fallback_max_age_s == 5.0
