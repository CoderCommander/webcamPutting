"""Tests for camera module — frame validation and fallback behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from birdman_putting.camera import Camera
from birdman_putting.config import CameraSettings


class TestIsBlackFrame:
    """Test the static black frame detection helper."""

    def test_pure_black_is_detected(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert Camera._is_black_frame(frame) is True

    def test_near_black_is_detected(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 2
        assert Camera._is_black_frame(frame) is True

    def test_dim_frame_is_not_black(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 10
        assert Camera._is_black_frame(frame) is False

    def test_normal_frame_is_not_black(self) -> None:
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        assert Camera._is_black_frame(frame) is False

    def test_threshold_boundary(self) -> None:
        # Just below threshold — should be black
        frame = np.full((100, 100, 3), 2, dtype=np.uint8)
        assert Camera._is_black_frame(frame) is True

        # Above threshold — should not be black
        frame = np.full((100, 100, 3), 4, dtype=np.uint8)
        assert Camera._is_black_frame(frame) is False


class TestValidateFrames:
    """Test the _validate_frames method with mocked capture."""

    def _make_camera(self, **kwargs: object) -> Camera:
        settings = CameraSettings(**kwargs)  # type: ignore[arg-type]
        return Camera(settings)

    @patch("birdman_putting.camera.time.sleep")
    def test_good_frames_pass(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        good_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, good_frame)
        camera._cap = mock_cap

        assert camera._validate_frames() is True

    @patch("birdman_putting.camera.time.sleep")
    def test_black_frames_fail(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, black_frame)
        camera._cap = mock_cap

        assert camera._validate_frames() is False

    @patch("birdman_putting.camera.time.sleep")
    def test_warmup_then_good_frame_passes(self, mock_sleep: MagicMock) -> None:
        """Simulate camera that produces black during warmup then good frames."""
        camera = self._make_camera()
        mock_cap = MagicMock()
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        good_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128

        # 5 warmup reads (discarded) + first validation read is good
        frames = [(True, black_frame)] * 5 + [(True, good_frame)]
        mock_cap.read.side_effect = frames
        camera._cap = mock_cap

        assert camera._validate_frames() is True

    @patch("birdman_putting.camera.time.sleep")
    def test_no_frames_fail(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        camera._cap = mock_cap

        assert camera._validate_frames() is False

    @patch("birdman_putting.camera.time.sleep")
    def test_no_cap_fails(self, mock_sleep: MagicMock) -> None:
        camera = self._make_camera()
        camera._cap = None
        assert camera._validate_frames() is False


def _make_camera(**kwargs: object) -> Camera:
    """Build a Camera without touching any real device."""
    settings = CameraSettings(**kwargs)  # type: ignore[arg-type]
    return Camera(settings)


def _color_frame(value: int = 100) -> np.ndarray:
    """A non-black BGR frame filled with a constant value."""
    return np.full((360, 640, 3), value, dtype=np.uint8)


class TestReadCopySemantics:
    """read() / read_latest() must return caller-owned copies (torn-frame fix)."""

    def _grabbing_camera(self) -> Camera:
        """A camera in 'grab thread running' state with a stored frame."""
        camera = _make_camera()
        camera._cap = MagicMock()  # non-None so read() does not early-return
        camera._grab_running = True
        camera._latest_frame = _color_frame(100)
        camera._frame_new = True
        return camera

    def test_read_returns_frame_when_fresh(self) -> None:
        camera = self._grabbing_camera()
        frame = camera.read()
        assert frame is not None
        assert frame.shape == (360, 640, 3)

    def test_read_returns_none_when_not_fresh(self) -> None:
        """Freshness contract: read() returns None when no new frame."""
        camera = self._grabbing_camera()
        camera._frame_new = False
        assert camera.read() is None

    def test_read_clears_freshness_flag(self) -> None:
        """A consumed frame is not re-delivered by read()."""
        camera = self._grabbing_camera()
        first = camera.read()
        assert first is not None
        # Second read with no new grab in between → None.
        assert camera.read() is None

    def test_read_returns_independent_copy(self) -> None:
        """Mutating read()'s result must not corrupt _latest_frame."""
        camera = self._grabbing_camera()
        frame = camera.read()
        assert frame is not None
        original = camera._latest_frame.copy()  # type: ignore[union-attr]
        frame[:] = 0  # simulate in-place normalize/flip on the returned frame
        # _latest_frame untouched by the mutation.
        assert np.array_equal(camera._latest_frame, original)  # type: ignore[arg-type]
        # And the returned frame is not the same object as the stored one.
        assert frame is not camera._latest_frame

    def test_read_latest_returns_frame_when_stale(self) -> None:
        """read_latest() ignores the freshness flag (one-shot calibration)."""
        camera = self._grabbing_camera()
        # Consume the fresh frame so _frame_new is False.
        assert camera.read() is not None
        assert camera._frame_new is False
        # read_latest still hands back the last frame.
        latest = camera.read_latest()
        assert latest is not None
        assert latest.shape == (360, 640, 3)

    def test_read_latest_does_not_clear_freshness(self) -> None:
        """read_latest() must not consult or clear _frame_new."""
        camera = self._grabbing_camera()
        assert camera._frame_new is True
        camera.read_latest()
        # Freshness flag untouched → a following read() still returns a frame.
        assert camera._frame_new is True
        assert camera.read() is not None

    def test_read_latest_returns_independent_copy(self) -> None:
        camera = self._grabbing_camera()
        latest = camera.read_latest()
        assert latest is not None
        original = camera._latest_frame.copy()  # type: ignore[union-attr]
        latest[:] = 0
        assert np.array_equal(camera._latest_frame, original)  # type: ignore[arg-type]
        assert latest is not camera._latest_frame

    def test_read_latest_none_when_never_captured(self) -> None:
        """Grab thread running but no frame produced yet → None (calibration)."""
        camera = _make_camera()
        camera._cap = MagicMock()
        camera._grab_running = True
        camera._latest_frame = None
        camera._frame_new = False
        assert camera.read_latest() is None

    def test_read_latest_none_when_no_cap_and_no_frame(self) -> None:
        camera = _make_camera()
        camera._cap = None
        camera._latest_frame = None
        assert camera.read_latest() is None


class TestGrabOnce:
    """The single grab iteration helper: failure counting + sleep + liveness."""

    def test_success_stores_frame_and_marks_fresh(self) -> None:
        camera = _make_camera()
        mock_cap = MagicMock()
        good = _color_frame(120)
        mock_cap.read.return_value = (True, good)
        camera._cap = mock_cap

        assert camera._grab_once() is True
        assert camera._frame_new is True
        assert camera._latest_frame is not None
        assert np.array_equal(camera._latest_frame, good)

    @patch("birdman_putting.camera.time.sleep")
    def test_failure_sleeps_instead_of_busy_spin(self, mock_sleep: MagicMock) -> None:
        camera = _make_camera()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        camera._cap = mock_cap

        assert camera._grab_once() is False
        mock_sleep.assert_called_once()
        # Sleeps a small, positive interval (not a busy spin, not seconds).
        (slept,) = mock_sleep.call_args[0]
        assert 0 < slept <= 0.05

    @patch("birdman_putting.camera.time.sleep")
    def test_failure_threshold_sets_unhealthy_flag(
        self, mock_sleep: MagicMock
    ) -> None:
        camera = _make_camera()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        camera._cap = mock_cap

        threshold = camera._GRAB_FAILURE_THRESHOLD
        # Just below threshold: still healthy.
        for _ in range(threshold - 1):
            camera._grab_once()
        assert camera.consecutive_failures == threshold - 1
        assert camera._read_failed is False
        assert camera.is_healthy is True

        # Crossing the threshold flips the liveness flag.
        camera._grab_once()
        assert camera.consecutive_failures == threshold
        assert camera._read_failed is True
        assert camera.is_healthy is False

    @patch("birdman_putting.camera.time.sleep")
    def test_success_resets_failure_counter(self, mock_sleep: MagicMock) -> None:
        camera = _make_camera()
        mock_cap = MagicMock()
        good = _color_frame(120)
        # Fail past the threshold, then succeed.
        fails = [(False, None)] * camera._GRAB_FAILURE_THRESHOLD
        mock_cap.read.side_effect = fails + [(True, good)]
        camera._cap = mock_cap

        for _ in range(camera._GRAB_FAILURE_THRESHOLD):
            camera._grab_once()
        assert camera._read_failed is True
        assert camera.consecutive_failures == camera._GRAB_FAILURE_THRESHOLD

        # A single success clears everything.
        assert camera._grab_once() is True
        assert camera.consecutive_failures == 0
        assert camera._read_failed is False
        assert camera.is_healthy is True

    def test_grab_once_with_no_cap_is_safe(self) -> None:
        camera = _make_camera()
        camera._cap = None
        # Should not raise; reports failure-ish without crashing.
        assert camera._grab_once() is False


class TestPropertyWriteSerialization:
    """update_settings must not let _cap.set() race the grab thread's read()."""

    def test_update_settings_uses_capture_lock(self) -> None:
        """_apply_camera_properties holds the capture lock around _cap.set()."""
        camera = _make_camera(saturation=0.5)
        observed = {"locked": None}

        mock_cap = MagicMock()

        def record_set(*_args: object, **_kwargs: object) -> bool:
            # While set() runs, the capture lock must be held.
            observed["locked"] = camera._capture_lock.locked()
            return True

        mock_cap.set.side_effect = record_set
        camera._cap = mock_cap

        camera.update_settings(CameraSettings(saturation=0.7))
        assert mock_cap.set.called
        assert observed["locked"] is True

    def test_grab_once_uses_capture_lock(self) -> None:
        """_grab_once holds the same capture lock around _cap.read()."""
        camera = _make_camera()
        observed = {"locked": None}
        mock_cap = MagicMock()

        def record_read() -> tuple[bool, np.ndarray]:
            observed["locked"] = camera._capture_lock.locked()
            return (True, _color_frame(90))

        mock_cap.read.side_effect = record_read
        camera._cap = mock_cap

        camera._grab_once()
        assert observed["locked"] is True


class TestReleaseIdempotent:
    """release() must be safe to call more than once."""

    def test_release_twice_does_not_raise(self) -> None:
        camera = _make_camera()
        mock_cap = MagicMock()
        camera._cap = mock_cap
        camera.release()
        # Second call: _cap is already None, must be a no-op (no exception).
        camera.release()
        assert camera._cap is None

    def test_release_with_no_cap_does_not_raise(self) -> None:
        camera = _make_camera()
        camera._cap = None
        camera.release()
        assert camera._cap is None


class TestHealthDefaults:
    """Fresh camera reports healthy with zero failures."""

    def test_initial_state_healthy(self) -> None:
        camera = _make_camera()
        assert camera.consecutive_failures == 0
        assert camera._read_failed is False
        assert camera.is_healthy is True
