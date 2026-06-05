"""Tests for camera module — frame validation and fallback behavior."""

from __future__ import annotations

import sys
import time
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


class _FakePSEyeCamera:
    """Stand-in for pseyepy.Camera used by the PS3 Eye tests.

    Returns an RGB frame (red channel low, blue channel high) so the
    RGB->BGR swap in the inline read can be verified.  Records whether end()
    was called.
    """

    RES_SMALL = 0  # 320x240 (QVGA)
    RES_LARGE = 1  # 640x480 (VGA)

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.ended = False
        # RGB frame: R=10, G=0, B=200 → after BGR swap, B=200, G=0, R=10.
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = 10   # R
        frame[:, :, 2] = 200  # B
        self._frame = frame

    def read(self) -> tuple[np.ndarray, float]:
        return self._frame, 0.0

    def end(self) -> None:
        self.ended = True


def _wait_for(predicate: object, timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until truthy or timeout; returns final truthiness."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.005)
    return bool(predicate())  # type: ignore[operator]


class TestPSEyeCamera:
    """PS3 Eye (pseyepy) source — fully mocked, no real hardware."""

    def _install_fake_pseyepy(self) -> MagicMock:
        """Inject a fake ``pseyepy`` module exposing ``Camera``."""
        fake_module = MagicMock()
        fake_module.Camera = _FakePSEyeCamera
        return fake_module

    def test_open_pseye_reports_open_and_streams_bgr_frames(self) -> None:
        fake_module = self._install_fake_pseyepy()
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(camera_type="pseye", pseye_resolution="qvga")
            try:
                assert camera.open_pseye() is True
                assert camera.is_open is True
                assert camera._is_pseye is True
                # QVGA dimensions reported.
                assert camera.frame_size == (320, 240)

                # open_pseye primes _latest_frame (no grab thread); read_latest()
                # hands back a BGR copy (240x320x3).
                assert _wait_for(lambda: camera.read_latest() is not None)
                frame = camera.read_latest()
                assert frame is not None
                assert frame.shape == (240, 320, 3)
                assert frame.dtype == np.uint8
                # RGB(10,0,200) -> BGR(200,0,10): channel 0 (B) is 200.
                assert frame[0, 0, 0] == 200
                assert frame[0, 0, 2] == 10
            finally:
                camera.release()

    def test_release_calls_pseye_end_and_is_idempotent(self) -> None:
        fake_module = self._install_fake_pseyepy()
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(camera_type="pseye", pseye_resolution="qvga")
            assert camera.open_pseye() is True
            pseye_handle = camera._pseye
            assert isinstance(pseye_handle, _FakePSEyeCamera)

            camera.release()
            assert pseye_handle.ended is True
            assert camera._pseye is None
            assert camera._is_pseye is False
            assert camera.is_open is False

            # Second release must be a no-op (idempotent), not raise.
            camera.release()
            assert camera._pseye is None

    def test_pseye_uses_inline_read_not_grab_thread(self) -> None:
        """PS3 Eye must be read inline (no grab thread) to avoid GIL starvation.

        pseyepy's read() blocks while holding the Python GIL, so running it on a
        dedicated grab thread starves the processing/UI threads.  open_pseye must
        therefore NOT start a grab thread, and read() must return BGR frames by
        reading the device inline.
        """
        fake_module = self._install_fake_pseyepy()
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(camera_type="pseye", pseye_resolution="qvga")
            try:
                assert camera.open_pseye() is True
                # No grab thread for the PS3 Eye.
                assert camera.is_grab_running is False
                assert camera.is_pseye is True
                # read() reads the device inline and returns a swapped BGR frame.
                frame = camera.read()
                assert frame is not None
                assert frame.shape == (240, 320, 3)
                assert frame[0, 0, 0] == 200  # B (from RGB B=200)
                assert frame[0, 0, 2] == 10   # R (from RGB R=10)
            finally:
                camera.release()

    def test_no_swap_keeps_rgb_order(self) -> None:
        """With pseye_swap_rb=False the inline read must not swap channels."""
        fake_module = self._install_fake_pseyepy()
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(
                camera_type="pseye", pseye_resolution="qvga", pseye_swap_rb=False
            )
            try:
                assert camera.open_pseye() is True
                assert _wait_for(lambda: camera.read_latest() is not None)
                frame = camera.read_latest()
                assert frame is not None
                # No swap: channel 0 stays R=10, channel 2 stays B=200.
                assert frame[0, 0, 0] == 10
                assert frame[0, 0, 2] == 200
            finally:
                camera.release()

    def test_vga_resolution_maps_to_res_large(self) -> None:
        fake_module = self._install_fake_pseyepy()
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(camera_type="pseye", pseye_resolution="vga")
            try:
                assert camera.open_pseye() is True
                # open_pseye records 640x480 for VGA from settings (the fake
                # frame is still 320x240, but the reported size comes from the
                # resolution mapping).
                assert camera.frame_size == (640, 480)
            finally:
                camera.release()

    def test_open_pseye_failure_returns_false(self) -> None:
        """If pseyepy.Camera construction raises, open_pseye reports False."""
        fake_module = MagicMock()
        fake_module.Camera = MagicMock(side_effect=RuntimeError("no device"))
        fake_module.Camera.RES_SMALL = 0
        fake_module.Camera.RES_LARGE = 1
        with patch.dict(sys.modules, {"pseyepy": fake_module}):
            camera = _make_camera(camera_type="pseye", pseye_resolution="qvga")
            assert camera.open_pseye() is False
            assert camera._is_pseye is False
            assert camera._pseye is None
            assert camera.is_open is False
            assert "PS3 Eye" in camera.status_message
