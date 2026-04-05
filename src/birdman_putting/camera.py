"""Camera capture abstraction wrapping OpenCV VideoCapture."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from birdman_putting.config import CameraSettings

logger = logging.getLogger(__name__)

# Camera property mapping: config field name -> OpenCV property ID
_CAMERA_PROPS = {
    "saturation": cv2.CAP_PROP_SATURATION,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "white_balance_blue": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
    "white_balance_red": cv2.CAP_PROP_WHITE_BALANCE_RED_V,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "hue": cv2.CAP_PROP_HUE,
    "gain": cv2.CAP_PROP_GAIN,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "gamma": cv2.CAP_PROP_GAMMA,
    "zoom": cv2.CAP_PROP_ZOOM,
    "focus": cv2.CAP_PROP_FOCUS,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
}

_WARMUP_FRAMES = 15           # frames to discard before validation (~0.75s at 30fps)
_WARMUP_DELAY = 0.05          # seconds between warmup reads
_FRAME_VALIDATE_ATTEMPTS = 5  # frames to check after warmup
_FRAME_VALIDATE_DELAY = 0.2   # seconds between validation reads


class Camera:
    """Manages video capture from webcam or video file.

    Handles MJPEG codec, FPS override, resolution, PS4 Eye decoding,
    and camera property management. Automatically falls back from
    MJPEG/DirectShow to the default backend if frames fail to arrive.
    """

    def __init__(self, settings: CameraSettings):
        self._settings = settings
        self._cap: cv2.VideoCapture | None = None
        self._video_file: bool = False
        self._fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._status_message: str = "Not started"
        self._rotation_matrix: np.ndarray | None = None  # Cached for per-frame rotation
        self._rotation_cached_for: float = 0.0  # Track which rotation value is cached

        # Threaded capture: a dedicated thread reads frames continuously
        # so camera.read() never blocks on the DirectShow/MSMF pipeline
        self._grab_thread: threading.Thread | None = None
        self._grab_running = False
        self._latest_frame: np.ndarray | None = None
        self._frame_new = False  # True when grab thread has a new frame
        self._frame_lock = threading.Lock()

        # Property changes queued from the main thread and applied on the
        # grab thread between frame reads. MSMF crashes if properties are
        # set from a different thread while capturing.
        self._pending_props: list[tuple[int, float]] = []
        self._props_lock = threading.Lock()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_width, self._frame_height

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def status_message(self) -> str:
        """Human-readable status of the last open attempt."""
        return self._status_message

    def open_webcam(self) -> bool:
        """Open the webcam with configured settings and frame validation.

        Always tries DirectShow first — even when it fails, it initializes
        the Windows camera subsystem so that the default backend can find
        DirectShow/MSMF devices. Without this warmup, only FFMPEG/obsensor
        backends are available and webcam open fails.

        Returns:
            True if camera opened successfully and producing frames.
        """
        s = self._settings
        self._video_file = False

        logger.debug(
            "open_webcam: index=%d, mjpeg=%s, width=%d, height=%d, fps_override=%d",
            s.webcam_index, s.mjpeg, s.width, s.height, s.fps_override,
        )

        # Always try DirectShow first — even if MJPEG is disabled, the
        # DirectShow attempt initializes the Windows camera subsystem.
        # Without this, the default backend only finds FFMPEG/obsensor.
        logger.debug("Trying DirectShow...")
        if self._try_open_mjpeg():
            self._read_properties()
            # Don't apply user camera properties here — _validate_frames()
            # forces auto-exposure so the camera can produce visible frames.
            # The grab thread applies real user settings when it reopens.
            if self._validate_frames():
                self._status_message = (
                    f"Connected (DirectShow {self._frame_width}x{self._frame_height}"
                    f" @ {self._fps:.0f}fps)"
                )
                logger.info("Camera ready: %s", self._status_message)
                return True
            else:
                logger.warning(
                    "DirectShow opened but frames are black or absent, "
                    "falling back to default backend"
                )
                self._release_cap()

        # Fallback — DirectShow failed (common) but initialized the subsystem
        logger.debug("Trying default backend...")
        if self._try_open_default():
            self._read_properties()
            if self._validate_frames():
                label = "fallback" if s.mjpeg else "default"
                self._status_message = (
                    f"Connected ({label} {self._frame_width}x{self._frame_height}"
                    f" @ {self._fps:.0f}fps)"
                )
                logger.info("Camera ready: %s", self._status_message)
                return True
            else:
                logger.error("Default backend opened but no frames arrived")
                self._release_cap()

        self._status_message = "Failed to open camera"
        logger.error(self._status_message)
        return False

    def _try_open_mjpeg(self) -> bool:
        """Try opening the camera with MJPEG codec via DirectShow.

        Returns:
            True if the capture device opened and accepted MJPEG codec
            (frames not yet validated).
        """
        s = self._settings
        logger.debug("DirectShow: opening camera index %d (cv2.CAP_DSHOW=%d, combined=%d)",
                      s.webcam_index, cv2.CAP_DSHOW, s.webcam_index + cv2.CAP_DSHOW)
        self._cap = cv2.VideoCapture(s.webcam_index + cv2.CAP_DSHOW)
        opened = self._cap is not None and self._cap.isOpened()
        logger.debug("DirectShow: isOpened=%s", opened)
        if not opened:
            logger.warning("Failed to open camera %d with DirectShow", s.webcam_index)
            return False

        # DirectShow requires: resolution -> codec -> FPS (in this order)
        res_w = s.width if s.width > 0 else 1280
        res_h = s.height if s.height > 0 else 720
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # type: ignore[attr-defined]
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Verify MJPEG codec actually took effect
        actual_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if actual_fourcc != fourcc:
            actual_str = "".join(
                chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)
            )
            logger.warning(
                "MJPEG codec not accepted by camera (requested MJPG, got %s), "
                "skipping MJPEG mode",
                actual_str,
            )
            self._release_cap()
            return False

        target_fps = s.fps_override if s.fps_override > 0 else 60
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        # Log actual FPS for diagnostic purposes
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "MJPEG camera: %dx%d @ %d fps requested (actual reported: %.0f fps)",
            res_w, res_h, target_fps, actual_fps,
        )
        if actual_fps > 0 and actual_fps < target_fps * 0.5:
            logger.warning(
                "Camera reports %.0f fps (requested %d) — MJPEG mode may not be working",
                actual_fps, target_fps,
            )

        # Minimize frame buffer latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def _try_open_default(self) -> bool:
        """Try opening the camera with the default backend (no DirectShow/MJPEG).

        Applies configured resolution and FPS so the fallback path still
        achieves the desired capture settings (e.g. 1280x720 @ 60fps).

        Returns:
            True if the capture device opened (frames not yet validated).
        """
        s = self._settings
        logger.debug("Default backend: opening camera index %d", s.webcam_index)
        self._cap = cv2.VideoCapture(s.webcam_index)
        opened = self._cap is not None and self._cap.isOpened()
        logger.debug("Default backend: isOpened=%s", opened)
        if not opened:
            logger.error("Failed to open camera %d with default backend", s.webcam_index)
            return False

        # Apply resolution and FPS (same defaults as MJPEG path)
        res_w = s.width if s.width > 0 else 1280
        res_h = s.height if s.height > 0 else 720
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

        target_fps = s.fps_override if s.fps_override > 0 else 60
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        logger.info(
            "Default backend camera: %dx%d @ %d fps requested",
            res_w, res_h, target_fps,
        )
        return True

    def _validate_frames(self) -> bool:
        """Confirm the camera can deliver frames (not stuck or disconnected).

        Only checks that cap.read() returns data — does NOT reject black
        frames. Some cameras (Razer Kiyo Pro) produce black frames for
        several seconds while auto-exposure converges; the grab thread
        will pick up real frames once the sensor settles.

        Returns:
            True if at least one frame was successfully read.
        """
        if self._cap is None:
            return False

        # Phase 1: Warmup — discard initial frames
        for i in range(1, _WARMUP_FRAMES + 1):
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.debug("Warmup frame %d: no frame", i)
            time.sleep(_WARMUP_DELAY)

        # Phase 2: Validate — just need one readable frame
        for attempt in range(1, _FRAME_VALIDATE_ATTEMPTS + 1):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                mean_val = float(np.mean(frame))
                logger.debug(
                    "Frame validation passed on attempt %d (mean=%.1f)",
                    attempt, mean_val,
                )
                if mean_val < 3.0:
                    logger.warning(
                        "Camera frames are dark (mean=%.1f) — auto-exposure "
                        "may still be converging", mean_val,
                    )
                return True
            logger.debug("Frame validation attempt %d: no frame", attempt)
            time.sleep(_FRAME_VALIDATE_DELAY)

        return False

    def _read_properties(self) -> None:
        """Read actual camera properties (FPS, resolution) after open.

        Handles PS4 Eye special settings and FPS=0 fallback.
        """
        if self._cap is None:
            return

        s = self._settings

        # PS4 Eye special settings
        if s.ps4:
            self._cap.set(cv2.CAP_PROP_FPS, 120)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1724)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 404)

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle FPS detection failure
        if self._fps == 0.0:
            self._cap.set(cv2.CAP_PROP_FPS, 60)
            self._fps = 60.0
            logger.warning("FPS detection returned 0, defaulting to 60")

        logger.info(
            "Camera properties: %dx%d @ %.1f fps",
            self._frame_width, self._frame_height, self._fps,
        )

    def _release_cap(self) -> None:
        """Release the current capture device without full cleanup logging."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def open_video(self, path: str | Path) -> bool:
        """Open a video file for testing.

        Returns:
            True if video opened successfully.
        """
        self._video_file = True
        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            logger.error("Failed to open video file: %s", path)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video opened: %s (%dx%d @ %.1f fps)",
            path, self._frame_width, self._frame_height, self._fps,
        )
        return True

    def start_grab_thread(self) -> None:
        """Start a dedicated thread that owns the camera and grabs frames.

        The camera is reopened on the grab thread so that DirectShow/MSMF
        COM objects belong to that thread's apartment. This prevents
        Windows from throttling capture when the main window loses focus.
        """
        if self._video_file or self._grab_running:
            return

        # Release the camera from the main thread — it will be reopened
        # on the grab thread with its own COM apartment
        self._grab_running = True
        self._grab_ready = threading.Event()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="camera-grab",
        )
        self._grab_thread.start()
        # Wait for the grab thread to reopen the camera
        self._grab_ready.wait(timeout=5)
        logger.info("Camera grab thread started")

    def start_async_open(self, on_ready: Callable[[], None] | None = None) -> None:
        """Open the camera on a background thread, then start the grab thread.

        Runs the full open_webcam() sequence (DirectShow warmup + default
        backend) on a background thread so the UI stays responsive. Once
        the camera is open, starts the grab thread for continuous capture.

        Args:
            on_ready: Optional callback invoked once the camera is open and
                the grab thread is capturing. Called from a background thread;
                use window.after() to schedule UI updates from this callback.
        """
        if self._video_file or self._grab_running:
            return

        self._status_message = "Connecting to camera..."
        self._on_ready_callback = on_ready

        def _open_and_start() -> None:
            import sys
            # Initialize COM on this thread — DirectShow and MSMF both
            # require COM. Without it, VideoCapture silently skips these
            # backends and only tries FFMPEG/obsensor (which fail).
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.ole32.CoInitializeEx(None, 0)  # type: ignore[attr-defined]
                except Exception:
                    pass

            if self.open_webcam():
                self.start_grab_thread()
                if self._on_ready_callback:
                    self._on_ready_callback()
            else:
                logger.error("Background camera open failed")
                if self._on_ready_callback:
                    self._on_ready_callback()

        thread = threading.Thread(
            target=_open_and_start, daemon=True, name="camera-open",
        )
        thread.start()

    def _grab_loop(self) -> None:
        """Continuously read frames from the camera into _latest_frame.

        Opens its own VideoCapture so COM objects are on this thread.
        """
        import sys

        # Set this thread to highest priority and prevent timer throttling
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.SetThreadPriority(
                    kernel32.GetCurrentThread(), 2,  # THREAD_PRIORITY_HIGHEST
                )
                # Initialize COM on this thread (MTA for DirectShow)
                ctypes.windll.ole32.CoInitializeEx(None, 0)  # type: ignore[attr-defined]
                # Set 1ms timer resolution to prevent background throttling
                ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Reuse the existing VideoCapture instead of reopening.
        # Reopening creates a fresh DirectShow connection that resets
        # camera firmware settings (Razer Synapse exposure, etc.) and
        # the Kiyo Pro starts black — OpenCV can't fix it via properties.
        # COM is initialized on this thread (MTA) which is sufficient for
        # cross-thread DirectShow/MSMF access.
        if self._cap is None or not self._cap.isOpened():
            logger.error("Grab thread: no open camera to use")
            self._grab_ready.set()
            self._grab_running = False
            return

        # Don't apply saved camera properties here — the camera already has
        # working exposure/brightness from the firmware (Synapse, auto-exposure).
        # Applying saved props like auto_exposure=1 + exposure=-5 would override
        # that and cause black frames. User can still adjust via Settings sliders
        # which queue property changes through _pending_props.
        logger.info("Grab thread: using existing capture (no reopen, no property override)")
        self._grab_ready.set()

        # Set up message pump for MSMF fallback (DirectShow doesn't need it)
        _pump_messages = None
        if sys.platform == "win32":
            try:
                import ctypes.wintypes as wt
                _msg = wt.MSG()
                PM_REMOVE = 0x0001

                def _pump_messages() -> None:
                    while user32.PeekMessageW(ctypes.byref(_msg), 0, 0, 0, PM_REMOVE):
                        user32.TranslateMessage(ctypes.byref(_msg))
                        user32.DispatchMessageW(ctypes.byref(_msg))

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            except Exception:
                _pump_messages = None

        grab_count = 0
        grab_start = time.perf_counter()
        while self._grab_running and self._cap is not None:
            if _pump_messages is not None:
                _pump_messages()
            # Apply any queued property changes from the main thread
            if self._pending_props:
                self._drain_pending_props()
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_new = True
                grab_count += 1
                now = time.perf_counter()
                if now - grab_start >= 30.0:
                    grab_fps = grab_count / (now - grab_start)
                    logger.info("Grab thread FPS: %.1f", grab_fps)
                    grab_count = 0
                    grab_start = now

    def stop_grab_thread(self) -> None:
        """Stop the grab thread."""
        self._grab_running = False
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2)
            self._grab_thread = None

    def read(self) -> np.ndarray | None:
        """Read a single frame from the capture source.

        If the grab thread is running, returns the latest grabbed frame
        (non-blocking). Otherwise reads directly from the capture device.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        if self._cap is None:
            return None

        # Use threaded grab if available
        if self._grab_running:
            with self._frame_lock:
                if not self._frame_new:
                    return None  # No new frame since last read
                frame = self._latest_frame
                self._frame_new = False
            if frame is None:
                return None
        else:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

        # PS4 Eye frame decoding
        if self._settings.ps4 and not self._video_file:
            frame = self._decode_ps4(frame)

        # Apply darkness normalization
        if self._settings.darkness > 0:
            d = self._settings.darkness
            cv2.normalize(frame, frame, 0 - d, 255 - d, norm_type=cv2.NORM_MINMAX)

        # Flip if configured
        if self._settings.flip_image and not self._video_file:
            frame = cv2.flip(frame, 1)

        # NOTE: Rotation is applied AFTER resize in the processing loop
        # to avoid warpAffine on the full 1280x720 frame (~4x slower).

        return frame

    def apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation correction to a frame (call after resize for speed).

        Returns the rotated frame, or the original if rotation is 0.
        """
        if self._settings.rotation == 0.0:
            return frame
        if (self._rotation_matrix is None
                or self._rotation_cached_for != self._settings.rotation):
            h, w = frame.shape[:2]
            self._rotation_matrix = cv2.getRotationMatrix2D(
                (w / 2, h / 2), self._settings.rotation, 1.0,
            )
            self._rotation_cached_for = self._settings.rotation
        return cv2.warpAffine(
            frame, self._rotation_matrix, (frame.shape[1], frame.shape[0]),
        )

    def release(self) -> None:
        """Release the video capture."""
        self.stop_grab_thread()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def apply_properties(self) -> None:
        """Re-apply all camera properties from current settings.

        When the grab thread is running, changes are queued and applied
        on the grab thread between frame reads. MSMF crashes if camera
        properties are set from a different thread while capturing.
        """
        if self._grab_running:
            props = self._collect_properties()
            with self._props_lock:
                self._pending_props.extend(props)
        else:
            self._apply_camera_properties()

    # Properties that use 0.0 as a valid/meaningful value and should
    # always be sent to the camera (not skipped when zero).
    _ALWAYS_APPLY = {"auto_exposure", "autofocus", "auto_wb", "focus"}

    def _collect_properties(self) -> list[tuple[int, float]]:
        """Collect all camera properties that should be applied."""
        props = []
        auto_exp = getattr(self._settings, "auto_exposure", 0.0)
        auto_focus = getattr(self._settings, "autofocus", 0.0)
        for field_name, prop_id in _CAMERA_PROPS.items():
            value = getattr(self._settings, field_name, 0.0)
            # Skip manual exposure/focus when auto mode is on — they conflict
            if field_name == "exposure" and auto_exp == 3.0:
                continue
            if field_name == "focus" and auto_focus == 1.0:
                continue
            if value != 0.0 or field_name in self._ALWAYS_APPLY:
                props.append((prop_id, value))
        return props

    def _drain_pending_props(self) -> None:
        """Apply any queued property changes (called from grab thread)."""
        with self._props_lock:
            pending = list(self._pending_props)
            self._pending_props.clear()
        for prop_id, value in pending:
            if self._cap is not None:
                self._cap.set(prop_id, value)
                # Reverse-lookup name for logging
                name = next(
                    (k for k, v in _CAMERA_PROPS.items() if v == prop_id),
                    str(prop_id),
                )
                logger.debug("Set camera %s = %s (from queue)", name, value)

    def _apply_camera_properties(self) -> None:
        """Apply all configured camera properties directly.

        Only call from the thread that owns the VideoCapture, or when the
        grab thread is not running.
        """
        if self._cap is None:
            return

        auto_exp = getattr(self._settings, "auto_exposure", 0.0)
        auto_focus = getattr(self._settings, "autofocus", 0.0)
        for field_name, prop_id in _CAMERA_PROPS.items():
            value = getattr(self._settings, field_name, 0.0)
            # Skip manual exposure/focus when auto mode is on — they conflict
            if field_name == "exposure" and auto_exp == 3.0:
                continue
            if field_name == "focus" and auto_focus == 1.0:
                continue
            if value != 0.0 or field_name in self._ALWAYS_APPLY:
                self._cap.set(prop_id, value)
                logger.debug("Set camera %s = %s", field_name, value)

    @staticmethod
    def _decode_ps4(frame: np.ndarray) -> np.ndarray:
        """Decode PS4 Eye camera frame (extract left stereo image)."""
        left = np.zeros((400, 632, 3), np.uint8)
        for i in range(min(400, frame.shape[0])):
            if frame.shape[1] >= 640 + 24:
                left[i] = frame[i, 32:640 + 24]
        return left
