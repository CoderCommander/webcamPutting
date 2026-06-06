"""Camera capture abstraction wrapping OpenCV VideoCapture."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

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

_WARMUP_FRAMES = 5            # frames to discard before brightness checking
_WARMUP_DELAY = 0.1           # seconds between warmup reads
_FRAME_VALIDATE_ATTEMPTS = 3  # frames to check after warmup
_FRAME_VALIDATE_DELAY = 0.1   # seconds between validation reads
_BLACK_FRAME_THRESHOLD = 3.0  # mean brightness below this = black frame

# Grab-thread liveness tuning.  On a read failure the grab loop sleeps a
# small interval (instead of busy-spinning at 100% CPU on a dead camera) and
# counts consecutive failures; once the count reaches the threshold the camera
# is flagged unhealthy so a mid-session disconnect becomes observable.
_GRAB_FAILURE_SLEEP = 0.005      # seconds to back off after a failed read
_GRAB_FAILURE_THRESHOLD = 30     # consecutive failures before flagging unhealthy


class Camera:
    """Manages video capture from webcam or video file.

    Handles MJPEG codec, FPS override, resolution, PS4 Eye decoding,
    and camera property management. Automatically falls back from
    MJPEG/DirectShow to the default backend if frames fail to arrive.
    """

    # Exposed as class attributes so callers/tests can read the tuning without
    # importing the module-level constants.
    _GRAB_FAILURE_SLEEP = _GRAB_FAILURE_SLEEP
    _GRAB_FAILURE_THRESHOLD = _GRAB_FAILURE_THRESHOLD

    def __init__(self, settings: CameraSettings):
        self._settings = settings
        self._cap: cv2.VideoCapture | None = None
        # Sony PS3 Eye capture handle (pseyepy.Camera).  Mutually exclusive
        # with the cv2 _cap path; selected via settings.camera_type == "pseye".
        # Typed Any because pseyepy ships no type stubs.
        self._pseye: Any = None
        self._is_pseye: bool = False
        # Last exposure/gain pushed to the PS3 Eye hardware.  The read thread
        # re-applies whenever the (shared) settings values change, so the UI can
        # tune exposure/gain live without reopening the camera.  -1 forces the
        # first apply.
        self._applied_exposure: int = -1
        self._applied_gain: int = -1
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

        # Serializes access to the underlying cv2.VideoCapture object, which is
        # NOT thread-safe.  The grab thread's _cap.read() and any UI-thread
        # _cap.set() (property writes) must be mutually exclusive.
        self._capture_lock = threading.Lock()

        # Grab-thread liveness: consecutive failed reads and a sticky failure
        # flag set once the failure count crosses _GRAB_FAILURE_THRESHOLD.
        self._consecutive_failures = 0
        self._read_failed = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_width, self._frame_height

    @property
    def is_open(self) -> bool:
        if self._is_pseye:
            return self._pseye is not None
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_grab_running(self) -> bool:
        """Whether the grab thread is actively capturing frames."""
        return self._grab_running

    @property
    def is_pseye(self) -> bool:
        """Whether the active source is a PS3 Eye (read inline, no grab thread).

        The processing loop uses this to know that ``read()`` returning None is
        a transient "no fresh frame yet" condition to retry, not a fatal
        disconnect — the same role :attr:`is_grab_running` plays for cv2 sources.
        """
        return self._is_pseye

    @property
    def status_message(self) -> str:
        """Human-readable status of the last open attempt."""
        return self._status_message

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failed grab-thread reads (0 when healthy)."""
        return self._consecutive_failures

    @property
    def is_healthy(self) -> bool:
        """False once the grab thread has hit too many consecutive read failures.

        Used to detect a mid-session camera disconnect.  Becomes True again
        as soon as a single read succeeds (the counter resets).
        """
        return not self._read_failed

    def update_settings(self, settings: CameraSettings) -> None:
        """Replace camera settings and re-apply properties."""
        self._settings = settings
        self._apply_camera_properties()

    def open_webcam(self) -> bool:
        """Open the webcam with configured settings and frame validation.

        Strategy: open with MSMF first to validate frames and configure
        camera properties (MSMF preserves Kiyo Pro firmware state).  Then
        attempt to upgrade to DirectShow for higher FPS (~60 vs ~30).
        DirectShow resets some cameras on first open, so the MSMF-first
        approach ensures the firmware is configured before the switch.

        Returns:
            True if camera opened successfully and producing frames.
        """
        s = self._settings
        self._video_file = False

        # Clear env var that may have been set by a previous failed attempt —
        # OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 can disable the entire
        # MSMF backend on some OpenCV builds.
        import os
        os.environ.pop("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", None)

        # Try MJPEG/DirectShow first if configured
        if s.mjpeg and self._try_open_mjpeg():
                if self._validate_frames():
                    self._read_properties()
                    self._apply_camera_properties()
                    self._status_message = (
                        f"Connected (MJPEG {self._frame_width}x{self._frame_height}"
                        f" @ {self._fps:.0f}fps)"
                    )
                    logger.info("Camera ready: %s", self._status_message)
                    return True
                else:
                    logger.warning(
                        "MJPEG opened but frames are black or absent, "
                        "falling back to default backend"
                    )
                    self._release_cap()

        # Open with MSMF (default) — validates frames and configures firmware
        if not self._try_open_default():
            self._status_message = "Failed to open camera"
            logger.error(self._status_message)
            return False

        if not self._validate_frames():
            logger.error("Default backend opened but no frames arrived")
            self._release_cap()
            self._status_message = "Failed to open camera"
            logger.error(self._status_message)
            return False

        self._read_properties()
        self._apply_camera_properties()

        # NOTE: We used to try a DirectShow upgrade here for higher FPS.
        # Disabled — DirectShow resets the Razer Kiyo Pro firmware, causing
        # black frames (per CLAUDE.md).  It also wastes ~10-20s of startup
        # time, AND has been observed to leave MSMF in a state where the
        # subsequent fallback re-open ends up bound to a different device
        # (e.g., the laptop's integrated webcam at the same MSMF slot).
        # Keep MSMF only.  If DirectShow upgrade ever becomes useful for
        # a non-Kiyo camera, add a config flag to re-enable it.
        self._status_message = (
            f"Connected (MSMF {self._frame_width}x{self._frame_height}"
            f" @ {self._fps:.0f}fps)"
        )

        logger.info("Camera ready: %s", self._status_message)
        return True

    def _try_open_mjpeg(self) -> bool:
        """Try opening the camera with MJPEG codec via DirectShow.

        Returns:
            True if the capture device opened and accepted MJPEG codec
            (frames not yet validated).
        """
        s = self._settings
        self._cap = cv2.VideoCapture(s.webcam_index + cv2.CAP_DSHOW)
        if self._cap is None or not self._cap.isOpened():
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
        """Try opening the camera with the default backend (MSMF on Windows).

        DirectShow cannot be used here because it resets the Razer Kiyo Pro
        firmware settings (exposure/white balance), causing black frames.
        MSMF preserves firmware state.  The grab thread runs a Windows
        message pump to prevent MSMF from throttling when backgrounded.

        Returns:
            True if the capture device opened (frames not yet validated).
        """
        s = self._settings

        # Try to disable MSMF D3D11 GPU acceleration to prevent throttling
        # when GPU-intensive apps (GSPro) are in the foreground.
        # Strategy: try params constructor first, then post-open set, then default.
        self._cap = None
        try:
            self._cap = cv2.VideoCapture(
                s.webcam_index,
                cv2.CAP_ANY,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE],
            )
            if self._cap is not None and self._cap.isOpened():
                logger.info("Camera opened with HW_ACCELERATION_NONE (CPU-only)")
            else:
                self._cap = None
        except Exception:
            self._cap = None

        if self._cap is None:
            self._cap = cv2.VideoCapture(s.webcam_index)
            if self._cap is None or not self._cap.isOpened():
                logger.error("Failed to open camera %d with default backend", s.webcam_index)
                return False
            # Try to disable HW acceleration after open
            self._cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            logger.info("Camera opened with default backend (set HW_ACCEL_NONE post-open)")

        res_w = s.width if s.width > 0 else 1280
        res_h = s.height if s.height > 0 else 720
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

        target_fps = s.fps_override if s.fps_override > 0 else 60
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        logger.info(
            "Default backend camera (CPU-only): %dx%d @ %d fps requested",
            res_w, res_h, target_fps,
        )
        return True

    def _try_upgrade_to_dshow(self) -> bool:
        """Attempt to switch from MSMF to DirectShow for higher FPS.

        Releases the current MSMF capture, opens DirectShow, and validates
        that frames still arrive.  If DirectShow fails, reopens MSMF as
        fallback.  Camera properties are re-applied after the switch.

        Returns:
            True if successfully upgraded to DirectShow.
        """
        s = self._settings
        res_w = s.width if s.width > 0 else 1280
        res_h = s.height if s.height > 0 else 720
        target_fps = s.fps_override if s.fps_override > 0 else 60

        logger.info("Attempting DirectShow upgrade for higher FPS...")
        self._release_cap()

        cap = cv2.VideoCapture(s.webcam_index + cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            logger.warning("DirectShow failed to open, falling back to MSMF")
            return self._fallback_to_msmf()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._apply_camera_properties()

        # Extended warmup — DirectShow may need time after firmware was
        # configured by MSMF.  Read up to 30 frames looking for non-black.
        for i in range(30):
            ret, frame = self._cap.read()
            if ret and frame is not None and not self._is_black_frame(frame):
                actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
                logger.info(
                    "DirectShow upgrade succeeded at frame %d (mean=%.1f, fps=%.0f)",
                    i, float(np.mean(frame)), actual_fps,
                )
                self._read_properties()
                return True
            time.sleep(0.05)

        # DirectShow gave only black frames — fall back
        logger.warning("DirectShow produced black frames, falling back to MSMF")
        self._release_cap()
        return self._fallback_to_msmf()

    def _fallback_to_msmf(self) -> bool:
        """Reopen camera with default MSMF backend after DirectShow failure."""
        if self._try_open_default():
            # Brief warmup — MSMF was just working, should be quick
            for _ in range(5):
                self._cap.read()  # type: ignore[union-attr]
                time.sleep(0.05)
            self._read_properties()
            self._apply_camera_properties()
            return False
        logger.error("MSMF fallback also failed")
        return False

    @staticmethod
    def _is_black_frame(frame: np.ndarray) -> bool:
        """Check if a frame is effectively black (all/nearly-all zero pixels)."""
        return float(np.mean(frame)) < _BLACK_FRAME_THRESHOLD

    def _validate_frames(self) -> bool:
        """Read test frames to confirm the camera is producing usable data.

        Phase 1: Discard warmup frames (auto-exposure settling).
        Phase 2: Validate that at least one frame is non-black.

        Returns:
            True if at least one non-black frame was successfully read.
        """
        if self._cap is None:
            return False

        # Phase 1: Warmup — read and discard frames for auto-exposure
        for i in range(1, _WARMUP_FRAMES + 1):
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.debug("Warmup frame %d: no frame", i)
            time.sleep(_WARMUP_DELAY)

        # Phase 2: Validate — require at least one non-black frame
        for attempt in range(1, _FRAME_VALIDATE_ATTEMPTS + 1):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if not self._is_black_frame(frame):
                    logger.debug(
                        "Frame validation passed on attempt %d (mean=%.1f)",
                        attempt, float(np.mean(frame)),
                    )
                    return True
                logger.debug(
                    "Frame validation attempt %d: black frame (mean=%.1f)",
                    attempt, float(np.mean(frame)),
                )
            else:
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

    def open_pseye(self) -> bool:
        """Open a Sony PS3 Eye camera via the ``pseyepy`` library.

        pseyepy manages its own asynchronous USB capture internally.  Unlike
        the cv2 path there is NO grab thread: the processing loop reads the
        camera inline via :meth:`read` -> :meth:`_read_pseye_frame` because
        pseyepy's ``read()`` blocks while holding the Python GIL (see
        :meth:`_read_pseye_frame` and the note below).  None of the
        cv2/MSMF/DirectShow machinery applies.

        Resolution maps "qvga" -> 320x240 (RES_SMALL, ~96fps capable) and
        "vga" -> 640x480 (RES_LARGE).  pseyepy returns RGB frames; the inline
        read converts them to BGR when ``pseye_swap_rb`` is set.

        Returns:
            True if the PS3 Eye opened (and a first frame was primed); False on
            any failure (library missing, no device), with a status message.
        """
        s = self._settings
        self._video_file = False
        try:
            from pseyepy import Camera as PSEyeCamera  # type: ignore[import-untyped]

            if s.pseye_resolution.lower() == "vga":
                resolution = PSEyeCamera.RES_LARGE
                width, height = 640, 480
            else:
                resolution = PSEyeCamera.RES_SMALL
                width, height = 320, 240

            self._pseye = PSEyeCamera(
                fps=s.pseye_fps,
                resolution=resolution,
                colour=True,
                exposure=s.pseye_exposure,
                gain=s.pseye_gain,
            )
            self._is_pseye = True
            self._frame_width = width
            self._frame_height = height
            self._fps = float(s.pseye_fps)

            # Warm up: discard a couple of frames so the async USB pipeline
            # has settled before processing begins.
            for _ in range(2):
                self._pseye.read()

            # NOTE: deliberately NO grab thread for the PS3 Eye.  pseyepy's
            # read() (Cython ps3eye_grab_frame -> FrameQueue::Dequeue) blocks on
            # a C++ condition variable for the entire ~16ms inter-frame wait
            # while STILL HOLDING the Python GIL (the call is not wrapped in
            # `with nogil`).  On a dedicated grab thread that monopolizes the GIL
            # and starves the processing + UI threads down to ~4fps.  Instead the
            # processing loop reads the PS3 Eye inline via read() ->
            # _read_pseye_frame(): with a single compute thread there is no GIL
            # fight and the blocking read simply paces the loop at camera rate.
            # The driver's 2-slot ring buffer drops stale frames gracefully, so
            # latency stays <= ~1 frame even if processing dips below 60fps.
            self._pseye_log_count = 0
            self._pseye_log_t0 = time.perf_counter()

            # Prime _latest_frame so one-shot calibration captures
            # (read_latest) work immediately, before the processing loop has
            # taken its first inline read.
            self._read_pseye_frame()

            self._status_message = (
                f"Connected (PS3 Eye {width}x{height} @ {s.pseye_fps}fps)"
            )
            logger.info("Camera ready: %s", self._status_message)
            return True
        except Exception:
            logger.exception("Failed to open PS3 Eye via pseyepy")
            self._pseye = None
            self._is_pseye = False
            self._status_message = "Failed to open PS3 Eye (pseyepy)"
            return False

    def start_grab_thread(self) -> None:
        """Start a dedicated thread that continuously reads frames.

        Reuses the existing VideoCapture opened on the main thread rather
        than reopening it.  Reopening resets camera firmware settings
        (Razer Kiyo Pro loses Synapse exposure config → black frames).
        DirectShow does not require COM apartment ownership for reads,
        so cross-thread access is safe.
        """
        # The PS3 Eye is read inline on the processing thread (see open_pseye):
        # never spin up a grab thread for it, or two threads would race on
        # pseyepy.read() and fight over the GIL.
        if self._video_file or self._grab_running or self._is_pseye:
            return

        self._grab_running = True
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="camera-grab",
        )
        self._grab_thread.start()
        logger.info("Camera grab thread started")

    def _grab_loop(self) -> None:
        """Continuously read frames from the camera into _latest_frame.

        Reuses the VideoCapture already opened on the main thread.
        Reopening would reset Razer Kiyo Pro firmware → black frames.

        NOTE: The MSMF message pump runs on the MAIN thread (via
        PuttingApp._start_msmf_pump), not here.  MSMF's internal windows
        are associated with the thread that opened the VideoCapture (main),
        so pumping on this background thread would have no effect.
        """
        import sys

        # Aggressively prevent Windows from throttling this thread when
        # the app is backgrounded (user clicks GSPro).
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.SetThreadPriority(
                    kernel32.GetCurrentThread(), 2,  # THREAD_PRIORITY_HIGHEST
                )
                # Set 1ms timer resolution to prevent background throttling
                ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]

                # Prevent power/idle throttling on this thread
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED,
                )

                # Register with Multimedia Class Scheduler for real-time
                # priority that persists when the app is backgrounded.
                try:
                    avrt = ctypes.windll.avrt  # type: ignore[attr-defined]
                    task_index = ctypes.c_ulong(0)
                    handle = avrt.AvSetMmThreadCharacteristicsW(
                        "Pro Audio", ctypes.byref(task_index),
                    )
                    if handle:
                        logger.info("MMCSS: registered grab thread as 'Pro Audio'")
                    else:
                        logger.debug("MMCSS registration failed")
                except Exception:
                    pass  # avrt.dll not available on all editions
            except Exception:
                pass

        # PS3 Eye (pseyepy) uses a completely different read path — it manages
        # its own async USB capture, so none of the cv2/MSMF code below applies.
        if self._is_pseye:
            self._grab_loop_pseye()
            return

        logger.info("Grab thread reading from existing capture")

        # Discard initial frames so camera properties can settle
        # (auto-exposure, white balance, etc. need a few frames to converge)
        _SETTLE_FRAMES = 15
        for i in range(_SETTLE_FRAMES):
            if not self._grab_running or self._cap is None:
                return
            self._cap.read()
        logger.info("Grab thread: discarded %d settle frames", _SETTLE_FRAMES)

        grab_count = 0
        grab_start = time.perf_counter()
        while self._grab_running and self._cap is not None:
            if self._grab_once():
                grab_count += 1
                now = time.perf_counter()
                if now - grab_start >= 30.0:
                    grab_fps = grab_count / (now - grab_start)
                    logger.info("Grab thread FPS: %.1f", grab_fps)
                    grab_count = 0
                    grab_start = now
            # On failure _grab_once() has already slept a small interval, so
            # the loop does not busy-spin at 100% CPU on a dead camera.

    def _grab_loop_pseye(self) -> None:
        """Grab loop for the PS3 Eye: read pseyepy frames into _latest_frame.

        pseyepy's ``read()`` returns ``(frame, timestamp)`` where ``frame`` is
        an HxWx3 contiguous uint8 RGB array.  We convert RGB->BGR when
        ``pseye_swap_rb`` is set (this app/OpenCV expect BGR), then store the
        frame under ``_frame_lock`` exactly like the cv2 path so ``read()`` /
        ``read_latest()`` work unchanged.  Failure accounting mirrors
        :meth:`_grab_once` (back-off sleep + consecutive-failure liveness flag).
        """
        logger.info("Grab thread reading from PS3 Eye (pseyepy)")
        swap_rb = self._settings.pseye_swap_rb

        grab_count = 0
        grab_start = time.perf_counter()
        while self._grab_running and self._pseye is not None:
            try:
                frame, _ts = self._pseye.read()
            except Exception:
                frame = None

            if frame is not None:
                if swap_rb:
                    # pseyepy returns RGB; OpenCV/this app expect BGR.
                    frame = np.ascontiguousarray(frame[:, :, ::-1])
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_new = True
                if self._consecutive_failures or self._read_failed:
                    self._consecutive_failures = 0
                    self._read_failed = False

                grab_count += 1
                now = time.perf_counter()
                if now - grab_start >= 30.0:
                    logger.info("Grab thread FPS: %.1f", grab_count / (now - grab_start))
                    grab_count = 0
                    grab_start = now
            else:
                self._record_grab_failure()

    def _grab_once(self) -> bool:
        """Perform one capture iteration: read a frame and update shared state.

        Reads under _capture_lock (cv2.VideoCapture is not thread-safe, so this
        serializes against UI-thread property writes) and stores the frame under
        _frame_lock.  Tracks consecutive read failures: each failure sleeps a
        small back-off interval and increments the counter; once the counter
        reaches _GRAB_FAILURE_THRESHOLD the camera is flagged unhealthy
        (self._read_failed) and a warning is logged.  A successful read resets
        the counter and clears the flag.

        Returns:
            True if a frame was read and stored, False otherwise.
        """
        cap = self._cap
        if cap is None:
            self._record_grab_failure()
            return False

        with self._capture_lock:
            if self._cap is None:  # released while waiting for the lock
                self._record_grab_failure()
                return False
            ret, frame = self._cap.read()

        if ret and frame is not None:
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_new = True
            # Healthy again — clear any prior failure state.
            if self._consecutive_failures or self._read_failed:
                self._consecutive_failures = 0
                self._read_failed = False
            return True

        self._record_grab_failure()
        return False

    def _record_grab_failure(self) -> None:
        """Account for a failed grab: back off, count, and flag if persistent."""
        time.sleep(_GRAB_FAILURE_SLEEP)
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= _GRAB_FAILURE_THRESHOLD
            and not self._read_failed
        ):
            self._read_failed = True
            logger.warning(
                "Camera grab failed %d consecutive reads — marking unhealthy "
                "(possible disconnect)",
                self._consecutive_failures,
            )

    def stop_grab_thread(self) -> None:
        """Stop the grab thread."""
        self._grab_running = False
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2)
            self._grab_thread = None

    def _read_pseye_frame(self) -> np.ndarray | None:
        """Read one frame directly from the PS3 Eye on the calling thread.

        Replaces the grab-thread path for pseyepy.  pseyepy's ``read()`` blocks
        for the full inter-frame interval while holding the Python GIL, so it is
        called here on the single processing thread (never a background thread)
        to avoid starving the rest of the app.  The frame is cached in
        ``_latest_frame`` so one-shot calibration captures (:meth:`read_latest`)
        keep working, mirroring the old grab-thread contract.

        Returns:
            BGR frame (RGB->BGR swap applied when ``pseye_swap_rb``), or None
            only on a genuine read failure (device gone / repeated exception).
        """
        if self._pseye is None:
            return None

        # Apply any live exposure/gain changes requested via the UI.  The UI
        # writes pseye_exposure/pseye_gain on the shared settings object; we push
        # them to the OV534 here so ALL camera access stays on this one thread
        # (concurrent libusb control + bulk transfers from two threads risk a
        # native crash).  Cheap int compares when nothing changed.
        if self._settings.pseye_exposure != self._applied_exposure:
            try:
                self._pseye.exposure = int(self._settings.pseye_exposure)
                self._applied_exposure = int(self._settings.pseye_exposure)
                logger.info("PS3 Eye exposure -> %d", self._applied_exposure)
            except Exception:
                logger.exception("Failed to set PS3 Eye exposure")
                self._applied_exposure = int(self._settings.pseye_exposure)
        if self._settings.pseye_gain != self._applied_gain:
            try:
                self._pseye.gain = int(self._settings.pseye_gain)
                self._applied_gain = int(self._settings.pseye_gain)
                logger.info("PS3 Eye gain -> %d", self._applied_gain)
            except Exception:
                logger.exception("Failed to set PS3 Eye gain")
                self._applied_gain = int(self._settings.pseye_gain)

        frame = None
        for _ in range(3):  # tolerate a transient hiccup without stopping the app
            try:
                frame, _ts = self._pseye.read()
            except Exception:
                frame = None
            if frame is not None:
                break
        if frame is None:
            self._record_grab_failure()
            return None

        if self._settings.pseye_swap_rb:
            # pseyepy returns RGB; OpenCV/this app expect BGR.
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        with self._frame_lock:
            self._latest_frame = frame
            self._frame_new = True
        if self._consecutive_failures or self._read_failed:
            self._consecutive_failures = 0
            self._read_failed = False

        # Periodic read-rate log (replaces the old "Grab thread FPS" line so the
        # PS3 Eye's delivered frame rate stays observable in the logs).
        self._pseye_log_count = getattr(self, "_pseye_log_count", 0) + 1
        t0 = getattr(self, "_pseye_log_t0", 0.0)
        now = time.perf_counter()
        if t0 <= 0.0:
            self._pseye_log_t0 = now
        elif now - t0 >= 30.0:
            logger.info(
                "PS3 Eye inline read FPS: %.1f", self._pseye_log_count / (now - t0)
            )
            self._pseye_log_count = 0
            self._pseye_log_t0 = now

        return frame

    def read(self) -> np.ndarray | None:
        """Read a single frame from the capture source.

        If the grab thread is running, returns a COPY of the latest grabbed
        frame (non-blocking) and clears the freshness flag.  Otherwise reads
        directly from the capture device.

        Freshness contract (relied on by the processing loop): when the grab
        thread is running and no NEW frame has arrived since the last call,
        this returns None.  The returned frame is always an independent copy,
        so the in-place operations below (normalize/flip) and any downstream
        processing cannot be corrupted by the grab thread overwriting
        ``_latest_frame`` concurrently.

        Returns:
            BGR frame as a caller-owned numpy array, or None if no fresh frame
            is available / the read failed.
        """
        # PS3 Eye has no cv2 _cap — its frames arrive via inline reads, so
        # only short-circuit on a missing _cap for the (non-pseye) cv2 path.
        if self._cap is None and not self._is_pseye:
            return None

        # PS3 Eye: read inline on the calling (processing) thread — there is no
        # grab thread for it (see open_pseye for the GIL rationale).
        if self._is_pseye:
            frame = self._read_pseye_frame()
            if frame is None:
                return None
            return self._post_process(frame)

        # Use threaded grab if available
        if self._grab_running:
            with self._frame_lock:
                if not self._frame_new or self._latest_frame is None:
                    return None  # No new frame since last read
                # Copy under the lock so the grab thread cannot mutate the
                # buffer (or rebind _latest_frame) out from under us.
                frame = self._latest_frame.copy()
                self._frame_new = False
        else:
            with self._capture_lock:
                if self._cap is None:
                    return None
                ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

        return self._post_process(frame)

    def read_latest(self) -> np.ndarray | None:
        """Return a COPY of the most-recently captured frame, ignoring freshness.

        Unlike :meth:`read`, this does NOT consult or clear the ``_frame_new``
        freshness flag — it always hands back the last frame the grab thread
        stored (or, when the grab thread is not running, reads one directly).
        It is the primitive used by one-shot UI calibration captures, which
        must not spuriously get None just because the processing thread already
        consumed the latest frame.

        Returns:
            A caller-owned BGR frame, or None only if no frame has ever been
            captured / the camera is not open.
        """
        if self._grab_running:
            # Grab thread owns _cap.read(); never issue a competing direct read
            # from this (calibration) thread.  Hand back the last stored frame,
            # or None if the grab thread hasn't produced one yet.
            with self._frame_lock:
                if self._latest_frame is None:
                    return None
                frame = self._latest_frame.copy()
        elif self._latest_frame is not None:
            # Grab thread idle but a frame was previously captured — reuse it.
            with self._frame_lock:
                frame = self._latest_frame.copy()
        else:
            # No grab thread and nothing captured yet: read once directly
            # (e.g. video-file mode, or a one-shot before the grab thread runs).
            if self._cap is None:
                return None
            with self._capture_lock:
                if self._cap is None:
                    return None
                ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

        # PS4 Eye frame decoding
        if self._settings.ps4 and not self._video_file:
            frame = self._decode_ps4(frame)

        return self._post_process(frame)

    def _post_process(self, frame: np.ndarray) -> np.ndarray:
        """Apply darkness normalization and configured flip to ``frame``.

        Operates in place where possible; the caller must already own
        ``frame`` (it is always a copy in :meth:`read`/:meth:`read_latest`).

        NOTE: Rotation is applied AFTER resize in the processing loop to avoid
        warpAffine on the full 1280x720 frame (~4x slower), so it is not done
        here.
        """
        # Apply darkness normalization
        if self._settings.darkness > 0:
            d = self._settings.darkness
            cv2.normalize(frame, frame, 0 - d, 255 - d, norm_type=cv2.NORM_MINMAX)

        # Flip if configured. flip_image = horizontal (code 1), flip_vertical =
        # vertical (code 0); both together == a 180° rotation (use when the
        # camera is mounted inverted, e.g. on a different port/orientation).
        if not self._video_file:
            if self._settings.flip_image:
                frame = cv2.flip(frame, 1)
            if self._settings.flip_vertical:
                frame = cv2.flip(frame, 0)

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
        """Release the capture source (cv2 webcam/video or PS3 Eye).

        Idempotent: safe to call when nothing is open or after a prior release.
        """
        self.stop_grab_thread()
        if self._is_pseye:
            if self._pseye is not None:
                try:
                    self._pseye.end()
                except Exception:
                    logger.exception("Error closing PS3 Eye")
                self._pseye = None
                logger.info("PS3 Eye released")
            self._is_pseye = False
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def apply_properties(self) -> None:
        """Re-apply all camera properties from current settings."""
        self._apply_camera_properties()

    def _apply_camera_properties(self) -> None:
        """Apply all configured camera properties.

        cv2.VideoCapture is not thread-safe, so the _cap.set() writes are
        serialized against the grab thread's _cap.read() via _capture_lock.
        We chose a dedicated lock over pausing/joining the grab thread because
        stopping the thread would force re-settling and, on the Razer Kiyo Pro,
        risk a firmware reset; the lock is far lower-risk and the property
        writes are quick.
        """
        if self._cap is None:
            return

        with self._capture_lock:
            if self._cap is None:  # released while waiting for the lock
                return
            # Auto-exposure MODE must be applied BEFORE the exposure VALUE.  On
            # most UVC/DirectShow cameras, setting CAP_PROP_EXPOSURE while still
            # in auto mode is silently ignored — and _CAMERA_PROPS happens to
            # list "exposure" before "auto_exposure".  Apply the mode first.
            ae = getattr(self._settings, "auto_exposure", 0.0)
            if ae != 0.0:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae)
                logger.debug("Set camera auto_exposure = %s (before exposure)", ae)
            for field_name, prop_id in _CAMERA_PROPS.items():
                if field_name == "auto_exposure":
                    continue  # already applied above, before exposure
                value = getattr(self._settings, field_name, 0.0)
                if value != 0.0:
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
