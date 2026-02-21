"""Camera capture abstraction wrapping OpenCV VideoCapture."""

from __future__ import annotations

import logging
import time
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

_WARMUP_FRAMES = 5            # frames to discard before brightness checking
_WARMUP_DELAY = 0.1           # seconds between warmup reads
_FRAME_VALIDATE_ATTEMPTS = 3  # frames to check after warmup
_FRAME_VALIDATE_DELAY = 0.1   # seconds between validation reads
_BLACK_FRAME_THRESHOLD = 3.0  # mean brightness below this = black frame


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

        Tries MJPEG/DirectShow first (if configured), validates that frames
        actually arrive, and falls back to the default backend on failure.

        Returns:
            True if camera opened successfully and producing frames.
        """
        s = self._settings
        self._video_file = False

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

        # Fallback (or primary if MJPEG not configured)
        if self._try_open_default():
            if self._validate_frames():
                self._read_properties()
                self._apply_camera_properties()
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
        """Try opening the camera with the default backend (no DirectShow/MJPEG).

        Applies configured resolution and FPS so the fallback path still
        achieves the desired capture settings (e.g. 1280x720 @ 60fps).

        Returns:
            True if the capture device opened (frames not yet validated).
        """
        s = self._settings
        self._cap = cv2.VideoCapture(s.webcam_index)
        if self._cap is None or not self._cap.isOpened():
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

    def read(self) -> np.ndarray | None:
        """Read a single frame from the capture source.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        if self._cap is None:
            return None

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

        return frame

    def release(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def apply_properties(self) -> None:
        """Re-apply all camera properties from current settings."""
        self._apply_camera_properties()

    def _apply_camera_properties(self) -> None:
        """Apply all configured camera properties."""
        if self._cap is None:
            return

        for field_name, prop_id in _CAMERA_PROPS.items():
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
