"""Camera capture abstraction wrapping OpenCV VideoCapture."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from webcam_putting.config import CameraSettings

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


class Camera:
    """Manages video capture from webcam or video file.

    Handles MJPEG codec, FPS override, resolution, PS4 Eye decoding,
    and camera property management.
    """

    def __init__(self, settings: CameraSettings):
        self._settings = settings
        self._cap: cv2.VideoCapture | None = None
        self._video_file: bool = False
        self._fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_width, self._frame_height

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open_webcam(self) -> bool:
        """Open the webcam with configured settings.

        Returns:
            True if camera opened successfully.
        """
        s = self._settings
        self._video_file = False

        if s.mjpeg:
            self._cap = cv2.VideoCapture(s.webcam_index + cv2.CAP_DSHOW)
            if self._cap is None or not self._cap.isOpened():
                logger.error("Failed to open camera %d with DirectShow", s.webcam_index)
                return False

            # DirectShow requires: resolution → codec → FPS (in this order)
            # Default to 720p @ 60fps for MJPEG when not explicitly configured
            res_w = s.width if s.width > 0 else 1280
            res_h = s.height if s.height > 0 else 720
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # type: ignore[attr-defined]
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            target_fps = s.fps_override if s.fps_override > 0 else 60
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
            logger.info("MJPEG camera: %dx%d @ %d fps requested", res_w, res_h, target_fps)

            # Minimize frame buffer latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap = cv2.VideoCapture(s.webcam_index)
            if self._cap is None or not self._cap.isOpened():
                logger.error("Failed to open camera %d", s.webcam_index)
                return False

        # PS4 Eye special settings
        if s.ps4:
            self._cap.set(cv2.CAP_PROP_FPS, 120)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1724)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 404)

        # Read actual properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle FPS detection failure
        if self._fps == 0.0:
            self._cap.set(cv2.CAP_PROP_FPS, 60)
            self._fps = 60.0
            logger.warning("FPS detection returned 0, defaulting to 60")

        logger.info(
            "Camera opened: %dx%d @ %.1f fps (backend=%s)",
            self._frame_width, self._frame_height, self._fps,
            self._cap.get(cv2.CAP_PROP_BACKEND),
        )

        # Apply camera properties
        self._apply_camera_properties()

        return True

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
