"""Application orchestrator — wires camera, detection, tracking, physics, and GSPro client.

Supports two display modes:
- GUI mode (default): CustomTkinter window with threaded frame processing
- Headless mode (--no-gui): OpenCV windows, original synchronous loop
"""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from birdman_putting.calibration import AutoCalibrator, CalibrationState
from birdman_putting.camera import Camera
from birdman_putting.color_presets import HSVRange, get_preset
from birdman_putting.config import AppConfig, save_config
from birdman_putting.detection import (
    BallDetector,
    generate_hsv_from_patch,
    resize_with_aspect_ratio,
)
from birdman_putting.gspro_client import GSProClient
from birdman_putting.physics import calculate_shot, estimate_putt_distance_feet
from birdman_putting.tracking import BallTracker, ShotState
from birdman_putting.ui.overlay import (
    draw_calibration_overlay,
    draw_overlay,
    project_trail,
)

if TYPE_CHECKING:
    from birdman_putting.mevo.detector import MevoDetector
    from birdman_putting.obs_controller import OBSController
    from birdman_putting.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class PuttingApp:
    """Main application coordinating all subsystems.

    In GUI mode, frame processing runs on a background thread and feeds
    annotated frames into a queue that the CustomTkinter window polls.
    In headless mode, everything runs synchronously on the main thread.
    """

    def __init__(
        self,
        config: AppConfig,
        video_path: str | None = None,
        debug: bool = False,
        headless: bool = False,
    ):
        self.config = config
        self._video_path = video_path
        self._debug = debug
        self._headless = headless

        # Resolve HSV range
        if config.ball.custom_hsv:
            self._hsv_range = HSVRange.from_dict(config.ball.custom_hsv)
        else:
            self._hsv_range = get_preset(config.ball.color_preset)

        # Initialize subsystems
        self._camera = Camera(config.camera)
        self._detector = BallDetector(
            hsv_range=self._hsv_range,
            min_radius=config.ball.min_radius,
            min_circularity=config.ball.min_circularity,
            morph_iterations=config.ball.morph_iterations,
        )
        self._tracker = BallTracker(
            zone=config.detection_zone,
            ball_settings=config.ball,
            shot_settings=config.shot,
            max_trail_points=config.overlay.max_trail_points,
        )
        self._gspro = GSProClient(config.connection, on_club_change=self._on_club_change)

        # FPS tracking
        self._fps_queue: deque[float] = deque(maxlen=30)
        self._actual_fps: float = 0.0

        # Adaptive frame skipping
        self._skip_counter: int = 0
        self._target_process_time: float = 1.0 / 30.0  # target 30fps processing
        self._fps_drop_start: float = 0.0  # Timestamp when FPS first dropped below 10

        # Headless color pick mode
        self._pick_mode = False
        self._pick_frame: np.ndarray | None = None

        # Threading
        self._running = False
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)  # type: ignore[type-arg]
        self._processing_thread: threading.Thread | None = None

        # Calibration
        self._calibrator: AutoCalibrator | None = None
        self._calibrating: bool = False

        # Mevo thread
        self._mevo_detector: MevoDetector | None = None
        self._mevo_thread: threading.Thread | None = None
        self._mevo_paused: bool = False  # Pause OCR during putting to free CPU

        # OBS controller
        self._obs: OBSController | None = None

        # Post-shot trail tracking: continue detecting ball after shot for trail
        self._post_shot_tracking = False
        self._post_shot_deadline: float = 0.0
        self._post_shot_radius: int = 0
        self._trail_clear_time: float = 0.0  # When to auto-clear the last-shot trail
        self._last_state_change: float = 0.0  # For stuck-state auto-reset

        # Angle calibration: measure HLA of a "straight" putt to auto-set rotation
        self._angle_cal_active = False
        self._angle_cal_samples: list[float] = []
        self._angle_cal_bg: np.ndarray | None = None  # Background frame for subtraction

        # GUI reference (set when run_gui is called)
        self._window: MainWindow | None = None

    def run(self) -> None:
        """Start the application in the appropriate mode."""
        self._set_high_priority()
        if self._headless:
            self._run_headless()
        else:
            self._run_gui()

    @staticmethod
    def _set_high_priority() -> None:
        """Set process priority to above-normal to prevent Windows throttling.

        When the Birdman window loses focus (e.g. user clicks GSPro),
        Windows deprioritizes background processes, starving the camera
        capture and processing threads.
        """
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            HIGH_PRIORITY_CLASS = 0x00000080
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
            logger.info("Process priority set to HIGH")

            # Disable Windows 11 EcoQoS / power throttling for this process
            try:
                import ctypes.wintypes as wt

                class POWER_THROTTLING_STATE(ctypes.Structure):
                    _fields_ = [
                        ("Version", wt.DWORD),
                        ("ControlMask", wt.DWORD),
                        ("StateMask", wt.DWORD),
                    ]

                ProcessPowerThrottling = 4
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1

                state = POWER_THROTTLING_STATE()
                state.Version = 1
                state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
                state.StateMask = 0  # 0 = HighQoS (disable throttling)

                # Use OpenProcess with explicit access rights
                pid = kernel32.GetCurrentProcessId()
                PROCESS_SET_INFORMATION = 0x0200
                h = kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid)
                if h:
                    result = kernel32.SetProcessInformation(
                        h, ProcessPowerThrottling,
                        ctypes.byref(state), ctypes.sizeof(state),
                    )
                    kernel32.CloseHandle(h)
                    if result:
                        logger.info("EcoQoS power throttling disabled")
                    else:
                        logger.debug("SetProcessInformation returned False")
            except Exception:
                pass  # Older Windows versions don't support this
        except Exception as e:
            logger.debug("Could not set process priority: %s", e)

    # ---- GUI Mode ----

    def _run_gui(self) -> None:
        """Start with CustomTkinter GUI on main thread, processing on background thread."""
        from birdman_putting.ui.main_window import MainWindow

        self._window = MainWindow(
            config=self.config,
            frame_queue=self._frame_queue,
            on_start=self._on_gui_start,
            on_stop=self._on_gui_stop,
            on_color_change=self._on_color_change,
            on_settings_changed=self._on_settings_changed,
            on_auto_zone=self._on_auto_zone,
            on_reset_putt=self.reset_putt,
            on_angle_cal=self._on_angle_cal,
        )

        # Auto-start if camera/video is available
        self._on_gui_start()

        # Run the tkinter main loop (blocks until window closes)
        self._window.mainloop()

        # Cleanup after window closes
        self._stop_processing()
        self._cleanup()

    def _on_gui_start(self) -> None:
        """Start camera capture and processing thread."""
        if self._running:
            return

        # Open camera or video
        if self._video_path:
            if not self._camera.open_video(self._video_path):
                logger.error("Failed to open video: %s", self._video_path)
                if self._window:
                    self._window.update_camera_status("Failed to open video", "error")
                return
        else:
            # Open on the main thread (which has COM from tkinter).
            # The DirectShow attempt takes ~30s and fails, but it
            # initializes the camera subsystem so the default backend works.
            logger.debug("Opening webcam (index=%d, mjpeg=%s)",
                         self.config.camera.webcam_index, self.config.camera.mjpeg)
            if not self._camera.open_webcam():
                logger.error("Failed to open webcam")
                if self._window:
                    self._window.update_camera_status(
                        self._camera.status_message, "error"
                    )
                return
            if self._window:
                self._window.update_camera_status(
                    self._camera.status_message, "ok"
                )

        # Start threaded camera capture (decoupled from tkinter event loop)
        self._camera.start_grab_thread()

        # Connect to GSPro
        connected = self._gspro.connect()
        if self._window:
            self._window.update_connection_status(connected)

        self._running = True

        # Connect to OBS if enabled
        self._start_obs()

        # Start Mevo thread if enabled
        self._start_mevo()

        # Start video display polling
        if self._window:
            self._window.start_video()

        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="processing"
        )
        self._processing_thread.start()

        logger.info("Processing started")

    def _on_gui_stop(self) -> None:
        """Stop processing and release camera."""
        self._stop_processing()
        if self._window:
            self._window.stop_video()

    def _on_color_change(self, preset_name: str) -> None:
        """Handle ball color preset change from UI."""
        self._hsv_range = get_preset(preset_name)
        self._detector.update_hsv(self._hsv_range)
        logger.info("HSV range updated to preset: %s", preset_name)

    def _on_settings_changed(self) -> None:
        """Handle settings dialog close — re-apply changed settings."""
        # Update detection zone
        self._tracker.zone = self.config.detection_zone

        # Update ball settings
        if self.config.ball.custom_hsv:
            self._hsv_range = HSVRange.from_dict(self.config.ball.custom_hsv)
        else:
            self._hsv_range = get_preset(self.config.ball.color_preset)
        self._detector.update_hsv(self._hsv_range)

        # Update camera settings and re-apply properties (focus, exposure, etc.)
        self._camera._settings = self.config.camera
        self._camera.apply_properties()

        logger.info("Settings reloaded")

    def reset_putt(self) -> None:
        """Force-reset the putt tracker to IDLE state and clear trail."""
        self._tracker.reset()
        self._tracker.last_shot_positions.clear()
        self._post_shot_tracking = False
        logger.info("Putt tracker manually reset")

    def _on_auto_zone(self) -> None:
        """Handle Auto Zone button — toggle calibration mode."""
        if self._calibrating:
            # Cancel calibration
            if self._calibrator:
                self._calibrator.cancel()
            self._calibrating = False
            if self._window:
                self._window.set_auto_zone_state(False)
            logger.info("Auto-calibration cancelled")
        else:
            # Start calibration
            direction = self.config.detection_zone.direction
            self._calibrator = AutoCalibrator(direction=direction)
            self._calibrator.start()
            self._calibrating = True
            self._tracker.reset()
            if self._window:
                self._window.set_auto_zone_state(True)
            logger.info("Auto-calibration started (direction=%s)", direction)

    def _on_angle_cal(self) -> None:
        """Handle Angle Cal button — toggle angle calibration mode.

        Captures a background frame (without the projected line) so that
        background subtraction can isolate the line from carpet/ambient light.
        """
        if self._angle_cal_active:
            self._angle_cal_active = False
            self._angle_cal_samples.clear()
            self._angle_cal_bg = None
            if self._window:
                self._window.set_angle_cal_state(False)
            logger.info("Angle calibration cancelled")
        else:
            # Capture current frame as background (line should NOT be projected yet).
            # Retry briefly — grab thread may not have a new frame ready yet.
            frame = None
            for _ in range(50):
                frame = self._camera.read()
                if frame is not None:
                    break
                time.sleep(0.02)
            if frame is not None:
                bg = resize_with_aspect_ratio(frame, width=640)
                self._angle_cal_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                logger.info("Angle cal: background frame captured")
            else:
                self._angle_cal_bg = None

            self._angle_cal_active = True
            self._angle_cal_samples.clear()
            self._tracker.reset()
            if self._window:
                self._window.set_angle_cal_state(True)
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_camera_status,
                        "Now project a straight line...", "watching",
                    )
            logger.info(
                "Angle calibration started — now project a straight white line "
                "from putting position through the gateway"
            )

    @staticmethod
    def _detect_line_angle(
        frame: np.ndarray,
        bg_gray: np.ndarray | None = None,
    ) -> float | None:
        """Detect the dominant straight line in the frame and return its angle.

        When bg_gray is provided, uses background subtraction to isolate the
        projected line from carpet, ambient light, and other bright surfaces.
        Falls back to absolute brightness thresholding when no background
        frame is available.

        Returns the angle in degrees of the longest detected line, or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if bg_gray is not None and bg_gray.shape == gray.shape:
            # Background subtraction: only pixels brighter than the background
            diff = cv2.subtract(gray, bg_gray)
            # Threshold the difference — the projected line adds ~50+ brightness
            _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        else:
            # Fallback: absolute brightness threshold
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Detect edges
        edges = cv2.Canny(mask, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=80, maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return None

        # Find the longest line
        best_len = 0.0
        best_angle = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > best_len:
                best_len = length
                # atan2(-dy, dx) with inverted Y for screen coords
                best_angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1))

        logger.debug(
            "Line detected: angle=%.2f°, length=%.0fpx", best_angle, best_len,
        )
        return best_angle

    def _process_angle_cal_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame during angle calibration — detect line, draw overlay.

        Accumulates angle samples over multiple frames and auto-applies
        when enough consistent readings are collected.
        """
        display = frame.copy()
        angle = self._detect_line_angle(frame, bg_gray=self._angle_cal_bg)

        h, w = display.shape[:2]
        if angle is not None:
            self._angle_cal_samples.append(angle)

            # Draw the detected line direction on the frame
            cx, cy = w // 2, h // 2
            length = min(w, h) // 3
            rad = math.radians(angle)
            x2 = int(cx + length * math.cos(rad))
            y2 = int(cy - length * math.sin(rad))  # invert Y
            cv2.line(display, (cx, cy), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

            n = len(self._angle_cal_samples)
            avg = sum(self._angle_cal_samples) / n
            status = f"ANGLE CAL: {angle:+.1f}° (avg {avg:+.1f}°, {n} samples)"
            cv2.putText(
                display, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )

            # Apply after 30 consistent frames (~0.5s at 60fps)
            if n >= 30:
                self._apply_angle_cal()
        else:
            cv2.putText(
                display, "ANGLE CAL: No line detected — project a white line",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_AA,
            )
            # Clear stale samples if line is lost
            self._angle_cal_samples.clear()

        # Green border to indicate calibration mode
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
        return display

    def _apply_angle_cal(self) -> None:
        """Apply angle calibration: adjust camera rotation to zero out line angle."""
        if not self._angle_cal_samples:
            return

        avg_angle = sum(self._angle_cal_samples) / len(self._angle_cal_samples)
        # A perfectly horizontal line should be 0°. The deviation IS the
        # camera rotation error.
        old_rotation = self.config.camera.rotation
        new_rotation = old_rotation - avg_angle
        new_rotation = max(-45.0, min(45.0, new_rotation))

        self.config.camera.rotation = new_rotation
        self._angle_cal_active = False
        self._angle_cal_samples.clear()
        self._angle_cal_bg = None

        save_config(self.config)
        logger.info(
            "Angle calibration applied: rotation %.1f° → %.1f° "
            "(line angle was %.2f°)",
            old_rotation, new_rotation, avg_angle,
        )

        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.set_angle_cal_state(False)
                self._window.after(
                    0, self._window.update_camera_status,
                    f"Cal done: rot={new_rotation:.1f}°", "ok",
                )

    def _processing_loop(self) -> None:
        """Background thread: capture → detect → track → annotate → queue."""
        # Raise this thread's priority so camera capture isn't starved
        # when Birdman is in the background
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                THREAD_PRIORITY_HIGHEST = 2
                handle = kernel32.GetCurrentThread()
                kernel32.SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)
                logger.info("Processing thread priority set to HIGHEST")
            except Exception:
                pass

        zone = self.config.detection_zone
        last_ui_update = 0.0

        while self._running:
            # Read frame
            frame = self._camera.read()
            if frame is None:
                if self._camera._grab_running:
                    # Threaded grab: no new frame yet, wait briefly and retry
                    time.sleep(0.001)
                    continue
                logger.warning("No frame received, stopping")
                self._running = False
                break

            frame_time = time.perf_counter()
            self._fps_queue.append(frame_time)

            # Calculate FPS (only counts actual new frames)
            if len(self._fps_queue) >= 2:
                elapsed = self._fps_queue[-1] - self._fps_queue[0]
                if elapsed > 0:
                    self._actual_fps = (len(self._fps_queue) - 1) / elapsed

            # FPS watchdog: auto-reset if FPS stays critically low
            if self._actual_fps > 0 and self._actual_fps < 10:
                if self._fps_drop_start == 0.0:
                    self._fps_drop_start = frame_time
                elif frame_time - self._fps_drop_start > 2.0:
                    logger.warning(
                        "FPS watchdog: %.1f FPS for >2s — auto-resetting",
                        self._actual_fps,
                    )
                    self._tracker.reset()
                    self._tracker.last_shot_positions.clear()
                    self._post_shot_tracking = False
                    self._trail_clear_time = 0.0
                    self._fps_drop_start = 0.0
            else:
                self._fps_drop_start = 0.0

            # Adaptive frame skipping: skip processing but keep capturing
            if self._skip_counter > 0:
                self._skip_counter -= 1
                continue

            # Resize for processing, then apply rotation on the smaller frame
            t0 = time.perf_counter()
            display_frame = resize_with_aspect_ratio(frame, width=640)
            display_frame = self._camera.apply_rotation(display_frame)

            # --- Angle calibration mode (line detection) ---
            if self._angle_cal_active:
                display_frame = self._process_angle_cal_frame(display_frame)
                try:
                    self._frame_queue.put_nowait(display_frame)
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        self._frame_queue.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self._frame_queue.put_nowait(display_frame)
                continue

            # --- Calibration mode ---
            if self._calibrating and self._calibrator:
                cal_detection = self._detector.detect_full_frame(
                    display_frame, timestamp=frame_time,
                )
                dh, dw = display_frame.shape[:2]
                cal_result = self._calibrator.update(cal_detection, dw, dh)

                # Draw calibration overlay
                state_text = f"AUTO ZONE: {self._calibrator.state.value}"
                ball_pos = (cal_detection.x, cal_detection.y) if cal_detection else None
                draw_calibration_overlay(display_frame, state_text, ball_pos)

                if cal_result is not None:
                    # Calibration complete — apply zone
                    self.config.detection_zone = cal_result.zone
                    self._tracker.zone = cal_result.zone
                    self._tracker.reset()
                    self._calibrating = False
                    save_config(self.config)
                    logger.info("Auto-calibration applied zone")
                    if self._window:
                        with contextlib.suppress(RuntimeError):
                            self._window.after(
                                0, self._window.set_auto_zone_state, False,
                            )
                elif self._calibrator.state == CalibrationState.FAILED:
                    self._calibrating = False
                    logger.warning("Auto-calibration failed")
                    if self._window:
                        with contextlib.suppress(RuntimeError):
                            self._window.after(
                                0, self._window.set_auto_zone_state, False,
                            )

                # Put frame and skip normal detect/track
                try:
                    self._frame_queue.put_nowait(display_frame)
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        self._frame_queue.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self._frame_queue.put_nowait(display_frame)
                continue

            # Widen detection zone when ball is in motion (extended tracking)
            if (
                self.config.shot.extended_tracking
                and self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
            ):
                detect_x1 = 0
                detect_x2 = display_frame.shape[1]
            else:
                detect_x1 = zone.start_x1
                detect_x2 = display_frame.shape[1]

            # Detect ball
            # When ball is in motion (STARTED/ENTERED):
            # - Widen Y range so vertical drift doesn't lose the ball
            # - Disable circularity filter (motion blur makes ball non-circular)
            ball_in_motion = self._tracker.state in (
                ShotState.STARTED, ShotState.ENTERED,
            )
            if ball_in_motion:
                det_y1 = max(0, zone.y1 - 80)
                det_y2 = min(display_frame.shape[0], zone.y2 + 80)
                saved_circ = self._detector.min_circularity
                self._detector.min_circularity = 0.0
            else:
                det_y1 = zone.y1
                det_y2 = zone.y2

            detection = self._detector.detect(
                frame=display_frame,
                zone_x1=detect_x1,
                zone_x2_limit=detect_x2,
                zone_y1=det_y1,
                zone_y2=det_y2,
                timestamp=frame_time,
                expected_radius=(
                    self._tracker.start_circle[2]
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                    else None
                ),
            )

            if ball_in_motion:
                self._detector.min_circularity = saved_circ

            # Track ball (with state transition logging)
            prev_state = self._tracker.state
            shot_result = self._tracker.update(detection)
            if self._tracker.state != prev_state:
                logger.info(
                    "Tracker: %s → %s", prev_state.value, self._tracker.state.value,
                )
                self._last_state_change = frame_time
            if shot_result is not None:
                logger.info("Shot result received from tracker")

            # Auto-reset if stuck in STARTED for >10s (no shot progress)
            # or stuck in BALL_DETECTED for >10s (can't stabilize)
            if (
                self._tracker.state in (ShotState.STARTED, ShotState.BALL_DETECTED)
                and self._last_state_change > 0
                and frame_time - self._last_state_change > 10.0
            ):
                    logger.warning(
                        "Auto-reset: stuck in %s for >10s",
                        self._tracker.state.value,
                    )
                    self._tracker.reset()
                    self._tracker.last_shot_positions.clear()
                    self._post_shot_tracking = False
                    self._last_state_change = frame_time
            # Log when ball is lost during active tracking
            if (
                detection is None
                and prev_state in (ShotState.STARTED, ShotState.ENTERED)
                and self._tracker.state == ShotState.IDLE
            ):
                logger.warning(
                    "Ball lost during %s — detection returned None "
                    "(frame: %dx%d, zone: x=%d-%d y=%d-%d)",
                    prev_state.value,
                    display_frame.shape[1], display_frame.shape[0],
                    detect_x1, detect_x2, zone.y1, zone.y2,
                )

            # Signal GSPro when ball is detected and ready
            # When Mevo is active, always report ball detected (Mevo handles it)
            if not self._mevo_detector:
                self._gspro.ball_detected = self._tracker.state not in (
                    ShotState.IDLE,
                )

            # Process completed shot
            if shot_result is not None:
                self._handle_shot(shot_result)
                # Begin post-shot tracking to extend the trail (only if
                # positions were stored — _handle_shot may return early)
                if self._tracker.last_shot_positions:
                    self._post_shot_tracking = True
                    self._post_shot_deadline = (
                        frame_time + self.config.overlay.trail_duration
                    )
                    self._post_shot_radius = shot_result.start_radius
                    # Auto-clear trail after duration (even without OBS idle)
                    self._trail_clear_time = (
                        frame_time + self.config.overlay.trail_duration
                    )

            # Post-shot trail extension
            if self._post_shot_tracking:
                if (frame_time >= self._post_shot_deadline
                        or self.config.overlay.projected_trail):
                    # Projected trail already calculated — no camera tracking needed
                    self._post_shot_tracking = False
                elif detection is None:
                    # Try to find ball near its last known position (not full frame)
                    last_positions = self._tracker.last_shot_positions
                    if len(last_positions) >= 2:
                        lx, ly = last_positions[-1]
                        # Search in a box around the last known position
                        margin = 80
                        search_y1 = max(0, ly - margin)
                        search_y2 = min(display_frame.shape[0], ly + margin)
                        search_x1 = max(0, lx - margin)
                        search_x2 = min(display_frame.shape[1], lx + margin)
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=search_x1,
                            zone_x2_limit=search_x2,
                            zone_y1=search_y1,
                            zone_y2=search_y2,
                            timestamp=frame_time,
                            expected_radius=self._post_shot_radius,
                            radius_tolerance=20,
                        )
                    if detection is not None:
                        self._tracker.last_shot_positions.append(
                            (detection.x, detection.y),
                        )
                    else:
                        self._post_shot_tracking = False
                elif detection is not None:
                    self._tracker.last_shot_positions.append(
                        (detection.x, detection.y),
                    )

            # Auto-clear stale trail to stop expensive glow rendering
            if (
                self._trail_clear_time > 0
                and frame_time >= self._trail_clear_time
                and self._tracker.state == ShotState.IDLE
            ):
                self._tracker.last_shot_positions.clear()
                self._trail_clear_time = 0.0

            # Build trail data for overlay
            active_trail: list[tuple[int, int]] = []
            if self._tracker.state in (ShotState.STARTED, ShotState.ENTERED):
                active_trail = [(x, y) for x, y, _t in self._tracker.positions]

            # Draw overlays onto display frame
            t_overlay = time.perf_counter()
            edit_mode = self._window.edit_zone_mode if self._window else False
            draw_overlay(
                frame=display_frame,
                zone=self.config.detection_zone,
                state=self._tracker.state,
                detection=detection,
                fps=self._actual_fps,
                connected=self._gspro.is_connected,
                connection_mode=self._gspro.mode,
                last_speed=self._tracker.last_shot_speed,
                last_hla=self._tracker.last_shot_hla,
                last_start=self._tracker.last_shot_start,
                last_end=self._tracker.last_shot_end,
                shot_count=self._tracker.shot_count,
                edit_mode=edit_mode,
                active_trail=active_trail,
                last_shot_trail=self._tracker.last_shot_positions,
                obs_overlay_mode=self.config.overlay.obs_overlay_mode,
                obs_show_zones=self.config.overlay.obs_show_zones,
                trail_color_name=self.config.overlay.trail_color,
                active_trail_color_name=self.config.overlay.active_trail_color,
            )

            # Log slow frames to diagnose FPS drops
            t_end = time.perf_counter()
            total_ms = (t_end - t0) * 1000
            if total_ms > 100:  # >100ms = slower than 10fps
                overlay_ms = (t_end - t_overlay) * 1000
                detect_ms = (t_overlay - t0) * 1000
                trail_len = len(self._tracker.last_shot_positions)
                logger.warning(
                    "Slow frame: %.0fms (detect=%.0fms, overlay=%.0fms) "
                    "state=%s trail=%d post_shot=%s",
                    total_ms, detect_ms, overlay_ms,
                    self._tracker.state.value, trail_len,
                    self._post_shot_tracking,
                )

            # Put frame into queue (drop old frames if queue is full)
            try:
                self._frame_queue.put_nowait(display_frame)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._frame_queue.get_nowait()
                with contextlib.suppress(queue.Full):
                    self._frame_queue.put_nowait(display_frame)

            # Adaptive skip: if processing is slow, skip next frame(s)
            process_duration = time.perf_counter() - frame_time
            if process_duration > self._target_process_time:
                frames_behind = int(process_duration / self._target_process_time)
                self._skip_counter = min(frames_behind, 2)

            # Periodic UI updates (~4 times per second for labels)
            if self._window and (frame_time - last_ui_update) > 0.25:
                last_ui_update = frame_time
                fps = self._actual_fps
                state = self._tracker.state.value
                connected = self._gspro.is_connected
                shot_count = self._tracker.shot_count
                # Schedule UI updates on the main thread
                with contextlib.suppress(RuntimeError):
                    self._window.after(0, self._window.update_fps, fps)
                    self._window.after(0, self._window.update_state, state)
                    self._window.after(
                        0, self._window.update_connection_status, connected
                    )
                    self._window.after(
                        0, self._window.update_shot_count, shot_count
                    )

    def _handle_shot(self, shot_result: object) -> None:
        """Process a completed shot — calculate physics, send to GSPro, update UI."""
        from birdman_putting.tracking import ShotResult
        if not isinstance(shot_result, ShotResult):
            return

        # Always store trail positions for the tracer (even if physics fails)
        self._tracker.last_shot_start = shot_result.start_position
        self._tracker.last_shot_end = shot_result.end_position

        # Build trail starting from the ball's resting position
        start = shot_result.start_position
        if self.config.overlay.projected_trail:
            self._tracker.last_shot_positions = project_trail(
                start=start,
                end=shot_result.end_position,
                frame_width=640,  # display_frame is always resized to 640
            )
        else:
            tracked = [(x, y) for x, y, _t in shot_result.positions]
            # Prepend the start position so the trail begins where the ball sat
            if tracked and tracked[0] != start:
                tracked.insert(0, start)
            self._tracker.last_shot_positions = tracked
        logger.info(
            "Shot captured: %d trail points, start=(%d,%d), end=(%d,%d), "
            "px_mm=%.4f, elapsed=%.4fs",
            len(self._tracker.last_shot_positions),
            *shot_result.start_position, *shot_result.end_position,
            shot_result.px_mm_ratio,
            shot_result.exit_time - shot_result.entry_time,
        )

        positions = list(shot_result.positions)
        is_rtl = self.config.detection_zone.direction == "right_to_left"
        shot_data = calculate_shot(
            start_pos=shot_result.start_position,
            end_pos=shot_result.end_position,
            entry_time=shot_result.entry_time,
            exit_time=shot_result.exit_time,
            px_mm_ratio=shot_result.px_mm_ratio,
            positions=positions,
            flip=self.config.camera.flip_image and self._video_path is None,
            reverse_x=is_rtl,
        )

        if not shot_data:
            logger.warning("Shot physics calculation failed — trail shown but no data")
            return

        s = self.config.shot
        in_range = (s.min_speed_mph <= shot_data.speed_mph <= s.max_speed_mph
                    and abs(shot_data.hla_degrees) <= s.max_hla_degrees)
        if not in_range:
            logger.info(
                "Shot out of range (%.2f MPH, %.2f HLA) - not sent",
                shot_data.speed_mph, shot_data.hla_degrees,
            )

        logger.info(
            "Shot #%d: %.2f MPH, HLA: %.2f, Dist: %.1f mm, Time: %.3f s%s",
            self._tracker.shot_count,
            shot_data.speed_mph, shot_data.hla_degrees,
            shot_data.distance_mm, shot_data.elapsed_seconds,
            "" if in_range else " (OUT OF RANGE)",
        )

        # Store speed/HLA for overlay display
        self._tracker.last_shot_speed = shot_data.speed_mph
        self._tracker.last_shot_hla = shot_data.hla_degrees

        # Estimate putt distance from speed and stimpmeter
        distance_ft = estimate_putt_distance_feet(
            shot_data.speed_mph, self.config.shot.stimpmeter,
        )

        # Only send valid shots to GSPro and OBS
        if in_range:
            speed = shot_data.speed_mph
            hla = shot_data.hla_degrees

            def _send() -> None:
                response = self._gspro.send_shot(speed, hla)
                if not response.success:
                    logger.warning("GSPro rejected shot: %s", response.message)

            threading.Thread(target=_send, daemon=True).start()

            if self._obs:
                self._obs.show_putt(shot_data.speed_mph, shot_data.hla_degrees)

        # Always update GUI shot display (even for out-of-range shots)
        if self._window:
            logger.info(
                "Scheduling GUI update: %.1f MPH, %.1f HLA, ~%.0f ft, shot #%d",
                shot_data.speed_mph, shot_data.hla_degrees,
                distance_ft, self._tracker.shot_count,
            )
            try:
                self._window.after(
                    0,
                    self._window.update_shot,
                    shot_data.speed_mph,
                    shot_data.hla_degrees,
                    self._tracker.shot_count,
                    distance_ft,
                )
            except RuntimeError as e:
                logger.error("Failed to schedule GUI update: %s", e)

    # ---- Headless Mode (OpenCV windows) ----

    def _run_headless(self) -> None:
        """Run with OpenCV windows only (no CustomTkinter)."""
        if self._video_path:
            if not self._camera.open_video(self._video_path):
                logger.error("Failed to open video: %s", self._video_path)
                return
        else:
            if not self._camera.open_webcam():
                logger.error("Failed to open webcam")
                return

        if not self._gspro.connect():
            logger.warning("Could not connect to GSPro. Shots will be logged only.")

        self._running = True

        # Connect to OBS if enabled
        self._start_obs()

        # Start Mevo thread if enabled
        self._start_mevo()

        logger.info("Putting app started (headless mode)")

        try:
            self._headless_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    def _on_headless_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Mouse callback for headless color pick mode."""
        if event != cv2.EVENT_LBUTTONDOWN or not self._pick_mode:
            return
        if self._pick_frame is None:
            return

        # Apply same blur + first HSV conversion as the detector, so that
        # generate_hsv_from_patch's internal BGR→HSV acts as the second
        # conversion, matching the detector's double-HSV color space.
        blurred = cv2.GaussianBlur(
            self._pick_frame, self._detector.blur_kernel, 0,
        )
        hsv_once = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv_range = generate_hsv_from_patch(hsv_once, x, y)
        logger.info("Picked HSV at (%d,%d): %s", x, y, hsv_range)
        self._hsv_range = hsv_range
        self._detector.update_hsv(hsv_range)
        self.config.ball.custom_hsv = hsv_range.to_dict()
        save_config(self.config)
        logger.info("Custom HSV saved to config")
        self._pick_mode = False

    def _headless_loop(self) -> None:
        """Synchronous main loop with cv2.imshow."""
        zone = self.config.detection_zone
        window_name = "Birdman Putting: Press q to exit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_headless_mouse)

        while self._running:
            frame_time = time.perf_counter()
            self._fps_queue.append(frame_time)

            if len(self._fps_queue) >= 2:
                elapsed = self._fps_queue[-1] - self._fps_queue[0]
                if elapsed > 0:
                    self._actual_fps = (len(self._fps_queue) - 1) / elapsed

            # Read frame (always read to drain camera buffer)
            frame = self._camera.read()
            if frame is None:
                break

            # Adaptive frame skipping
            if self._skip_counter > 0:
                self._skip_counter -= 1
                cv2.waitKey(1)
                continue

            display_frame = resize_with_aspect_ratio(frame, width=640)
            display_frame = self._camera.apply_rotation(display_frame)
            self._pick_frame = display_frame.copy()

            # --- Calibration mode (headless) ---
            if self._calibrating and self._calibrator:
                cal_detection = self._detector.detect_full_frame(
                    display_frame, timestamp=frame_time,
                )
                dh, dw = display_frame.shape[:2]
                cal_result = self._calibrator.update(cal_detection, dw, dh)

                state_text = f"AUTO ZONE: {self._calibrator.state.value}"
                ball_pos = (
                    (cal_detection.x, cal_detection.y) if cal_detection else None
                )
                draw_calibration_overlay(display_frame, state_text, ball_pos)

                if cal_result is not None:
                    self.config.detection_zone = cal_result.zone
                    zone = cal_result.zone
                    self._tracker.zone = cal_result.zone
                    self._tracker.reset()
                    self._calibrating = False
                    save_config(self.config)
                    logger.info("Auto-calibration applied zone (headless)")
                elif self._calibrator.state == CalibrationState.FAILED:
                    self._calibrating = False
                    logger.warning("Auto-calibration failed (headless)")

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running = False
                continue

            # Widen detection zone when ball is in motion (extended tracking)
            if (
                self.config.shot.extended_tracking
                and self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
            ):
                detect_x1 = 0
                detect_x2 = display_frame.shape[1]
            else:
                detect_x1 = zone.start_x1
                detect_x2 = display_frame.shape[1]

            ball_in_motion = self._tracker.state in (
                ShotState.STARTED, ShotState.ENTERED,
            )
            if ball_in_motion:
                det_y1 = max(0, zone.y1 - 80)
                det_y2 = min(display_frame.shape[0], zone.y2 + 80)
                saved_circ = self._detector.min_circularity
                self._detector.min_circularity = 0.0
            else:
                det_y1 = zone.y1
                det_y2 = zone.y2

            detection = self._detector.detect(
                frame=display_frame,
                zone_x1=detect_x1,
                zone_x2_limit=detect_x2,
                zone_y1=det_y1,
                zone_y2=det_y2,
                timestamp=frame_time,
                expected_radius=(
                    self._tracker.start_circle[2]
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                    else None
                ),
            )

            if ball_in_motion:
                self._detector.min_circularity = saved_circ

            shot_result = self._tracker.update(detection)

            # Signal GSPro when ball is detected and ready
            if not self._mevo_detector:
                self._gspro.ball_detected = self._tracker.state not in (
                    ShotState.IDLE,
                )

            if shot_result is not None:
                self._handle_shot(shot_result)
                if self._tracker.last_shot_positions:
                    self._post_shot_tracking = True
                    self._post_shot_deadline = (
                        frame_time + self.config.overlay.trail_duration
                    )
                    self._post_shot_radius = shot_result.start_radius
                    self._trail_clear_time = (
                        frame_time + self.config.overlay.trail_duration
                    )

            # Post-shot trail extension
            if self._post_shot_tracking:
                if (frame_time >= self._post_shot_deadline
                        or self.config.overlay.projected_trail):
                    self._post_shot_tracking = False
                elif detection is None:
                    last_positions = self._tracker.last_shot_positions
                    if len(last_positions) >= 2:
                        lx, ly = last_positions[-1]
                        margin = 80
                        search_y1 = max(0, ly - margin)
                        search_y2 = min(display_frame.shape[0], ly + margin)
                        search_x1 = max(0, lx - margin)
                        search_x2 = min(display_frame.shape[1], lx + margin)
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=search_x1,
                            zone_x2_limit=search_x2,
                            zone_y1=search_y1,
                            zone_y2=search_y2,
                            timestamp=frame_time,
                            expected_radius=self._post_shot_radius,
                            radius_tolerance=20,
                        )
                    if detection is not None:
                        self._tracker.last_shot_positions.append(
                            (detection.x, detection.y),
                        )
                    else:
                        self._post_shot_tracking = False
                elif detection is not None:
                    self._tracker.last_shot_positions.append(
                        (detection.x, detection.y),
                    )

            # Auto-clear stale trail
            if (
                self._trail_clear_time > 0
                and frame_time >= self._trail_clear_time
                and self._tracker.state == ShotState.IDLE
            ):
                self._tracker.last_shot_positions.clear()
                self._trail_clear_time = 0.0

            # Build trail data for overlay
            active_trail: list[tuple[int, int]] = []
            if self._tracker.state in (ShotState.STARTED, ShotState.ENTERED):
                active_trail = [(x, y) for x, y, _t in self._tracker.positions]

            draw_overlay(
                frame=display_frame,
                zone=zone,
                state=self._tracker.state,
                detection=detection,
                fps=self._actual_fps,
                connected=self._gspro.is_connected,
                connection_mode=self._gspro.mode,
                last_speed=self._tracker.last_shot_speed,
                last_hla=self._tracker.last_shot_hla,
                last_start=self._tracker.last_shot_start,
                last_end=self._tracker.last_shot_end,
                shot_count=self._tracker.shot_count,
                active_trail=active_trail,
                last_shot_trail=self._tracker.last_shot_positions,
                obs_overlay_mode=self.config.overlay.obs_overlay_mode,
                obs_show_zones=self.config.overlay.obs_show_zones,
                trail_color_name=self.config.overlay.trail_color,
                active_trail_color_name=self.config.overlay.active_trail_color,
                headless=True,
            )

            if self._pick_mode:
                cv2.putText(
                    display_frame, "CLICK ON BALL TO PICK COLOR",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

            cv2.imshow(window_name, display_frame)

            # Adaptive skip calculation
            process_duration = time.perf_counter() - frame_time
            if process_duration > self._target_process_time:
                frames_behind = int(process_duration / self._target_process_time)
                self._skip_counter = min(frames_behind, 2)

            if self._debug:
                mask = self._detector.get_mask(
                    display_frame, zone.start_x1, 640, zone.y1, zone.y2
                )
                cv2.imshow("Debug Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
            elif key == ord("d"):
                self._debug = not self._debug
                if not self._debug:
                    cv2.destroyWindow("Debug Mask")
            elif key == ord("a"):
                self._on_auto_zone()
            elif key == ord("r"):
                self.reset_putt()
            elif key == ord("c"):
                self._pick_mode = not self._pick_mode
                logger.info("Color pick mode: %s", "ON" if self._pick_mode else "OFF")

    # ---- OBS ----

    def _start_obs(self) -> None:
        """Connect to OBS WebSocket if enabled."""
        if not self.config.obs.enabled:
            return

        from birdman_putting.obs_controller import OBSController

        self._obs = OBSController(self.config.obs, on_idle=self._on_obs_idle)
        if not self._obs.connect():
            logger.warning("OBS connection failed — overlay disabled")
            self._obs = None

    # GSPro club codes that should use FS Golf PC chipping mode
    _CHIPPING_CLUBS = {"SW", "LW", "AW", "GW", "PW"}
    _last_club: str = ""  # Debounce duplicate club change messages

    def _on_club_change(self, club: str) -> None:
        """Called when GSPro sends a club selection (code 201).

        Automatically switches OBS scenes and FS Golf PC swing mode.
        Debounces duplicate messages (GSPro often sends the same club twice).
        """
        club_upper = club.upper()
        if club_upper == self._last_club:
            return  # Duplicate — skip
        self._last_club = club_upper

        is_putter = club_upper in ("PT", "PUTTER")

        # Pause/resume Mevo OCR — no point running Tesseract during putting
        if self._mevo_detector:
            if is_putter and not self._mevo_paused:
                self._mevo_paused = True
                logger.info("Mevo OCR paused (putter selected)")
            elif not is_putter and self._mevo_paused:
                self._mevo_paused = False
                logger.info("Mevo OCR resumed (%s selected)", club)

        # OBS scene switching
        if self._obs is not None and self.config.obs.auto_scene_switch:
            if is_putter:
                self._obs.switch_to_putt()
            else:
                self._obs.switch_to_main()

        # FS Golf PC chipping/full swing mode
        capture = getattr(self, "_mevo_capture", None)
        if capture is not None and not is_putter:
            if club_upper in self._CHIPPING_CLUBS:
                capture.send_key("c")
                logger.info("FS Golf PC → Chipping mode (%s)", club)
            else:
                capture.send_key("f")
                logger.info("FS Golf PC → Full Swing mode (%s)", club)

    def _on_obs_idle(self) -> None:
        """Called when OBS transitions back to idle scene — clear trail."""
        self._tracker.last_shot_positions.clear()
        self._post_shot_tracking = False
        logger.debug("Trail cleared on OBS idle transition")

    def _stop_obs(self) -> None:
        """Disconnect from OBS."""
        if self._obs is not None:
            self._obs.disconnect()
            self._obs = None

    # ---- Mevo ----

    def _start_mevo(self) -> None:
        """Start Mevo OCR thread if enabled."""
        if not self.config.mevo.enabled:
            return

        try:
            from birdman_putting.mevo.detector import MevoDetector, build_rois
            from birdman_putting.mevo.ocr import MevoOCR
            from birdman_putting.mevo.screenshot import WindowCapture
        except (ImportError, OSError) as e:
            logger.warning("Mevo dependencies not available: %s", e)
            if self._window:
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_mevo_status, f"Error: {e}", "error",
                    )
            return

        rois = build_rois(self.config.mevo.rois)
        if not rois:
            logger.warning("No Mevo ROIs configured — Mevo disabled")
            if self._window:
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_mevo_status, "No ROIs configured", "error",
                    )
            return

        capture = WindowCapture(self.config.mevo.window_title)

        # Widen FS Golf window so all columns (including right-side metrics) are visible
        if capture.find_window():
            capture.widen()

            # Scale ROIs if the current capture size differs from calibration
            cal_w = self.config.mevo.cal_width
            cal_h = self.config.mevo.cal_height
            if cal_w > 0 and cal_h > 0:
                test_frame = capture.capture()
                if test_frame is not None:
                    cur_h, cur_w = test_frame.shape[:2]
                    if cur_w != cal_w or cur_h != cal_h:
                        sx = cur_w / cal_w
                        sy = cur_h / cal_h
                        for roi in rois:
                            roi.x = int(roi.x * sx)
                            roi.y = int(roi.y * sy)
                            roi.width = int(roi.width * sx)
                            roi.height = int(roi.height * sy)
                        logger.info(
                            "Scaled %d ROIs: cal %dx%d → current %dx%d (%.2fx, %.2fy)",
                            len(rois), cal_w, cal_h, cur_w, cur_h, sx, sy,
                        )

        tessdata = self.config.mevo.tessdata_dir or None
        ocr = MevoOCR(rois=rois, tessdata_dir=tessdata)
        self._mevo_detector = MevoDetector(self.config.mevo, ocr, capture)
        self._mevo_capture = capture  # Keep reference for cleanup

        self._mevo_thread = threading.Thread(
            target=self._mevo_loop, daemon=True, name="mevo",
        )
        self._mevo_thread.start()
        self._gspro.ball_detected = True  # Mevo always has ball ready
        logger.info("Mevo OCR thread started (window='%s')", self.config.mevo.window_title)

        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.after(
                    0, self._window.update_mevo_status, "Watching...", "watching",
                )

    def _mevo_loop(self) -> None:
        """Background thread: poll Mevo display for new shots."""
        interval = self.config.mevo.poll_interval
        while self._running:
            if self._mevo_paused:
                time.sleep(0.5)
                continue
            start = time.perf_counter()
            if self._mevo_detector:
                shot = self._mevo_detector.poll()
                if shot is not None:
                    self._handle_mevo_shot(shot)
            elapsed = time.perf_counter() - start
            remaining = max(0.01, interval - elapsed)
            time.sleep(remaining)

    def _handle_mevo_shot(self, shot: object) -> None:
        """Process a Mevo shot — send full data to GSPro, update UI."""
        from birdman_putting.mevo.detector import MevoShotData

        if not isinstance(shot, MevoShotData):
            return

        logger.info(
            "Mevo shot: %.1f mph, VLA=%.1f, HLA=%.1f, Spin=%d",
            shot.ball_speed, shot.launch_angle, shot.launch_direction,
            int(shot.spin_rate),
        )

        # Only relay to GSPro if configured (disable when LM connects directly)
        if self.config.mevo.send_to_gspro:
            self._gspro._shot_cooldown = 3

            def _send_mevo() -> None:
                response = self._gspro.send_full_shot(
                    ball_speed=shot.ball_speed,
                    vla=shot.launch_angle,
                    hla=shot.launch_direction,
                    total_spin=shot.spin_rate,
                    spin_axis=shot.spin_axis,
                    back_spin=shot.back_spin,
                    side_spin=shot.side_spin,
                    club_speed=shot.club_speed,
                    carry_distance=shot.carry_distance,
                    aoa=shot.aoa,
                    club_path=shot.club_path,
                    dynamic_loft=shot.dynamic_loft,
                    face_to_target=shot.face_to_target,
                    lateral_impact=shot.lateral_impact,
                    vertical_impact=shot.vertical_impact,
                )
                logger.info("Mevo → GSPro: %s", "OK" if response.success else response.message)

            threading.Thread(target=_send_mevo, daemon=True).start()

        # Show Mevo shot data on OBS overlay
        if self._obs:
            self._obs.show_mevo_shot(shot)

        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.after(
                    0, self._window.update_mevo_status, "Shot detected!", "ok",
                )
                self._window.after(
                    0, self._window.update_shot,
                    shot.ball_speed, shot.launch_direction,
                    self._gspro.shot_number,
                )

    def _stop_mevo(self) -> None:
        """Stop the Mevo thread and restore FS Golf window."""
        if self._mevo_thread is not None:
            self._mevo_thread.join(timeout=5)
            self._mevo_thread = None
        self._mevo_detector = None
        # Restore FS Golf window to original size
        capture = getattr(self, "_mevo_capture", None)
        if capture is not None:
            capture.restore()
            self._mevo_capture = None

    # ---- Shared ----

    def _stop_processing(self) -> None:
        """Signal processing thread to stop and wait."""
        self._running = False
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

    def _cleanup(self) -> None:
        """Release all resources."""
        self._stop_mevo()
        self._stop_obs()
        self._camera.release()
        self._gspro.disconnect()
        cv2.destroyAllWindows()
        logger.info("Putting app stopped")
