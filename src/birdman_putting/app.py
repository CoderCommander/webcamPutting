"""Application orchestrator — wires camera, detection, tracking, physics, and GSPro client.

Supports two display modes:
- GUI mode (default): CustomTkinter window with threaded frame processing
- Headless mode (--no-gui): OpenCV windows, original synchronous loop
"""

from __future__ import annotations

import contextlib
import logging
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
from birdman_putting.physics import calculate_shot
from birdman_putting.tracking import BallTracker, ShotState
from birdman_putting.ui.overlay import draw_calibration_overlay, draw_overlay

if TYPE_CHECKING:
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
        )
        self._tracker = BallTracker(
            zone=config.detection_zone,
            ball_settings=config.ball,
            shot_settings=config.shot,
        )
        self._gspro = GSProClient(config.connection)

        # FPS tracking
        self._fps_queue: deque[float] = deque(maxlen=30)
        self._actual_fps: float = 0.0

        # Adaptive frame skipping
        self._skip_counter: int = 0
        self._target_process_time: float = 1.0 / 30.0  # target 30fps processing

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

        # GUI reference (set when run_gui is called)
        self._window: MainWindow | None = None

    def run(self) -> None:
        """Start the application in the appropriate mode."""
        if self._headless:
            self._run_headless()
        else:
            self._run_gui()

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

        # Connect to GSPro
        connected = self._gspro.connect()
        if self._window:
            self._window.update_connection_status(connected)

        self._running = True

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

    def _processing_loop(self) -> None:
        """Background thread: capture → detect → track → annotate → queue."""
        zone = self.config.detection_zone
        last_ui_update = 0.0

        while self._running:
            frame_time = time.perf_counter()
            self._fps_queue.append(frame_time)

            # Calculate FPS
            if len(self._fps_queue) >= 2:
                elapsed = self._fps_queue[-1] - self._fps_queue[0]
                if elapsed > 0:
                    self._actual_fps = (len(self._fps_queue) - 1) / elapsed

            # Read frame (always read to drain camera buffer)
            frame = self._camera.read()
            if frame is None:
                logger.warning("No frame received, stopping")
                self._running = False
                break

            # Adaptive frame skipping: skip processing but keep capturing
            if self._skip_counter > 0:
                self._skip_counter -= 1
                continue

            # Resize for processing
            display_frame = resize_with_aspect_ratio(frame, width=640)

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

            # Detect ball
            detection = self._detector.detect(
                frame=display_frame,
                zone_x1=zone.start_x1,
                zone_x2_limit=640,
                zone_y1=zone.y1,
                zone_y2=zone.y2,
                timestamp=frame_time,
                expected_radius=(
                    self._tracker.start_circle[2]
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                    else None
                ),
            )

            # Track ball
            shot_result = self._tracker.update(detection)

            # Process completed shot
            if shot_result is not None:
                self._handle_shot(shot_result)

            # Build trail data for overlay
            active_trail: list[tuple[int, int]] = []
            if self._tracker.state in (ShotState.STARTED, ShotState.ENTERED):
                active_trail = [(x, y) for x, y, _t in self._tracker.positions]

            # Draw overlays onto display frame
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
            return

        s = self.config.shot
        if not (s.min_speed_mph <= shot_data.speed_mph <= s.max_speed_mph
                and abs(shot_data.hla_degrees) <= s.max_hla_degrees):
            logger.info(
                "Shot out of range (%.2f MPH, %.2f HLA) - not sent",
                shot_data.speed_mph, shot_data.hla_degrees,
            )
            return

        logger.info(
            "Shot #%d: %.2f MPH, HLA: %.2f, Dist: %.1f mm, Time: %.3f s",
            self._tracker.shot_count,
            shot_data.speed_mph, shot_data.hla_degrees,
            shot_data.distance_mm, shot_data.elapsed_seconds,
        )

        # Store for overlay display
        self._tracker.last_shot_speed = shot_data.speed_mph
        self._tracker.last_shot_hla = shot_data.hla_degrees
        self._tracker.last_shot_start = shot_result.start_position
        self._tracker.last_shot_end = shot_result.end_position
        self._tracker.last_shot_positions = [
            (x, y) for x, y, _t in shot_result.positions
        ]

        # Send to GSPro
        response = self._gspro.send_shot(shot_data.speed_mph, shot_data.hla_degrees)
        if not response.success:
            logger.warning("GSPro rejected shot: %s", response.message)

        # Update GUI shot display
        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.after(
                    0,
                    self._window.update_shot,
                    shot_data.speed_mph,
                    shot_data.hla_degrees,
                    self._tracker.shot_count,
                )

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

        hsv_range = generate_hsv_from_patch(self._pick_frame, x, y)
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

            detection = self._detector.detect(
                frame=display_frame,
                zone_x1=zone.start_x1,
                zone_x2_limit=640,
                zone_y1=zone.y1,
                zone_y2=zone.y2,
                timestamp=frame_time,
                expected_radius=(
                    self._tracker.start_circle[2]
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                    else None
                ),
            )

            shot_result = self._tracker.update(detection)
            if shot_result is not None:
                self._handle_shot(shot_result)

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
            elif key == ord("c"):
                self._pick_mode = not self._pick_mode
                logger.info("Color pick mode: %s", "ON" if self._pick_mode else "OFF")

    # ---- Shared ----

    def _stop_processing(self) -> None:
        """Signal processing thread to stop and wait."""
        self._running = False
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None

    def _cleanup(self) -> None:
        """Release all resources."""
        self._camera.release()
        self._gspro.disconnect()
        cv2.destroyAllWindows()
        logger.info("Putting app stopped")
