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
from birdman_putting.physics import (
    ShotData,
    calculate_shot,
    estimate_putt_distance_feet,
    ppf_from_ball_radius,
    speed_from_launch_velocity,
    speed_from_trajectory_fit,
    speed_from_visible_distance,
    target_speed_for_distance,
)
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
        self._obs_calibration_grid: bool = False

        # Mevo thread
        self._mevo_detector: MevoDetector | None = None
        self._mevo_thread: threading.Thread | None = None
        self._mevo_paused: bool = False  # Pause OCR during putting to free CPU
        # Most recent Mevo reading stashed for putt-fallback use.
        # Tuple of (timestamp, ball_speed_mph, hla_deg). None when no
        # reading has been captured during the current session.
        self._mevo_last: tuple[float, float, float] | None = None
        self._mevo_last_lock = threading.Lock()

        # OBS controller
        self._obs: OBSController | None = None

        # Post-shot trail tracking: continue detecting ball after shot for trail
        self._post_shot_tracking = False
        self._post_shot_deadline: float = 0.0
        self._post_shot_radius: int = 0
        self._trail_clear_time: float = 0.0  # When to auto-clear the last-shot trail
        self._last_shot_time: float = 0.0  # When the last shot completed (for trail fade)
        self._obs_window_visible = False  # Track separate OBS overlay window state
        self._last_state_change: float = 0.0  # For stuck-state auto-reset

        # Angle calibration: measure HLA of a "straight" putt to auto-set rotation
        self._angle_cal_active = False
        self._angle_cal_samples: list[float] = []
        self._angle_cal_bg: np.ndarray | None = None  # Background frame for subtraction

        # Distance calibration: putt known distances to derive pixels_per_foot
        self._dist_cal_active = False
        self._dist_cal_phase = 0  # 0 = ~5ft phase, 1 = ~10ft phase
        self._dist_cal_phase_labels = ["~5 ft", "~10 ft"]
        self._dist_cal_samples: list[tuple[float, float]] = []  # (pixel_distance, actual_ft)
        self._dist_cal_putts_per_phase = 3
        self._dist_cal_pending_px: float = 0.0  # pixel distance awaiting user input

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

    def _start_msmf_pump(self) -> None:
        """Prevent Windows from throttling camera capture when backgrounded.

        Sets execution state flags so Windows knows this process is doing
        real-time media work and should not be power-throttled or sleep.
        The primary fix for GPU-related throttling (GSPro taking the GPU)
        is disabling D3D11 HW acceleration in the camera open path.
        """
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            prev = kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED,
            )
            if prev:
                logger.info("SetThreadExecutionState: anti-throttle active")
        except Exception:
            pass

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
            on_dist_cal=self._on_dist_cal,
            on_auto_cal=self._on_auto_cal,
            on_obs_auto_cal=self._on_obs_auto_cal,
            on_cork_cal=self._on_cork_cal,
            on_reconnect_gspro=self.reconnect_gspro,
            on_obs_calibrate=self.toggle_obs_calibration,
        )

        # Auto-start if camera/video is available
        self._on_gui_start()

        # Start a Windows message pump on the main thread.
        # MSMF (Media Foundation) associates its internal windows with the
        # thread that opened the VideoCapture (the main thread).  When this
        # window loses focus (user clicks GSPro), Windows stops dispatching
        # messages to it, and MSMF throttles frame delivery to ~2 fps.
        # Pumping messages here keeps MSMF fed even when backgrounded.
        self._start_msmf_pump()

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
        self._camera.update_settings(self.config.camera)

        logger.info("Settings reloaded")

    def reset_putt(self) -> None:
        """Force-reset the putt tracker to IDLE state and clear trail."""
        self._tracker.reset()
        self._tracker.last_shot_positions.clear()
        self._post_shot_tracking = False
        logger.info("Putt tracker manually reset")

    def reconnect_gspro(self) -> None:
        """Disconnect and reconnect GSPro (camera/OBS/Mevo stay running)."""
        import threading as _threading

        def _do_reconnect() -> None:
            logger.info("Reconnecting to GSPro...")
            self._gspro.disconnect()
            connected = self._gspro.connect()
            if self._window:
                try:
                    self._window.after(0, self._window.update_connection_status, connected)
                except RuntimeError:
                    pass
            logger.info("GSPro reconnect %s", "succeeded" if connected else "failed")

        _threading.Thread(target=_do_reconnect, daemon=True, name="gspro-reconnect").start()

    def toggle_obs_calibration(self) -> None:
        """Toggle the OBS calibration grid overlay on/off."""
        self._obs_calibration_grid = not self._obs_calibration_grid
        logger.info("OBS calibration grid: %s",
                     "ON" if self._obs_calibration_grid else "OFF")

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
            # Use read_latest() — a one-shot capture must not spuriously get
            # None just because the processing thread already consumed the
            # most-recent frame; freshness is irrelevant for a background grab.
            frame = self._camera.read_latest()
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

    def _on_dist_cal(self) -> None:
        """Toggle distance calibration mode.

        Guides the user through putting at approximate distances (~5ft, ~10ft).
        After each putt, a dialog asks for the actual distance. This lets
        the user putt naturally without needing to hit exact marks.
        """
        if self._dist_cal_active:
            # Cancel
            self._dist_cal_active = False
            self._dist_cal_phase = 0
            self._dist_cal_samples = []
            self._dist_cal_pending_speed = 0.0
            if self._window:
                self._window.set_dist_cal_state(False)
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_camera_status,
                        "Calibration cancelled", "idle",
                    )
            logger.info("Distance calibration cancelled")
            return

        # Guard: don't start if another calibration is active
        if self._angle_cal_active or getattr(self, "_calibrating", False):
            logger.warning("Cannot start distance cal while another calibration is active")
            return

        self._dist_cal_active = True
        self._dist_cal_phase = 0
        self._dist_cal_samples = []
        self._dist_cal_pending_speed = 0.0
        self._tracker.reset()
        if self._window:
            self._window.set_dist_cal_state(True)
            self._show_dist_cal_phase_prompt()
        logger.info("Distance calibration started — putt %s", self._dist_cal_phase_labels[0])

    def _on_auto_cal(self) -> None:
        """Compute pixels_per_foot from the detected ball radius.

        Unlike the roll-based Dist Cal wizard (which depends on the user
        correctly estimating each putt's roll distance AND is biased by
        the constant pixel distance at which shots complete), this reads
        the in-frame ball radius and uses the known 21.335mm golf ball
        radius as a physical reference. One click, no putting required.

        Requires the ball to be detected and resting in the start zone.
        """
        if self._angle_cal_active or self._dist_cal_active:
            logger.warning("Cannot run Auto Cal while another calibration is active")
            if self._window:
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_camera_status,
                        "Auto Cal: cancel other cal first", "error",
                    )
            return

        # Pull the ball radius the tracker has locked onto at rest. This
        # is the radius from the stable start-position detection, so it
        # is the ball sitting on the mat — the correct reference object.
        # Only populated once the tracker reaches STARTED state (ball
        # stable for `start_stability_frames` consecutive frames).
        start_circle = self._tracker.start_circle  # (x, y, radius)
        radius_px = start_circle[2] if start_circle else 0

        if radius_px <= 0:
            logger.warning(
                "Auto Cal: no stable ball lock — place ball in start zone "
                "and wait for the detection ring to appear first",
            )
            if self._window:
                with contextlib.suppress(RuntimeError):
                    self._window.after(
                        0, self._window.update_camera_status,
                        "Auto Cal: place ball & wait for lock", "error",
                    )
            return

        ppf = ppf_from_ball_radius(float(radius_px))
        self.config.shot.pixels_per_foot = round(ppf, 2)
        self.config.shot.speed_calibration_factor = 1.0
        # Clear any stale per-x markers — radius-based ppf is a single
        # global value, so leftover markers from a prior OBS Cal would
        # contradict it.
        self.config.shot.calibration_markers = []
        save_config(self.config)

        logger.info(
            "Auto Cal complete: ball radius=%dpx → pixels_per_foot=%.2f",
            radius_px, ppf,
        )
        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.after(
                    0, self._window.update_camera_status,
                    f"Auto Cal: ppf={ppf:.1f} (r={radius_px}px)", "ok",
                )

    def _on_obs_auto_cal(self) -> None:
        """Calibrate pixels_per_foot from a projected 1-ft marker scene.

        Workflow:
          1. Switch OBS to the configured calibration scene (which projects
             evenly-spaced bright markers along the putt line at 1-ft intervals)
          2. Wait briefly for the projection to settle
          3. Capture a fresh frame from the camera, applying the same resize
             and rotation the tracker uses
          4. Detect the projected markers via brightness threshold
          5. Median inter-marker pixel distance == pixels_per_foot
          6. Save config and switch OBS back to the prior scene

        This is the only calibration method that's accurate for the Kiyo
        Pro (or any wide-angle / non-perpendicular camera): it measures
        the ground-plane scale directly, instead of inferring it from the
        ball radius (which assumes a perpendicular overhead view).
        """
        from birdman_putting.detection import resize_with_aspect_ratio
        from birdman_putting.ppf_calibration import detect_calibration_markers

        if self._angle_cal_active or self._dist_cal_active:
            self._notify_cam_status("Auto Cal: cancel other cal first", "error")
            return

        if self._obs is None or not self._obs.is_connected:
            self._notify_cam_status(
                "OBS Auto Cal needs OBS connected — enable [obs]", "error",
            )
            logger.warning("OBS Auto Cal: OBS controller not connected")
            return

        if not self._running or self._camera is None:
            self._notify_cam_status("Start the camera first", "error")
            return

        cal_scene = self.config.obs.calibration_scene
        prior_scene = self._obs.current_scene()
        logger.info(
            "OBS Auto Cal: switching to '%s' (will restore '%s')",
            cal_scene, prior_scene,
        )

        if not self._obs.switch_to_scene(cal_scene):
            self._notify_cam_status(
                f"OBS Auto Cal: scene '{cal_scene}' not found", "error",
            )
            return

        # Settle: give projector + camera + auto-exposure a moment
        time.sleep(0.6)

        # Capture a frame from the live grab thread (avoid touching the
        # capture device directly) and run the tracker's normal preprocess.
        # read_latest() returns the most-recent frame regardless of freshness
        # — this one-shot capture must not get None just because the
        # processing thread already consumed the latest frame.
        try:
            frame = self._camera.read_latest()
            if frame is None:
                self._notify_cam_status(
                    "OBS Auto Cal: no frame from camera", "error",
                )
                self._obs.switch_to_scene(prior_scene or self.config.obs.idle_scene)
                return
            display = resize_with_aspect_ratio(frame, width=640)
            display = self._camera.apply_rotation(display)

            zone = self.config.detection_zone
            band_y_center = (zone.y1 + zone.y2) // 2
            # Cyan hue-range filter: the projector renders calibration
            # markers in cyan, so we ignore the orange ball + any orange
            # OBS overlays (shot trails, etc.) that might be in frame.
            result = detect_calibration_markers(
                display,
                band_y_center=band_y_center,
                band_half_height=60,
                hue_range=(75, 105),
                min_saturation=80,
            )
            # Drop any detections that fall inside the start zone X range
            # (extra safety — the ball lives there during calibration).
            filtered = [
                (cx, cy) for (cx, cy) in result.centers
                if not (zone.start_x1 <= cx <= zone.start_x2)
            ]
            if len(filtered) != len(result.centers):
                logger.info(
                    "OBS Auto Cal: filtered %d detections inside start zone",
                    len(result.centers) - len(filtered),
                )
                from birdman_putting.ppf_calibration import MarkerDetectionResult
                import numpy as _np
                diffs = [
                    filtered[i + 1][0] - filtered[i][0]
                    for i in range(len(filtered) - 1)
                ]
                result = MarkerDetectionResult(
                    centers=filtered,
                    median_spacing_px=float(_np.median(diffs)) if diffs else None,
                    mean_spacing_px=float(_np.mean(diffs)) if diffs else None,
                    diffs=diffs,
                )
        finally:
            # Always restore the prior scene, even on error
            self._obs.switch_to_scene(prior_scene or self.config.obs.idle_scene)

        if result.median_spacing_px is None or len(result.centers) < 2:
            # Save the analyzed frame to disk so the failure can be
            # diagnosed visually (wrong band Y? markers too dim? scene
            # didn't switch in time?).
            import cv2 as _cv2
            debug_path = "obs_cal_debug.png"
            try:
                # Mark the detection band on a copy
                annotated = display.copy()
                _cv2.rectangle(
                    annotated,
                    (0, max(0, band_y_center - 60)),
                    (annotated.shape[1] - 1, min(annotated.shape[0] - 1, band_y_center + 60)),
                    (0, 255, 255), 1,
                )
                _cv2.line(
                    annotated, (0, band_y_center),
                    (annotated.shape[1] - 1, band_y_center),
                    (255, 0, 255), 1,
                )
                _cv2.imwrite(debug_path, annotated)
                logger.info("OBS Auto Cal: saved debug frame to %s", debug_path)
            except Exception:
                logger.exception("Failed to save debug frame")

            self._notify_cam_status(
                f"OBS Auto Cal: only {len(result.centers)} markers found — "
                "check projection",
                "error",
            )
            logger.warning(
                "OBS Auto Cal: %d markers detected (need 2+) — "
                "see obs_cal_debug.png",
                len(result.centers),
            )
            return

        ppf = result.median_spacing_px
        self.config.shot.pixels_per_foot = round(ppf, 2)
        self.config.shot.speed_calibration_factor = 1.0

        # Filter out outlier-spacing markers — same logic the OBS-Cal
        # detector applies to the median.  Drop the first marker if the
        # gap to the next marker is >2x the median (i.e., we picked up a
        # straggler from outside the calibration row, like the orange
        # ball near the start zone).  Then store the marker X-positions
        # for per-x ppf interpolation in the trajectory fit.
        marker_xs = [float(cx) for (cx, _cy) in result.centers]
        # Drop leading/trailing markers that are way out of pattern
        if len(marker_xs) >= 3 and result.diffs:
            median_d = result.median_spacing_px or 0
            if median_d > 0:
                # Walk inward, dropping endpoints whose spacing is >2x median
                while (len(marker_xs) >= 3
                       and (marker_xs[1] - marker_xs[0]) > 2.0 * median_d):
                    marker_xs.pop(0)
                while (len(marker_xs) >= 3
                       and (marker_xs[-1] - marker_xs[-2]) > 2.0 * median_d):
                    marker_xs.pop()

        self.config.shot.calibration_markers = [round(x, 2) for x in marker_xs]
        save_config(self.config)

        diffs_str = ", ".join(f"{d:.0f}" for d in result.diffs)
        logger.info(
            "OBS Auto Cal: %d markers, spacings=[%s], median=%.2f -> "
            "pixels_per_foot=%.2f, calibration_markers=%s",
            len(result.centers), diffs_str, ppf, ppf,
            [round(x, 1) for x in marker_xs],
        )
        self._notify_cam_status(
            f"OBS Cal: ppf={ppf:.1f} ({len(result.centers)} markers, per-x cal stored)",
            "ok",
        )

    def _on_cork_cal(self) -> None:
        """Calibrate pixels_per_foot from physical bright markers
        (corks, white tape, anything bright) placed at 1-ft intervals
        along the putt line.  Works without a projector — useful when
        the projector doesn't cover the camera's full FOV.

        Workflow:
          1. User places 6-8 white/bright markers at exactly 1 ft
             intervals along the putt path
          2. User clicks Cork Cal
          3. Birdman captures a frame, detects bright spots in the
             detection-zone Y band (white-luma threshold, no hue filter)
          4. Saves marker X-positions to config for per-x ppf interp
        """
        from birdman_putting.detection import resize_with_aspect_ratio
        from birdman_putting.ppf_calibration import detect_calibration_markers

        if self._angle_cal_active or self._dist_cal_active:
            self._notify_cam_status("Cork Cal: cancel other cal first", "error")
            return
        if not self._running or self._camera is None:
            self._notify_cam_status("Cork Cal: start the camera first", "error")
            return

        # One-shot capture: read_latest() returns the most-recent frame
        # regardless of freshness, so we don't spuriously get None just
        # because the processing thread already consumed the latest frame.
        frame = self._camera.read_latest()
        if frame is None:
            self._notify_cam_status("Cork Cal: no frame from camera", "error")
            return
        display = resize_with_aspect_ratio(frame, width=640)
        display = self._camera.apply_rotation(display)

        zone = self.config.detection_zone
        band_y_center = (zone.y1 + zone.y2) // 2
        # No hue filter — accept any bright (white/cork) marker.
        result = detect_calibration_markers(
            display,
            band_y_center=band_y_center,
            band_half_height=60,
            luma_threshold=180,
        )
        # Drop detections inside the start zone (the ball itself is
        # there during cal and would be picked up as a marker).
        filtered = [
            (cx, cy) for (cx, cy) in result.centers
            if not (zone.start_x1 <= cx <= zone.start_x2)
        ]
        if len(filtered) != len(result.centers):
            logger.info(
                "Cork Cal: filtered %d detections inside start zone",
                len(result.centers) - len(filtered),
            )

        if len(filtered) < 2:
            # Save debug frame for visual inspection
            import cv2 as _cv2
            try:
                annotated = display.copy()
                _cv2.rectangle(
                    annotated,
                    (0, max(0, band_y_center - 60)),
                    (annotated.shape[1] - 1,
                     min(annotated.shape[0] - 1, band_y_center + 60)),
                    (0, 255, 255), 1,
                )
                _cv2.imwrite("cork_cal_debug.png", annotated)
                logger.info("Cork Cal: saved debug frame to cork_cal_debug.png")
            except Exception:
                pass
            self._notify_cam_status(
                f"Cork Cal: only {len(filtered)} markers found — "
                "check corks/lighting",
                "error",
            )
            return

        # Compute spacings, pick median ppf
        marker_xs = sorted(float(cx) for (cx, _cy) in filtered)
        diffs = [marker_xs[i + 1] - marker_xs[i]
                 for i in range(len(marker_xs) - 1)]
        # Robust median: drop spacings >2x the smallest plausible one
        # (handles corks placed at non-uniform spacing or stray detections)
        sorted_diffs = sorted(diffs)
        p25 = sorted_diffs[len(sorted_diffs) // 4] if sorted_diffs else 0
        kept = [d for d in diffs if d <= p25 * 1.5] if p25 > 0 else diffs
        median_d = float(np.median(kept)) if kept else float(np.median(diffs))

        self.config.shot.pixels_per_foot = round(median_d, 2)
        self.config.shot.speed_calibration_factor = 1.0
        self.config.shot.calibration_markers = [round(x, 2) for x in marker_xs]
        save_config(self.config)

        diffs_str = ", ".join(f"{d:.0f}" for d in diffs)
        logger.info(
            "Cork Cal: %d markers, spacings=[%s], median=%.2f -> "
            "pixels_per_foot=%.2f, calibration_markers=%s",
            len(filtered), diffs_str, median_d, median_d,
            [round(x, 1) for x in marker_xs],
        )
        self._notify_cam_status(
            f"Cork Cal: ppf={median_d:.1f} ({len(filtered)} markers, full-FOV cal)",
            "ok",
        )

    def _notify_cam_status(self, status: str, state: str) -> None:
        """Push a status update to the camera-status strip if the GUI exists."""
        if self._window is None:
            return
        with contextlib.suppress(RuntimeError):
            self._window.after(
                0, self._window.update_camera_status, status, state,
            )

    def _show_dist_cal_phase_prompt(self) -> None:
        """Update all UI elements to show the current calibration phase."""
        if not self._window:
            return
        phase = self._dist_cal_phase
        label = self._dist_cal_phase_labels[phase]
        phase_start = phase * self._dist_cal_putts_per_phase
        phase_n = len(self._dist_cal_samples) - phase_start
        total_phases = len(self._dist_cal_phase_labels)

        with contextlib.suppress(RuntimeError):
            self._window.after(
                0, self._window.show_cal_phase,
                f"STEP {phase + 1} of {total_phases}",
                f"Putt {label} ({phase_n}/{self._dist_cal_putts_per_phase})",
            )
            self._window.after(
                0, self._window.update_camera_status,
                f"CALIBRATING: Putt {label} — {phase_n}/{self._dist_cal_putts_per_phase} done",
                "watching",
            )

    def _process_dist_cal_shot(self, pixel_distance: float) -> None:
        """Record a calibration putt and prompt user for actual distance."""
        if pixel_distance < 5:
            return  # Too small to be a real putt

        self._dist_cal_pending_px = pixel_distance
        phase_label = self._dist_cal_phase_labels[self._dist_cal_phase]

        logger.info("Dist cal: putt traveled %.0f pixels — prompting for distance", pixel_distance)

        if self._window:
            with contextlib.suppress(RuntimeError):
                self._window.after(0, self._show_dist_cal_dialog, phase_label)

    def _show_dist_cal_dialog(self, phase_label: str) -> None:
        """Show a dialog asking the user for the actual putt distance."""
        import customtkinter as ctk

        phase = self._dist_cal_phase + 1
        total = len(self._dist_cal_phase_labels)

        dialog = ctk.CTkInputDialog(
            text=(
                f"Step {phase}/{total}: How many feet did that putt roll?\n"
                f"(aiming for {phase_label})"
            ),
            title=f"Distance Cal — Step {phase}/{total}",
        )
        result = dialog.get_input()

        if result is None or not self._dist_cal_active:
            return

        try:
            actual_ft = float(result)
        except ValueError:
            logger.warning("Dist cal: invalid input '%s', skipping putt", result)
            return

        if actual_ft <= 0:
            logger.warning("Dist cal: distance must be positive, skipping")
            return

        self._dist_cal_samples.append((self._dist_cal_pending_px, actual_ft))
        self._dist_cal_pending_px = 0.0

        # Count putts in current phase
        phase_start = self._dist_cal_phase * self._dist_cal_putts_per_phase
        phase_n = len(self._dist_cal_samples) - phase_start

        logger.info(
            "Dist cal: recorded %.1f ft at %.0f px — step %d putt %d/%d",
            actual_ft, self._dist_cal_samples[-1][0],
            self._dist_cal_phase + 1, phase_n, self._dist_cal_putts_per_phase,
        )

        if phase_n < self._dist_cal_putts_per_phase:
            self._show_dist_cal_phase_prompt()
        elif self._dist_cal_phase + 1 < len(self._dist_cal_phase_labels):
            self._dist_cal_phase += 1
            self._tracker.reset()
            self._show_dist_cal_phase_prompt()
            logger.info("Dist cal: advancing to step %d — %s",
                        self._dist_cal_phase + 1,
                        self._dist_cal_phase_labels[self._dist_cal_phase])
        else:
            self._apply_dist_cal()

    def _apply_dist_cal(self) -> None:
        """Compute pixels_per_foot from collected samples and save."""
        ratios: list[float] = []
        for pixel_dist, actual_ft in self._dist_cal_samples:
            if pixel_dist > 5 and actual_ft > 0:
                ratios.append(pixel_dist / actual_ft)

        if not ratios:
            logger.error("Dist cal: no valid samples — aborting")
            self._dist_cal_active = False
            return

        avg_ppf = sum(ratios) / len(ratios)

        self.config.shot.pixels_per_foot = round(avg_ppf, 2)
        # Reset speed_calibration_factor since pixels_per_foot replaces it
        self.config.shot.speed_calibration_factor = 1.0
        save_config(self.config)

        self._dist_cal_active = False
        self._dist_cal_phase = 0
        self._dist_cal_samples = []

        if self._window:
            self._window.set_dist_cal_state(False)
            with contextlib.suppress(RuntimeError):
                self._window.after(
                    0, self._window.update_camera_status,
                    f"Cal done: {avg_ppf:.1f} px/ft", "ok",
                )

        logger.info(
            "Distance calibration complete: pixels_per_foot=%.2f (from %d samples)",
            avg_ppf, len(ratios),
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
        # when Birdman is in the background.  THREAD_PRIORITY_HIGHEST alone
        # is not enough — Windows EcoQoS / background throttling can still
        # drop a HIGHEST-priority thread to ~5-9fps when the window loses
        # focus.  MMCSS "Pro Audio" gives real-time priority that survives
        # background throttling, matching what the grab thread already does.
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                THREAD_PRIORITY_HIGHEST = 2
                handle = kernel32.GetCurrentThread()
                kernel32.SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)
                logger.info("Processing thread priority set to HIGHEST")

                # Register with Multimedia Class Scheduler for real-time
                # priority that survives Windows EcoQoS / background
                # throttling.  Without this, processing FPS collapses to
                # single digits whenever Birdman loses window focus.
                try:
                    avrt = ctypes.windll.avrt  # type: ignore[attr-defined]
                    task_index = ctypes.c_ulong(0)
                    mmcss_handle = avrt.AvSetMmThreadCharacteristicsW(
                        "Pro Audio", ctypes.byref(task_index),
                    )
                    if mmcss_handle:
                        logger.info(
                            "MMCSS: registered processing thread as 'Pro Audio'",
                        )
                    else:
                        logger.debug("MMCSS registration failed for processing")
                except Exception:
                    pass  # avrt.dll not available on all editions
            except Exception:
                pass

        zone = self.config.detection_zone
        last_ui_update = 0.0

        try:
            while self._running:
                # Read frame
                frame = self._camera.read()
                if frame is None:
                    if self._camera.is_grab_running:
                        # Threaded grab: no new frame yet, wait briefly and retry
                        time.sleep(0.001)
                        continue
                    logger.warning("No frame received, stopping")
                    self._running = False
                    break
                saved_circ = None
                try:

                    frame_time = time.perf_counter()
                    self._fps_queue.append(frame_time)

                    # Calculate FPS (only counts actual new frames)
                    if len(self._fps_queue) >= 2:
                        elapsed = self._fps_queue[-1] - self._fps_queue[0]
                        if elapsed > 0:
                            self._actual_fps = (len(self._fps_queue) - 1) / elapsed

                    # FPS watchdog: auto-reset tracker if FPS stays critically low
                    # AND the tracker is stuck in a non-IDLE state for an extended
                    # period.  We do NOT reset from IDLE/BALL_DETECTED/STARTED — the
                    # user may be in the middle of placing a ball, and resetting
                    # mid-stability-accumulation prevents detection from ever
                    # locking on a slow CPU.  Only ENTERED-stuck is suspicious;
                    # the ENTERED state already has its own 2s timeout (tracker.py)
                    # so the watchdog is now mostly belt-and-suspenders for
                    # genuinely degenerate cases.
                    _WATCHDOG_FPS_THRESHOLD = 3.0   # was 10 — only react to severe drops
                    _WATCHDOG_HOLD_SECONDS = 30.0   # was 2 — give detection time to settle
                    if self._actual_fps > 0 and self._actual_fps < _WATCHDOG_FPS_THRESHOLD:
                        tracker_stuck_state = self._tracker.state in (
                            ShotState.ENTERED,
                        )
                        if tracker_stuck_state:
                            if self._fps_drop_start == 0.0:
                                self._fps_drop_start = frame_time
                            elif frame_time - self._fps_drop_start > _WATCHDOG_HOLD_SECONDS:
                                logger.warning(
                                    "FPS watchdog: %.1f FPS for >%ds while stuck in "
                                    "ENTERED — resetting tracker",
                                    self._actual_fps, int(_WATCHDOG_HOLD_SECONDS),
                                )
                                self._tracker.reset()
                                self._post_shot_tracking = False
                                self._fps_drop_start = 0.0
                        else:
                            self._fps_drop_start = 0.0
                    else:
                        self._fps_drop_start = 0.0

                    # Adaptive frame skipping: skip processing but keep capturing.
                    # Never skip while a putt is live (STARTED/ENTERED) or the
                    # post-shot tracer is drawing — every motion frame counts for
                    # an accurate fit and a smooth projected trail (even under
                    # OBS encoding load).
                    if self._skip_counter > 0 and not (
                        self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                        or self._post_shot_tracking
                    ):
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

                    # Set detection area based on state and tracking mode
                    if (
                        self.config.shot.extended_tracking
                        and self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                    ):
                        detect_x1 = 0
                        detect_x2 = display_frame.shape[1]
                    elif self._tracker.state == ShotState.ENTERED:
                        # Wide search to track ball exiting past gateway
                        detect_x1 = zone.start_x1
                        detect_x2 = display_frame.shape[1]
                    else:
                        detect_x1 = zone.start_x1
                        detect_x2 = display_frame.shape[1]

                    # Widen Y range + disable circularity when ball is moving (STARTED
                    # or ENTERED) to handle motion blur during roll.  The original
                    # cam-putting-py (which Springbok uses) doesn't use a circularity
                    # filter at all — motion blur drops circularity to 0.3-0.5, which
                    # was causing birdman to reject valid in-flight detections and
                    # leave the tracker with only sparse trail points.
                    if self._tracker.state == ShotState.ENTERED:
                        det_y1 = max(0, zone.y1 - 50)
                        det_y2 = min(display_frame.shape[0], zone.y2 + 50)
                        saved_circ = self._detector.min_circularity
                        self._detector.min_circularity = 0.0
                    elif self._tracker.state == ShotState.STARTED:
                        det_y1 = max(0, zone.y1 - 30)
                        det_y2 = min(display_frame.shape[0], zone.y2 + 30)
                        saved_circ = self._detector.min_circularity
                        self._detector.min_circularity = 0.0
                    else:
                        det_y1 = zone.y1
                        det_y2 = zone.y2
                        saved_circ = None

                    expected_r = (
                        self._tracker.start_circle[2]
                        if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                        else None
                    )
                    # Expected ball position for inter-frame continuity during motion
                    # states (STARTED/ENTERED).  Prefer the most-recent tracked
                    # position (deque tail) — during ENTERED the ball has rolled away
                    # from start, so start_circle is stale; positions[-1] is where it
                    # actually was last seen.  Fall back to start_circle when the trail
                    # is empty.  Left as None for IDLE/BALL_DETECTED so first-detection
                    # keeps its largest-contour behavior.
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED):
                        _trail = self._tracker.positions
                        if _trail:
                            _lx, _ly, _ = _trail[-1]
                            expected_pos: tuple[int, int] | None = (int(_lx), int(_ly))
                        else:
                            expected_pos = self._tracker.start_circle[:2]
                    else:
                        expected_pos = None

                    # Two-pass detection in STARTED state:
                    # 1. Search start zone only — if ball still there, use that
                    # 2. Only if ball NOT found at start, search wider area for gateway
                    # This prevents noise contours in the gateway area from
                    # triggering false ENTERED transitions while the ball is stationary.
                    if self._tracker.state == ShotState.STARTED:
                        # Pass 1: start zone only
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=zone.start_x1,
                            zone_x2_limit=zone.start_x2,
                            zone_y1=det_y1,
                            zone_y2=det_y2,
                            timestamp=frame_time,
                            expected_radius=expected_r,
                            expected_pos=expected_pos,
                        )
                        if detection is None:
                            # Ball not in start zone — search full width for gateway
                            # crossing.  A fast putt can jump past the gateway in one
                            # frame, so we must search the entire frame, not just a
                            # narrow band around the gateway.
                            detection = self._detector.detect(
                                frame=display_frame,
                                zone_x1=detect_x1,
                                zone_x2_limit=detect_x2,
                                zone_y1=det_y1,
                                zone_y2=det_y2,
                                timestamp=frame_time,
                                expected_radius=expected_r,
                                expected_pos=expected_pos,
                            )
                    else:
                        # ENTERED passes expected_pos (set above); IDLE/BALL_DETECTED
                        # leaves it None so first-detection stays largest-contour.
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=detect_x1,
                            zone_x2_limit=detect_x2,
                            zone_y1=det_y1,
                            zone_y2=det_y2,
                            timestamp=frame_time,
                            expected_radius=expected_r,
                            expected_pos=expected_pos,
                        )

                    # Skip tracker updates when a non-putter club is selected.
                    # Prevents false putt detections from full-swing motion in
                    # or near the detection zone.  Mevo handles those shots.
                    if not self.is_putting_mode:
                        if self._tracker.state != ShotState.IDLE:
                            self._tracker.reset()
                        detection = None  # Clear for downstream rendering
                        shot_result = None
                        prev_state = self._tracker.state
                    else:
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
                    # or stuck in BALL_DETECTED for >10s (can't stabilize).
                    # Compare against the tracker's last_activity_time (same
                    # perf_counter clock as frame_time): it advances on genuine
                    # progress (state transitions and real re-starts) but STAYS
                    # FROZEN when a stationary ball's redundant re-starts are
                    # suppressed — so a truly stuck ball trips this reset while a
                    # ball that keeps genuinely re-arming keeps it alive.
                    if (
                        self._tracker.state in (ShotState.STARTED, ShotState.BALL_DETECTED)
                        and self._tracker.last_activity_time > 0
                        and frame_time - self._tracker.last_activity_time > 10.0
                    ):
                            logger.warning(
                                "Auto-reset: stuck in %s for >10s (no tracker activity)",
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
                            self._last_shot_time = frame_time
                            ov = self.config.overlay
                            total = ov.trail_peak_time + ov.trail_fade_time
                            self._post_shot_deadline = frame_time + total
                            self._post_shot_radius = shot_result.start_radius
                            # Auto-clear trail after peak + fade
                            self._trail_clear_time = (
                                frame_time + total
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

                    # Compute trail brightness based on peak + fade timing
                    trail_brightness = 1.0
                    if self._last_shot_time > 0 and self._tracker.last_shot_positions:
                        elapsed_since_shot = frame_time - self._last_shot_time
                        peak = self.config.overlay.trail_peak_time
                        fade = self.config.overlay.trail_fade_time
                        if elapsed_since_shot <= peak:
                            trail_brightness = 1.0
                        elif fade > 0:
                            trail_brightness = max(0.0, 1.0 - (elapsed_since_shot - peak) / fade)
                        else:
                            trail_brightness = 0.0

                    overlay_kwargs = dict(
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
                        obs_show_zones=self.config.overlay.obs_show_zones,
                        obs_calibration_grid=self._obs_calibration_grid,
                        trail_color_name=self.config.overlay.trail_color,
                        active_trail_color_name=self.config.overlay.active_trail_color,
                        trail_brightness=trail_brightness,
                    )

                    draw_overlay(
                        frame=display_frame,
                        obs_overlay_mode=self.config.overlay.obs_overlay_mode,
                        **overlay_kwargs,
                    )
                    output_frame = display_frame

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
                        self._frame_queue.put_nowait(output_frame)
                    except queue.Full:
                        with contextlib.suppress(queue.Empty):
                            self._frame_queue.get_nowait()
                        with contextlib.suppress(queue.Full):
                            self._frame_queue.put_nowait(output_frame)

                    # Adaptive skip: if processing is slow, skip next frame(s) —
                    # but never during a live putt or the post-shot tracer, so a
                    # motion frame is never dropped under load (e.g. OBS encoding).
                    process_duration = time.perf_counter() - frame_time
                    _putt_active = (
                        self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                        or self._post_shot_tracking
                    )
                    if process_duration > self._target_process_time and not _putt_active:
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
                except Exception:
                    logger.exception(
                        "Processing loop: error processing frame — skipping"
                    )
                    continue
                finally:
                    if saved_circ is not None:
                        self._detector.min_circularity = saved_circ
        except Exception:
            logger.exception("Processing loop crashed — stopping")
            self._running = False

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

        # Calibrated-speed path: if pixels_per_foot is set, derive MPH
        # using the ground-plane scale instead of ball-radius timing.
        # Two estimators:
        #   - speed_from_launch_velocity: measures pixels/sec during the
        #     first ~6 frames of motion. Doesn't saturate on long putts,
        #     so 30/50-ft putts read correctly.
        #   - speed_from_visible_distance: PutTrak-style. Reads the full
        #     visible pixel distance as "feet rolled" — saturates at the
        #     frame edge, so long putts read low.
        # We take max(launch, dist): for short putts they agree; for
        # long putts launch wins (dist-based is clipped). If the launch
        # window is too short to be trustworthy, launch returns 0 and
        # dist-based owns the result (preserves today's behavior for
        # very fast 1-2-frame putts).
        ppf = self.config.shot.pixels_per_foot
        if ppf > 0:
            dx = shot_result.end_position[0] - shot_result.start_position[0]
            dy = shot_result.end_position[1] - shot_result.start_position[1]
            distance_px = math.sqrt(dx * dx + dy * dy)
            stimp = self.config.shot.stimpmeter

            # Trajectory fit: solves x(t) = v0·t − ½·a·t² over ALL
            # motion samples, where a is the stimp deceleration.  Far
            # more robust than picking a launch window — uses the full
            # post-exit traversal (now ~20-30 frames instead of ~6).
            # When per-x calibration markers are available (from OBS
            # Cal), they're used for piecewise-linear pixel→ft mapping
            # to correct fisheye / off-axis ppf variation.
            cal_markers = self.config.shot.calibration_markers or None
            fit_speed, fit_dbg = speed_from_trajectory_fit(
                list(shot_result.positions), ppf, stimp,
                calibration_markers=cal_markers,
            )
            launch_speed, launch_dbg = speed_from_launch_velocity(
                list(shot_result.positions), ppf,
            )
            dist_speed = speed_from_visible_distance(distance_px, ppf, stimp)

            # Prefer trajectory fit when we have enough data; fall back
            # to max(launch, dist) for sparse trails (1-2 motion frames).
            if fit_speed > 0:
                final_speed = fit_speed
                speed_source = "fit"
            else:
                final_speed = max(launch_speed, dist_speed)
                speed_source = "launch/dist"

            # Over-read safety cap: a sparse-trail fallback (fit failed) measures
            # speed from only 1-2 frames and can wildly over-read a firm putt as
            # a 50-100 ft rocket. Bound the FALLBACK to a realistic putt distance
            # so a dropped-out putt can't send a monster to GSPro. The
            # trajectory-fit path is trusted and never capped. (Stopgap until the
            # processing frame rate is raised so firm putts reach the fit.)
            _MAX_FALLBACK_PUTT_FT = 50.0
            fallback_capped = False
            if speed_source != "fit" and final_speed > 0:
                if estimate_putt_distance_feet(final_speed, stimp) > _MAX_FALLBACK_PUTT_FT:
                    final_speed = target_speed_for_distance(_MAX_FALLBACK_PUTT_FT, stimp)
                    fallback_capped = True

            if final_speed > 0:
                est_ft = estimate_putt_distance_feet(final_speed, stimp)
                logger.info(
                    "Calibrated speed: %.2f MPH (source=%s, fit=%.2f, "
                    "launch=%.2f, dist=%.2f; visible=%.0fpx, est_roll=%.1fft) "
                    "[fit_n=%d fit_tau=%.3fs fit_travel=%.2fft "
                    "fit_per_x=%s fit_reason=%s | "
                    "launch_window_dt=%.3fs launch_reason=%s]",
                    final_speed, speed_source, fit_speed, launch_speed, dist_speed,
                    distance_px, est_ft,
                    fit_dbg["n_motion"], fit_dbg["tau_max"],
                    fit_dbg["travel_max_ft"],
                    fit_dbg.get("uses_markers", False), fit_dbg["reason"],
                    launch_dbg["window_dt"], launch_dbg["reason"],
                )
                shot_data = ShotData(
                    speed_mph=round(final_speed, 2),
                    hla_degrees=shot_data.hla_degrees,
                    distance_mm=shot_data.distance_mm,
                    elapsed_seconds=shot_data.elapsed_seconds,
                )

            # Mevo putt-fallback: when both fit and launch fail (no
            # motion / dropped-frame artifact / etc.), we may have
            # nothing useful from the webcam. If Mevo radar caught the
            # shot, prefer that.
            cap_hit = "cap" in launch_dbg.get("reason", "")
            no_launch = fit_speed <= 0 and launch_speed <= 0
            under_min = final_speed < self.config.shot.min_speed_mph
            low_confidence = cap_hit or fallback_capped or (no_launch and under_min)
            if low_confidence:
                mevo_reading = self._recent_mevo_reading()
                if mevo_reading is not None:
                    mevo_mph, mevo_hla = mevo_reading
                    logger.info(
                        "Mevo putt fallback engaged: webcam=%.2f MPH "
                        "(reason: cap_hit=%s, no_launch=%s, under_min=%s) "
                        "-> using Mevo %.2f MPH, HLA=%.2f",
                        final_speed, cap_hit, no_launch, under_min,
                        mevo_mph, mevo_hla,
                    )
                    shot_data = ShotData(
                        speed_mph=round(mevo_mph, 2),
                        hla_degrees=round(mevo_hla, 2),
                        distance_mm=shot_data.distance_mm,
                        elapsed_seconds=shot_data.elapsed_seconds,
                    )
                    # Clear the cached reading so the same shot isn't
                    # reused if another webcam event fires shortly after.
                    with self._mevo_last_lock:
                        self._mevo_last = None

        # Distance calibration: record pixel distance and skip GSPro send
        if self._dist_cal_active:
            dx = shot_result.end_position[0] - shot_result.start_position[0]
            dy = shot_result.end_position[1] - shot_result.start_position[1]
            pixel_dist = math.sqrt(dx * dx + dy * dy)

            # Reject saturated samples.  The tracker has TWO completion
            # paths and they yield different start_position values:
            #
            #   1. ENTERED → exit:  start_position = gateway entry pos.
            #      Exit triggers when travel_from_entry >= min_exit_distance_px,
            #      so end_x ≈ entry_x + min_exit. Saturated.
            #
            #   2. past_gateway:    start_position = ball-rest pos.
            #      Triggers when ball jumps past gateway in one frame.
            #      Saturated unless ball ROLLED to a stop in-frame
            #      (which would have completed via ENTERED-timeout, not
            #      via exit trigger — those have actual end positions).
            #
            # The robust check: any shot that completed via an exit
            # trigger (vs. timeout / ball-stop detection) is saturated
            # for calibration purposes, because end_x is pinned at the
            # threshold not at the ball's rest. We detect this by
            # checking whether end_x is approximately at (entry_x + min_exit)
            # OR (start_x_rest + (gateway_x - start_x_rest) + min_exit) —
            # both reduce to "end_x is near a fixed offset from gateway".
            min_exit = float(self.config.shot.min_exit_distance_px)
            z = self.config.detection_zone
            if z.direction == "left_to_right":
                gateway_x = z.start_x2 + z.gateway_width
                # Saturation band: end_x within ~15% of (gateway_x + min_exit)
                expected_satur_end_x = gateway_x + min_exit
                end_x_distance_from_satur = abs(
                    shot_result.end_position[0] - expected_satur_end_x,
                )
            else:
                gateway_x = z.start_x1 - z.gateway_width
                expected_satur_end_x = gateway_x - min_exit
                end_x_distance_from_satur = abs(
                    shot_result.end_position[0] - expected_satur_end_x,
                )
            saturation_margin = min_exit * 0.30  # ±15 px for min_exit=50
            travel = abs(shot_result.end_position[0] - shot_result.start_position[0])
            if end_x_distance_from_satur <= saturation_margin:
                logger.warning(
                    "Dist Cal: rejecting saturated sample "
                    "(end_x=%d ≈ saturation band %d±%d, travel=%dpx). "
                    "Shot completed at the exit trigger — end position is "
                    "fixed at (gateway+%dpx), so this sample tells us nothing "
                    "about the real putt distance. Use Auto Cal for "
                    "ball-radius-based calibration, OR putt softer so the "
                    "ball stops inside the frame.",
                    int(shot_result.end_position[0]),
                    int(expected_satur_end_x), int(saturation_margin),
                    int(travel), int(min_exit),
                )
                if self._window:
                    with contextlib.suppress(RuntimeError):
                        self._window.after(
                            0, self._window.update_camera_status,
                            "Sample rejected (saturated) — putt softer or use Auto Cal",
                            "error",
                        )
                return

            self._process_dist_cal_shot(pixel_dist)
            return

        # Apply speed calibration factor (from Dist Cal wizard)
        if self.config.shot.speed_calibration_factor != 1.0:
            shot_data = ShotData(
                speed_mph=round(shot_data.speed_mph * self.config.shot.speed_calibration_factor, 2),
                hla_degrees=shot_data.hla_degrees,
                distance_mm=shot_data.distance_mm,
                elapsed_seconds=shot_data.elapsed_seconds,
            )

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

        try:
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
                saved_circ = None
                try:

                    # Adaptive frame skipping — never skip during a live putt or
                    # the post-shot tracer (every motion frame counts).
                    if self._skip_counter > 0 and not (
                        self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                        or self._post_shot_tracking
                    ):
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

                    # Set detection area based on state and tracking mode
                    if (
                        self.config.shot.extended_tracking
                        and self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                    ):
                        detect_x1 = 0
                        detect_x2 = display_frame.shape[1]
                    elif self._tracker.state == ShotState.ENTERED:
                        detect_x1 = zone.start_x1
                        detect_x2 = display_frame.shape[1]
                    else:
                        detect_x1 = zone.start_x1
                        detect_x2 = display_frame.shape[1]

                    if self._tracker.state == ShotState.ENTERED:
                        det_y1 = max(0, zone.y1 - 50)
                        det_y2 = min(display_frame.shape[0], zone.y2 + 50)
                        saved_circ = self._detector.min_circularity
                        self._detector.min_circularity = 0.0
                    elif self._tracker.state == ShotState.STARTED:
                        det_y1 = max(0, zone.y1 - 30)
                        det_y2 = min(display_frame.shape[0], zone.y2 + 30)
                        saved_circ = self._detector.min_circularity
                        self._detector.min_circularity = 0.0
                    else:
                        det_y1 = zone.y1
                        det_y2 = zone.y2
                        saved_circ = None

                    expected_r = (
                        self._tracker.start_circle[2]
                        if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED)
                        else None
                    )
                    # Expected ball position for inter-frame continuity during motion
                    # states (STARTED/ENTERED).  Prefer the most-recent tracked
                    # position (deque tail); fall back to start_circle when the trail
                    # is empty.  Left as None for IDLE/BALL_DETECTED so first-detection
                    # keeps its largest-contour behavior.
                    if self._tracker.state not in (ShotState.IDLE, ShotState.BALL_DETECTED):
                        _trail = self._tracker.positions
                        if _trail:
                            _lx, _ly, _ = _trail[-1]
                            expected_pos: tuple[int, int] | None = (int(_lx), int(_ly))
                        else:
                            expected_pos = self._tracker.start_circle[:2]
                    else:
                        expected_pos = None

                    # Two-pass detection in STARTED state (headless)
                    if self._tracker.state == ShotState.STARTED:
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=zone.start_x1,
                            zone_x2_limit=zone.start_x2,
                            zone_y1=det_y1,
                            zone_y2=det_y2,
                            timestamp=frame_time,
                            expected_radius=expected_r,
                            expected_pos=expected_pos,
                        )
                        if detection is None:
                            detection = self._detector.detect(
                                frame=display_frame,
                                zone_x1=detect_x1,
                                zone_x2_limit=detect_x2,
                                zone_y1=det_y1,
                                zone_y2=det_y2,
                                timestamp=frame_time,
                                expected_radius=expected_r,
                                expected_pos=expected_pos,
                            )
                    else:
                        # ENTERED passes expected_pos (set above); IDLE/BALL_DETECTED
                        # leaves it None so first-detection stays largest-contour.
                        detection = self._detector.detect(
                            frame=display_frame,
                            zone_x1=detect_x1,
                            zone_x2_limit=detect_x2,
                            zone_y1=det_y1,
                            zone_y2=det_y2,
                            timestamp=frame_time,
                            expected_radius=expected_r,
                            expected_pos=expected_pos,
                        )

                    # Skip tracker updates when a non-putter club is selected
                    if not self.is_putting_mode:
                        if self._tracker.state != ShotState.IDLE:
                            self._tracker.reset()
                        shot_result = None
                    else:
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

                    overlay_kwargs_hl = dict(
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
                        obs_show_zones=self.config.overlay.obs_show_zones,
                        obs_calibration_grid=self._obs_calibration_grid,
                        trail_color_name=self.config.overlay.trail_color,
                        active_trail_color_name=self.config.overlay.active_trail_color,
                        headless=True,
                    )

                    draw_overlay(
                        frame=display_frame,
                        obs_overlay_mode=self.config.overlay.obs_overlay_mode,
                        **overlay_kwargs_hl,
                    )

                    if self._pick_mode:
                        cv2.putText(
                            display_frame, "CLICK ON BALL TO PICK COLOR",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        )

                    cv2.imshow(window_name, display_frame)

                    # Adaptive skip calculation — never during a live putt/tracer.
                    process_duration = time.perf_counter() - frame_time
                    _putt_active = (
                        self._tracker.state in (ShotState.STARTED, ShotState.ENTERED)
                        or self._post_shot_tracking
                    )
                    if process_duration > self._target_process_time and not _putt_active:
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
                except Exception:
                    logger.exception(
                        "Headless loop: error processing frame — skipping"
                    )
                    continue
                finally:
                    if saved_circ is not None:
                        self._detector.min_circularity = saved_circ
        except Exception:
            logger.exception("Headless processing loop crashed — stopping")
            self._running = False

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

    # LW is always Chipping regardless of distance
    _ALWAYS_CHIPPING_CLUBS = {"LW"}
    # Other wedges use Chipping only when within _CHIPPING_MAX_YARDS
    _DISTANCE_CHIPPING_CLUBS = {"SW", "AW", "GW", "PW"}
    _PUTTER_CODES = ("PT", "PUTTER")
    _last_club: str = ""  # Debounce duplicate club change messages

    @property
    def is_putting_mode(self) -> bool:
        """Whether webcam putt tracking should be active.

        True when the GSPro-selected club is a putter, OR when no club
        has been reported yet (safe default so the user can still putt
        before touching the sim).  False for all non-putter clubs — the
        webcam tracker is paused during full swings so that motion near
        the putting zone does not register as false putts.
        """
        if not self._last_club:
            return True  # No club selected yet — allow putting
        return self._last_club.upper() in self._PUTTER_CODES

    _CHIPPING_MAX_YARDS: float = 30.0  # Chip when within this distance with a wedge (not LW)

    def _on_club_change(self, club: str, distance_to_target: float = 0.0) -> None:
        """Called when GSPro sends a club selection (code 201).

        Automatically switches OBS scenes and FS Golf PC swing mode.
        OBS/Mevo changes are debounced on duplicate clubs, but the FSG
        key send always fires — the user may manually toggle FSG between
        shots, and birdman has no way to observe that.  Re-sending the
        same key is a no-op if FSG is already in the right mode.

        Args:
            club: Club code from GSPro (e.g. "PT", "SW", "DR").
            distance_to_target: Distance to hole in yards (from GSPro).
        """
        club_upper = club.upper()
        club_changed = club_upper != self._last_club
        self._last_club = club_upper

        is_putter = club_upper in ("PT", "PUTTER")

        # Pause/resume Mevo OCR — no point running Tesseract during putting
        # UNLESS putt_fallback is enabled (then we want fresh readings
        # available when the webcam misses a fast putt).
        # Only act on actual club changes to avoid log spam.
        if club_changed and self._mevo_detector:
            want_pause = is_putter and not self.config.mevo.putt_fallback
            if want_pause and not self._mevo_paused:
                self._mevo_paused = True
                logger.info("Mevo OCR paused (putter selected)")
            elif not want_pause and self._mevo_paused:
                self._mevo_paused = False
                reason = "putt_fallback on" if is_putter else f"{club} selected"
                logger.info("Mevo OCR resumed (%s)", reason)

        # OBS scene switching — only on actual club changes
        if club_changed and self._obs is not None and self.config.obs.auto_scene_switch:
            if is_putter:
                self._obs.switch_to_putt()
            else:
                self._obs.switch_to_main()

        # FS Golf PC chipping/full swing mode:
        # - LW (lob wedge): always Chipping regardless of distance
        # - SW/AW/GW/PW: Chipping only when within _CHIPPING_MAX_YARDS
        # - All other non-putter clubs: Full Swing
        capture = getattr(self, "_mevo_capture", None)
        if capture is not None and not is_putter:
            if club_upper in self._ALWAYS_CHIPPING_CLUBS:
                use_chipping = True
            elif club_upper in self._DISTANCE_CHIPPING_CLUBS:
                # Require a known, short distance.  distance=0 means
                # GSPro hasn't resolved it yet — default to Full Swing.
                use_chipping = 0 < distance_to_target <= self._CHIPPING_MAX_YARDS
            else:
                use_chipping = False
            if use_chipping:
                capture.send_key("c")
                logger.info(
                    "FS Golf PC → Chipping mode (%s, %.1f yds to target)",
                    club, distance_to_target,
                )
            else:
                capture.send_key("f")
                logger.info(
                    "FS Golf PC → Full Swing mode (%s, %.1f yds to target)",
                    club, distance_to_target,
                )

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

    def _recent_mevo_reading(self) -> tuple[float, float] | None:
        """Return (ball_speed_mph, hla_deg) if a fresh Mevo reading exists.

        Freshness is governed by ``mevo.putt_fallback_max_age_s``.
        Returns None if no reading, expired, or fallback disabled.
        """
        if not self.config.mevo.putt_fallback:
            return None
        with self._mevo_last_lock:
            reading = self._mevo_last
        if reading is None:
            return None
        ts, mph, hla = reading
        age = time.perf_counter() - ts
        if age > self.config.mevo.putt_fallback_max_age_s:
            return None
        return mph, hla

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

        # Stash the reading for potential putt-fallback use. Always store,
        # regardless of mode — we'll use it only when webcam asks.
        with self._mevo_last_lock:
            self._mevo_last = (
                time.perf_counter(),
                float(shot.ball_speed),
                float(shot.launch_direction),
            )

        # In putting mode with fallback enabled, DON'T forward to GSPro here —
        # the webcam path will decide whether to use this reading. Forwarding
        # twice would cause duplicate/conflicting shots.
        if self.is_putting_mode and self.config.mevo.putt_fallback:
            logger.debug("Mevo reading stashed for putt fallback (not forwarded)")
            return

        # Only relay to GSPro if configured (disable when LM connects directly)
        if self.config.mevo.send_to_gspro:
            self._gspro.set_shot_cooldown(3)

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
        # Drain frame queue to free numpy arrays
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        cv2.destroyAllWindows()
        logger.info("Putting app stopped")
