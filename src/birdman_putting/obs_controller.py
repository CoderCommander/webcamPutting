"""OBS WebSocket controller for displaying shot data on projector overlays."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdman_putting.config import OBSSettings
    from birdman_putting.mevo.detector import MevoShotData

logger = logging.getLogger(__name__)


class OBSController:
    """Controls OBS via WebSocket to display shot data.

    Uses obsws-python (OBS WebSocket v5) to switch scenes and update
    text sources with shot metrics after full swings and putts.
    """

    def __init__(self, settings: OBSSettings) -> None:
        self._settings = settings
        self._client: object | None = None
        self._idle_timer: threading.Timer | None = None
        self._created_sources: set[str] = set()  # Track auto-created text sources

    def connect(self) -> bool:
        """Connect to OBS WebSocket server.

        Returns:
            True if connection successful.
        """
        try:
            import obsws_python as obs

            self._client = obs.ReqClient(
                host=self._settings.host,
                port=self._settings.port,
                password=self._settings.password or None,
            )
            logger.info(
                "Connected to OBS at %s:%d",
                self._settings.host, self._settings.port,
            )

            # Log available scenes so user can configure correct names
            try:
                cl: obs.ReqClient = self._client  # type: ignore[assignment]
                resp = cl.get_scene_list()
                names = [s["sceneName"] for s in resp.scenes]  # type: ignore[union-attr]
                logger.info("OBS scenes available: %s", names)

                # Warn about missing configured scenes
                for label, name in [
                    ("mevo_scene", self._settings.mevo_scene),
                    ("putt_scene", self._settings.putt_scene),
                    ("idle_scene", self._settings.idle_scene),
                ]:
                    if name not in names:
                        logger.warning(
                            "OBS scene '%s' (config: %s) not found — "
                            "create it in OBS or update config.toml [obs] %s",
                            name, label, label,
                        )
            except Exception:
                pass  # Non-fatal — scene listing is informational

            return True
        except Exception as e:
            logger.error("Failed to connect to OBS: %s", e)
            self._client = None
            return False

    def disconnect(self) -> None:
        """Disconnect from OBS."""
        self._cancel_idle_timer()
        if self._client is not None:
            try:
                self._client.base_client.ws.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._client = None
            logger.info("Disconnected from OBS")

    def show_mevo_shot(self, shot: MevoShotData) -> None:
        """Switch to Mevo scene and populate text sources with shot data."""
        if self._client is None:
            return

        self._cancel_idle_timer()

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]

            # Switch to Mevo scene
            scene = self._settings.mevo_scene
            cl.set_current_program_scene(scene)

            # Update text sources with shot data
            self._set_text(cl, "BallSpeed", f"{shot.ball_speed:.1f}", scene)
            self._set_text(cl, "LaunchAngle", f"{shot.launch_angle:.1f}", scene)
            self._set_text(cl, "LaunchDirection", f"{shot.launch_direction:+.1f}", scene)
            self._set_text(cl, "SpinRate", f"{int(shot.spin_rate)}", scene)
            self._set_text(cl, "SpinAxis", f"{shot.spin_axis:+.1f}", scene)

            if shot.club_speed > 0:
                self._set_text(cl, "ClubSpeed", f"{shot.club_speed:.1f}", scene)
            if shot.smash_factor > 0:
                self._set_text(cl, "SmashFactor", f"{shot.smash_factor:.2f}", scene)
            if shot.carry_distance > 0:
                self._set_text(cl, "CarryDistance", f"{shot.carry_distance:.0f}", scene)
            if shot.total_distance > 0:
                self._set_text(cl, "TotalDistance", f"{shot.total_distance:.0f}", scene)
            if shot.apex_height > 0:
                self._set_text(cl, "ApexHeight", f"{shot.apex_height:.0f}", scene)
            if shot.flight_time > 0:
                self._set_text(cl, "FlightTime", f"{shot.flight_time:.1f}", scene)
            if shot.descent_angle > 0:
                self._set_text(cl, "DescentAngle", f"{shot.descent_angle:.1f}", scene)
            if shot.curve != 0:
                self._set_text(cl, "Curve", f"{shot.curve:+.1f}", scene)
            if shot.roll_distance > 0:
                self._set_text(cl, "RollDistance", f"{shot.roll_distance:.0f}", scene)

            logger.info("OBS: Mevo shot data displayed")
        except Exception as e:
            logger.error("OBS: Failed to show Mevo shot: %s", e)

        self._schedule_idle()

    def show_putt(self, speed: float, hla: float) -> None:
        """Switch to putt scene and populate text sources."""
        if self._client is None:
            return

        self._cancel_idle_timer()

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]

            scene = self._settings.putt_scene
            cl.set_current_program_scene(scene)
            self._set_text(cl, "PuttSpeed", f"{speed:.1f}", scene)
            self._set_text(cl, "PuttHLA", f"{hla:+.1f}", scene)

            logger.info("OBS: Putt data displayed")
        except Exception as e:
            logger.error("OBS: Failed to show putt: %s", e)

        self._schedule_idle()

    def show_idle(self) -> None:
        """Switch back to idle/default scene."""
        if self._client is None:
            return

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            cl.set_current_program_scene(self._settings.idle_scene)
            logger.info("OBS: Switched to idle scene")
        except Exception as e:
            logger.error("OBS: Failed to switch to idle scene: %s", e)

    def _set_text(self, client: object, source_name: str, text: str, scene_name: str) -> None:
        """Update an OBS text source, auto-creating it if missing."""
        try:
            client.set_input_settings(  # type: ignore[union-attr]
                source_name, {"text": text}, overlay=True,
            )
        except Exception:
            # Source doesn't exist — try to create it
            if source_name in self._created_sources:
                return  # Already failed to create, don't retry
            self._create_text_source(client, source_name, text, scene_name)

    def _create_text_source(
        self, client: object, source_name: str, text: str, scene_name: str,
    ) -> None:
        """Create a GDI+ (Windows) or FreeType2 (macOS/Linux) text source."""
        import sys

        settings = {
            "text": text,
            "font": {"face": "Arial", "size": 36, "style": "Bold"},
        }

        # Windows uses GDI+, macOS/Linux use FreeType2
        if sys.platform == "win32":
            input_kinds = ["text_gdiplus_v2", "text_gdiplus"]
        else:
            input_kinds = ["text_ft2_source_v2", "text_ft2_source"]

        for kind in input_kinds:
            try:
                client.create_input(  # type: ignore[union-attr]
                    scene_name, source_name, kind, settings, True,
                )
                self._created_sources.add(source_name)
                logger.info(
                    "OBS: Auto-created text source '%s' (%s) in '%s'",
                    source_name, kind, scene_name,
                )
                return
            except Exception:
                continue  # Try next input kind

        # All kinds failed
        self._created_sources.add(source_name)  # Don't retry
        logger.warning("OBS: Could not create text source '%s' in '%s'", source_name, scene_name)

    def _schedule_idle(self) -> None:
        """Schedule a return to idle scene after display_duration."""
        self._idle_timer = threading.Timer(
            self._settings.display_duration, self.show_idle,
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        """Cancel any pending idle timer."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
