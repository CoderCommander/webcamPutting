"""OBS WebSocket controller for displaying shot data on projector overlays."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
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

    def __init__(
        self,
        settings: OBSSettings,
        on_idle: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        self._client: object | None = None
        self._idle_timer: threading.Timer | None = None
        self._created_sources: set[str] = set()  # Track auto-created text sources
        self._on_idle = on_idle  # Called when transitioning back to idle scene
        # Reconnect machinery: obsws calls run on several threads (GSPro
        # listener, mevo poll, idle Timer, UI) -- serialize client swaps and
        # rate-limit reconnect attempts so an OBS restart mid-session heals
        # itself instead of silently killing scene switching forever.
        self._client_lock = threading.Lock()
        self._last_connect_attempt: float = 0.0
        self._closed = False  # set by disconnect(); blocks auto-reconnect
        # Scene that failed to apply while OBS was down: replayed once on the
        # next successful (re)connect so the projector is not stuck on the
        # wrong view for a whole putting spell.
        self._pending_scene: str | None = None

    def connect(self) -> bool:
        """Connect to OBS WebSocket server.

        Returns:
            True if connection successful.
        """
        try:
            import obsws_python as obs

            # timeout=3 is essential: obsws-python defaults to timeout=None,
            # and a hung-but-alive OBS then blocks the CALLING thread forever
            # -- measured blocking the GSPro listener thread (club gating,
            # Mevo pause/resume, FS Golf key sends all stall behind it).
            self._client = obs.ReqClient(
                host=self._settings.host,
                port=self._settings.port,
                password=self._settings.password or None,
                timeout=3,
            )
            self._closed = False
            pending, self._pending_scene = self._pending_scene, None
            if pending:
                try:
                    self._client.set_current_program_scene(pending)
                    logger.info("OBS: applied pending scene '%s'", pending)
                except Exception:
                    logger.warning("OBS: pending scene '%s' failed", pending)
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
        """Disconnect from OBS (and stop auto-reconnecting)."""
        self._closed = True
        self._cancel_idle_timer()
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.base_client.ws.close()  # type: ignore[union-attr]
            self._client = None
            logger.info("Disconnected from OBS")

    def _mark_disconnected(self) -> None:
        """Drop the client after a failed request so the next call reconnects.

        Without this, a dead client object lived forever: after an OBS
        restart every scene switch failed silently for the rest of the
        session, with Stop/Start (and a 50-90s camera reopen) the only cure.
        """
        with self._client_lock:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._client.base_client.ws.close()  # type: ignore[attr-defined]
                self._client = None
        logger.warning("OBS request failed -- connection dropped, will retry")

    def _ensure_client(self) -> bool:
        """Return True if a client exists, lazily reconnecting at most every 5s.

        Non-blocking for concurrent callers: if another thread is already
        reconnecting, return False immediately rather than queueing a second
        3s connect behind it.
        """
        if self._client is not None:
            return True
        if self._closed:
            return False
        import time as _time
        if not self._client_lock.acquire(blocking=False):
            return False
        try:
            if self._client is not None:
                return True
            now = _time.monotonic()
            if now - self._last_connect_attempt < 5.0:
                return False
            self._last_connect_attempt = now
        finally:
            self._client_lock.release()
        # connect() runs outside the lock: it blocks up to ~3s.
        return self.connect()

    def show_mevo_shot(self, shot: MevoShotData) -> None:
        """Switch to Mevo scene and populate text sources with shot data."""
        if not self._ensure_client():
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
            self._mark_disconnected()

        self._schedule_idle()

    def show_putt(self, speed: float, hla: float) -> None:
        """Switch to putt scene and populate text sources.

        When auto_scene_switch is on, the scene is already switched by the
        club change callback — skip text updates and idle timer entirely.
        """
        if not self._ensure_client():
            return

        if self._settings.auto_scene_switch:
            # Scene already switched by club change — no text overlay needed
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
            self._mark_disconnected()

        self._schedule_idle()

    def show_idle(self) -> None:
        """Switch back to idle/default scene and clear trail."""
        if not self._ensure_client():
            return

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            cl.set_current_program_scene(self._settings.idle_scene)
            logger.info("OBS: Switched to idle scene")
        except Exception as e:
            logger.error("OBS: Failed to switch to idle scene: %s", e)
            self._mark_disconnected()

        if self._on_idle:
            self._on_idle()

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def current_scene(self) -> str | None:
        """Return the active program scene name, or None on failure."""
        if self._client is None:
            return None
        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            resp = cl.get_current_program_scene()
            return getattr(resp, "current_program_scene_name", None) or getattr(
                resp, "scene_name", None,
            )
        except Exception:
            self._mark_disconnected()
            return None

    def switch_to_scene(self, scene_name: str) -> bool:
        """Switch OBS to the named scene. Returns True on success."""
        if not scene_name or not self._ensure_client():
            return False
        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            cl.set_current_program_scene(scene_name)
            logger.info("OBS: Switched to scene '%s'", scene_name)
            return True
        except Exception as e:
            logger.error("OBS: Failed to switch to scene '%s': %s", scene_name, e)
            self._mark_disconnected()
            return False

    def switch_to_putt(self) -> None:
        """Switch to putt scene (no idle timer — stays until club changes)."""
        if not self._ensure_client():
            self._pending_scene = self._settings.putt_scene
            return

        self._cancel_idle_timer()

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            cl.set_current_program_scene(self._settings.putt_scene)
            logger.info("OBS: Switched to putt scene (putter selected)")
        except Exception as e:
            logger.error("OBS: Failed to switch to putt scene: %s", e)
            self._pending_scene = self._settings.putt_scene
            self._mark_disconnected()

    def switch_to_main(self) -> None:
        """Switch to main/idle scene (non-putter club selected)."""
        if not self._ensure_client():
            self._pending_scene = self._settings.idle_scene
            return

        self._cancel_idle_timer()

        try:
            import obsws_python as obs  # noqa: F811

            cl: obs.ReqClient = self._client  # type: ignore[assignment]
            cl.set_current_program_scene(self._settings.idle_scene)
            logger.info("OBS: Switched to main scene (non-putter selected)")
        except Exception as e:
            logger.error("OBS: Failed to switch to main scene: %s", e)
            self._pending_scene = self._settings.idle_scene
            self._mark_disconnected()

        if self._on_idle:
            self._on_idle()

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
        """Create a text source in OBS, trying all available text input kinds."""
        settings = {
            "text": text,
            "font": {"face": "Arial", "size": 36, "style": "Bold"},
        }

        # Query OBS for available input kinds and find text-related ones
        input_kinds: list[str] = []
        try:
            resp = client.get_input_kind_list()  # type: ignore[union-attr]
            all_kinds = resp.input_kinds  # type: ignore[union-attr]
            # Look for text-related kinds (gdiplus, freetype, etc.)
            input_kinds = [
                k for k in all_kinds
                if "text" in k.lower() or "gdi" in k.lower() or "ft2" in k.lower()
            ]
            if input_kinds:
                logger.debug("OBS text input kinds available: %s", input_kinds)
        except Exception:
            pass  # Fall back to hardcoded list

        # Fall back to common names if query returned nothing
        if not input_kinds:
            import sys
            if sys.platform == "win32":
                input_kinds = [
                    "text_gdiplus_v3", "text_gdiplus_v2", "text_gdiplus",
                ]
            else:
                input_kinds = [
                    "text_ft2_source_v2", "text_ft2_source",
                ]

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
        logger.warning(
            "OBS: Could not create text source '%s' in '%s' "
            "(tried: %s)",
            source_name, scene_name, input_kinds,
        )

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
