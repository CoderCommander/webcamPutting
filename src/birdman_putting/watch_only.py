"""Lightweight GSPro→OBS scene-switching daemon.

When another launch monitor (e.g. the Flightscope API tool) is the primary
source for shot data on GSPro Open Connect v1 (port 921), birdman in
watch-only mode connects alongside as a passive observer. It receives
GSPro's club-selection broadcasts (code 201) and drives the configured
OBS scene transitions — no camera, no Mevo OCR, no shot processing.

This relies on GSPro broadcasting code 201 events to all connected clients,
not just the primary LM. If your tool blocks that broadcast, this won't
get events; the log will show "Connected to GSPro" but no club selections.

Usage:
    python -m birdman_putting --watch-only
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import replace

from birdman_putting.config import AppConfig
from birdman_putting.gspro_client import GSProClient
from birdman_putting.gspro_watcher import GSProWatcher
from birdman_putting.obs_controller import OBSController

logger = logging.getLogger(__name__)


_PUTTER_CLUBS = {"PT", "PUTTER"}
# Use a different DeviceID so GSPro tracks us as a separate client and
# doesn't confuse our heartbeats with the primary launch monitor's.
_WATCH_DEVICE_ID = "BirdmanWatchOnly"


class _WatchOnlyApp:
    """Holds shared state for the watch-only daemon."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # Override DeviceID to avoid clobbering the primary LM's identity.
        watch_conn = replace(config.connection, device_id=_WATCH_DEVICE_ID)
        self._gspro = GSProClient(watch_conn, on_club_change=self._on_club_change)
        self._watcher: GSProWatcher | None = None
        self._obs: OBSController | None = None
        self._last_club: str = ""
        self._stop = threading.Event()

    # ---- GSPro club-change handler ----

    def _on_club_change(self, club: str) -> None:
        """Drive OBS scene switching on GSPro club selection."""
        club_upper = (club or "").upper()
        if club_upper == self._last_club:
            return  # Debounce duplicate events
        self._last_club = club_upper

        is_putter = club_upper in _PUTTER_CLUBS
        logger.info(
            "Club change: %s -> %s scene",
            club_upper, "putt" if is_putter else "main",
        )

        if self._obs is None or not self._config.obs.auto_scene_switch:
            return

        try:
            if is_putter:
                self._obs.switch_to_putt()
            else:
                self._obs.switch_to_main()
        except Exception:
            logger.exception("OBS scene switch failed")

    # ---- Lifecycle ----

    def start(self) -> bool:
        """Connect to GSPro and OBS. Returns True on success."""
        # OBS is the actual point of this mode — fail loudly if unreachable.
        if not self._config.obs.enabled:
            logger.error(
                "OBS is disabled in config ([obs] enabled = false). "
                "Watch-only mode has nothing to do — enable OBS first.",
            )
            return False

        self._obs = OBSController(self._config.obs)
        if not self._obs.connect():
            logger.error(
                "Failed to connect to OBS at %s:%d. "
                "Is OBS running with WebSocket enabled?",
                self._config.obs.host, self._config.obs.port,
            )
            return False

        # GSPro Open Connect listener — best-effort; tolerates being down.
        # The client background-reconnects with backoff.
        if not self._gspro.connect():
            logger.warning(
                "GSPro Open Connect not reachable at %s:%d — will retry "
                "in background. If your launch-monitor connector doesn't "
                "use port 921, this will never connect; rely on the GSPro "
                "window watcher (enable [gspro_watcher]) instead.",
                self._config.connection.gspro_host,
                self._config.connection.gspro_port,
            )

        # GSPro window watcher — OCR-based fallback signal source.
        if self._config.gspro_watcher.enabled:
            self._watcher = GSProWatcher(
                self._config.gspro_watcher,
                on_club_change=self._on_club_change,
            )
            if not self._watcher.start():
                logger.warning(
                    "GSPro window watcher failed to start — see prior errors. "
                    "Run --calibrate-gspro-club to set the club ROI.",
                )
                self._watcher = None
        else:
            logger.info(
                "GSPro window watcher disabled. If port 921 isn't reachable, "
                "enable it: set [gspro_watcher].enabled = true and run "
                "--calibrate-gspro-club to define the club ROI.",
            )

        logger.info(
            "Watch-only mode active. DeviceID=%s. "
            "Sources: GSPro 921 listener=%s, window watcher=%s. "
            "Will switch OBS putt_scene='%s' / idle_scene='%s'.",
            _WATCH_DEVICE_ID,
            "on" if self._gspro.is_connected else "pending",
            "on" if self._watcher is not None else "off",
            self._config.obs.putt_scene,
            self._config.obs.idle_scene,
        )
        return True

    def run_forever(self) -> None:
        """Block until interrupted, then clean up."""
        try:
            while not self._stop.is_set():
                self._stop.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._stop.set()

    def _cleanup(self) -> None:
        if self._watcher is not None:
            with contextlib.suppress(Exception):
                self._watcher.stop()
        with contextlib.suppress(Exception):
            self._gspro.disconnect()
        if self._obs is not None:
            with contextlib.suppress(Exception):
                self._obs.disconnect()


def run_watch_only(config: AppConfig) -> None:
    """Entry point — runs until Ctrl-C."""
    app = _WatchOnlyApp(config)
    if not app.start():
        return
    try:
        app.run_forever()
    except Exception:
        logger.exception("Watch-only mode crashed")
        app.stop()
