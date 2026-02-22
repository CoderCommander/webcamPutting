"""GSPro Open Connect v1 API client and HTTP middleware fallback."""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum

from birdman_putting.config import ConnectionSettings

logger = logging.getLogger(__name__)


class ConnectionMode(Enum):
    GSPRO_DIRECT = "gspro_direct"
    HTTP_MIDDLEWARE = "http_middleware"


@dataclass
class GSProResponse:
    """Response from GSPro after sending a shot."""

    success: bool
    message: str = ""


class GSProClient:
    """Manages connection to GSPro for sending shot data.

    Supports two modes:
    - gspro_direct: TCP socket to GSPro Open Connect v1 API (port 921)
    - http_middleware: HTTP POST to legacy middleware connector (port 8888)
    """

    def __init__(self, settings: ConnectionSettings):
        self._settings = settings
        self._socket: socket.socket | None = None
        self._shot_number: int = 0
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        if self._settings.mode == ConnectionMode.HTTP_MIDDLEWARE.value:
            return True  # HTTP is stateless, always "connected"
        return self._connected.is_set()

    @property
    def mode(self) -> str:
        return self._settings.mode

    @property
    def shot_number(self) -> int:
        return self._shot_number

    def connect(self) -> bool:
        """Establish connection to GSPro.

        Returns:
            True if connection successful.
        """
        if self._settings.mode == ConnectionMode.HTTP_MIDDLEWARE.value:
            logger.info("Using HTTP middleware mode (%s)", self._settings.http_url)
            return True

        return self._connect_socket()

    def disconnect(self) -> None:
        """Close connection and stop heartbeat."""
        self._running = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=10)
            self._heartbeat_thread = None

        with self._lock:
            if self._socket is not None:
                with contextlib.suppress(OSError):
                    self._socket.close()
                self._socket = None
                self._connected.clear()
                logger.info("Disconnected from GSPro")

    def send_shot(self, speed_mph: float, hla_degrees: float) -> GSProResponse:
        """Send shot data to GSPro.

        Args:
            speed_mph: Ball speed in MPH.
            hla_degrees: Horizontal launch angle in degrees.

        Returns:
            GSProResponse indicating success or failure.
        """
        if self._settings.mode == ConnectionMode.HTTP_MIDDLEWARE.value:
            return self._send_http(speed_mph, hla_degrees)
        else:
            return self._send_socket(speed_mph, hla_degrees)

    def send_full_shot(
        self,
        ball_speed: float,
        vla: float,
        hla: float,
        total_spin: float,
        spin_axis: float,
        back_spin: float,
        side_spin: float,
        club_speed: float = 0.0,
    ) -> GSProResponse:
        """Send full shot data from a launch monitor to GSPro.

        Args:
            ball_speed: Ball speed in MPH.
            vla: Vertical launch angle in degrees.
            hla: Horizontal launch angle in degrees.
            total_spin: Total spin in RPM.
            spin_axis: Spin axis in degrees.
            back_spin: Back spin in RPM.
            side_spin: Side spin in RPM.
            club_speed: Club head speed in MPH (0 if unavailable).

        Returns:
            GSProResponse indicating success or failure.
        """
        if not self._connected.is_set() and not self._connect_socket():
            return GSProResponse(success=False, message="Not connected to GSPro")

        self._shot_number += 1
        message = self._build_full_shot_message(
            ball_speed, vla, hla, total_spin, spin_axis,
            back_spin, side_spin, club_speed,
        )
        return self._send_json(message)

    # --- GSPro Direct (Socket) ---

    def _connect_socket(self) -> bool:
        """Connect to GSPro Open Connect API via TCP socket."""
        host = self._settings.gspro_host
        port = self._settings.gspro_port

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(10.0)

            with self._lock:
                self._socket = sock

            self._connected.set()
            logger.info("Connected to GSPro at %s:%d", host, port)

            # Start heartbeat thread
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

            return True

        except (OSError, ConnectionRefusedError) as e:
            logger.error("Failed to connect to GSPro at %s:%d: %s", host, port, e)
            return False

    def _send_socket(self, speed_mph: float, hla_degrees: float) -> GSProResponse:
        """Send shot via TCP socket."""
        if not self._connected.is_set() and not self._connect_socket():
            return GSProResponse(success=False, message="Not connected to GSPro")

        self._shot_number += 1
        message = self._build_shot_message(speed_mph, hla_degrees)
        return self._send_json(message)

    def _send_json(self, message: dict[str, object]) -> GSProResponse:
        """Send a JSON message over the socket."""
        with self._lock:
            if self._socket is None:
                return GSProResponse(success=False, message="Socket not connected")

            try:
                data = json.dumps(message).encode("utf-8")
                self._socket.sendall(data)

                # Read response
                response_data = self._socket.recv(4096)
                if response_data:
                    response = json.loads(response_data.decode("utf-8"))
                    code = response.get("Code", -1)
                    msg = response.get("Message", "")
                    if code == 200:
                        logger.info("GSPro accepted shot: %s", msg)
                        return GSProResponse(success=True, message=msg)
                    else:
                        logger.warning("GSPro rejected shot (code %d): %s", code, msg)
                        return GSProResponse(success=False, message=msg)

                return GSProResponse(success=True)

            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to send to GSPro: %s", e)
                self._connected.clear()
                return GSProResponse(success=False, message=str(e))

    def _build_shot_message(self, speed_mph: float, hla_degrees: float) -> dict[str, object]:
        """Build GSPro Open Connect v1 shot message for putting."""
        return {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": {
                "Speed": round(speed_mph, 2),
                "SpinAxis": 0.0,
                "TotalSpin": 0.0,
                "BackSpin": 0.0,
                "SideSpin": 0.0,
                "HLA": round(hla_degrees, 2),
                "VLA": 0.0,
                "CarryDistance": 0.0,
            },
            "ShotDataOptions": {
                "ContainsBallData": True,
                "ContainsClubData": False,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": True,
                "IsHeartBeat": False,
            },
        }

    def _build_full_shot_message(
        self,
        ball_speed: float,
        vla: float,
        hla: float,
        total_spin: float,
        spin_axis: float,
        back_spin: float,
        side_spin: float,
        club_speed: float,
    ) -> dict[str, object]:
        """Build GSPro Open Connect v1 message with full ball data."""
        return {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": {
                "Speed": round(ball_speed, 2),
                "SpinAxis": round(spin_axis, 2),
                "TotalSpin": round(total_spin, 2),
                "BackSpin": round(back_spin, 2),
                "SideSpin": round(side_spin, 2),
                "HLA": round(hla, 2),
                "VLA": round(vla, 2),
                "CarryDistance": 0.0,
            },
            "ShotDataOptions": {
                "ContainsBallData": True,
                "ContainsClubData": club_speed > 0,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": True,
                "IsHeartBeat": False,
            },
        }

    def _build_heartbeat_message(self) -> dict[str, object]:
        """Build GSPro heartbeat message."""
        return {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": {
                "Speed": 0.0,
                "SpinAxis": 0.0,
                "TotalSpin": 0.0,
                "BackSpin": 0.0,
                "SideSpin": 0.0,
                "HLA": 0.0,
                "VLA": 0.0,
                "CarryDistance": 0.0,
            },
            "ShotDataOptions": {
                "ContainsBallData": False,
                "ContainsClubData": False,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": False,
                "IsHeartBeat": True,
            },
        }

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to maintain GSPro connection."""
        backoff = 1.0
        while self._running:
            time.sleep(self._settings.heartbeat_interval)

            if not self._running:
                break

            if self._connected.is_set():
                message = self._build_heartbeat_message()
                result = self._send_json(message)
                if not result.success:
                    logger.warning("Heartbeat failed, will attempt reconnect")
                    self._connected.clear()
                    backoff = 1.0
            else:
                # Attempt reconnect with exponential backoff
                logger.info("Attempting reconnect (backoff=%.1fs)", backoff)
                if self._connect_socket():
                    backoff = 1.0
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    # --- HTTP Middleware (Legacy) ---

    def _send_http(self, speed_mph: float, hla_degrees: float) -> GSProResponse:
        """Send shot via HTTP POST to legacy middleware."""
        data = {
            "ballData": {
                "BallSpeed": f"{speed_mph:.2f}",
                "TotalSpin": 0,
                "LaunchDirection": f"{hla_degrees:.2f}",
            }
        }

        try:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                self._settings.http_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                logger.info("HTTP response: %s", result.get("result", "OK"))
                return GSProResponse(success=True, message=str(result))

        except urllib.error.URLError as e:
            logger.error("HTTP POST failed: %s", e)
            return GSProResponse(success=False, message=str(e))
        except Exception as e:
            logger.error("HTTP POST error: %s", e)
            return GSProResponse(success=False, message=str(e))
