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
        self._ball_detected = False  # Set True when ball is ready for shot
        self._shot_cooldown: int = 0  # Heartbeat cycles to skip after a shot

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

    @property
    def ball_detected(self) -> bool:
        return self._ball_detected

    @ball_detected.setter
    def ball_detected(self, value: bool) -> None:
        self._ball_detected = value

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
        carry_distance: float = 0.0,
        aoa: float = 0.0,
        club_path: float = 0.0,
        dynamic_loft: float = 0.0,
        face_to_target: float = 0.0,
        lateral_impact: float = 0.0,
        vertical_impact: float = 0.0,
    ) -> GSProResponse:
        """Send full shot data from a launch monitor to GSPro.

        Returns:
            GSProResponse indicating success or failure.
        """
        if not self._connected.is_set() and not self._connect_socket():
            return GSProResponse(success=False, message="Not connected to GSPro")

        self._shot_number += 1
        self._shot_cooldown = 3  # Skip 3 heartbeat cycles (~15s) after shot
        message = self._build_full_shot_message(
            ball_speed, vla, hla, total_spin, spin_axis,
            back_spin, side_spin, club_speed,
            carry_distance=carry_distance,
            aoa=aoa, club_path=club_path, dynamic_loft=dynamic_loft,
            face_to_target=face_to_target,
            lateral_impact=lateral_impact, vertical_impact=vertical_impact,
        )
        return self._send_json(message)

    # --- GSPro Direct (Socket) ---

    def _open_socket(self) -> bool:
        """Open a TCP socket to GSPro (no heartbeat thread).

        Used by both initial connect and reconnects from the heartbeat loop.
        """
        host = self._settings.gspro_host
        port = self._settings.gspro_port

        # Close any existing socket
        with self._lock:
            if self._socket is not None:
                with contextlib.suppress(OSError):
                    self._socket.close()
                self._socket = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(10.0)

            with self._lock:
                self._socket = sock

            self._connected.set()
            logger.info("Connected to GSPro at %s:%d", host, port)
            return True

        except (OSError, ConnectionRefusedError) as e:
            logger.error("Failed to connect to GSPro at %s:%d: %s", host, port, e)
            return False

    def _connect_socket(self) -> bool:
        """Connect to GSPro and start the heartbeat thread."""
        if not self._open_socket():
            return False

        # Only start heartbeat if not already running
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

        return True

    def _send_socket(self, speed_mph: float, hla_degrees: float) -> GSProResponse:
        """Send shot via TCP socket."""
        if not self._connected.is_set() and not self._connect_socket():
            return GSProResponse(success=False, message="Not connected to GSPro")

        self._shot_number += 1
        self._shot_cooldown = 3  # Skip 3 heartbeat cycles (~15s) after shot
        message = self._build_shot_message(speed_mph, hla_degrees)
        return self._send_json(message)

    def _send_json(self, message: dict[str, object]) -> GSProResponse:
        """Send a JSON message over the socket and wait briefly for response.

        Uses a short recv timeout (2s) since GSPro sometimes processes
        shots without responding. A recv timeout is treated as likely
        success — the connection is NOT killed.
        """
        with self._lock:
            if self._socket is None:
                return GSProResponse(success=False, message="Socket not connected")

            try:
                data = json.dumps(message).encode("utf-8")
                self._socket.sendall(data)
            except OSError as e:
                logger.error("Failed to send to GSPro: %s", e)
                self._connected.clear()
                return GSProResponse(success=False, message=str(e))

            # Try to read response with a short timeout
            try:
                self._socket.settimeout(2.0)
                response_data = self._socket.recv(4096)
                self._socket.settimeout(10.0)

                if response_data:
                    response = json.loads(response_data.decode("utf-8"))
                    code = response.get("Code", -1)
                    msg = response.get("Message", "")
                    if code == 200:
                        logger.info("GSPro accepted shot: %s", msg)
                        return GSProResponse(success=True, message=msg)
                    logger.warning("GSPro rejected shot (code %d): %s", code, msg)
                    return GSProResponse(success=False, message=msg)

                return GSProResponse(success=True)

            except TimeoutError:
                # GSPro often processes shots without responding — treat as OK
                self._socket.settimeout(10.0)
                logger.debug("GSPro recv timeout (shot likely accepted)")
                return GSProResponse(success=True, message="sent (no response)")
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to read GSPro response: %s", e)
                self._connected.clear()
                return GSProResponse(success=False, message=str(e))

    def _send_heartbeat_json(self, message: dict[str, object]) -> GSProResponse:
        """Send a heartbeat message without waiting for a response.

        GSPro does not respond to heartbeats, so blocking on recv causes
        a 10-second timeout that cascades into connection failures.
        """
        with self._lock:
            if self._socket is None:
                return GSProResponse(success=False, message="Socket not connected")

            try:
                data = json.dumps(message).encode("utf-8")
                self._socket.sendall(data)

                # Drain any pending response data without blocking
                self._socket.setblocking(False)
                try:
                    self._socket.recv(4096)
                except BlockingIOError:
                    pass  # No data available — expected for heartbeats
                finally:
                    self._socket.setblocking(True)
                    self._socket.settimeout(10.0)

                return GSProResponse(success=True)

            except OSError as e:
                logger.error("Heartbeat send failed: %s", e)
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
        carry_distance: float = 0.0,
        aoa: float = 0.0,
        club_path: float = 0.0,
        dynamic_loft: float = 0.0,
        face_to_target: float = 0.0,
        lateral_impact: float = 0.0,
        vertical_impact: float = 0.0,
    ) -> dict[str, object]:
        """Build GSPro Open Connect v1 message with full ball + club data."""
        has_club = club_speed > 0 or aoa != 0 or club_path != 0
        ball_data: dict[str, object] = {
            "Speed": round(ball_speed, 2),
            "SpinAxis": round(spin_axis, 2),
            "TotalSpin": round(total_spin, 2),
            "BackSpin": round(back_spin, 2),
            "SideSpin": round(side_spin, 2),
            "HLA": round(hla, 2),
            "VLA": round(vla, 2),
        }
        if carry_distance > 0:
            ball_data["CarryDistance"] = round(carry_distance, 1)

        club_data: dict[str, object] = {}
        if club_speed > 0:
            club_data["Speed"] = round(club_speed, 2)
        if aoa != 0:
            club_data["AngleOfAttack"] = round(aoa, 2)
        if club_path != 0:
            club_data["Path"] = round(club_path, 2)
        if dynamic_loft != 0:
            club_data["Loft"] = round(dynamic_loft, 2)
        if face_to_target != 0:
            club_data["FaceToTarget"] = round(face_to_target, 2)
        if lateral_impact != 0:
            club_data["HorizontalFaceImpact"] = round(lateral_impact, 2)
        if vertical_impact != 0:
            club_data["VerticalFaceImpact"] = round(vertical_impact, 2)

        msg: dict[str, object] = {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": ball_data,
            "ShotDataOptions": {
                "ContainsBallData": True,
                "ContainsClubData": has_club,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": True,
                "IsHeartBeat": False,
            },
        }
        if club_data:
            msg["ClubData"] = club_data
        return msg

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
            },
            "ShotDataOptions": {
                "ContainsBallData": False,
                "ContainsClubData": False,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": self._ball_detected,
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

            # Skip heartbeat briefly after a shot so GSPro can process it
            # without a zero-data heartbeat arriving immediately after
            if self._shot_cooldown > 0:
                self._shot_cooldown -= 1
                continue

            if self._connected.is_set():
                message = self._build_heartbeat_message()
                result = self._send_heartbeat_json(message)
                if not result.success:
                    logger.warning("Heartbeat failed, will attempt reconnect")
                    self._connected.clear()
                    backoff = 1.0
            else:
                # Reconnect using _open_socket (not _connect_socket which
                # would spawn another heartbeat thread)
                logger.info("Attempting reconnect (backoff=%.1fs)", backoff)
                if self._open_socket():
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
