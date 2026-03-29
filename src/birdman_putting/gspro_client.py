"""GSPro Open Connect v1 API client and HTTP middleware fallback."""

from __future__ import annotations

import contextlib
import json
import logging
import select
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
            sock.settimeout(2.0)  # Match MLM2PRO connector timeout

            with self._lock:
                self._socket = sock

            self._connected.set()
            logger.info("Connected to GSPro at %s:%d", host, port)
            return True

        except (OSError, ConnectionRefusedError) as e:
            logger.error("Failed to connect to GSPro at %s:%d: %s", host, port, e)
            return False

    def _connect_socket(self) -> bool:
        """Connect to GSPro.

        No heartbeat thread — the springbok MLM2PRO connector (defacto
        standard) does not send heartbeats. The socket stays alive from
        the TCP connection alone. Heartbeats with zero-data BallData
        were interfering with GSPro's shot processing.
        """
        if not self._open_socket():
            return False
        self._running = True

        # Send initial ready signal so GSPro shows "Ready" in OpenAPI
        ready_msg = {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": {
                "Speed": 0.0,
                "SpinAxis": 0.0,
                "TotalSpin": 0.0,
                "HLA": 0.0,
                "VLA": 0.0,
                "Backspin": 0.0,
                "SideSpin": 0.0,
                "CarryDistance": 0,
            },
            "ShotDataOptions": {
                "ContainsBallData": False,
                "ContainsClubData": False,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": True,
                "IsHeartBeat": True,
            },
        }
        self._send_json(ready_msg)
        logger.info("Sent ready signal to GSPro")

        # Start a listener thread for incoming GSPro messages (club selection etc.)
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._heartbeat_thread = threading.Thread(
                target=self._message_listener, daemon=True,
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
        """Send a JSON message over the socket (send only, no recv).

        Matches the springbok connector's launch_ball() behavior:
        socket.write(JSON) with no response wait. Responses are read
        by the _message_listener thread to avoid race conditions.
        """
        with self._lock:
            if self._socket is None:
                return GSProResponse(success=False, message="Socket not connected")

            try:
                data = json.dumps(message).encode("utf-8")
                self._socket.sendall(data)
                logger.info("Shot sent to GSPro (shot #%d)", self._shot_number)
                return GSProResponse(success=True, message="sent")

            except OSError as e:
                logger.error("Failed to send to GSPro: %s", e)
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
                    self._socket.settimeout(2.0)

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
                "HLA": round(hla_degrees, 2),
                "VLA": 0.0,
                "Backspin": 0.0,
                "SideSpin": 0.0,
                "CarryDistance": 0,
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
        """Build GSPro Open Connect v1 message with full ball + club data.

        Format matches the MLM2PRO-GSPro-Connector's to_gspro() format
        (springbok/MLM2PRO-GSPro-Connector ball_data.py).
        """
        msg: dict[str, object] = {
            "DeviceID": self._settings.device_id,
            "Units": "Yards",
            "ShotNumber": self._shot_number,
            "APIversion": "1",
            "BallData": {
                "Speed": round(ball_speed, 2),
                "SpinAxis": round(spin_axis, 2),
                "TotalSpin": round(total_spin, 2),
                "HLA": round(hla, 2),
                "VLA": round(vla, 2),
                "Backspin": round(back_spin, 2),
                "SideSpin": round(side_spin, 2),
                "CarryDistance": round(carry_distance, 1),
            },
            "ClubData": {
                "Speed": round(club_speed, 2),
                "AngleOfAttack": round(aoa, 2),
                "FaceToTarget": round(face_to_target, 2),
                "Lie": 0,
                "Loft": round(dynamic_loft, 2),
                "Path": round(club_path, 2),
                "SpeedAtImpact": round(club_speed, 2),
                "VerticalFaceImpact": round(vertical_impact, 2),
                "HorizontalFaceImpact": round(lateral_impact, 2),
                "ClosureRate": 0,
            },
            "ShotDataOptions": {
                "ContainsBallData": True,
                "ContainsClubData": True,
                "LaunchMonitorIsReady": True,
                "LaunchMonitorBallDetected": True,
                "IsHeartBeat": False,
            },
        }
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

    def _message_listener(self) -> None:
        """Listen for incoming GSPro messages (club selection, etc.).

        Matches the springbok connector's check_for_message() pattern:
        non-blocking select() to read incoming data without sending
        heartbeats. The connection stays alive via TCP keepalive.
        """
        backoff = 1.0
        while self._running:
            if not self._connected.is_set():
                # Reconnect
                logger.info("Attempting reconnect (backoff=%.1fs)", backoff)
                if self._open_socket():
                    backoff = 1.0
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                continue

            # Check for incoming messages from GSPro (non-blocking)
            with self._lock:
                sock = self._socket
            if sock is None:
                time.sleep(1)
                continue

            try:
                readable, _, _ = select.select([sock], [], [], 1.0)
                if readable:
                    data = sock.recv(2048)
                    if len(data) == 0:
                        logger.warning("GSPro closed the connection")
                        self._connected.clear()
                        continue
                    # Parse messages (may be concatenated)
                    for part in data.decode("utf-8").replace("}{", "}|{").split("|"):
                        try:
                            msg = json.loads(part)
                            code = msg.get("Code", -1)
                            if code == 201:
                                club = msg.get("Player", {}).get("Club", "")
                                logger.info("GSPro club selected: %s", club)
                            else:
                                logger.debug("GSPro message (code %d): %s", code, msg)
                        except json.JSONDecodeError:
                            pass
            except OSError as e:
                logger.error("GSPro listener error: %s", e)
                self._connected.clear()

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
