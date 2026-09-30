"""GSPro Open Connect v1 API client and HTTP middleware fallback."""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import select
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from birdman_putting.config import ConnectionSettings

logger = logging.getLogger(__name__)

# Winsock / errno codes that indicate the socket is simply gone. Treat these
# as a *clean* disconnect (reconnect quietly) rather than an unexpected error.
#   10053 WSAECONNABORTED  - software caused connection abort
#   10054 WSAECONNRESET    - connection reset by peer (forcibly closed)
#   10038 WSAENOTSOCK      - operation on something that is not a socket
#                            (happens when the fd is closed under a blocked read)
#   10058 WSAESHUTDOWN     - cannot send/recv after socket shutdown
_CLEAN_DISCONNECT_CODES = frozenset(
    {
        10053,
        10054,
        10038,
        10058,
        errno.EBADF,
        errno.ENOTSOCK,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ESHUTDOWN,
        errno.EPIPE,
    }
)


def _is_clean_disconnect(exc: OSError) -> bool:
    """Return True if ``exc`` represents an expected socket teardown.

    Covers ``ConnectionResetError``/``ConnectionAbortedError``/``TimeoutError``
    (subclasses of ``OSError``) plus any ``OSError`` whose ``winerror`` or
    ``errno`` is a known "socket is gone" code.
    """
    if isinstance(
        exc, (ConnectionResetError, ConnectionAbortedError, TimeoutError, BrokenPipeError)
    ):
        return True
    code = getattr(exc, "winerror", None)
    if code in _CLEAN_DISCONNECT_CODES:
        return True
    return exc.errno in _CLEAN_DISCONNECT_CODES


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

    def __init__(
        self,
        settings: ConnectionSettings,
        on_club_change: Callable[[str, float], None] | None = None,
    ):
        self._settings = settings
        self._socket: socket.socket | None = None
        self._shot_number: int = 0
        self._connected = threading.Event()
        self._lock = threading.Lock()
        # Serializes socket teardown/replacement so the listener-driven
        # reconnect and an external disconnect()/reconnect cannot run at the
        # same time and leak or stomp on each other's sockets.
        self._teardown_lock = threading.Lock()
        self._listener_thread: threading.Thread | None = None
        self._running = False
        # Wakes the listener out of its reconnect backoff sleep immediately on
        # disconnect().  The old time.sleep(30) was uninterruptible: a
        # Reconnect click while the listener slept spawned a SECOND listener
        # (two threads recv'ing into one unlocked buffer), and window close
        # hung for exactly the join timeout.
        self._wake = threading.Event()
        # Generation token: bumped by disconnect(); a listener whose captured
        # generation is stale exits instead of racing a newly-spawned one.
        self._listener_gen = 0
        self._ball_detected = False  # Set True when ball is ready for shot
        self._on_club_change = on_club_change  # Called with club name on GSPro code 201
        # Persistent receive buffer for streaming JSON framing across recv()s.
        self._recv_buffer = b""

    def set_shot_cooldown(self, cycles: int) -> None:
        """Deprecated no-op retained for backwards compatibility.

        The post-shot cooldown is handled by the tracker (see
        ``ShotSettings.post_shot_cooldown``); this client no longer sends
        heartbeats, so there are no cycles to skip. Kept callable because
        ``app.py`` still invokes it.
        """
        return None

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
        """Close the connection and stop the listener thread.

        Signals the listener to stop, then tears the socket down. The teardown
        does ``shutdown(SHUT_RDWR)`` *before* ``close()`` so a listener thread
        blocked in ``select()``/``recv()`` is woken promptly and observes a
        clean disconnect instead of a "not a socket" error.
        """
        self._running = False
        self._listener_gen += 1  # any current listener is now stale and exits
        self._connected.clear()
        self._wake.set()  # break the listener out of a backoff wait instantly
        # Wake the listener and release the socket. Serialized so we never race
        # the listener's own reconnect teardown.
        self._teardown_socket(log_msg="Disconnected from GSPro")

        thread = self._listener_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
            if thread.is_alive():
                # Keep the reference: nulling it here let connect() spawn a
                # SECOND listener while this one still ran.  With the wake
                # event + <=2s socket waits this path should not occur.
                logger.warning("GSPro listener did not exit within 10s")
            else:
                self._listener_thread = None

    def _teardown_socket(self, log_msg: str | None = None) -> None:
        """Shut down and close the current socket, if any (idempotent).

        Holds ``_teardown_lock`` for the whole operation so an external
        ``disconnect()`` and the listener-driven reconnect cannot both be
        manipulating the socket at once.
        """
        with self._teardown_lock:
            with self._lock:
                sock = self._socket
                self._socket = None
            if sock is None:
                return
            # shutdown() first to unblock any reader; ignore errors if the
            # peer already closed (the socket may not be connected).
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()
            if log_msg:
                logger.info(log_msg)

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
        if not self._connected.is_set():
            # Do NOT reconnect from the send path: the listener owns
            # reconnection, and a concurrent _connect_socket here raced its
            # _open_socket (leaked established sockets — reproduced).  Only
            # bootstrap when no listener exists at all (standalone use).
            t = self._listener_thread
            if (t is not None and t.is_alive()) or not self._connect_socket():
                return GSProResponse(
                    success=False,
                    message="Not connected to GSPro (reconnecting in background)",
                )

        self._shot_number += 1
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

        Used by both initial connect and reconnects from the listener loop.
        Any previous socket is torn down (shutdown + close) first, under the
        teardown lock so we never race a concurrent disconnect.
        """
        host = self._settings.gspro_host
        port = self._settings.gspro_port

        # Tear down any existing socket and start with a fresh receive buffer.
        self._teardown_socket()
        self._reset_recv_buffer()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(2.0)  # Match MLM2PRO connector timeout

            # TCP keepalive: on localhost a dead GSPro RSTs instantly, but
            # over the LAN (two-machine split) a slept/power-lost GSPro box
            # otherwise leaves is_connected=True forever -- and the first
            # putt logs "Shot sent to GSPro" for a shot that never arrives.
            # 10s idle, 3s interval => dead peer detected in ~20s.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            with contextlib.suppress(AttributeError, OSError):
                # non-Windows / unsupported: plain SO_KEEPALIVE stands
                sock.ioctl(  # type: ignore[attr-defined]
                    socket.SIO_KEEPALIVE_VALS, (1, 10000, 3000),
                )

            with self._lock:
                self._socket = sock

            # A disconnect() may have raced in while we were connecting; if so,
            # don't leak the freshly-opened socket.
            if not self._running:
                self._teardown_socket()
                return False

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
        # Mark running before opening so the "do we still want a socket?"
        # invariant (_running True ⇒ keep the socket) holds during connect.
        self._running = True
        self._wake.clear()
        opened = self._open_socket()
        if opened:
            self._send_ready_signal()
        else:
            # Do NOT give up: start the listener anyway so its backoff loop
            # keeps retrying in the background.  Previously a failed FIRST
            # connect set _running=False and never started the listener --
            # the "will retry in background" log in watch-only mode was
            # false, and with 921 down at launch the client was dead until
            # an app restart.
            logger.warning(
                "GSPro not reachable at startup -- listener will keep "
                "retrying in the background",
            )

        # Start a listener thread for incoming GSPro messages (club selection etc.)
        if self._listener_thread is None or not self._listener_thread.is_alive():
            self._listener_thread = threading.Thread(
                target=self._message_listener, daemon=True,
            )
            self._listener_thread.start()

        return opened

    def _send_ready_signal(self) -> None:
        """Send the LaunchMonitorIsReady handshake on a fresh socket.

        Extracted so RECONNECTS send it too: reconnects go through
        _open_socket(), which never sent it, so after a GSPro restart the
        client showed connected but GSPro never re-armed the LM (reproduced:
        0 bytes received post-reconnect).
        """
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

    def _send_socket(self, speed_mph: float, hla_degrees: float) -> GSProResponse:
        """Send shot via TCP socket."""
        if not self._connected.is_set():
            # Do NOT reconnect from the send path: the listener owns
            # reconnection, and a concurrent _connect_socket here raced its
            # _open_socket (leaked established sockets — reproduced).  Only
            # bootstrap when no listener exists at all (standalone use).
            t = self._listener_thread
            if (t is not None and t.is_alive()) or not self._connect_socket():
                return GSProResponse(
                    success=False,
                    message="Not connected to GSPro (reconnecting in background)",
                )

        self._shot_number += 1
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
                logger.info("Shot sent to GSPro (shot #%d, %d bytes)", self._shot_number, len(data))
                logger.debug("GSPro payload: %s", data.decode("utf-8"))
                return GSProResponse(success=True, message="sent")

            except OSError as e:
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

    def _reset_recv_buffer(self) -> None:
        """Discard any buffered, partially-received bytes.

        Called on (re)connect so stale fragments from a dropped socket cannot
        corrupt framing on the new one.
        """
        self._recv_buffer = b""

    def _handle_message(self, msg: dict[str, object]) -> None:
        """Dispatch a single decoded GSPro message."""
        code = msg.get("Code", -1)
        if code == 201:
            player = msg.get("Player", {})
            if isinstance(player, dict):
                club = player.get("Club", "")
                raw_distance = player.get("DistanceToTarget", 0)
            else:
                club, raw_distance = "", 0
            # DistanceToTarget drives the FS Golf chipping/full-swing decision
            # for the distance-gated wedges (see app._on_club_change). It was
            # previously never read, so distance always defaulted to 0.0 and
            # "0 < distance <= 30" was ALWAYS False — i.e. those clubs could
            # never select Chipping regardless of how close the pin was.
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError):
                distance = 0.0
            # GSPro reports DistanceToTarget in YARDS off the green and in
            # FEET once on it, and the message carries no units field — so log
            # the whole Player object to make the active unit inspectable.
            logger.info(
                "GSPro club selected: %s (%.1f to target) raw_player=%s",
                club, distance, player,
            )
            if self._on_club_change and club:
                # NEVER let a callback exception escape: this runs on the
                # socket listener thread, and an uncaught raise kills it
                # permanently — no further club updates, no OBS switching and
                # no reconnect, while is_connected still reports True.
                try:
                    self._on_club_change(club, distance)
                except Exception:
                    logger.exception("on_club_change callback raised")
        else:
            logger.debug("GSPro message (code %s): %s", code, msg)

    def _process_recv_data(self, data: bytes) -> int:
        """Append ``data`` to the receive buffer and dispatch complete objects.

        GSPro frames messages as bare, concatenated JSON objects with no
        delimiter, and a single TCP ``recv`` may contain a partial object, one
        object, or several. We accumulate bytes in ``self._recv_buffer`` and use
        ``json.JSONDecoder.raw_decode`` to peel off one complete object at a
        time, leaving any incomplete trailing bytes buffered for the next read.

        This is string-aware (braces inside string values do not confuse it)
        and never drops an object split across two reads.

        Returns:
            The number of complete messages dispatched this call.
        """
        if data:
            self._recv_buffer += data

        # Decode against the text form so raw_decode's index lines up with
        # character positions; re-encode the remainder to keep the buffer bytes.
        try:
            text = self._recv_buffer.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start >= len(self._recv_buffer) - 4:
                # A multi-byte character split across recvs — wait for the rest.
                return 0
            # Invalid bytes MID-buffer never become decodable: strict decode
            # would fail forever and every later valid message would be lost
            # until reconnect.  Decode leniently; the resync below skips the
            # replacement garbage.
            logger.warning(
                "GSPro framing: invalid UTF-8 at byte %d — decoding leniently",
                e.start,
            )
            text = self._recv_buffer.decode("utf-8", errors="replace")

        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        dispatched = 0
        while idx < n:
            # Skip inter-object whitespace.
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            try:
                msg, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError as e:
                if text[idx] != "{":
                    # Garbage prefix: it can never become a valid object, and
                    # keeping it poisoned the framing PERMANENTLY (reproduced:
                    # 17 stray bytes → every later club message dropped).
                    # Resync to the next possible object start.
                    nxt = text.find("{", idx)
                    if nxt == -1:
                        logger.warning(
                            "GSPro framing: discarded %d unparseable chars",
                            n - idx,
                        )
                        idx = n
                    else:
                        logger.warning(
                            "GSPro framing: skipped %d garbage chars", nxt - idx,
                        )
                        idx = nxt
                    continue
                # Starts with '{': usually an INCOMPLETE object (error at or
                # near the buffer end) — wait for more data.  But a malformed
                # COMPLETE object (e.g. lenient-decode replacement chars mid-
                # object) errors far from the end and would otherwise be kept
                # forever, swallowing every later message until the size cap
                # (~1,870 club messages measured).  GSPro objects are a few
                # hundred bytes and recv chunks are 2048, so an error more
                # than one chunk from the end cannot be simple truncation.
                if e.pos < n - 2048:
                    nxt = text.find("{", idx + 1)
                    logger.warning(
                        "GSPro framing: malformed object at %d (err at %d) — "
                        "resyncing", idx, e.pos,
                    )
                    idx = nxt if nxt != -1 else n
                    continue
                break
            idx = end
            if isinstance(msg, dict):
                self._handle_message(msg)
                dispatched += 1
            else:
                logger.debug("Ignoring non-object GSPro message: %r", msg)

        # Persist whatever we could not fully parse — with a hard cap so a
        # malformed-but-'{'-prefixed stream cannot grow the buffer unbounded.
        remainder = text[idx:]
        if len(remainder) > 65536:
            logger.warning(
                "GSPro framing: dropping %d-char unparseable buffer",
                len(remainder),
            )
            remainder = ""
        self._recv_buffer = remainder.encode("utf-8")
        return dispatched

    def _listen_once(self, sock: socket.socket) -> bool:
        """Run one select/recv/frame/dispatch cycle on ``sock``.

        Returns:
            True if the connection is still alive after this cycle; False if it
            was closed or dropped (the caller should reconnect). On a dropped
            connection this clears the connected flag and logs at INFO (no
            traceback) for expected socket teardown. Genuinely unexpected
            exceptions are logged at ERROR with a traceback and also reported as
            "connection gone" so the loop recovers.
        """
        try:
            readable, _, _ = select.select([sock], [], [], 1.0)
            if not readable:
                return True  # Nothing to read this tick; connection still up.
            data = sock.recv(2048)
        except (ConnectionResetError, ConnectionAbortedError, TimeoutError, BrokenPipeError):
            logger.info("GSPro connection dropped; reconnecting")
            self._connected.clear()
            return False
        except OSError as e:
            if _is_clean_disconnect(e):
                logger.info("GSPro connection dropped; reconnecting")
                self._connected.clear()
                return False
            logger.error("GSPro listener error: %s", e, exc_info=True)
            self._connected.clear()
            return False
        except Exception as e:  # noqa: BLE001 - report unexpected types loudly
            logger.error("GSPro listener error: %s", e, exc_info=True)
            self._connected.clear()
            return False

        if len(data) == 0:
            # Peer performed an orderly shutdown.
            logger.info("GSPro connection dropped; reconnecting")
            self._connected.clear()
            return False

        try:
            self._process_recv_data(data)
        except Exception:  # noqa: BLE001 — a dispatch bug must not kill the listener
            logger.exception("GSPro message dispatch failed — continuing")
        return True

    def _message_listener(self) -> None:
        """Listener thread: read incoming GSPro messages and auto-reconnect.

        Matches the springbok connector's check_for_message() pattern: a
        non-blocking ``select()`` reads incoming data without sending
        heartbeats; the connection stays alive via TCP keepalive. This thread
        is the sole owner of socket teardown during reconnects — external
        ``disconnect()`` only signals intent and shuts the socket down to wake
        this loop.
        """
        gen = self._listener_gen
        backoff = 1.0
        while self._running and gen == self._listener_gen:
            if not self._connected.is_set():
                logger.info("Attempting reconnect (backoff=%.1fs)", backoff)
                if self._open_socket():
                    backoff = 1.0
                    # Re-arm GSPro: a reconnect that skips the ready signal
                    # leaves GSPro not listening to this LM (reproduced).
                    self._send_ready_signal()
                else:
                    # Event-based wait: disconnect() sets _wake so shutdown
                    # never sits out a 30s sleep, and no second listener can
                    # be spawned while this one is unkillable mid-backoff.
                    self._wake.wait(timeout=backoff)
                    backoff = min(backoff * 2, 30.0)
                continue

            with self._lock:
                sock = self._socket
            if sock is None:
                # disconnect()/teardown cleared it out from under us.
                if not self._running:
                    break
                self._connected.clear()
                continue

            if not self._listen_once(sock):
                # Connection gone; loop will reconnect (unless we're stopping).
                continue

        logger.info("GSPro listener thread exiting")

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
