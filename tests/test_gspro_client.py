"""Tests for GSPro client."""

import errno
import json
import logging

import pytest

from birdman_putting.config import ConnectionSettings
from birdman_putting.gspro_client import GSProClient


class TestShotMessageFormat:
    def test_gspro_direct_message_format(self):
        """Verify shot message matches GSPro Open Connect v1 spec."""
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings)
        client._shot_number = 5

        msg = client._build_shot_message(8.5, -2.3)

        assert msg["DeviceID"] == "BirdmanPutting"
        assert msg["Units"] == "Yards"
        assert msg["ShotNumber"] == 5
        assert msg["APIversion"] == "1"

        ball = msg["BallData"]
        assert ball["Speed"] == 8.5
        assert ball["HLA"] == -2.3
        assert ball["VLA"] == 0.0
        assert ball["TotalSpin"] == 0.0
        assert ball["SpinAxis"] == 0.0
        assert ball["Backspin"] == 0.0
        assert ball["SideSpin"] == 0.0

        opts = msg["ShotDataOptions"]
        assert opts["ContainsBallData"] is True
        assert opts["ContainsClubData"] is False
        assert opts["LaunchMonitorIsReady"] is True
        assert opts["LaunchMonitorBallDetected"] is True
        assert opts["IsHeartBeat"] is False

    def test_shot_number_not_incremented_by_message_build(self):
        """Building a message should not change shot number."""
        settings = ConnectionSettings()
        client = GSProClient(settings)
        client._shot_number = 3

        client._build_shot_message(5.0, 0.0)
        assert client._shot_number == 3

    def test_message_is_valid_json(self):
        """Ensure message serializes to valid JSON."""
        settings = ConnectionSettings()
        client = GSProClient(settings)

        msg = client._build_shot_message(12.5, -3.7)
        json_str = json.dumps(msg)
        parsed = json.loads(json_str)

        assert parsed["BallData"]["Speed"] == 12.5
        assert parsed["BallData"]["HLA"] == -3.7


class TestFullShotMessageFormat:
    def test_full_shot_message_has_all_ball_data(self):
        """Verify full shot message includes VLA, spin, etc."""
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings)
        client._shot_number = 1

        msg = client._build_full_shot_message(
            ball_speed=120.5, vla=12.3, hla=-1.5,
            total_spin=3000.0, spin_axis=15.0,
            back_spin=2898.0, side_spin=776.0,
            club_speed=95.0,
        )

        assert msg["DeviceID"] == "BirdmanPutting"
        assert msg["APIversion"] == "1"

        ball = msg["BallData"]
        assert ball["Speed"] == 120.5
        assert ball["VLA"] == 12.3
        assert ball["HLA"] == -1.5
        assert ball["TotalSpin"] == 3000.0
        assert ball["SpinAxis"] == 15.0
        assert ball["Backspin"] == 2898.0
        assert ball["SideSpin"] == 776.0

        opts = msg["ShotDataOptions"]
        assert opts["ContainsBallData"] is True
        assert opts["ContainsClubData"] is True
        assert opts["LaunchMonitorIsReady"] is True
        assert opts["IsHeartBeat"] is False

        club = msg["ClubData"]
        assert club["Speed"] == 95.0

    def test_full_shot_no_club_speed(self):
        """ClubData always present (MLM2PRO format), Speed=0 when unknown."""
        settings = ConnectionSettings()
        client = GSProClient(settings)
        client._shot_number = 1

        msg = client._build_full_shot_message(
            ball_speed=100.0, vla=10.0, hla=0.0,
            total_spin=2500.0, spin_axis=0.0,
            back_spin=2500.0, side_spin=0.0,
            club_speed=0.0,
        )
        # MLM2PRO format always includes ClubData
        assert msg["ShotDataOptions"]["ContainsClubData"] is True
        assert msg["ClubData"]["Speed"] == 0.0

    def test_full_shot_values_rounded(self):
        """Values should be rounded to 2 decimal places."""
        settings = ConnectionSettings()
        client = GSProClient(settings)
        client._shot_number = 1

        msg = client._build_full_shot_message(
            ball_speed=120.5678, vla=12.3456, hla=-1.5678,
            total_spin=3000.1234, spin_axis=15.6789,
            back_spin=2898.1234, side_spin=776.5678,
            club_speed=95.1234,
        )
        ball = msg["BallData"]
        assert ball["Speed"] == 120.57
        assert ball["VLA"] == 12.35
        assert ball["HLA"] == -1.57


class TestHTTPMiddlewareFormat:
    def test_http_mode_is_connected(self):
        """HTTP mode is always 'connected' (stateless)."""
        settings = ConnectionSettings(mode="http_middleware")
        client = GSProClient(settings)
        assert client.is_connected is True

    def test_mode_property(self):
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings)
        assert client.mode == "gspro_direct"

        settings2 = ConnectionSettings(mode="http_middleware")
        client2 = GSProClient(settings2)
        assert client2.mode == "http_middleware"


# --- Fakes / helpers for listener tests ---------------------------------


class _FakeSocket:
    """Minimal stand-in for a connected TCP socket.

    ``recv_script`` is a list of either ``bytes`` (returned from recv) or an
    ``Exception`` instance (raised from recv). When the script is exhausted,
    recv returns b"" (peer closed) by default.
    """

    def __init__(self, recv_script=None):
        self._recv_script = list(recv_script or [])
        self.shutdown_calls = []
        self.closed = False
        self.sent = []

    def recv(self, _bufsize):
        if not self._recv_script:
            return b""
        item = self._recv_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def sendall(self, data):
        self.sent.append(data)

    def shutdown(self, how):
        self.shutdown_calls.append(how)

    def close(self):
        self.closed = True


def _make_client():
    settings = ConnectionSettings(mode="gspro_direct")
    return GSProClient(settings)


def _force_readable(monkeypatch, sock):
    """Make the gspro_client module's select.select report ``sock`` readable.

    Real ``select.select`` on Windows only accepts real OS sockets, so a fake
    socket cannot be passed through it. Patching select lets ``_listen_once``
    exercise its real recv/frame/dispatch and exception-handling path against
    the fake, without any actual network or file descriptor.
    """
    from birdman_putting import gspro_client as _mod

    monkeypatch.setattr(
        _mod.select, "select", lambda r, w, x, t=None: (list(r), [], [])
    )


class TestListenerFraming:
    def test_single_object_split_across_two_chunks(self):
        """A JSON object delivered in two recv chunks yields exactly one message."""
        client = _make_client()
        dispatched = []
        client._handle_message = dispatched.append  # type: ignore[assignment]

        obj = json.dumps({"Code": 200, "Message": "hi"})
        first, second = obj[: len(obj) // 2], obj[len(obj) // 2:]

        n1 = client._process_recv_data(first.encode("utf-8"))
        assert n1 == 0  # incomplete — nothing dispatched yet
        assert dispatched == []

        n2 = client._process_recv_data(second.encode("utf-8"))
        assert n2 == 1
        assert len(dispatched) == 1
        assert dispatched[0]["Code"] == 200
        assert dispatched[0]["Message"] == "hi"

    def test_two_concatenated_objects_in_one_chunk(self):
        """Two back-to-back objects in one recv yield two messages."""
        client = _make_client()
        dispatched = []
        client._handle_message = dispatched.append  # type: ignore[assignment]

        blob = (
            json.dumps({"Code": 200, "Message": "a"})
            + json.dumps({"Code": 201, "Player": {"Club": "PT"}})
        ).encode("utf-8")

        n = client._process_recv_data(blob)
        assert n == 2
        assert [m["Code"] for m in dispatched] == [200, 201]
        assert dispatched[1]["Player"]["Club"] == "PT"

    def test_brace_inside_string_does_not_corrupt_parsing(self):
        """A literal '}{' inside a string value must not split the object."""
        client = _make_client()
        dispatched = []
        client._handle_message = dispatched.append  # type: ignore[assignment]

        blob = json.dumps({"Code": 200, "Message": "a}{b"}).encode("utf-8")
        n = client._process_recv_data(blob)
        assert n == 1
        assert dispatched[0]["Message"] == "a}{b"

    def test_buffer_persists_then_completes_with_trailing_partial(self):
        """Complete object + trailing partial: dispatch one, keep remainder."""
        client = _make_client()
        dispatched = []
        client._handle_message = dispatched.append  # type: ignore[assignment]

        full = json.dumps({"Code": 1})
        partial_start = '{"Code": 2, "x":'  # incomplete second object
        n = client._process_recv_data((full + partial_start).encode("utf-8"))
        assert n == 1
        assert dispatched[0]["Code"] == 1

        rest = ' 99}'
        n2 = client._process_recv_data(rest.encode("utf-8"))
        assert n2 == 1
        assert dispatched[1]["Code"] == 2
        assert dispatched[1]["x"] == 99

    def test_club_change_callback_invoked_via_handle_message(self):
        """Code 201 with a Club drives the on_club_change callback."""
        clubs = []
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings, on_club_change=clubs.append)

        blob = json.dumps({"Code": 201, "Player": {"Club": "DR"}}).encode("utf-8")
        client._process_recv_data(blob)
        assert clubs == ["DR"]


class TestListenerCleanDisconnect:
    def test_connection_reset_is_clean_not_error(self, caplog, monkeypatch):
        """recv() ConnectionResetError → handled cleanly, no ERROR traceback."""
        client = _make_client()
        client._running = True
        client._connected.set()
        sock = _FakeSocket(recv_script=[ConnectionResetError(10054, "forcibly closed")])
        _force_readable(monkeypatch, sock)

        with caplog.at_level(logging.DEBUG, logger="birdman_putting.gspro_client"):
            result = client._listen_once(sock)

        # The iteration reports the connection is gone.
        assert result is False
        # Connected flag cleared so the loop will reconnect.
        assert client._connected.is_set() is False
        # No ERROR-level record with a traceback for this expected path.
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [], f"unexpected error logs: {[r.getMessage() for r in errors]}"
        # An informational reconnect message is emitted instead.
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("reconnect" in m.lower() or "dropped" in m.lower() for m in infos)

    def test_winerror_10038_is_clean(self, caplog, monkeypatch):
        """OSError winerror=10038 (not a socket) → clean disconnect, no traceback."""
        client = _make_client()
        client._running = True
        client._connected.set()
        exc = OSError("not a socket")
        exc.winerror = 10038
        sock = _FakeSocket(recv_script=[exc])
        _force_readable(monkeypatch, sock)

        with caplog.at_level(logging.DEBUG, logger="birdman_putting.gspro_client"):
            result = client._listen_once(sock)

        assert result is False
        assert client._connected.is_set() is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [], f"unexpected error logs: {[r.getMessage() for r in errors]}"

    def test_ebadf_oserror_is_clean(self, caplog, monkeypatch):
        """OSError errno=EBADF → clean disconnect (covers shutdown/close race)."""
        client = _make_client()
        client._running = True
        client._connected.set()
        exc = OSError(errno.EBADF, "bad file descriptor")
        sock = _FakeSocket(recv_script=[exc])
        _force_readable(monkeypatch, sock)

        with caplog.at_level(logging.DEBUG, logger="birdman_putting.gspro_client"):
            result = client._listen_once(sock)

        assert result is False
        assert client._connected.is_set() is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []

    def test_peer_closed_empty_recv_is_clean(self, caplog, monkeypatch):
        """recv() returning b'' (peer closed) → clean disconnect, no traceback."""
        client = _make_client()
        client._running = True
        client._connected.set()
        sock = _FakeSocket(recv_script=[b""])
        _force_readable(monkeypatch, sock)

        with caplog.at_level(logging.DEBUG, logger="birdman_putting.gspro_client"):
            result = client._listen_once(sock)

        assert result is False
        assert client._connected.is_set() is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []

    def test_unexpected_exception_still_logs_error(self, caplog, monkeypatch):
        """A genuinely unexpected exception type is logged at ERROR with traceback."""
        client = _make_client()
        client._running = True
        client._connected.set()
        sock = _FakeSocket(recv_script=[ValueError("totally unexpected")])
        _force_readable(monkeypatch, sock)

        with caplog.at_level(logging.DEBUG, logger="birdman_putting.gspro_client"):
            result = client._listen_once(sock)

        assert result is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "unexpected exception should be logged at ERROR"
        assert any(r.exc_info for r in errors), "ERROR should include a traceback"


class TestListenerBufferReset:
    def test_reconnect_resets_recv_buffer(self):
        """Stale partial bytes from a dropped connection must not leak forward."""
        client = _make_client()
        dispatched = []
        client._handle_message = dispatched.append  # type: ignore[assignment]

        # Feed a partial object, then simulate reconnect clearing the buffer.
        client._process_recv_data(b'{"Code": 5, "x":')
        assert dispatched == []
        client._reset_recv_buffer()

        # A fresh complete object after reset must parse on its own.
        client._process_recv_data(json.dumps({"Code": 9}).encode("utf-8"))
        assert len(dispatched) == 1
        assert dispatched[0]["Code"] == 9


class TestTeardownRace:
    def test_disconnect_shuts_down_before_close(self, monkeypatch):
        """disconnect() shuts the socket down (to unblock reader) then closes it."""
        client = _make_client()
        sock = _FakeSocket()
        with client._lock:
            client._socket = sock
        client._connected.set()
        client._running = True

        client.disconnect()

        # shutdown(SHUT_RDWR) called before close to wake a blocked recv/select.
        import socket as _socket
        assert sock.shutdown_calls == [_socket.SHUT_RDWR]
        assert sock.closed is True
        assert client._connected.is_set() is False

    def test_set_shot_cooldown_is_callable_noop(self):
        """Public set_shot_cooldown is preserved for external callers (app.py)."""
        client = _make_client()
        # Must not raise; kept as a no-op stub for backwards compatibility.
        client.set_shot_cooldown(3)


class TestPublicApiPreserved:
    def test_listener_thread_attribute_exists(self):
        """Private listener thread attr exists under its (renamed) name."""
        client = _make_client()
        assert hasattr(client, "_listener_thread")

    def test_public_methods_present(self):
        client = _make_client()
        for name in (
            "connect", "disconnect", "send_shot", "send_full_shot",
            "is_connected", "mode", "shot_number", "ball_detected",
        ):
            assert hasattr(client, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
