"""Tests for GSPro client."""

import json

from webcam_putting.config import ConnectionSettings
from webcam_putting.gspro_client import GSProClient


class TestShotMessageFormat:
    def test_gspro_direct_message_format(self):
        """Verify shot message matches GSPro Open Connect v1 spec."""
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings)
        client._shot_number = 5

        msg = client._build_shot_message(8.5, -2.3)

        assert msg["DeviceID"] == "WebcamPutting"
        assert msg["Units"] == "Yards"
        assert msg["ShotNumber"] == 5
        assert msg["APIversion"] == "1"

        ball = msg["BallData"]
        assert ball["Speed"] == 8.5
        assert ball["HLA"] == -2.3
        assert ball["VLA"] == 0.0
        assert ball["TotalSpin"] == 0.0
        assert ball["SpinAxis"] == 0.0
        assert ball["BackSpin"] == 0.0
        assert ball["SideSpin"] == 0.0

        opts = msg["ShotDataOptions"]
        assert opts["ContainsBallData"] is True
        assert opts["ContainsClubData"] is False
        assert opts["LaunchMonitorIsReady"] is True
        assert opts["LaunchMonitorBallDetected"] is True
        assert opts["IsHeartBeat"] is False

    def test_heartbeat_message_format(self):
        """Verify heartbeat has IsHeartBeat=True."""
        settings = ConnectionSettings(mode="gspro_direct")
        client = GSProClient(settings)

        msg = client._build_heartbeat_message()

        assert msg["ShotDataOptions"]["IsHeartBeat"] is True
        assert msg["ShotDataOptions"]["ContainsBallData"] is False
        assert msg["ShotDataOptions"]["LaunchMonitorBallDetected"] is False
        assert msg["ShotDataOptions"]["LaunchMonitorIsReady"] is True

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
