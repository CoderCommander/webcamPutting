"""Tests for GSPro client."""

import json

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
        assert ball["BackSpin"] == 2898.0
        assert ball["SideSpin"] == 776.0

        opts = msg["ShotDataOptions"]
        assert opts["ContainsBallData"] is True
        assert opts["ContainsClubData"] is True  # club_speed > 0
        assert opts["IsHeartBeat"] is False

    def test_full_shot_no_club_speed(self):
        """ContainsClubData should be False when club_speed is 0."""
        settings = ConnectionSettings()
        client = GSProClient(settings)
        client._shot_number = 1

        msg = client._build_full_shot_message(
            ball_speed=100.0, vla=10.0, hla=0.0,
            total_spin=2500.0, spin_axis=0.0,
            back_spin=2500.0, side_spin=0.0,
            club_speed=0.0,
        )
        assert msg["ShotDataOptions"]["ContainsClubData"] is False

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
