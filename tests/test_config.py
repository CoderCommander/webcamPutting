"""Tests for config loading, saving, and migration."""

from pathlib import Path

from birdman_putting.config import (
    AppConfig,
    load_config,
    migrate_from_ini,
    save_config,
)


class TestConfigRoundTrip:
    def test_save_and_load_defaults(self, tmp_path: Path):
        config = AppConfig()
        config_path = tmp_path / "config.toml"

        save_config(config, config_path)
        loaded = load_config(config_path)

        assert loaded.detection_zone.start_x1 == config.detection_zone.start_x1
        assert loaded.detection_zone.start_x2 == config.detection_zone.start_x2
        assert loaded.camera.mjpeg == config.camera.mjpeg
        assert loaded.ball.color_preset == config.ball.color_preset
        assert loaded.connection.gspro_port == config.connection.gspro_port
        assert loaded.shot.min_speed_mph == config.shot.min_speed_mph

    def test_save_and_load_custom_values(self, tmp_path: Path):
        config = AppConfig()
        config.detection_zone.start_x1 = 42
        config.camera.webcam_index = 3
        config.ball.color_preset = "orange2"
        config.connection.mode = "http_middleware"
        config.shot.max_speed_mph = 30.0

        config_path = tmp_path / "config.toml"
        save_config(config, config_path)
        loaded = load_config(config_path)

        assert loaded.detection_zone.start_x1 == 42
        assert loaded.camera.webcam_index == 3
        assert loaded.ball.color_preset == "orange2"
        assert loaded.connection.mode == "http_middleware"
        assert loaded.shot.max_speed_mph == 30.0

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        config = load_config(tmp_path / "nonexistent.toml")
        assert config.detection_zone.start_x1 == 10
        assert config.ball.color_preset == "yellow"

    def test_load_partial_config(self, tmp_path: Path):
        """Config with only some sections should fill in defaults for the rest."""
        config_path = tmp_path / "partial.toml"
        config_path.write_text('[camera]\nwebcam_index = 5\n')

        loaded = load_config(config_path)
        assert loaded.camera.webcam_index == 5
        # Other sections should be defaults
        assert loaded.detection_zone.start_x1 == 10
        assert loaded.ball.color_preset == "yellow"


class TestINIMigration:
    def test_migrate_basic_settings(self, tmp_path: Path):
        ini_path = tmp_path / "config.ini"
        ini_path.write_text(
            "[putting]\n"
            "startx1 = 20\n"
            "startx2 = 200\n"
            "y1 = 150\n"
            "y2 = 400\n"
            "radius = 12\n"
            "flip = 1\n"
            "mjpeg = 1\n"
            "fps = 60\n"
        )

        config = migrate_from_ini(ini_path)

        assert config.detection_zone.start_x1 == 20
        assert config.detection_zone.start_x2 == 200
        assert config.detection_zone.y1 == 150
        assert config.detection_zone.y2 == 400
        assert config.ball.fixed_radius == 12
        assert config.camera.flip_image is True
        assert config.camera.mjpeg is True
        assert config.camera.fps_override == 60

    def test_migrate_custom_hsv(self, tmp_path: Path):
        ini_path = tmp_path / "config.ini"
        ini_path.write_text(
            "[putting]\n"
            "customhsv = {'hmin': 3, 'smin': 181, 'vmin': 134,"
            " 'hmax': 40, 'smax': 255, 'vmax': 255}\n"
        )

        config = migrate_from_ini(ini_path)
        assert config.ball.custom_hsv is not None
        assert config.ball.custom_hsv["hmin"] == 3
        assert config.ball.custom_hsv["smax"] == 255

    def test_migrate_camera_properties(self, tmp_path: Path):
        ini_path = tmp_path / "config.ini"
        ini_path.write_text(
            "[putting]\n"
            "saturation = 128.0\n"
            "exposure = -7.0\n"
            "brightness = 100.0\n"
        )

        config = migrate_from_ini(ini_path)
        assert config.camera.saturation == 128.0
        assert config.camera.exposure == -7.0
        assert config.camera.brightness == 100.0

    def test_migrate_missing_file(self, tmp_path: Path):
        """Missing INI file should return defaults."""
        config = migrate_from_ini(tmp_path / "missing.ini")
        assert config.detection_zone.start_x1 == 10


class TestCrashSafeConfig:
    """The config save path must survive a crash mid-save.

    Regression: save_config used open(path, "wb"), truncating in place.  A
    crash mid-save left a stump that load_config silently interpreted as
    FULL DEFAULTS -- which the UI autosave then persisted, permanently
    wiping ppf/zone/markers/ROIs.
    """

    @staticmethod
    def _custom_config() -> AppConfig:
        config = AppConfig()
        config.shot.pixels_per_foot = 73.0
        config.detection_zone.start_x1 = 98
        config.camera.exposure = -7.0
        return config

    def test_save_writes_backup(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        save_config(self._custom_config(), path)
        # Second save rolls the first file to .bak
        save_config(self._custom_config(), path)
        assert (tmp_path / "config.toml.bak").exists()

    def test_truncated_config_is_moved_aside_not_defaulted(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        save_config(self._custom_config(), path)
        save_config(self._custom_config(), path)  # ensure .bak exists

        # Simulate a crash mid-save: truncate mid-key to an UNPARSEABLE stump.
        # (A stump cut exactly at a line boundary parses as valid partial TOML
        # and is out of scope: the atomic os.replace save makes torn writes of
        # the main file impossible going forward.)
        path.write_bytes(b"[shot]\npixels_per_foot = ")

        loaded = load_config(path)

        # The corrupt file must be moved aside, never left in place where the
        # autosave would overwrite it...
        assert not path.exists()
        corrupts = list(tmp_path.glob("config.toml.corrupt-*"))
        assert corrupts, "corrupt file was not preserved aside"
        # ...and the last good calibration must be RESTORED from the backup,
        # not silently replaced with defaults.
        assert loaded.shot.pixels_per_foot == 73.0
        assert loaded.detection_zone.start_x1 == 98

    def test_truncated_config_without_backup_uses_defaults(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_bytes(b"\x00\x01garbage")
        loaded = load_config(path)
        assert loaded.shot.pixels_per_foot == AppConfig().shot.pixels_per_foot
        assert list(tmp_path.glob("config.toml.corrupt-*"))

    def test_no_temp_files_left_behind(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        save_config(self._custom_config(), path)
        assert not list(tmp_path.glob("*.tmp"))

    def test_concurrent_saves_do_not_corrupt_or_raise(self, tmp_path: Path):
        """Two real threads hammering save_config on one path (the UI autosave
        vs processing-thread scenario). Regression: 17.5% PermissionError rate
        at os.replace before the module-level save lock + retry."""
        import threading

        path = tmp_path / "config.toml"
        errors: list[BaseException] = []

        def hammer() -> None:
            cfg = self._custom_config()
            for _ in range(60):
                try:
                    save_config(cfg, path)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[:3]
        assert load_config(path).shot.pixels_per_foot == 73.0
        assert not list(tmp_path.glob("*.tmp"))
