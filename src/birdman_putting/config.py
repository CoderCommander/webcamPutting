"""Configuration system using dataclasses and TOML persistence."""

from __future__ import annotations

import logging
import sys
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

APP_NAME = "birdman-putting"
CONFIG_DIR = Path(user_config_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class DetectionZone:
    """Ball detection zone rectangle coordinates."""

    start_x1: int = 10
    start_x2: int = 180
    y1: int = 180
    y2: int = 450
    gateway_width: int = 10  # Width of the detection gateway past start_x2
    direction: str = "left_to_right"  # "left_to_right" or "right_to_left"


@dataclass
class CameraSettings:
    """Camera capture configuration."""

    webcam_index: int = 0
    mjpeg: bool = True
    fps_override: int = 0
    width: int = 0
    height: int = 0
    flip_image: bool = False
    flip_view: bool = False
    rotation: int = 0  # Degrees (-45 to +45), clockwise-positive
    darkness: int = 0
    ps4: bool = False

    # Camera properties (0.0 = use camera default)
    saturation: float = 0.0
    exposure: float = 0.0
    auto_wb: float = 0.0
    white_balance_blue: float = 0.0
    white_balance_red: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    hue: float = 0.0
    gain: float = 0.0
    sharpness: float = 0.0
    auto_exposure: float = 0.0
    gamma: float = 0.0
    zoom: float = 0.0
    focus: float = 0.0
    autofocus: float = 0.0


@dataclass
class BallSettings:
    """Ball detection settings."""

    color_preset: str = "yellow"
    custom_hsv: dict[str, int] | None = None
    fixed_radius: int = 0  # 0 = auto-detect
    min_radius: int = 5
    start_stability_frames: int = 15  # Frames needed for stable start detection
    start_position_tolerance: int = 3  # Pixels tolerance for position clustering


@dataclass
class ShotSettings:
    """Shot detection and validation thresholds."""

    min_speed_mph: float = 0.5
    max_speed_mph: float = 25.0
    min_time_seconds: float = 0.5
    max_hla_degrees: float = 40.0
    hla_consistency_threshold: float = 30.0
    min_exit_distance_px: int = 50  # Minimum pixel distance beyond gateway to count as exit


@dataclass
class ConnectionSettings:
    """GSPro connection configuration."""

    mode: str = "gspro_direct"  # "gspro_direct" or "http_middleware"
    gspro_host: str = "127.0.0.1"
    gspro_port: int = 921
    http_url: str = "http://127.0.0.1:8888/putting"
    device_id: str = "BirdmanPutting"
    heartbeat_interval: float = 5.0  # seconds


@dataclass
class ReplaySettings:
    """Replay camera configuration."""

    enabled: bool = False
    show_replay: bool = True
    camera_index: int = 0
    ps4: bool = False
    duration_seconds: float = 3.0


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection_zone: DetectionZone = field(default_factory=DetectionZone)
    camera: CameraSettings = field(default_factory=CameraSettings)
    ball: BallSettings = field(default_factory=BallSettings)
    shot: ShotSettings = field(default_factory=ShotSettings)
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)
    replay: ReplaySettings = field(default_factory=ReplaySettings)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a dict, handling nested dataclasses.

    Excludes None values since TOML doesn't support them.
    """
    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        if hasattr(value, "__dataclass_fields__"):
            result[f.name] = _dataclass_to_dict(value)
        else:
            result[f.name] = value
    return result


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass from a dict, handling nested dataclasses and missing keys."""
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            value = data[f.name]
            # Check if the field type is a dataclass
            field_type = f.type if isinstance(f.type, type) else None
            if field_type is None and isinstance(f.type, str):
                # Resolve string annotations
                import birdman_putting.config as mod
                field_type = getattr(mod, f.type, None)
            if (
                field_type
                and hasattr(field_type, "__dataclass_fields__")
                and isinstance(value, dict)
            ):
                kwargs[f.name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from TOML file, falling back to defaults for missing values."""
    config_path = path or CONFIG_FILE

    if not config_path.exists():
        logger.info("No config file found at %s, using defaults", config_path)
        return AppConfig()

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning("Failed to read config file %s, using defaults", config_path, exc_info=True)
        return AppConfig()

    config = AppConfig()
    for section_field in fields(AppConfig):
        if section_field.name in data and isinstance(data[section_field.name], dict):
            section_cls = type(getattr(config, section_field.name))
            section_defaults = getattr(config, section_field.name)
            section_data = {}
            for sf in fields(section_defaults):
                if sf.name in data[section_field.name]:
                    section_data[sf.name] = data[section_field.name][sf.name]
                else:
                    section_data[sf.name] = getattr(section_defaults, sf.name)
            setattr(config, section_cls.__name__, section_cls(**section_data))
            # Actually set on the config object
            object.__setattr__(config, section_field.name, section_cls(**section_data))

    return config


def save_config(config: AppConfig, path: Path | None = None) -> None:
    """Save config to TOML file."""
    config_path = path or CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = _dataclass_to_dict(config)

    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)

    logger.info("Config saved to %s", config_path)


def migrate_from_ini(ini_path: Path, config: AppConfig | None = None) -> AppConfig:
    """Migrate settings from old config.ini format to AppConfig."""
    parser = ConfigParser()
    parser.read(ini_path)
    cfg = config or AppConfig()

    if not parser.has_section("putting"):
        return cfg

    def get_int(key: str, default: int) -> int:
        return int(parser.get("putting", key)) if parser.has_option("putting", key) else default

    def get_float(key: str, default: float) -> float:
        return float(parser.get("putting", key)) if parser.has_option("putting", key) else default

    # Detection zone
    cfg.detection_zone.start_x1 = get_int("startx1", cfg.detection_zone.start_x1)
    cfg.detection_zone.start_x2 = get_int("startx2", cfg.detection_zone.start_x2)
    cfg.detection_zone.y1 = get_int("y1", cfg.detection_zone.y1)
    cfg.detection_zone.y2 = get_int("y2", cfg.detection_zone.y2)

    # Camera
    cfg.camera.flip_image = bool(get_int("flip", 0))
    cfg.camera.flip_view = bool(get_int("flipview", 0))
    cfg.camera.darkness = get_int("darkness", 0)
    cfg.camera.mjpeg = bool(get_int("mjpeg", 1))
    cfg.camera.ps4 = bool(get_int("ps4", 0))
    cfg.camera.fps_override = get_int("fps", 0)
    cfg.camera.height = get_int("height", 0)
    cfg.camera.width = get_int("width", 0)

    # Camera properties
    cfg.camera.saturation = get_float("saturation", 0.0)
    cfg.camera.exposure = get_float("exposure", 0.0)
    cfg.camera.auto_wb = get_float("autowb", 0.0)
    cfg.camera.white_balance_blue = get_float("whiteBalanceBlue", 0.0)
    cfg.camera.white_balance_red = get_float("whiteBalanceRed", 0.0)
    cfg.camera.brightness = get_float("brightness", 0.0)
    cfg.camera.contrast = get_float("contrast", 0.0)
    cfg.camera.hue = get_float("hue", 0.0)
    cfg.camera.gain = get_float("gain", 0.0)
    cfg.camera.sharpness = get_float("sharpness", 0.0)
    cfg.camera.auto_exposure = get_float("autoexposure", 0.0)
    cfg.camera.gamma = get_float("gamma", 0.0)
    cfg.camera.zoom = get_float("zoom", 0.0)
    cfg.camera.focus = get_float("focus", 0.0)
    cfg.camera.autofocus = get_float("autofocus", 0.0)

    # Ball
    cfg.ball.fixed_radius = get_int("radius", 0)

    # Custom HSV
    if parser.has_option("putting", "customhsv"):
        import ast
        try:
            cfg.ball.custom_hsv = ast.literal_eval(parser.get("putting", "customhsv"))
        except (ValueError, SyntaxError):
            logger.warning("Could not parse customhsv from INI file")

    # Replay
    cfg.replay.show_replay = bool(get_int("showreplay", 0))
    cfg.replay.enabled = bool(get_int("replaycam", 0))
    cfg.replay.camera_index = get_int("replaycamindex", 0)
    cfg.replay.ps4 = bool(get_int("replaycamps4", 0))

    logger.info("Migrated config from %s", ini_path)
    return cfg
