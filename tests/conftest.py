"""Shared test fixtures."""

import cv2
import numpy as np
import pytest

from birdman_putting.config import AppConfig, BallSettings, DetectionZone, ShotSettings


@pytest.fixture
def blank_frame() -> np.ndarray:
    """640x360 black BGR frame."""
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_with_yellow_ball(blank_frame: np.ndarray) -> np.ndarray:
    """Frame with a yellow circle at (100, 250) radius 15."""
    # Yellow in BGR: (0, 255, 255)
    cv2.circle(blank_frame, (100, 250), 15, (0, 255, 255), -1)
    return blank_frame


@pytest.fixture
def frame_with_orange_ball(blank_frame: np.ndarray) -> np.ndarray:
    """Frame with an orange circle at (150, 300) radius 12."""
    # Orange in BGR: (0, 165, 255)
    cv2.circle(blank_frame, (150, 300), 12, (0, 165, 255), -1)
    return blank_frame


@pytest.fixture
def default_config() -> AppConfig:
    """Default application config."""
    return AppConfig()


@pytest.fixture
def detection_zone() -> DetectionZone:
    """Default detection zone."""
    return DetectionZone()


@pytest.fixture
def ball_settings() -> BallSettings:
    """Default ball settings with fast stabilization for tests."""
    return BallSettings(start_stability_frames=3, start_position_tolerance=5)


@pytest.fixture
def shot_settings() -> ShotSettings:
    """Default shot settings."""
    return ShotSettings()
