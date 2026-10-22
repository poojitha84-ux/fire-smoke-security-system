"""
Pytest configuration and shared fixtures for Fire & Smoke Security System.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, List
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig


@dataclass
class MockBoxItem:
    cls: Any
    conf: Any
    xyxy: Any


class MockBoxes:
    def __init__(self, boxes: List[MockBoxItem]):
        self.boxes = boxes

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)


class MockTorchItem:
    def __init__(self, val):
        self.val = val

    def item(self):
        return self.val


class MockTorchArray:
    def __init__(self, arr):
        self.arr = np.array(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class MockYOLOResult:
    def __init__(self, boxes: List[dict], names: dict):
        self.names = names
        mock_items = []
        for b in boxes:
            mock_items.append(
                MockBoxItem(
                    cls=[MockTorchItem(b["cls"])],
                    conf=[MockTorchItem(b["conf"])],
                    xyxy=[MockTorchArray(b["xyxy"])],
                )
            )
        self.boxes = MockBoxes(mock_items)


class MockYOLOModel:
    """Mock YOLO model for deterministic offline unit testing."""
    def __init__(self, names: dict):
        self.names = names
        self.next_detections: List[dict] = []

    def set_predictions(self, detections: List[dict]):
        self.next_detections = detections

    def predict(self, frame, **kwargs):
        return [MockYOLOResult(self.next_detections, self.names)]


@pytest.fixture
def sample_config(tmp_path) -> AppConfig:
    config = AppConfig()
    config.BASE_DIR = tmp_path
    config.MODELS_DIR = tmp_path / "models"
    config.CAMERA_SOURCE = 0
    config.CAMERA_ID = "TEST-CAM-01"
    config.CAMERA_LOCATION = "Lab Alpha"
    config.FIRE_CONFIDENCE_THRESHOLD = 0.45
    config.SMOKE_CONFIDENCE_THRESHOLD = 0.40
    config.TEMPORAL_WINDOW_SIZE = 8
    config.TEMPORAL_CONFIRM_COUNT = 5
    config.CLEAR_CONSECUTIVE_FRAMES = 10
    config.ALERT_COOLDOWN_SECONDS = 5.0
    config.ENABLE_AUDIO_ALARM = True
    config.ENABLE_SCREEN_REJECTION = True
    config.SCREEN_OVERLAP_REJECT_THRESHOLD = 0.65
    config.SCREEN_OVERLAP_UNCERTAIN_THRESHOLD = 0.25
    return config


@pytest.fixture
def mock_fire_smoke_model() -> MockYOLOModel:
    return MockYOLOModel(names={0: "smoke", 1: "fire"})


@pytest.fixture
def mock_screen_model() -> MockYOLOModel:
    return MockYOLOModel(names={67: "cell phone", 62: "tv", 63: "laptop"})


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)
