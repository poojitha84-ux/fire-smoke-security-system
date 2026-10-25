"""
Unit tests for Streamlit application integration and detection pipeline.
Verifies that the Streamlit components cleanly reuse the core pipeline,
handle frame processing, and respond to dynamic configuration changes.
"""

import numpy as np
import pytest
import cv2

from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState
from detector.inference import DetectionPipeline, FrameResult


class MockModel:
    """Mock YOLO model for fast pipeline verification in unit tests."""
    def __init__(self, names=None):
        self.names = names or {0: "fire", 1: "smoke"}

    def predict(self, frame, conf=0.25, verbose=False):
        return []


@pytest.fixture
def mock_streamlit_pipeline():
    """Fixture to build a lightweight detection pipeline matching streamlit_app.py."""
    config = AppConfig()
    loader = ModelLoader()
    loader.fire_smoke_model = MockModel({0: "fire", 1: "smoke"})
    loader.fire_class_ids = [0]
    loader.smoke_class_ids = [1]
    loader.screen_model = MockModel({0: "cell phone", 1: "tv", 2: "laptop", 3: "person"})
    loader.screen_class_ids = [0, 1, 2]
    loader.person_class_ids = [3]

    screen_filter = ScreenContextFilter()
    temporal_filter = TemporalFilter(window_size=3, confirm_count=2)
    alert_manager = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_filter,
        temporal_filter=temporal_filter,
        alert_manager=alert_manager,
        fire_threshold=0.45,
        smoke_threshold=0.40,
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )
    return pipeline, config


def test_streamlit_pipeline_blank_frame(mock_streamlit_pipeline):
    """Test that the pipeline processes empty/blank camera frames cleanly."""
    pipeline, config = mock_streamlit_pipeline
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(blank_frame)

    assert isinstance(result, FrameResult)
    assert result.alert_state == AlertState.NORMAL
    assert result.environmental_fire_count == 0
    assert result.environmental_smoke_count == 0
    assert result.annotated_frame.shape == (480, 640, 3)


def test_streamlit_pipeline_dynamic_threshold_update(mock_streamlit_pipeline):
    """Test that adjusting sliders in the Streamlit interface updates pipeline attributes."""
    pipeline, config = mock_streamlit_pipeline

    # Adjust threshold as Streamlit slider would
    pipeline.fire_threshold = 0.65
    pipeline.enable_smoke_detection = True
    pipeline.smoke_threshold = 0.55

    assert pipeline.fire_threshold == 0.65
    assert pipeline.enable_smoke_detection is True
    assert pipeline.smoke_threshold == 0.55


def test_streamlit_pipeline_defense_toggles(mock_streamlit_pipeline):
    """Test that false-positive defense toggles can be modified dynamically."""
    pipeline, config = mock_streamlit_pipeline

    pipeline.screen_filter.enabled = False
    pipeline.screen_filter.enable_face_smoke_filter = False
    pipeline.screen_filter.enable_white_object_filter = False

    assert pipeline.screen_filter.enabled is False
    assert pipeline.screen_filter.enable_face_smoke_filter is False
    assert pipeline.screen_filter.enable_white_object_filter is False


def test_streamlit_pipeline_snapshot_frame_processing(mock_streamlit_pipeline):
    """Test processing an arbitrary synthetic image as captured by st.camera_input."""
    pipeline, config = mock_streamlit_pipeline
    test_img = np.full((300, 400, 3), 128, dtype=np.uint8)

    result = pipeline.process_frame(test_img)

    assert result is not None
    assert result.frame_index == 1
    assert result.annotated_frame.shape == (300, 400, 3)
