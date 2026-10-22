"""
Integration and Unit tests for Detection Pipeline, Model Loader, and Config.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from config import AppConfig
from detector.model_loader import ModelLoader, ModelLoadError
from detector.screen_context import ScreenContextFilter, HazardStatus
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState
from detector.inference import DetectionPipeline, FrameResult


def test_invalid_config_validation():
    cfg = AppConfig()
    cfg.FIRE_CONFIDENCE_THRESHOLD = 1.5  # Invalid (> 1.0)
    cfg.TEMPORAL_CONFIRM_COUNT = 15
    cfg.TEMPORAL_WINDOW_SIZE = 8       # Invalid (confirm > window)
    cfg.ALERT_COOLDOWN_SECONDS = -5.0  # Invalid (< 0)

    errors = cfg.validate()
    assert len(errors) >= 3
    assert any("FIRE_CONFIDENCE_THRESHOLD" in e for e in errors)
    assert any("TEMPORAL_CONFIRM_COUNT" in e for e in errors)
    assert any("ALERT_COOLDOWN_SECONDS" in e for e in errors)


def test_missing_model_weights_raises_descriptive_error(tmp_path):
    loader = ModelLoader(models_dir=tmp_path / "empty_models")
    non_existent = tmp_path / "does_not_exist.pt"

    with patch("detector.model_loader.ModelLoader.download_file", side_effect=ModelLoadError("Network offline")):
        with pytest.raises(ModelLoadError) as exc_info:
            loader.load_fire_smoke_model(str(non_existent))
        assert "unavailable" in str(exc_info.value).lower()


def test_class_mapping_and_confidence_threshold_filtering(sample_config, mock_fire_smoke_model, dummy_frame):
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    screen_filter = ScreenContextFilter(enabled=False)
    temporal_filter = TemporalFilter()
    alert_manager = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_filter,
        temporal_filter=temporal_filter,
        alert_manager=alert_manager,
        fire_threshold=0.50,
        smoke_threshold=0.40,
    )

    # Mock predictions:
    # 1. Fire with conf 0.40 (BELOW fire threshold 0.50 -> must be filtered out!)
    # 2. Fire with conf 0.75 (ABOVE fire threshold 0.50 -> kept!)
    # 3. Smoke with conf 0.35 (BELOW smoke threshold 0.40 -> filtered out!)
    # 4. Smoke with conf 0.65 (ABOVE smoke threshold 0.40 -> kept!)
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.40, "xyxy": [10, 10, 50, 50]},
        {"cls": 1, "conf": 0.75, "xyxy": [100, 100, 200, 200]},
        {"cls": 0, "conf": 0.35, "xyxy": [250, 250, 300, 300]},
        {"cls": 0, "conf": 0.65, "xyxy": [350, 350, 450, 450]},
    ])

    result: FrameResult = pipeline.process_frame(dummy_frame)

    assert result.environmental_fire_count == 1
    assert result.environmental_smoke_count == 1
    assert result.raw_hazard_count == 2  # Only 2 passed threshold


def test_end_to_end_pipeline_with_phone_rejection(sample_config, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67]  # cell phone

    screen_filter = ScreenContextFilter(enabled=True, reject_threshold=0.65)
    temporal_filter = TemporalFilter(window_size=5, confirm_count=3)
    alert_manager = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_filter,
        temporal_filter=temporal_filter,
        alert_manager=alert_manager,
        fire_threshold=0.45,
    )

    # Case: Phone detected at [100, 100, 300, 400]
    mock_screen_model.set_predictions([
        {"cls": 67, "conf": 0.90, "xyxy": [100, 100, 300, 400]}
    ])

    # Fire displayed on phone at [120, 120, 280, 380] (completely inside)
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [120, 120, 280, 380]}
    ])

    res = pipeline.process_frame(dummy_frame)

    # Must be rejected!
    assert res.screen_rejected_count == 1
    assert res.environmental_fire_count == 0
    assert res.fire_state == HazardState.NORMAL
    assert res.alert_state == AlertState.NORMAL
    assert pipeline.total_screen_false_positives_blocked == 1
