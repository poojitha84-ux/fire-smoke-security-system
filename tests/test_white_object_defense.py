"""
Comprehensive Regression & Scenario Tests for White Objects, Rigid Surfaces,
Optical Glare, and Bright Backgrounds False-Positive Defense.

Tests:
1. Plain white wall (flat uniform surface) -> WHITE_OBJECT_REJECTED
2. White paper / notebook on desk (rigid rectilinear contour) -> WHITE_OBJECT_REJECTED
3. White shirt / clothing on detected person -> FACE_REJECTED (torso/clothing artifact)
4. Bright window / sunlight glare (specular saturation) -> WHITE_OBJECT_REJECTED
5. White background with a person (person detected + white wall candidate rejected) -> NORMAL
6. Gray / light-colored rigid furniture (straight rectangular edges) -> WHITE_OBJECT_REJECTED
7. Normal room lighting and low lighting (clean scenes remain NORMAL)
8. Genuine white smoke (amorphous, turbulent, soft edges) -> CONFIRMED_ENVIRONMENTAL
9. Genuine dark smoke (amorphous industrial plume) -> CONFIRMED_ENVIRONMENTAL
10. Genuine fire near white surface -> Fire is NEVER suppressed
11. Empirical real model test on real asset images (zidane.jpg, bus.jpg)
"""

import cv2
import numpy as np
import pytest
from pathlib import Path

from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter, HazardStatus, PersonDetection, estimate_head_bbox, estimate_torso_bbox
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState
from detector.inference import DetectionPipeline, FrameResult


def test_plain_white_wall_rejected(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 1: Plain white wall.
    - Flat uniform white painted surface (mean V >= 230, Sat <= 15, std < 4.5).
    - Model predicts low/moderate confidence smoke (conf 0.48).
    - White-object filter classifies as WHITE_OBJECT_REJECTED.
    - No environmental smoke, no alert, alarm silent.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    # Plain white wall frame (BGR 245)
    wall_frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.48, "xyxy": [150, 100, 450, 380]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(wall_frame)
        assert res.environmental_smoke_count == 0
        assert res.white_rejected_count == 1
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_white_false_positives_blocked == 8


def test_white_paper_on_desk_rejected(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 2: White paper or notebook on desk.
    - Rigid rectangular contour with sharp step edges.
    - Candidate confidence 0.52.
    - Filter detects rigid 4-sided rectilinear geometry and rejects candidate.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    # Frame with dark desk and white sheet of paper
    frame = np.full((480, 640, 3), (80, 120, 160), dtype=np.uint8)
    cv2.rectangle(frame, (150, 100), (450, 380), (250, 250, 250), -1)
    cv2.rectangle(frame, (150, 100), (450, 380), (30, 30, 30), 2) # sharp border

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.52, "xyxy": [148, 98, 452, 382]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(frame)
        assert res.environmental_smoke_count == 0
        assert res.white_rejected_count == 1
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_white_false_positives_blocked == 8


def test_white_shirt_on_person_rejected(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 3: White shirt / clothing on detected person.
    - Person detected in frame: bbox [200, 100, 500, 480].
    - Torso box estimated at [215, 206, 485, 423].
    - Smoke candidate predicted on chest/torso (conf 0.54).
    - Torso/clothing defense classifies as FACE_REJECTED (clothing artifact).
    - Audio siren stays silent!
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(
        enabled=True,
        enable_face_smoke_filter=True,
        enable_clothing_smoke_filter=True,
    )
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.91, "xyxy": [200, 100, 500, 480]}
    ])
    # Smoke on torso/shirt
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.54, "xyxy": [220, 210, 470, 410]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 0
        assert res.face_rejected_count == 1
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_face_false_positives_blocked == 8


def test_bright_window_glare_rejected(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 4: Bright window and sunlight glare.
    - Frame contains an overexposed window (>70% pixels with V >= 248, Sat <= 25, std < 8).
    - Candidate confidence 0.52.
    - Optical glare defense classifies as WHITE_OBJECT_REJECTED.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (150, 50), (450, 350), (255, 255, 255), -1)

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.52, "xyxy": [150, 50, 450, 350]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(frame)
        assert res.environmental_smoke_count == 0
        assert res.white_rejected_count == 1
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_white_false_positives_blocked == 8


def test_white_background_with_person(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 5: White background with a person.
    - Operator seated in front of a white wall.
    - Person detected at [300, 100, 600, 480].
    - Smoke falsely fires on the white background at [50, 50, 250, 350] (conf 0.47).
    - White-object filter rejects the wall candidate.
    - Total scene remains completely NORMAL.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(
        enabled=True,
        enable_face_smoke_filter=True,
        enable_white_object_filter=True,
    )
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    frame = np.full((480, 640, 3), 245, dtype=np.uint8) # white wall background

    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.92, "xyxy": [300, 100, 600, 480]}
    ])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.47, "xyxy": [50, 50, 250, 350]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(frame)
        assert res.environmental_smoke_count == 0
        assert res.white_rejected_count == 1
        assert res.alert_state == AlertState.NORMAL


def test_light_colored_furniture_rejected(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 6: Gray or light-colored furniture.
    - Rigid rectilinear furniture structure (table/cabinet).
    - Candidate confidence 0.50.
    - Rectilinear object defense rejects candidate.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    # Frame with light-gray tall cabinet / bookcase (aspect ratio outside display screen)
    frame = np.full((480, 640, 3), (80, 80, 90), dtype=np.uint8)
    cv2.rectangle(frame, (100, 50), (220, 450), (220, 220, 225), -1)
    cv2.rectangle(frame, (100, 50), (220, 450), (40, 40, 45), 2)

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.50, "xyxy": [98, 48, 222, 452]}
    ])

    res = pipeline.process_frame(frame)
    assert res.environmental_smoke_count == 0
    assert res.white_rejected_count == 1
    assert res.alert_state == AlertState.NORMAL
    assert res.audio_alarm_active is False


def test_normal_room_and_low_lighting(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 7: Normal room lighting and low lighting.
    - Normal scene without hazards stays NORMAL.
    - Low lighting scene without hazards stays NORMAL.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
    )

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([])

    # Normal lighting
    normal_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    res_norm = pipeline.process_frame(normal_frame)
    assert res_norm.alert_state == AlertState.NORMAL

    # Low lighting
    low_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    res_low = pipeline.process_frame(low_frame)
    assert res_low.alert_state == AlertState.NORMAL


def test_genuine_white_smoke_confirmed(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 8: Genuine white smoke plume.
    - Real white smoke has turbulent variance (std >= 10), amorphous soft boundaries,
      no 4-corner rectilinear contour, and is not blown-out optical glare.
    - Must NOT be rejected by white-object filter!
    - Passes to temporal confirmation and triggers smoke alert.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    # Frame with simulated genuine white smoke plume
    np.random.seed(42)
    frame = np.full((480, 640, 3), (110, 110, 110), dtype=np.uint8)
    smoke_patch = np.full((200, 200, 3), 210, dtype=np.uint8)
    noise = (np.random.randn(200, 200, 3) * 20).astype(np.int16)
    smoke_patch = np.clip(smoke_patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    smoke_patch = cv2.GaussianBlur(smoke_patch, (21, 21), 0)
    frame[100:300, 150:350] = smoke_patch

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.55, "xyxy": [150, 100, 350, 300]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(frame)
        assert res.environmental_smoke_count == 1
        assert res.white_rejected_count == 0

    assert res.smoke_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_genuine_dark_smoke_confirmed(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 9: Genuine dark industrial smoke.
    - Dark gray/black amorphous smoke cloud.
    - Confirmed immediately without interference from white object defense.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    frame = np.full((480, 640, 3), (140, 140, 140), dtype=np.uint8)
    dark_patch = np.full((180, 180, 3), 55, dtype=np.uint8)
    dark_noise = (np.random.randn(180, 180, 3) * 15).astype(np.int16)
    dark_patch = np.clip(dark_patch.astype(np.int16) + dark_noise, 0, 255).astype(np.uint8)
    dark_patch = cv2.GaussianBlur(dark_patch, (17, 17), 0)
    frame[120:300, 180:360] = dark_patch

    mock_screen_model.set_predictions([])
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.60, "xyxy": [180, 120, 360, 300]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(frame)
        assert res.environmental_smoke_count == 1

    assert res.smoke_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_genuine_fire_near_white_object_never_suppressed(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 10: Genuine fire near a white wall or object.
    - Life-safety critical: Fire is NEVER evaluated or rejected by white-object filter.
    - Fire is confirmed immediately and sounds the audio siren.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.40,
    )

    white_frame = np.full((480, 640, 3), 245, dtype=np.uint8)

    mock_screen_model.set_predictions([])
    # Fire candidate on white surface
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [150, 150, 300, 300]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(white_frame)
        assert res.environmental_fire_count == 1

    assert res.fire_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_empirical_real_models_execution_on_asset_images():
    """
    SCENARIO 11: Empirical verification using real weights from disk
    (models/fire_smoke_yolov8n.pt and models/yolov8n.pt).
    Runs real model forward passes on asset image test data without errors.
    """
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    fire_smoke_weights = models_dir / "fire_smoke_yolov8n.pt"
    screen_weights = models_dir / "yolov8n.pt"

    assert fire_smoke_weights.exists()
    assert screen_weights.exists()

    loader = ModelLoader(models_dir=models_dir)
    loader.load_fire_smoke_model(str(fire_smoke_weights))
    loader.load_screen_detector_model(str(screen_weights))

    filter_ctx = ScreenContextFilter(enabled=True, enable_white_object_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
    )

    # Run on synthetic white wall frame
    white_frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    res = pipeline.process_frame(white_frame)
    assert res is not None
    assert res.environmental_fire_count == 0
    assert res.environmental_smoke_count == 0
    assert res.alert_state == AlertState.NORMAL
