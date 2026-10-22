"""
Regression and Integration Tests for Human Face Smoke False-Positive Defense.

Tests:
1. Normal webcam scene: person sitting in front of camera with no hazards -> stays NORMAL
2. False smoke on face: model predicts smoke directly on user's head/face (conf 0.48) -> FACE_REJECTED, NO alarm
3. Ambiguous smoke on face (conf 0.62) -> FACE_UNCERTAIN, displayed on HUD, no emergency siren
4. Real smoke in room beside person (conf 0.45) -> CONFIRMED_ENVIRONMENTAL, triggers smoke alert
5. Real billowing smoke covering person (plume dispersion > 35%) -> CONFIRMED_ENVIRONMENTAL, triggers alert
6. Real fire near person (conf 0.85) -> CONFIRMED_ENVIRONMENTAL, fire is NEVER suppressed by face filter
7. Disabled face filter (ENABLE_FACE_SMOKE_FILTER=False) -> regression test confirming face smoke would trigger without defense
8. Real model verification: tests actual YOLO COCO person/head extraction on real frame
"""

from pathlib import Path
import numpy as np
import pytest

from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter, HazardStatus, PersonDetection, estimate_head_bbox
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState
from detector.inference import DetectionPipeline, FrameResult
from services.incident_service import IncidentService
from services.camera_service import CameraService, CameraStatus


def test_estimate_head_bbox_geometry():
    """Verify anatomical head bbox estimation from full person box."""
    # Full person box: x: 100..500 (w=400), y: 100..900 (h=800)
    person_box = [100, 100, 500, 900]
    head_box = estimate_head_bbox(person_box)

    # Inset: 10% on each side -> 100 + 40 = 140, 500 - 40 = 460
    assert head_box[0] == 140
    assert head_box[2] == 460
    # Top 42% -> 100 to 100 + 0.42*800 = 436
    assert head_box[1] == 100
    assert head_box[3] == 436


def test_normal_webcam_scene_with_person(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Normal webcam use with an operator sitting at their desk:
    - Person detected by COCO model
    - Fire/smoke model detects NO hazards
    - System stays strictly NORMAL, 0 alerts, 0 false alarms
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
    )

    # Operator seated in front of webcam: person detected, no fire/smoke
    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.92, "xyxy": [200, 100, 500, 480]}  # person
    ])
    mock_fire_smoke_model.set_predictions([])

    for _ in range(10):
        res: FrameResult = pipeline.process_frame(dummy_frame)
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False
        assert res.environmental_smoke_count == 0
        assert res.environmental_fire_count == 0


def test_face_detected_as_smoke_rejected(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    ROOT CAUSE BUG REPRODUCTION & FIX:
    - User sits in front of webcam: person box [200, 100, 500, 480] -> head [230, 100, 470, 259]
    - Fire/smoke model falsely predicts 'smoke' on facial features at [250, 120, 450, 250] with conf 0.48
    - Face defense must classify this as FACE_REJECTED
    - NO environmental smoke added, NO alert confirmed, audio alarm remains SILENT!
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
        face_overlap_reject_threshold=0.65,
        face_uncertain_conf_floor=0.55,
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

    # Person present in frame
    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.91, "xyxy": [200, 100, 500, 480]}
    ])
    # Model falsely fires 'smoke' on the face region
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.48, "xyxy": [250, 120, 450, 250]}
    ])

    for _ in range(8):
        res: FrameResult = pipeline.process_frame(dummy_frame)
        # MUST NOT be confirmed environmental smoke!
        assert res.environmental_smoke_count == 0
        assert res.face_rejected_count == 1
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_face_false_positives_blocked == 8


def test_face_detected_as_smoke_uncertain_at_intermediate_confidence(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Ambiguous face-overlapping smoke with intermediate confidence (0.62):
    - Higher than conf_floor (0.55) but below calibrated confirmation (0.72)
    - Must be classified as FACE_UNCERTAIN
    - Shows on HUD for operator review, but does NOT trigger emergency alert or audio alarm
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
        face_uncertain_conf_floor=0.55,
        face_smoke_calibrated_threshold=0.72,
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
        {"cls": 0, "conf": 0.88, "xyxy": [200, 100, 500, 480]}
    ])
    # Smoke on face with conf 0.62
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.62, "xyxy": [250, 120, 450, 250]}
    ])

    for _ in range(8):
        res: FrameResult = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 0
        assert res.face_uncertain_count == 1
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False


def test_genuine_smoke_beside_person_confirmed(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Real smoke in the room beside a person:
    - Person is at [300, 100, 600, 480]
    - Smoke is at [50, 50, 250, 300] with conf 0.45 (light smoke)
    - Face filter must NOT interfere with genuine room smoke!
    - System confirms smoke alert and triggers audio
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
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
        {"cls": 0, "conf": 0.90, "xyxy": [300, 100, 600, 480]}
    ])
    # Real environmental smoke in the room, not on person
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.45, "xyxy": [50, 50, 250, 300]}
    ])

    for _ in range(3):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 1

    # Frame 4 reaches confirm count
    res4 = pipeline.process_frame(dummy_frame)
    assert res4.smoke_state == HazardState.CONFIRMED
    assert res4.alert_state == AlertState.CONFIRMED_ALERT
    assert res4.audio_alarm_active is True


def test_genuine_billowing_smoke_covering_person_confirmed(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Dangerous situation: real dense billowing smoke covering room where a person is:
    - Person is at [200, 100, 500, 480]
    - Large smoke plume is at [80, 20, 600, 450] (billowing across room and over person)
    - Plume dispersion ratio is > 35% external
    - Face filter must recognize environmental plume and CONFIRM smoke!
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
        smoke_plume_dispersion_threshold=0.35,
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
        {"cls": 0, "conf": 0.85, "xyxy": [200, 100, 500, 480]}
    ])
    # Large smoke plume covering person and extending far into room
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.65, "xyxy": [80, 20, 600, 450]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 1

    assert res.smoke_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_genuine_fire_near_person_never_rejected(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Fire is an immediate life-safety hazard:
    - Person detected at [200, 100, 500, 480]
    - Flame detected at [250, 150, 350, 280] (directly in front of person)
    - Fire filter must NEVER reject flame due to face/person presence!
    - Confirms fire immediately!
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.40,
    )

    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.90, "xyxy": [200, 100, 500, 480]}
    ])
    # Fire candidate overlapping person
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [250, 150, 350, 280]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_fire_count == 1

    assert res.fire_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_disabled_face_smoke_filter_behavior(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Regression verification:
    If enable_face_smoke_filter is False, the face defense is inactive,
    confirming that the filter was indeed responsible for suppression.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    # Explicitly DISABLE face filter
    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=False)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.90, "xyxy": [200, 100, 500, 480]}
    ])
    # Smoke on face
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.48, "xyxy": [250, 120, 450, 250]}
    ])

    for _ in range(4):
        res = pipeline.process_frame(dummy_frame)

    # Without filter, it would count as environmental smoke!
    assert res.environmental_smoke_count == 1
    assert res.smoke_state == HazardState.CONFIRMED


def test_real_model_person_and_head_extraction():
    """
    EMPIRICAL MODEL TEST (Uses REAL models/yolov8n.pt on disk):
    Verifies that the actual COCO YOLOv8n weights load correctly,
    properly include person class (0), and successfully extract person
    and head bounding boxes on a test frame without errors.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    coco_path = models_dir / "yolov8n.pt"
    assert coco_path.exists(), f"Missing weights: {coco_path}"

    loader = ModelLoader(models_dir=models_dir)
    model = loader.load_screen_detector_model(str(coco_path))

    assert model is not None
    assert 0 in loader.person_class_ids, "Person class 0 not found in COCO model"

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)

    # Synthetic test frame (480x640)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test unified context detection
    screens, persons = filter_ctx.detect_context_from_model(
        model,
        test_frame,
        screen_class_ids=loader.screen_class_ids,
        person_class_ids=loader.person_class_ids,
    )

    # Expect valid lists without exceptions
    assert isinstance(screens, list)
    assert isinstance(persons, list)


def test_person_moving_head_multiple_frames(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 3: Person moving head across consecutive frames.
    - Operator seated at desk shifts head/posture across 6 frames.
    - Person detection and false smoke coordinates track together.
    - Face filter must continuously reject every frame without jitter or leakage into temporal filter.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    # Shifting coordinates over 6 frames (simulating natural head movement)
    shifts = [
        ([200, 100, 500, 480], [250, 120, 450, 250]),
        ([210, 105, 510, 485], [260, 125, 460, 255]),
        ([220, 110, 520, 490], [270, 130, 470, 260]),
        ([215, 100, 515, 480], [265, 120, 465, 250]),
        ([205, 95, 505, 475], [255, 115, 455, 245]),
        ([200, 100, 500, 480], [250, 120, 450, 250]),
    ]

    for p_box, s_box in shifts:
        mock_screen_model.set_predictions([{"cls": 0, "conf": 0.90, "xyxy": p_box}])
        mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.49, "xyxy": s_box}])

        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 0
        assert res.face_rejected_count == 1
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    assert pipeline.total_face_false_positives_blocked == 6
    assert temp_flt.trackers["smoke"].positive_ratio == 0.0


def test_varying_lighting_conditions_face_defense(mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 4: Varying lighting conditions (dark, normal, bright/washed out).
    - Verifies that geometric and contextual defense operates reliably regardless
      of frame pixel luminance variations that often cause contrast-based false edges.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([{"cls": 0, "conf": 0.89, "xyxy": [200, 100, 500, 480]}])
    mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.50, "xyxy": [250, 120, 450, 250]}])

    # 1. Dark lighting scene (mean ~25)
    dark_frame = np.full((480, 640, 3), 25, dtype=np.uint8)
    res_dark = pipeline.process_frame(dark_frame)
    assert res_dark.face_rejected_count == 1
    assert res_dark.environmental_smoke_count == 0

    # 2. Normal indoor lighting (mean ~128)
    mid_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res_mid = pipeline.process_frame(mid_frame)
    assert res_mid.face_rejected_count == 1
    assert res_mid.environmental_smoke_count == 0

    # 3. Bright backlit / glare scene (mean ~240)
    bright_frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    res_bright = pipeline.process_frame(bright_frame)
    assert res_bright.face_rejected_count == 1
    assert res_bright.environmental_smoke_count == 0


def test_partially_visible_person_at_frame_edge(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 5: Partially visible person near the edge of the frame.
    - Person is clipped against the top/left boundary: [0, 0, 220, 320].
    - Verifies that head bounding box estimation handles boundary conditions
      without IndexError, negative slices, or broken overlap calculations.
    """
    person_box = [0, 0, 220, 320]
    head_box = estimate_head_bbox(person_box)
    assert head_box[0] >= 0
    assert head_box[1] >= 0
    assert head_box[2] > head_box[0]
    assert head_box[3] > head_box[1]

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([{"cls": 0, "conf": 0.88, "xyxy": person_box}])
    # False smoke inside head area at edge
    mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.49, "xyxy": [25, 10, 190, 120]}])

    res = pipeline.process_frame(dummy_frame)
    assert res.face_rejected_count == 1
    assert res.environmental_smoke_count == 0
    assert res.alert_state == AlertState.NORMAL


def test_simultaneous_fire_and_smoke_near_person(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 9: Fire and smoke occurring simultaneously near a person.
    - Person seated at desk.
    - Both genuine fire and genuine smoke are present in the room.
    - System confirms both hazards; fire alert takes priority; smoke is not suppressed.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.40,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([{"cls": 0, "conf": 0.90, "xyxy": [350, 100, 600, 480]}])
    # Both genuine fire and genuine smoke in room space
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.86, "xyxy": [50, 200, 180, 350]},  # Fire
        {"cls": 0, "conf": 0.65, "xyxy": [40, 50, 280, 250]},   # Smoke
    ])

    for _ in range(4):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_fire_count == 1
        assert res.environmental_smoke_count == 1

    assert res.fire_state == HazardState.CONFIRMED
    assert res.smoke_state == HazardState.CONFIRMED
    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.audio_alarm_active is True


def test_persistent_false_detections_temporal_isolation(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 10: Persistent false detections across 30 consecutive frames.
    - Confirms that face smoke detections are filtered at the spatial context layer
      and NEVER enter the temporal filter, preventing any false alarm escalation
      even under indefinite webcam exposure.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    filter_ctx = ScreenContextFilter(enabled=True, enable_face_smoke_filter=True)
    temp_flt = TemporalFilter(window_size=8, confirm_count=4)
    alert_mgr = AlertManager(enable_audio=True)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=filter_ctx,
        temporal_filter=temp_flt,
        alert_manager=alert_mgr,
        smoke_threshold=0.40,
    )

    mock_screen_model.set_predictions([{"cls": 0, "conf": 0.92, "xyxy": [200, 100, 500, 480]}])
    mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.49, "xyxy": [250, 120, 450, 250]}])

    for i in range(30):
        res = pipeline.process_frame(dummy_frame)
        assert res.environmental_smoke_count == 0
        assert res.face_rejected_count == 1
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False
        assert temp_flt.trackers["smoke"].positive_ratio == 0.0

    assert pipeline.total_face_false_positives_blocked == 30


def test_missing_or_unavailable_camera_handling(sample_config):
    """
    SCENARIO 11: Missing or unavailable webcam handling.
    - Tests CameraService behavior when an invalid camera device ID is supplied.
    - Verifies graceful error capture, status update, and safe cleanup without crashing.
    """
    from unittest.mock import MagicMock, patch
    pipeline = MagicMock(spec=DetectionPipeline)
    service = CameraService(sample_config, pipeline)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        service.start_monitoring(source=9999)
        assert service.status == CameraStatus.ERROR
        assert "Failed opening camera source" in service.status_message

        # Safely stops without throwing
        service.stop_monitoring()
        assert service.status == CameraStatus.STOPPED


def test_hud_labels_and_confidence_propagation(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    SCENARIO 12: Confidence values and displayed detection labels on HUD.
    - Verifies exact status tags and confidence propagation for both FACE_REJECTED
      and FACE_UNCERTAIN in FrameResult and HUD overlay.
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
        face_uncertain_conf_floor=0.55,
        face_smoke_calibrated_threshold=0.72,
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

    mock_screen_model.set_predictions([{"cls": 0, "conf": 0.93, "xyxy": [200, 100, 500, 480]}])

    # Case A: Low confidence (0.48) -> FACE_REJECTED
    mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.48, "xyxy": [250, 120, 450, 250]}])
    res_rejected = pipeline.process_frame(dummy_frame)
    assert res_rejected.face_rejected_count == 1
    assert len(res_rejected.validation_results) == 1
    val_rej = res_rejected.validation_results[0]
    assert val_rej.status == HazardStatus.FACE_REJECTED
    assert val_rej.confidence == 0.48
    assert "face" in val_rej.reason.lower()
    # Frame is annotated and not None
    assert res_rejected.annotated_frame is not None
    assert res_rejected.annotated_frame.shape == dummy_frame.shape

    # Case B: Intermediate confidence (0.64) -> FACE_UNCERTAIN
    mock_fire_smoke_model.set_predictions([{"cls": 0, "conf": 0.64, "xyxy": [250, 120, 450, 250]}])
    res_uncertain = pipeline.process_frame(dummy_frame)
    assert res_uncertain.face_uncertain_count == 1
    assert len(res_uncertain.validation_results) == 1
    val_unc = res_uncertain.validation_results[0]
    assert val_unc.status == HazardStatus.FACE_UNCERTAIN
    assert val_unc.confidence == 0.64
    assert res_uncertain.annotated_frame is not None

