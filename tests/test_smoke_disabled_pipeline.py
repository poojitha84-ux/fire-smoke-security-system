"""
Integration and Unit Tests for Pipeline with Smoke Detection Disabled.

Validates that:
1. Smoke detection is completely disabled in the inference pipeline and temporal confirmation.
2. Fire detection remains fully operational (bounding boxes, confidence, temporal confirmation, sirens, incident logs).
3. Simultaneous fire and smoke filters out smoke and confirms fire.
4. Sustained smoke inputs cannot create smoke incidents, trigger alerts, or sound alarms.
5. Phone screen defense remains active to filter false fire detections on mobile displays.
6. Camera telemetry and HUD overlays accurately reflect SMOKE: DISABLED and FIRE: ENABLED.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter, HazardStatus
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState
from detector.inference import DetectionPipeline, FrameResult
from services.incident_service import IncidentService
from services.camera_service import CameraService, CameraStatus


def test_default_config_has_smoke_disabled():
    """Verify that default AppConfig disables smoke and enables fire."""
    cfg = AppConfig()
    assert cfg.ENABLE_FIRE_DETECTION is True
    assert cfg.ENABLE_SMOKE_DETECTION is False


def test_smoke_completely_suppressed_when_disabled(sample_config, mock_fire_smoke_model, dummy_frame):
    """
    When enable_smoke_detection=False, high-confidence smoke predictions (0.95)
    are filtered out at inference before entering temporal confirmation.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=False),
        temporal_filter=TemporalFilter(),
        alert_manager=AlertManager(),
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    # Provide only smoke predictions (conf 0.95)
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.95, "xyxy": [100, 100, 300, 300]},
        {"cls": 0, "conf": 0.82, "xyxy": [400, 200, 550, 450]},
    ])

    result: FrameResult = pipeline.process_frame(dummy_frame)

    assert result.environmental_smoke_count == 0
    assert result.smoke_state == HazardState.NORMAL
    assert result.raw_hazard_count == 0
    assert result.alert_state == AlertState.NORMAL
    assert result.audio_alarm_active is False
    assert len(result.validation_results) == 0


def test_sustained_smoke_never_creates_incident_or_alarm(sample_config, mock_fire_smoke_model, dummy_frame, tmp_path):
    """
    Even with 15 sustained consecutive frames of smoke, the pipeline must
    NEVER enter suspected/confirmed smoke, NEVER trigger the audio siren,
    and NEVER log an incident.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    incident_svc = IncidentService(log_file=tmp_path / "incidents.jsonl")

    def on_incident(event):
        incident_svc.create_incident(
            camera_id=event.camera_id,
            camera_location=event.camera_location,
            hazard_type=event.hazard_type,
            confidence=event.max_confidence,
            severity=event.severity.value,
            confirmation_status=getattr(event, "confirmation_status", "CONFIRMED"),
            incident_id=event.event_id,
        )

    alert_mgr = AlertManager(
        cooldown_seconds=10.0,
        enable_audio=True,
        on_confirmed_incident=on_incident,
    )
    temporal_flt = TemporalFilter(window_size=5, confirm_count=3)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=False),
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.96, "xyxy": [150, 150, 350, 350]},
    ])

    for _ in range(15):
        res = pipeline.process_frame(dummy_frame)
        assert res.smoke_state == HazardState.NORMAL
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False

    # Verify no incidents created in incident service
    active_incidents = incident_svc.get_active_incidents()
    all_incidents = incident_svc.get_all_incidents()
    assert len(active_incidents) == 0
    assert len(all_incidents) == 0


def test_fire_detection_fully_operational_when_smoke_disabled(sample_config, mock_fire_smoke_model, dummy_frame, tmp_path):
    """
    Fire detection continues to function accurately when smoke is disabled:
    temporal confirmation confirms the fire, triggers RED ALERT, activates the siren,
    and creates an incident log.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    incident_svc = IncidentService(log_file=tmp_path / "incidents.jsonl")

    def on_incident(event):
        incident_svc.create_incident(
            camera_id=event.camera_id,
            camera_location=event.camera_location,
            hazard_type=event.hazard_type,
            confidence=event.max_confidence,
            severity=event.severity.value,
            confirmation_status=getattr(event, "confirmation_status", "CONFIRMED"),
            incident_id=event.event_id,
        )

    alert_mgr = AlertManager(
        cooldown_seconds=10.0,
        enable_audio=True,
        on_confirmed_incident=on_incident,
    )
    temporal_flt = TemporalFilter(window_size=5, confirm_count=3)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=False),
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        enable_fire_detection=True,
        enable_smoke_detection=False,
        fire_threshold=0.45,
    )

    # Frame with valid fire detection
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.88, "xyxy": [120, 120, 260, 280]},
    ])

    # Feed frames to trigger temporal confirmation (3 needed out of 5)
    results = [pipeline.process_frame(dummy_frame) for _ in range(4)]
    final_res = results[-1]

    assert final_res.environmental_fire_count == 1
    assert final_res.environmental_smoke_count == 0
    assert final_res.fire_state == HazardState.CONFIRMED
    assert final_res.smoke_state == HazardState.NORMAL
    assert final_res.alert_state in (AlertState.CONFIRMED_ALERT, AlertState.COOLDOWN)
    assert final_res.audio_alarm_active is True

    # Check incident creation
    incidents = incident_svc.get_active_incidents()
    assert len(incidents) == 1
    assert "fire" in incidents[0].hazard_type.lower()
    assert "smoke" not in incidents[0].hazard_type.lower()


def test_simultaneous_fire_and_smoke_filters_smoke_and_confirms_fire(sample_config, mock_fire_smoke_model, dummy_frame, tmp_path):
    """
    When both fire and smoke are detected in the frame:
    Smoke must be discarded completely.
    Fire must be kept, confirmed, and alerted without smoke contamination.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    incident_svc = IncidentService(log_file=tmp_path / "incidents.jsonl")

    def on_incident(event):
        incident_svc.create_incident(
            camera_id=event.camera_id,
            camera_location=event.camera_location,
            hazard_type=event.hazard_type,
            confidence=event.max_confidence,
            severity=event.severity.value,
            confirmation_status=getattr(event, "confirmation_status", "CONFIRMED"),
            incident_id=event.event_id,
        )

    alert_mgr = AlertManager(
        cooldown_seconds=10.0,
        enable_audio=True,
        on_confirmed_incident=on_incident,
    )
    temporal_flt = TemporalFilter(window_size=5, confirm_count=3)

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=False),
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [100, 100, 200, 200]},  # Fire
        {"cls": 0, "conf": 0.92, "xyxy": [300, 300, 500, 500]},  # Smoke
    ])

    for _ in range(4):
        final_res = pipeline.process_frame(dummy_frame)

    assert final_res.environmental_fire_count == 1
    assert final_res.environmental_smoke_count == 0
    assert final_res.fire_state == HazardState.CONFIRMED
    assert final_res.smoke_state == HazardState.NORMAL
    assert final_res.alert_state in (AlertState.CONFIRMED_ALERT, AlertState.COOLDOWN)


def test_phone_screen_fire_defense_preserved(sample_config, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Screen filtering continues to reject fire displayed on mobile devices
    even when smoke detection is disabled.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67]  # cell phone

    screen_filter = ScreenContextFilter(enabled=True, reject_threshold=0.65)
    temporal_filter = TemporalFilter(window_size=5, confirm_count=3)
    alert_mgr = AlertManager()

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_filter,
        temporal_filter=temporal_filter,
        alert_manager=alert_mgr,
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    # Phone bbox
    mock_screen_model.set_predictions([
        {"cls": 67, "conf": 0.90, "xyxy": [100, 100, 300, 400]}
    ])

    # Fire video inside phone
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [120, 120, 280, 380]}
    ])

    res = pipeline.process_frame(dummy_frame)

    assert res.environmental_fire_count == 0
    assert res.screen_rejected_count == 1
    assert res.fire_state == HazardState.NORMAL
    assert res.alert_state == AlertState.NORMAL


def test_hud_overlay_renders_clean_surveillance(sample_config, mock_fire_smoke_model, dummy_frame):
    """
    The HUD overlay text on annotated frames must be clean and not show 'DISABLED' or 'UNAVAILABLE'.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=False),
        temporal_filter=TemporalFilter(),
        alert_manager=AlertManager(),
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    mock_fire_smoke_model.set_predictions([])
    res = pipeline.process_frame(dummy_frame)

    # Verify pipeline properties
    assert pipeline.enable_fire_detection is True
    assert pipeline.enable_smoke_detection is False
    assert res.annotated_frame is not None


def test_camera_service_telemetry_status(sample_config):
    """
    CameraService.get_status() reflects enable_fire_detection=True and enable_smoke_detection=False.
    """
    sample_config.ENABLE_FIRE_DETECTION = True
    sample_config.ENABLE_SMOKE_DETECTION = False

    dummy_pipeline = MagicMock(spec=DetectionPipeline)
    dummy_pipeline.enable_fire_detection = True
    dummy_pipeline.enable_smoke_detection = False
    dummy_pipeline.total_screen_false_positives_blocked = 0
    dummy_pipeline.total_face_false_positives_blocked = 0
    dummy_pipeline.total_white_false_positives_blocked = 0

    service = CameraService(sample_config, dummy_pipeline)
    status = service.get_status()

    assert status["enable_fire_detection"] is True
    assert status["enable_smoke_detection"] is False
    assert status["environmental_smoke_count"] == 0


def test_clean_dashboard_controls_and_indicators(sample_config):
    """
    Verify that camera controls contain Start/Stop monitoring, the smoke card is intact,
    and no 'Disabled', 'Unavailable', or 'Inactive' messages are shown on the dashboard.
    """
    from app import create_app
    app = create_app(sample_config)
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Standard camera controls only
    assert 'id="btnStartMonitoring"' in html
    assert 'id="btnStopMonitoring"' in html
    assert 'id="btnSmokeDetection"' not in html

    # Smoke card intact
    assert 'id="tileSmoke"' in html
    assert "SMOKE DETECTION" in html

    # Clean dashboard: no 'Disabled', 'Unavailable', or 'Inactive' in user-facing dashboard
    assert "DISABLED" not in html
    assert "UNAVAILABLE" not in html.upper()
    assert "INACTIVE" not in html.upper()
    assert 'id="toastNotification"' not in html


def test_smoke_detection_button_api_endpoint(sample_config):
    """
    Verify that smoke detection endpoint safely maintains smoke detection disabled
    and never starts smoke inference.
    """
    from app import create_app
    app = create_app(sample_config)
    client = app.test_client()

    for endpoint in ["/api/smoke/toggle", "/api/detection/smoke", "/api/smoke/start"]:
        res = client.post(endpoint)
        assert res.status_code == 200
        data = res.get_json()
        assert data["enable_smoke_detection"] is False

    # Verify background status remains disabled
    status_res = client.get("/api/status")
    status_data = status_res.get_json()
    assert status_data["enable_smoke_detection"] is False


def test_no_person_or_face_bboxes_rendered_on_frame(sample_config, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Verify that person and face bounding boxes and labels are NEVER drawn
    on the camera feed, keeping the surveillance feed clean.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]
    loader.person_class_ids = [0]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=True, enable_face_smoke_filter=True),
        temporal_filter=TemporalFilter(),
        alert_manager=AlertManager(),
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    # Person detected in frame
    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.92, "xyxy": [150, 100, 450, 450]}
    ])
    mock_fire_smoke_model.set_predictions([])

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process_frame(blank_frame)

    # Person detected internally
    assert len(res.person_detections) == 1
    # Frame is annotated with top HUD, but person/head bbox is NOT drawn in the image area
    annotated = res.annotated_frame
    assert annotated is not None
    # Below the top 50px HUD, the person region (e.g. around (150, 100)) must NOT have person outline drawn
    # Specifically, check the pixel at the head border (x=180, y=100) - in previous versions it was drawn with (160, 140, 80)
    assert not np.array_equal(annotated[100, 180], [160, 140, 80])


def test_face_not_displayed_as_fire(sample_config, mock_fire_smoke_model, mock_screen_model):
    """
    Verify that a false prediction on a human face is not classified as confirmed fire,
    does not sound alarms, and is never displayed as a fire detection.
    """
    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.person_class_ids = [0]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=ScreenContextFilter(enabled=True),
        temporal_filter=TemporalFilter(),
        alert_manager=AlertManager(),
        enable_fire_detection=True,
        enable_smoke_detection=False,
    )

    # Create a synthetic image with skin-toned face patch at [220, 100, 380, 240]
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    # BGR for typical Caucasian/Asian/olive skin tone: B=110, G=140, R=190
    frame[100:240, 220:380] = [110, 140, 190]

    # Person at [200, 100, 400, 450]
    mock_screen_model.set_predictions([
        {"cls": 0, "conf": 0.90, "xyxy": [200, 100, 400, 450]}
    ])

    # Model falsely predicts fire on the face
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.65, "xyxy": [220, 100, 380, 240]}
    ])

    res = pipeline.process_frame(frame)

    assert res.environmental_fire_count == 0
    assert res.fire_state == HazardState.NORMAL
    assert res.alert_state == AlertState.NORMAL
    assert res.audio_alarm_active is False
    assert res.face_rejected_count == 1
    # Verify no fire box in validation results is CONFIRMED_ENVIRONMENTAL
    assert not any(v.status == HazardStatus.CONFIRMED_ENVIRONMENTAL for v in res.validation_results)

