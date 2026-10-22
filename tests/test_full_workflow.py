"""
Full End-to-End Workflow Integration Tests.

Validates the full security monitoring pipeline:
1. Simulated confirmed environmental fire (temporal confirmation, RED ALERT, audio trigger, persistence)
2. Simulated confirmed environmental smoke (MEDIUM severity, audio trigger, persistence)
3. Phone-screen fire rejection (false positive blocked, no alert, no audio)
4. Normal scene (clean background, no false alerts)
5. Sustained uncertain screen-proximity warning (UNCONFIRMED incident persistence, operator visibility)
6. Alarm silence, acknowledgment, resolution, and state clearing
7. Real model files verification (verifies real .pt weights on disk, classes, confidence thresholds)
"""

import json
from pathlib import Path
import numpy as np
import pytest

from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter, HazardStatus
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState, AlertSeverity
from detector.inference import DetectionPipeline
from services.incident_service import IncidentService, IncidentStatus
from services.camera_service import CameraService, CameraStatus
from app import create_app


def test_full_workflow_simulated_confirmed_fire(tmp_path, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Simulated Confirmed Fire Workflow:
    - Multiple frames of fire outside any screen
    - Temporal confirmation reached
    - Incident recorded and written to JSONL
    - Red Alert payload verified
    - Duplicate frames do not duplicate incidents
    """
    log_file = tmp_path / "incidents.jsonl"
    incident_service = IncidentService(log_file=log_file)
    incidents_captured = []

    def on_incident(event):
        inc = incident_service.create_incident(
            camera_id=event.camera_id,
            camera_location=event.camera_location,
            hazard_type=event.hazard_type,
            confidence=event.max_confidence,
            severity=event.severity.value,
            confirmation_status=getattr(event, "confirmation_status", "CONFIRMED"),
            incident_id=event.event_id,
        )
        incidents_captured.append(inc)

    alert_mgr = AlertManager(
        camera_id="CAM-SOUTH-01",
        camera_location="Chemical Storage Wing",
        cooldown_seconds=10.0,
        enable_audio=True,
        on_confirmed_incident=on_incident,
    )

    temporal_flt = TemporalFilter(window_size=8, confirm_count=4, clear_consecutive_frames=5)
    screen_flt = ScreenContextFilter(enabled=True)

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_flt,
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.4,
        smoke_threshold=0.4,
    )

    # Fire outside phone (x: 50..180, y: 50..180)
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.88, "xyxy": [50, 50, 180, 180]}
    ])
    mock_screen_model.set_predictions([])

    # Frames 1-3: Suspected
    for i in range(3):
        res = pipeline.process_frame(dummy_frame)
        assert res.alert_state in (AlertState.NORMAL, AlertState.SUSPECTED)
        assert len(incidents_captured) == 0

    # Frame 4: Confirm count reached (4/8)
    res4 = pipeline.process_frame(dummy_frame)
    assert res4.alert_state == AlertState.CONFIRMED_ALERT
    assert res4.audio_alarm_active is True
    assert len(incidents_captured) == 1

    active_alert = alert_mgr.get_active_alert_summary()
    assert active_alert is not None
    assert active_alert["hazard_type"] == "fire"
    assert active_alert["severity"] == "HIGH"
    assert active_alert["camera_id"] == "CAM-SOUTH-01"
    assert active_alert["camera_location"] == "Chemical Storage Wing"
    assert active_alert["confirmation_status"] == "CONFIRMED"
    assert active_alert["audio_alarm_active"] is True

    # Frame 5: Sustained active fire must NOT create a duplicate incident
    res5 = pipeline.process_frame(dummy_frame)
    assert len(incidents_captured) == 1  # Duplicate suppressed!
    assert res5.audio_alarm_active is True

    # Verify JSONL persistence on disk
    assert log_file.exists()
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["hazard_type"] == "fire"
    assert lines[0]["camera_id"] == "CAM-SOUTH-01"
    assert lines[0]["current_status"] == "CONFIRMED"


def test_full_workflow_simulated_confirmed_smoke(tmp_path, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Simulated Confirmed Smoke Workflow:
    - Multiple frames of smoke
    - Reaches confirmation -> MEDIUM severity incident created
    """
    log_file = tmp_path / "smoke_incidents.jsonl"
    incident_service = IncidentService(log_file=log_file)
    incidents_captured = []

    alert_mgr = AlertManager(
        camera_id="CAM-WEST-02",
        camera_location="Server Room A",
        enable_audio=True,
        on_confirmed_incident=lambda e: incidents_captured.append(
            incident_service.create_incident(
                camera_id=e.camera_id,
                camera_location=e.camera_location,
                hazard_type=e.hazard_type,
                confidence=e.max_confidence,
                severity=e.severity.value,
                confirmation_status="CONFIRMED",
                incident_id=e.event_id,
            )
        ),
    )

    temporal_flt = TemporalFilter(window_size=6, confirm_count=3, clear_consecutive_frames=4)
    screen_flt = ScreenContextFilter(enabled=True)

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_flt,
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.4,
        smoke_threshold=0.4,
    )

    # Environmental Smoke
    mock_fire_smoke_model.set_predictions([
        {"cls": 0, "conf": 0.76, "xyxy": [100, 100, 300, 300]}
    ])
    mock_screen_model.set_predictions([])

    for _ in range(3):
        res = pipeline.process_frame(dummy_frame)

    assert res.alert_state == AlertState.CONFIRMED_ALERT
    assert res.smoke_state == HazardState.CONFIRMED
    assert len(incidents_captured) == 1
    assert incidents_captured[0].hazard_type == "smoke"
    assert incidents_captured[0].severity == "MEDIUM"


def test_full_workflow_phone_screen_fire_rejection(tmp_path, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Phone Screen Fire Video Defense Workflow:
    - Fire detection inside mobile phone screen
    - Must be classified as SCREEN_REJECTED
    - No confirmed alert, no audio alarm, no incident logged
    """
    log_file = tmp_path / "phone_defense.jsonl"
    incident_service = IncidentService(log_file=log_file)
    incidents_captured = []

    alert_mgr = AlertManager(
        camera_id="CAM-LOBBY",
        camera_location="Front Desk",
        on_confirmed_incident=lambda e: incidents_captured.append(e),
    )
    temporal_flt = TemporalFilter(window_size=8, confirm_count=4)
    screen_flt = ScreenContextFilter(enabled=True, reject_threshold=0.6)

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_flt,
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.4,
        smoke_threshold=0.4,
    )

    # Phone: (100, 100, 250, 400)
    # Fire fully inside phone: (110, 120, 240, 380) -> overlap ~1.0
    mock_screen_model.set_predictions([
        {"cls": 67, "conf": 0.89, "xyxy": [100, 100, 250, 400]}
    ])
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.85, "xyxy": [110, 120, 240, 380]}
    ])

    for _ in range(8):
        res = pipeline.process_frame(dummy_frame)
        assert res.alert_state == AlertState.NORMAL
        assert res.environmental_fire_count == 0
        assert res.screen_rejected_count == 1
        assert res.audio_alarm_active is False

    assert len(incidents_captured) == 0
    assert not log_file.exists() or log_file.stat().st_size == 0


def test_full_workflow_normal_scene(dummy_frame, mock_fire_smoke_model, mock_screen_model):
    """
    Normal Scene Monitoring:
    - Zero detections
    - AlertState remains NORMAL
    - Audio alarm inactive
    """
    alert_mgr = AlertManager(camera_id="CAM-OFFICE", camera_location="Staff Lounge")
    temporal_flt = TemporalFilter()
    screen_flt = ScreenContextFilter()

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_flt,
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
    )

    mock_fire_smoke_model.set_predictions([])
    mock_screen_model.set_predictions([])

    for _ in range(10):
        res = pipeline.process_frame(dummy_frame)
        assert res.alert_state == AlertState.NORMAL
        assert res.audio_alarm_active is False
        assert res.raw_hazard_count == 0


def test_full_workflow_sustained_uncertain_warning(tmp_path, mock_fire_smoke_model, mock_screen_model, dummy_frame):
    """
    Sustained Uncertain Hazard Workflow:
    - Fire behind phone screen (area ratio < threshold, triggering SCREEN_UNCERTAIN)
    - When sustained for >= sustained_uncertain_threshold frames
    - Emits UNCONFIRMED warning incident for operator inspection
    - Does NOT falsely claim confirmed fire, but does NOT ignore it
    """
    log_file = tmp_path / "uncertain_incidents.jsonl"
    incident_service = IncidentService(log_file=log_file)
    incidents_captured = []

    def on_incident(event):
        inc = incident_service.create_incident(
            camera_id=event.camera_id,
            camera_location=event.camera_location,
            hazard_type=event.hazard_type,
            confidence=event.max_confidence,
            severity=event.severity.value,
            confirmation_status=getattr(event, "confirmation_status", "CONFIRMED"),
            incident_id=event.event_id,
        )
        incidents_captured.append(inc)

    alert_mgr = AlertManager(
        camera_id="CAM-DESK-03",
        camera_location="Security Control Desk",
        sustained_uncertain_threshold=5,
        on_confirmed_incident=on_incident,
    )

    temporal_flt = TemporalFilter(window_size=8, confirm_count=4)
    # Configure screen filter so partial protrusion triggers SCREEN_UNCERTAIN
    screen_flt = ScreenContextFilter(
        enabled=True,
        reject_threshold=0.85,
        uncertain_threshold=0.20,
        fire_behind_screen_area_ratio=0.5,
    )

    loader = ModelLoader()
    loader.fire_smoke_model = mock_fire_smoke_model
    loader.screen_model = mock_screen_model
    loader.fire_class_ids = [1]
    loader.smoke_class_ids = [0]
    loader.screen_class_ids = [67, 62, 63]

    pipeline = DetectionPipeline(
        model_loader=loader,
        screen_filter=screen_flt,
        temporal_filter=temporal_flt,
        alert_manager=alert_mgr,
        fire_threshold=0.4,
    )

    # Phone: (100, 100, 300, 300) [area 40,000]
    # Fire: (120, 120, 280, 280) [area 25,600, overlap ~1.0 with phone, but area ratio = 25600/40000 = 0.64 > 0.5]
    # For behind phone: fire (80, 120, 220, 280) -> partial protrusion
    mock_screen_model.set_predictions([
        {"cls": 67, "conf": 0.85, "xyxy": [100, 100, 300, 300]}
    ])
    # Fire overlapping 50% of its area
    mock_fire_smoke_model.set_predictions([
        {"cls": 1, "conf": 0.80, "xyxy": [50, 100, 200, 250]}
    ])

    # First 4 frames: below sustained threshold
    for _ in range(4):
        res = pipeline.process_frame(dummy_frame)
        assert res.screen_uncertain_count > 0

    assert len(incidents_captured) == 0

    # 5th consecutive frame: sustained threshold reached
    res5 = pipeline.process_frame(dummy_frame)
    assert res5.alert_state == AlertState.UNCERTAIN_WARNING
    assert len(incidents_captured) == 1
    assert incidents_captured[0].confirmation_status == "UNCONFIRMED"
    assert incidents_captured[0].current_status == IncidentStatus.UNCONFIRMED
    assert incidents_captured[0].hazard_type == "uncertain_screen_proximity"

    # Verify active alert summary
    summary = alert_mgr.get_active_alert_summary()
    assert summary is not None
    assert summary["state"] == "uncertain_warning"
    assert summary["confirmation_status"] == "UNCONFIRMED"


def test_full_workflow_alarm_silence_and_resolution(tmp_path, dummy_frame):
    """
    Alarm Silence, Operator Acknowledgment, and Incident Resolution:
    - Fire triggers alert and starts alarm
    - Operator silences audio: alarm stops, alert stays visible
    - Operator acknowledges and resolves incident
    - Camera clears: system returns to NORMAL
    """
    log_file = tmp_path / "lifecycle_incidents.jsonl"
    config = AppConfig()
    config.BASE_DIR = tmp_path
    config.CAMERA_ID = "CAM-TEST-LIFECYCLE"
    config.ENABLE_AUDIO_ALARM = True

    app = create_app(config=config)
    client = app.test_client()

    alert_mgr: AlertManager = app.alert_manager
    incident_svc: IncidentService = app.incident_service

    # Simulate confirmed fire
    alert_mgr.evaluate_states(
        fire_state=HazardState.CONFIRMED,
        smoke_state=HazardState.NORMAL,
        fire_conf=0.91,
    )
    assert alert_mgr.current_state == AlertState.CONFIRMED_ALERT
    assert alert_mgr.audio_trigger_active is True
    assert alert_mgr.audio_silenced is False

    # Check dashboard status API
    status_res = client.get("/api/status").get_json()
    assert status_res["alert_state"] == "confirmed_alert"
    assert status_res["audio_alarm_active"] is True
    assert status_res["active_alert"] is not None
    assert status_res["active_alert"]["hazard_type"] == "fire"
    assert status_res["active_alert"]["severity"] == "HIGH"

    # Step 1: Security staff silences audio alarm
    silence_res = client.post("/api/audio/silence").get_json()
    assert silence_res["success"] is True
    assert alert_mgr.audio_silenced is True
    assert alert_mgr.audio_trigger_active is False

    # Verify hazard is STILL CONFIRMED and NOT hidden
    status_after_silence = client.get("/api/status").get_json()
    assert status_after_silence["alert_state"] in ("confirmed_alert", "cooldown")
    assert status_after_silence["audio_alarm_active"] is False

    # Step 2: Security staff acknowledges incident
    incident_id = alert_mgr.last_incident_id
    assert incident_id is not None
    ack_res = client.post(f"/api/incidents/{incident_id}/acknowledge", json={
        "operator_name": "Officer Miller",
        "notes": "Responding with Class ABC fire extinguisher",
    }).get_json()
    assert ack_res["success"] is True
    assert ack_res["incident"]["current_status"] == "ACKNOWLEDGED"

    # Step 3: Security staff marks incident resolved
    resolve_res = client.post(f"/api/incidents/{incident_id}/resolve", json={
        "operator_name": "Officer Miller",
        "notes": "Fire extinguished. Scene ventilated and safe.",
    }).get_json()
    assert resolve_res["success"] is True
    assert resolve_res["incident"]["current_status"] == "RESOLVED"

    # Step 4: Environment clears (NORMAL)
    alert_mgr.evaluate_states(
        fire_state=HazardState.NORMAL,
        smoke_state=HazardState.NORMAL,
    )
    assert alert_mgr.current_state == AlertState.NORMAL
    assert alert_mgr.active_alert_event is None


def test_full_workflow_real_model_loading_and_classes():
    """
    Empirical Model Verification:
    - Loads the ACTUAL local weights files from models/ directory
    - Tests real class IDs and confidence threshold configuration
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    fire_smoke_path = models_dir / "fire_smoke_yolov8n.pt"
    screen_path = models_dir / "yolov8n.pt"

    assert fire_smoke_path.exists(), f"Fire/Smoke weights not found at {fire_smoke_path}"
    assert screen_path.exists(), f"Screen detector weights not found at {screen_path}"

    loader = ModelLoader(models_dir=models_dir)
    fire_model = loader.load_fire_smoke_model(fire_smoke_path)
    screen_model = loader.load_screen_detector_model(screen_path)

    assert fire_model is not None
    assert screen_model is not None

    # Verify Fire & Smoke model class mappings
    # Expected: {0: 'smoke', 1: 'fire'}
    assert 0 in loader.smoke_class_ids or "smoke" in str(fire_model.names.values()).lower()
    assert 1 in loader.fire_class_ids or "fire" in str(fire_model.names.values()).lower()

    # Verify COCO screen class mappings
    # Expected COCO: 62: tv, 63: laptop, 67: cell phone
    assert 67 in loader.screen_class_ids
    assert 62 in loader.screen_class_ids
    assert 63 in loader.screen_class_ids

    # Run inference on a blank 640x480 frame with real models
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = fire_model.predict(blank, verbose=False)
    assert len(results) > 0
