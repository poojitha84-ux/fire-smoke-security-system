"""
Unit tests for Alert Manager and State Machine.
"""

import time
import pytest

from detector.alert_manager import AlertManager, AlertState, AlertSeverity, AlertEvent
from detector.temporal_filter import HazardState


def test_severity_and_event_generation():
    events = []

    def on_incident(event: AlertEvent):
        events.append(event)

    mgr = AlertManager(
        camera_id="CAM-TEST",
        camera_location="Room 101",
        cooldown_seconds=10.0,
        enable_audio=True,
        on_confirmed_incident=on_incident,
    )

    # 1. Fire confirmed alone -> HIGH severity
    state = mgr.evaluate_states(
        fire_state=HazardState.CONFIRMED,
        smoke_state=HazardState.NORMAL,
        fire_conf=0.88,
        smoke_conf=0.0,
    )
    assert state == AlertState.CONFIRMED_ALERT
    assert len(events) == 1
    assert events[0].hazard_type == "fire"
    assert events[0].severity == AlertSeverity.HIGH
    assert events[0].camera_id == "CAM-TEST"
    assert mgr.audio_trigger_active is True


def test_dual_fire_and_smoke_critical_severity():
    events = []
    mgr = AlertManager(
        camera_id="CAM-TEST",
        camera_location="Basement",
        cooldown_seconds=10.0,
        on_confirmed_incident=lambda e: events.append(e),
    )

    mgr.evaluate_states(
        fire_state=HazardState.CONFIRMED,
        smoke_state=HazardState.CONFIRMED,
        fire_conf=0.9,
        smoke_conf=0.85,
    )
    assert len(events) == 1
    assert events[0].hazard_type == "fire_and_smoke"
    assert events[0].severity == AlertSeverity.CRITICAL


def test_alert_cooldown_suppresses_duplicate_notifications():
    """Duplicate confirmed frames must enter cooldown without re-alerting."""
    events = []
    mgr = AlertManager(
        camera_id="CAM-TEST",
        camera_location="Hallway",
        cooldown_seconds=5.0,
        on_confirmed_incident=lambda e: events.append(e),
    )

    # First trigger
    mgr.evaluate_states(fire_state=HazardState.CONFIRMED, smoke_state=HazardState.NORMAL, fire_conf=0.85)
    assert len(events) == 1

    # Immediate second frame: enters cooldown, NO new event emitted!
    state2 = mgr.evaluate_states(fire_state=HazardState.CONFIRMED, smoke_state=HazardState.NORMAL, fire_conf=0.85)
    assert state2 in (AlertState.CONFIRMED_ALERT, AlertState.COOLDOWN)
    assert len(events) == 1  # Suppressed!


def test_silence_audio_alarm():
    mgr = AlertManager(camera_id="CAM-TEST", camera_location="Lobby", enable_audio=True)

    mgr.evaluate_states(fire_state=HazardState.CONFIRMED, smoke_state=HazardState.NORMAL, fire_conf=0.85)
    assert mgr.audio_trigger_active is True

    # Operator silences
    mgr.silence_audio()
    assert mgr.audio_silenced is True
    assert mgr.audio_trigger_active is False
