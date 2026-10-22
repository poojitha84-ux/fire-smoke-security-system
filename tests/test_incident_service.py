"""
Unit tests for Incident Management Service.
"""

from pathlib import Path
import pytest

from services.incident_service import IncidentService, IncidentStatus, NotificationDeliveryStatus


def test_create_and_retrieve_incident(tmp_path):
    log_file = tmp_path / "incidents.jsonl"
    service = IncidentService(log_file=log_file)

    inc = service.create_incident(
        camera_id="CAM-FOYER-01",
        camera_location="Foyer Entrance",
        hazard_type="fire",
        confidence=0.88,
        severity="HIGH",
        confirmation_status="CONFIRMED",
    )

    assert inc.incident_id.startswith("INC-")
    assert inc.hazard_type == "fire"
    assert inc.severity == "HIGH"
    assert inc.current_status == IncidentStatus.CONFIRMED
    assert inc.notification_status == NotificationDeliveryStatus.NOT_CONFIGURED

    # Retrieve by ID
    retrieved = service.get_incident(inc.incident_id)
    assert retrieved is not None
    assert retrieved.camera_location == "Foyer Entrance"


def test_incident_acknowledgement_and_resolution(tmp_path):
    log_file = tmp_path / "incidents.jsonl"
    service = IncidentService(log_file=log_file)

    inc = service.create_incident(
        camera_id="CAM-02",
        camera_location="Server Room",
        hazard_type="smoke",
        confidence=0.74,
        severity="MEDIUM",
    )

    # Acknowledge
    ack = service.acknowledge_incident(inc.incident_id, operator_name="Officer Smith", notes="Investigating smoke sensor")
    assert ack is not None
    assert ack.current_status == IncidentStatus.ACKNOWLEDGED
    assert ack.acknowledged_by == "Officer Smith"
    assert ack.acknowledged_at is not None

    # Resolve
    res = service.resolve_incident(inc.incident_id, operator_name="Officer Smith", notes="Burnt toaster isolated. All clear.")
    assert res is not None
    assert res.current_status == IncidentStatus.RESOLVED
    assert res.resolution_notes == "Burnt toaster isolated. All clear."

    # Active incidents list should now be empty
    active = service.get_active_incidents()
    assert len(active) == 0


def test_notification_delivery_update(tmp_path):
    log_file = tmp_path / "incidents.jsonl"
    service = IncidentService(log_file=log_file)

    inc = service.create_incident(
        camera_id="CAM-01",
        camera_location="East Corridor",
        hazard_type="fire",
        confidence=0.91,
        severity="HIGH",
    )

    service.update_notification_status(
        inc.incident_id,
        NotificationDeliveryStatus.DELIVERED,
        details="Delivered to security-lead@domain.com",
    )

    updated = service.get_incident(inc.incident_id)
    assert updated.notification_status == NotificationDeliveryStatus.DELIVERED
    assert "security-lead@domain.com" in updated.notification_details
