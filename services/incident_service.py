"""
Security Incident Management Service.

Maintains life-cycle of confirmed and unconfirmed visual hazard incidents:
- Unique incident ID generation
- Tracking camera ID, location, hazard type, confidence, severity
- Security operator acknowledgement and resolution workflow
- Notification delivery status tracking
- Thread-safe persistence and logging
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


class IncidentStatus(str, Enum):
    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class NotificationDeliveryStatus(str, Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


@dataclass
class IncidentReport:
    """Detailed record of a visual fire/smoke hazard incident."""
    incident_id: str
    timestamp: str
    camera_id: str
    camera_location: str
    hazard_type: str                         # "fire", "smoke", "fire_and_smoke"
    confidence: float
    severity: str                            # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    confirmation_status: str                 # "CONFIRMED" or "UNCONFIRMED"
    current_status: IncidentStatus = IncidentStatus.CONFIRMED
    notification_status: NotificationDeliveryStatus = NotificationDeliveryStatus.NOT_CONFIGURED
    notification_details: str = "External notifications unconfigured"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["current_status"] = self.current_status.value
        d["notification_status"] = self.notification_status.value
        return d


class IncidentService:
    """Thread-safe incident manager and audit logger."""

    def __init__(self, log_file: Optional[Path] = None):
        self.lock = threading.Lock()
        self.incidents: Dict[str, IncidentReport] = {}
        self.log_file = log_file

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing_log()

    def create_incident(
        self,
        camera_id: str,
        camera_location: str,
        hazard_type: str,
        confidence: float,
        severity: str,
        confirmation_status: str = "CONFIRMED",
        metadata: Optional[Dict[str, Any]] = None,
        incident_id: Optional[str] = None,
    ) -> IncidentReport:
        """Create a new security incident report."""
        with self.lock:
            ts = datetime.now(timezone.utc)
            inc_id = incident_id or f"INC-{ts.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

            incident = IncidentReport(
                incident_id=inc_id,
                timestamp=ts.isoformat(),
                camera_id=camera_id,
                camera_location=camera_location,
                hazard_type=hazard_type,
                confidence=round(confidence, 4),
                severity=severity,
                confirmation_status=confirmation_status,
                current_status=IncidentStatus.CONFIRMED if confirmation_status == "CONFIRMED" else IncidentStatus.UNCONFIRMED,
                notification_status=NotificationDeliveryStatus.NOT_CONFIGURED,
                metadata=metadata or {},
            )

            self.incidents[inc_id] = incident
            self._append_to_log(incident)
            logger.info(f"New incident recorded: {inc_id} ({hazard_type.upper()} at {camera_location})")
            return incident

    def acknowledge_incident(
        self,
        incident_id: str,
        operator_name: str = "Security Operator",
        notes: str = "",
    ) -> Optional[IncidentReport]:
        """Mark incident as acknowledged by security staff."""
        with self.lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return None
            incident.current_status = IncidentStatus.ACKNOWLEDGED
            incident.acknowledged_by = operator_name
            incident.acknowledged_at = datetime.now(timezone.utc).isoformat()
            if notes:
                incident.metadata["ack_notes"] = notes
            self._append_to_log(incident)
            logger.info(f"Incident {incident_id} acknowledged by {operator_name}")
            return incident

    def resolve_incident(
        self,
        incident_id: str,
        operator_name: str = "Security Operator",
        notes: str = "Hazard extinguished / verified secure",
    ) -> Optional[IncidentReport]:
        """Mark incident as resolved."""
        with self.lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return None
            incident.current_status = IncidentStatus.RESOLVED
            incident.resolved_by = operator_name
            incident.resolved_at = datetime.now(timezone.utc).isoformat()
            incident.resolution_notes = notes
            self._append_to_log(incident)
            logger.info(f"Incident {incident_id} resolved by {operator_name}: {notes}")
            return incident

    def update_notification_status(
        self,
        incident_id: str,
        status: NotificationDeliveryStatus,
        details: str = "",
    ) -> Optional[IncidentReport]:
        """Update external notification delivery status."""
        with self.lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return None
            incident.notification_status = status
            if details:
                incident.notification_details = details
            self._append_to_log(incident)
            return incident

    def get_incident(self, incident_id: str) -> Optional[IncidentReport]:
        with self.lock:
            return self.incidents.get(incident_id)

    def get_all_incidents(self, limit: int = 100) -> List[IncidentReport]:
        with self.lock:
            # Sorted by timestamp descending
            sorted_list = sorted(
                self.incidents.values(),
                key=lambda x: x.timestamp,
                reverse=True,
            )
            return sorted_list[:limit]

    def get_active_incidents(self) -> List[IncidentReport]:
        """Returns incidents that are not yet RESOLVED."""
        with self.lock:
            return [
                inc for inc in self.incidents.values()
                if inc.current_status in (IncidentStatus.CONFIRMED, IncidentStatus.ACKNOWLEDGED, IncidentStatus.UNCONFIRMED)
            ]

    def _append_to_log(self, incident: IncidentReport):
        if not self.log_file:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(incident.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed appending incident to file {self.log_file}: {e}")

    def _load_existing_log(self):
        if not self.log_file or not self.log_file.exists():
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        inc = IncidentReport(
                            incident_id=data["incident_id"],
                            timestamp=data["timestamp"],
                            camera_id=data["camera_id"],
                            camera_location=data["camera_location"],
                            hazard_type=data["hazard_type"],
                            confidence=data["confidence"],
                            severity=data["severity"],
                            confirmation_status=data["confirmation_status"],
                            current_status=IncidentStatus(data.get("current_status", "CONFIRMED")),
                            notification_status=NotificationDeliveryStatus(data.get("notification_status", "NOT_CONFIGURED")),
                            notification_details=data.get("notification_details", ""),
                            acknowledged_by=data.get("acknowledged_by"),
                            acknowledged_at=data.get("acknowledged_at"),
                            resolved_by=data.get("resolved_by"),
                            resolved_at=data.get("resolved_at"),
                            resolution_notes=data.get("resolution_notes"),
                            metadata=data.get("metadata", {}),
                        )
                        self.incidents[inc.incident_id] = inc
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Error reading existing incident log: {e}")
