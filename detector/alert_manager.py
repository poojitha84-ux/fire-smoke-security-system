"""
Alert Manager and Hazard State Machine.

Coordinates state transitions between NORMAL, SUSPECTED, CONFIRMED_ALERT, and COOLDOWN.
Calculates severity, prevents alert spam/duplication, controls alarm audio state,
and generates structured incident event triggers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from .temporal_filter import HazardState

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertState(str, Enum):
    NORMAL = "normal"
    SUSPECTED = "suspected"
    CONFIRMED_ALERT = "confirmed_alert"
    UNCERTAIN_WARNING = "uncertain_warning"
    COOLDOWN = "cooldown"


@dataclass
class AlertEvent:
    """Structured event emitted when hazard is confirmed or state transitions."""
    event_id: str
    timestamp: str
    hazard_type: str                   # "fire", "smoke", "fire_and_smoke", or "uncertain_screen_proximity"
    severity: AlertSeverity
    max_confidence: float
    state: AlertState
    camera_id: str
    camera_location: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confirmation_status: str = "CONFIRMED"


class AlertManager:
    """
    Manages security alerting lifecycle, duplicate suppression, and cooldown logic.
    """

    def __init__(
        self,
        camera_id: str = "CAM-01",
        camera_location: str = "Main Building",
        cooldown_seconds: float = 30.0,
        enable_audio: bool = True,
        audio_cooldown_seconds: float = 10.0,
        sustained_uncertain_threshold: int = 5,
        on_confirmed_incident: Optional[Callable[[AlertEvent], None]] = None,
    ):
        self.camera_id = camera_id
        self.camera_location = camera_location
        self.cooldown_seconds = cooldown_seconds
        self.enable_audio = enable_audio
        self.audio_cooldown_seconds = audio_cooldown_seconds
        self.sustained_uncertain_threshold = sustained_uncertain_threshold
        self.on_confirmed_incident = on_confirmed_incident

        self.current_state: AlertState = AlertState.NORMAL
        self.last_alert_time: float = 0.0
        self.last_audio_alarm_time: float = 0.0
        self.audio_silenced: bool = False
        self.audio_trigger_active: bool = False

        self.consecutive_uncertain_frames: int = 0
        self.uncertain_warning_emitted: bool = False
        self.active_alert_event: Optional[AlertEvent] = None

        self.last_incident_id: Optional[str] = None
        self.event_log: List[Dict[str, Any]] = []

    def evaluate_states(
        self,
        fire_state: HazardState,
        smoke_state: HazardState,
        fire_conf: float = 0.0,
        smoke_conf: float = 0.0,
        uncertain_active: bool = False,
        screen_rejections_count: int = 0,
    ) -> AlertState:
        """
        Evaluate current hazard states and manage alert dispatch.
        """
        now = time.time()
        time_since_last_alert = now - self.last_alert_time

        is_fire_confirmed = fire_state == HazardState.CONFIRMED
        is_smoke_confirmed = smoke_state == HazardState.CONFIRMED
        is_hazard_confirmed = is_fire_confirmed or is_smoke_confirmed

        is_suspected = (fire_state == HazardState.SUSPECTED or smoke_state == HazardState.SUSPECTED)

        if is_hazard_confirmed:
            self.consecutive_uncertain_frames = 0
            self.uncertain_warning_emitted = False

            # Hazard is confirmed by temporal filter
            if self.current_state != AlertState.CONFIRMED_ALERT and self.current_state != AlertState.COOLDOWN:
                # Fresh confirmed incident!
                self._trigger_confirmed_alert(
                    is_fire_confirmed, is_smoke_confirmed, fire_conf, smoke_conf
                )
            else:
                # Still confirmed in subsequent frames: maintain audio alarm if not silenced
                if self.enable_audio and not self.audio_silenced:
                    self.audio_trigger_active = True

                if self.current_state == AlertState.COOLDOWN:
                    # In cooldown: if cooldown elapsed, transition back to active alert
                    if time_since_last_alert >= self.cooldown_seconds:
                        self.current_state = AlertState.CONFIRMED_ALERT
                        logger.info("Alert cooldown expired; hazard still actively confirmed.")
                elif self.current_state == AlertState.CONFIRMED_ALERT:
                    # Check if we should enter cooldown to prevent duplicate notifications
                    if time_since_last_alert < self.cooldown_seconds:
                        self.current_state = AlertState.COOLDOWN

        elif uncertain_active:
            # No confirmed environmental hazard, but uncertain detection active (e.g. fire/smoke near or behind screen)
            self.consecutive_uncertain_frames += 1
            if self.consecutive_uncertain_frames >= self.sustained_uncertain_threshold:
                if not self.uncertain_warning_emitted:
                    self._trigger_uncertain_warning()
                else:
                    self.current_state = AlertState.UNCERTAIN_WARNING
            elif self.current_state not in (AlertState.CONFIRMED_ALERT, AlertState.COOLDOWN):
                if is_suspected:
                    self.current_state = AlertState.SUSPECTED

        elif is_suspected:
            self.consecutive_uncertain_frames = 0
            self.uncertain_warning_emitted = False
            if self.current_state == AlertState.NORMAL:
                self.current_state = AlertState.SUSPECTED

        else:
            # Both states are NORMAL and no uncertain hazard active
            self.consecutive_uncertain_frames = 0
            self.uncertain_warning_emitted = False

            if self.current_state in (AlertState.CONFIRMED_ALERT, AlertState.COOLDOWN, AlertState.SUSPECTED, AlertState.UNCERTAIN_WARNING):
                self._record_state_change(
                    from_state=self.current_state,
                    to_state=AlertState.NORMAL,
                    reason="Environmental hazard cleared; normal conditions restored.",
                )
                self.current_state = AlertState.NORMAL
                self.audio_trigger_active = False
                self.audio_silenced = False
                self.active_alert_event = None

        return self.current_state

    def _trigger_confirmed_alert(
        self,
        is_fire: bool,
        is_smoke: bool,
        fire_conf: float,
        smoke_conf: float,
    ):
        now = time.time()
        self.last_alert_time = now
        self.current_state = AlertState.CONFIRMED_ALERT

        if is_fire and is_smoke:
            hazard_type = "fire_and_smoke"
            severity = AlertSeverity.CRITICAL
            conf = max(fire_conf, smoke_conf)
            desc = "CRITICAL HAZARD: Both active fire and smoke confirmed by visual monitoring."
        elif is_fire:
            hazard_type = "fire"
            severity = AlertSeverity.HIGH
            conf = fire_conf
            desc = "HIGH HAZARD: Genuine environmental fire confirmed by visual monitoring."
        else:
            hazard_type = "smoke"
            severity = AlertSeverity.MEDIUM
            conf = smoke_conf
            desc = "MEDIUM HAZARD: Environmental smoke accumulation confirmed by visual monitoring."

        event_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        self.last_incident_id = event_id

        # Trigger audio alarm if enabled and not silenced
        if self.enable_audio and not self.audio_silenced:
            if (now - self.last_audio_alarm_time) >= self.audio_cooldown_seconds:
                self.audio_trigger_active = True
                self.last_audio_alarm_time = now

        alert_event = AlertEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hazard_type=hazard_type,
            severity=severity,
            max_confidence=conf,
            state=AlertState.CONFIRMED_ALERT,
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            description=desc,
            metadata={"fire_conf": fire_conf, "smoke_conf": smoke_conf},
            confirmation_status="CONFIRMED",
        )
        self.active_alert_event = alert_event

        self._record_state_change(
            from_state=AlertState.NORMAL,
            to_state=AlertState.CONFIRMED_ALERT,
            reason=f"Confirmed {hazard_type} alert ({severity.value})",
        )

        logger.warning(f"SECURITY ALERT CONFIRMED: {alert_event}")

        if self.on_confirmed_incident:
            try:
                self.on_confirmed_incident(alert_event)
            except Exception as e:
                logger.error(f"Error executing on_confirmed_incident callback: {e}")

    def _trigger_uncertain_warning(self):
        """Trigger sustained warning for hazard in close screen proximity or potential occlusion."""
        now = time.time()
        self.last_alert_time = now
        self.current_state = AlertState.UNCERTAIN_WARNING
        self.uncertain_warning_emitted = True

        event_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        self.last_incident_id = event_id

        alert_event = AlertEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hazard_type="uncertain_screen_proximity",
            severity=AlertSeverity.MEDIUM,
            max_confidence=0.0,
            state=AlertState.UNCERTAIN_WARNING,
            camera_id=self.camera_id,
            camera_location=self.camera_location,
            description="SUSTAINED UNCERTAIN HAZARD: Fire/smoke detection sustained near or behind screen device. Visual verification recommended.",
            metadata={"consecutive_uncertain_frames": self.consecutive_uncertain_frames},
            confirmation_status="UNCONFIRMED",
        )
        self.active_alert_event = alert_event

        self._record_state_change(
            from_state=AlertState.NORMAL,
            to_state=AlertState.UNCERTAIN_WARNING,
            reason=f"Sustained uncertain hazard for {self.consecutive_uncertain_frames} frames",
        )

        logger.warning(f"SUSTAINED UNCERTAIN WARNING: {alert_event}")

        if self.on_confirmed_incident:
            try:
                self.on_confirmed_incident(alert_event)
            except Exception as e:
                logger.error(f"Error executing on_confirmed_incident callback for uncertain warning: {e}")

    def get_active_alert_summary(self) -> Optional[Dict[str, Any]]:
        """Returns active hazard details for prominent dashboard presentation."""
        if self.active_alert_event and self.current_state in (
            AlertState.CONFIRMED_ALERT,
            AlertState.COOLDOWN,
            AlertState.UNCERTAIN_WARNING,
        ):
            return {
                "event_id": self.active_alert_event.event_id,
                "timestamp": self.active_alert_event.timestamp,
                "hazard_type": self.active_alert_event.hazard_type,
                "severity": self.active_alert_event.severity.value,
                "max_confidence": round(self.active_alert_event.max_confidence, 4),
                "state": self.current_state.value,
                "camera_id": self.active_alert_event.camera_id,
                "camera_location": self.active_alert_event.camera_location,
                "description": self.active_alert_event.description,
                "confirmation_status": getattr(self.active_alert_event, "confirmation_status", "CONFIRMED"),
                "audio_alarm_active": self.audio_trigger_active,
                "audio_silenced": self.audio_silenced,
            }
        return None

    def _record_state_change(self, from_state: AlertState, to_state: AlertState, reason: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason,
        }
        self.event_log.append(entry)
        if len(self.event_log) > 200:
            self.event_log.pop(0)

    def silence_audio(self):
        """Security staff silence control."""
        self.audio_silenced = True
        self.audio_trigger_active = False
        logger.info("Audio alarm manually silenced by security operator.")

    def enable_audio_alarm(self, enabled: bool = True):
        self.enable_audio = enabled
        if not enabled:
            self.audio_trigger_active = False

    def reset(self):
        self.current_state = AlertState.NORMAL
        self.last_alert_time = 0.0
        self.audio_silenced = False
        self.audio_trigger_active = False
        self.last_incident_id = None
        self.consecutive_uncertain_frames = 0
        self.uncertain_warning_emitted = False
        self.active_alert_event = None
