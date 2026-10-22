"""
Security External Notification Service.

Supports:
- SMTP email alerting with TLS/SSL
- Secure Webhook dispatch (Slack / MS Teams / Custom Security Gateway)
- Retry logic with backoff
- Clear status tracking (NOT_CONFIGURED, PENDING, DELIVERED, FAILED)
- Asynchronous dispatch without blocking video capture or inference loops
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import email.message
import json
import logging
import smtplib
import time
from typing import Callable, Optional
import urllib.request
import urllib.error

from config import AppConfig
from .incident_service import IncidentReport, NotificationDeliveryStatus

logger = logging.getLogger(__name__)


@dataclass
class NotificationResult:
    """Outcome of notification delivery attempt."""
    success: bool
    status: NotificationDeliveryStatus
    message: str
    timestamp: str
    retries_attempted: int = 0


class NotificationService:
    """Manages secure notification dispatch to external security endpoints."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="NotifyWorker")

    def is_configured(self) -> bool:
        """Check whether external notifications are configured."""
        if not self.config.ENABLE_NOTIFICATIONS:
            return False

        if self.config.NOTIFICATION_TYPE == "smtp":
            return bool(self.config.SMTP_SERVER and self.config.ALERT_EMAIL_TO)
        elif self.config.NOTIFICATION_TYPE == "webhook":
            return bool(self.config.WEBHOOK_URL)
        return False

    def send_notification(self, incident: IncidentReport) -> NotificationResult:
        """
        Synchronously attempt to deliver notification.
        """
        ts = datetime.now(timezone.utc).isoformat()

        if not self.config.ENABLE_NOTIFICATIONS:
            return NotificationResult(
                success=False,
                status=NotificationDeliveryStatus.NOT_CONFIGURED,
                message="Notifications disabled in system settings.",
                timestamp=ts,
            )

        if not self.is_configured():
            return NotificationResult(
                success=False,
                status=NotificationDeliveryStatus.NOT_CONFIGURED,
                message=f"Missing configuration for notification type: {self.config.NOTIFICATION_TYPE}",
                timestamp=ts,
            )

        if self.config.NOTIFICATION_TYPE == "smtp":
            return self._send_smtp(incident)
        elif self.config.NOTIFICATION_TYPE == "webhook":
            return self._send_webhook(incident)
        else:
            return NotificationResult(
                success=False,
                status=NotificationDeliveryStatus.FAILED,
                message=f"Unsupported notification type: {self.config.NOTIFICATION_TYPE}",
                timestamp=ts,
            )

    def dispatch_async(
        self,
        incident: IncidentReport,
        on_complete: Optional[Callable[[IncidentReport, NotificationResult], None]] = None,
    ):
        """
        Non-blocking dispatch running on thread pool.
        """
        def _task():
            res = self.send_notification(incident)
            if on_complete:
                try:
                    on_complete(incident, res)
                except Exception as e:
                    logger.error(f"Error in on_complete callback: {e}")

        self.executor.submit(_task)

    def _send_smtp(self, incident: IncidentReport) -> NotificationResult:
        ts = datetime.now(timezone.utc).isoformat()
        attempts = 0
        max_attempts = max(1, self.config.WEBHOOK_MAX_RETRIES)
        last_error = ""

        msg = email.message.EmailMessage()
        msg["Subject"] = f"[{incident.severity}] Security Alert: {incident.hazard_type.upper()} Detected"
        msg["From"] = self.config.ALERT_EMAIL_FROM or "security-alerts@local.monitor"
        msg["To"] = self.config.ALERT_EMAIL_TO

        body = (
            f"SECURITY INCIDENT ALERT\n"
            f"----------------------------------------\n"
            f"Incident ID   : {incident.incident_id}\n"
            f"Time (UTC)    : {incident.timestamp}\n"
            f"Camera        : {incident.camera_id} ({incident.camera_location})\n"
            f"Hazard Type   : {incident.hazard_type.upper()}\n"
            f"Confidence    : {incident.confidence * 100:.1f}%\n"
            f"Severity      : {incident.severity}\n"
            f"Status        : {incident.confirmation_status}\n"
            f"----------------------------------------\n"
            f"Please verify security dashboard immediately.\n"
        )
        msg.set_content(body)

        while attempts < max_attempts:
            attempts += 1
            try:
                server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT, timeout=8.0)
                if self.config.SMTP_USE_TLS:
                    server.starttls()
                if self.config.SMTP_USERNAME and self.config.SMTP_PASSWORD:
                    server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
                server.send_message(msg)
                server.quit()
                logger.info(f"SMTP notification delivered to {self.config.ALERT_EMAIL_TO} for {incident.incident_id}")
                return NotificationResult(
                    success=True,
                    status=NotificationDeliveryStatus.DELIVERED,
                    message=f"Delivered to {self.config.ALERT_EMAIL_TO}",
                    timestamp=ts,
                    retries_attempted=attempts - 1,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"SMTP attempt {attempts} failed: {e}")
                time.sleep(1.0)

        return NotificationResult(
            success=False,
            status=NotificationDeliveryStatus.FAILED,
            message=f"SMTP delivery failed after {attempts} attempts: {last_error}",
            timestamp=ts,
            retries_attempted=attempts,
        )

    def _send_webhook(self, incident: IncidentReport) -> NotificationResult:
        ts = datetime.now(timezone.utc).isoformat()
        attempts = 0
        max_attempts = max(1, self.config.WEBHOOK_MAX_RETRIES)
        last_error = ""

        payload = {
            "incident_id": incident.incident_id,
            "timestamp": incident.timestamp,
            "camera_id": incident.camera_id,
            "camera_location": incident.camera_location,
            "hazard_type": incident.hazard_type,
            "confidence": incident.confidence,
            "severity": incident.severity,
            "confirmation_status": incident.confirmation_status,
            "system": "Fire & Smoke Real-Time Security Monitor",
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.config.WEBHOOK_URL,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "FireSmokeSecurityMonitor/1.0"},
            method="POST",
        )

        while attempts < max_attempts:
            attempts += 1
            try:
                with urllib.request.urlopen(req, timeout=self.config.WEBHOOK_TIMEOUT_SECONDS) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(f"Webhook alert successfully delivered to {self.config.WEBHOOK_URL} for {incident.incident_id}")
                        return NotificationResult(
                            success=True,
                            status=NotificationDeliveryStatus.DELIVERED,
                            message=f"Webhook delivered (HTTP {resp.status})",
                            timestamp=ts,
                            retries_attempted=attempts - 1,
                        )
                    else:
                        last_error = f"HTTP {resp.status}"
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Webhook attempt {attempts} failed: {e}")
                time.sleep(1.0)

        return NotificationResult(
            success=False,
            status=NotificationDeliveryStatus.FAILED,
            message=f"Webhook failed after {attempts} attempts: {last_error}",
            timestamp=ts,
            retries_attempted=attempts,
        )

    def shutdown(self):
        self.executor.shutdown(wait=False)
