"""
Unit tests for External Security Notification Service.
"""

from unittest.mock import MagicMock, patch
import pytest

from config import AppConfig
from services.incident_service import IncidentReport, IncidentStatus, NotificationDeliveryStatus
from services.notification_service import NotificationService, NotificationResult


@pytest.fixture
def sample_incident() -> IncidentReport:
    return IncidentReport(
        incident_id="INC-TEST-001",
        timestamp="2026-10-22T08:00:00Z",
        camera_id="CAM-01",
        camera_location="Auditorium",
        hazard_type="fire",
        confidence=0.89,
        severity="HIGH",
        confirmation_status="CONFIRMED",
    )


def test_notifications_disabled(sample_config, sample_incident):
    sample_config.ENABLE_NOTIFICATIONS = False
    service = NotificationService(sample_config)

    res = service.send_notification(sample_incident)
    assert res.success is False
    assert res.status == NotificationDeliveryStatus.NOT_CONFIGURED
    assert "disabled" in res.message.lower()


def test_unconfigured_notification_credentials(sample_config, sample_incident):
    sample_config.ENABLE_NOTIFICATIONS = True
    sample_config.NOTIFICATION_TYPE = "smtp"
    sample_config.SMTP_SERVER = ""  # Unconfigured!
    sample_config.ALERT_EMAIL_TO = ""

    service = NotificationService(sample_config)
    assert service.is_configured() is False

    res = service.send_notification(sample_incident)
    assert res.success is False
    assert res.status == NotificationDeliveryStatus.NOT_CONFIGURED
    assert "missing configuration" in res.message.lower()


def test_webhook_delivery_success(sample_config, sample_incident):
    sample_config.ENABLE_NOTIFICATIONS = True
    sample_config.NOTIFICATION_TYPE = "webhook"
    sample_config.WEBHOOK_URL = "https://security-gateway.example.com/webhook"

    service = NotificationService(sample_config)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = service.send_notification(sample_incident)
        assert res.success is True
        assert res.status == NotificationDeliveryStatus.DELIVERED
        assert "HTTP 200" in res.message


def test_webhook_delivery_failure(sample_config, sample_incident):
    sample_config.ENABLE_NOTIFICATIONS = True
    sample_config.NOTIFICATION_TYPE = "webhook"
    sample_config.WEBHOOK_URL = "https://invalid-gateway.example.com"
    sample_config.WEBHOOK_MAX_RETRIES = 1

    service = NotificationService(sample_config)

    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = service.send_notification(sample_incident)
        assert res.success is False
        assert res.status == NotificationDeliveryStatus.FAILED
        assert "Connection refused" in res.message


def test_smtp_delivery_success(sample_config, sample_incident):
    sample_config.ENABLE_NOTIFICATIONS = True
    sample_config.NOTIFICATION_TYPE = "smtp"
    sample_config.SMTP_SERVER = "smtp.security.local"
    sample_config.ALERT_EMAIL_TO = "soc@security.local"

    service = NotificationService(sample_config)

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp):
        res = service.send_notification(sample_incident)
        assert res.success is True
        assert res.status == NotificationDeliveryStatus.DELIVERED
        mock_smtp.send_message.assert_called_once()
