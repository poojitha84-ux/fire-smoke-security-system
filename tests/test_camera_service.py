"""
Unit tests for Camera Service and Hardware Lifecycle.
"""

from unittest.mock import MagicMock, patch
import pytest

from services.camera_service import CameraService, CameraStatus
from detector.inference import DetectionPipeline


@pytest.fixture
def dummy_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=DetectionPipeline)
    pipeline.total_screen_false_positives_blocked = 0
    return pipeline


def test_camera_stopped_by_default(sample_config, dummy_pipeline):
    service = CameraService(sample_config, dummy_pipeline)
    assert service.status == CameraStatus.STOPPED
    jpeg = service.get_jpeg_frame()
    # JPEG magic bytes: FF D8
    assert jpeg.startswith(b"\xff\xd8")


def test_camera_start_stop_resource_cleanup(sample_config, dummy_pipeline):
    service = CameraService(sample_config, dummy_pipeline)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np_dummy := MagicMock())
    np_dummy.size = 100

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # Start
        service.start_monitoring(source=0)
        assert service.status in (CameraStatus.CONNECTING, CameraStatus.CONNECTED)

        # Stop
        service.stop_monitoring()
        assert service.status == CameraStatus.STOPPED
        # Verify release was called
        mock_cap.release.assert_called()


def test_camera_unavailable_error_handling(sample_config, dummy_pipeline):
    sample_config.CAMERA_SOURCE = "rtsp://invalid-ip-camera:554/live"
    service = CameraService(sample_config, dummy_pipeline)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False  # Failed to open!

    with patch("cv2.VideoCapture", return_value=mock_cap):
        service._open_camera()
        assert service.status == CameraStatus.ERROR
        assert "Failed opening camera source" in service.status_message


def test_camera_disconnection_state(sample_config, dummy_pipeline):
    service = CameraService(sample_config, dummy_pipeline)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)  # Stream disconnected!

    service._cap = mock_cap
    service.consecutive_read_failures = 10
    service.status = CameraStatus.DISCONNECTED
    service.status_message = "Camera stream disconnected"

    status = service.get_status()
    assert status["status"] == "DISCONNECTED"
    assert status["is_monitoring"] is False
