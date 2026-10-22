"""
Camera Capture and Stream Service.

Controls hardware webcam and RTSP/IP CCTV camera lifecycle:
- Start/stop monitoring controls
- Threaded background capture and inference loop
- Automatic resource cleanup and camera release
- Camera disconnection detection and reconnection handling
- MJPEG streaming generator for Flask web dashboard
"""

from enum import Enum
import logging
import threading
import time
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from config import AppConfig
from detector.inference import DetectionPipeline, FrameResult

logger = logging.getLogger(__name__)


class CameraStatus(str, Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class CameraService:
    """Manages live camera feeds, inference execution, and resource cleanup."""

    def __init__(self, config: AppConfig, pipeline: DetectionPipeline):
        self.config = config
        self.pipeline = pipeline

        self.source: Union[int, str] = config.CAMERA_SOURCE
        self.status: CameraStatus = CameraStatus.STOPPED
        self.status_message: str = "Monitoring inactive"

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.latest_result: Optional[FrameResult] = None
        self.latest_frame_jpeg: Optional[bytes] = None
        self.last_frame_timestamp: float = 0.0
        self.consecutive_read_failures: int = 0
        self.connection_attempts: int = 0

    def start_monitoring(self, source: Optional[Union[int, str]] = None) -> bool:
        """Start live camera capture and monitoring loop."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning("Camera monitoring is already running.")
                return True

            if source is not None:
                self.source = source

            # Sync pipeline detection flags with active configuration
            if hasattr(self.pipeline, "enable_smoke_detection"):
                self.pipeline.enable_smoke_detection = getattr(self.config, "ENABLE_SMOKE_DETECTION", False)
            if hasattr(self.pipeline, "enable_fire_detection"):
                self.pipeline.enable_fire_detection = getattr(self.config, "ENABLE_FIRE_DETECTION", True)

            self._stop_event.clear()
            self.status = CameraStatus.CONNECTING
            self.status_message = f"Connecting to camera source: {self.source}"
            self.consecutive_read_failures = 0
            self.connection_attempts = 0

            self._thread = threading.Thread(
                target=self._capture_loop,
                name="CameraCaptureWorker",
                daemon=True,
            )
            self._thread.start()
            logger.info(f"Started camera monitoring thread for source: {self.source}")
            return True

    def stop_monitoring(self):
        """Stop camera capture and release hardware resources cleanly."""
        logger.info("Stopping camera monitoring and releasing resources...")
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            self._thread = None

        with self._lock:
            self._cleanup_camera()
            self.status = CameraStatus.STOPPED
            self.status_message = "Monitoring stopped by user"
            self.latest_frame_jpeg = None
            self.latest_result = None

        logger.info("Camera resources released successfully.")

    def _cleanup_camera(self):
        """Internal camera release."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            finally:
                self._cap = None

    def _open_camera(self) -> bool:
        """Attempt opening camera capture."""
        self._cleanup_camera()
        self.connection_attempts += 1
        logger.info(f"Attempting to open camera source {self.source} (attempt {self.connection_attempts})...")

        try:
            # If source is digit string, convert to int
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)

            # Choose backend: on Windows MSMF or DSHOW works well for webcams
            if isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(src)
            else:
                cap = cv2.VideoCapture(src)

            if cap.isOpened():
                if isinstance(src, int):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_FRAME_HEIGHT)
                self._cap = cap
                self.status = CameraStatus.CONNECTED
                self.status_message = f"Connected to camera {self.config.CAMERA_ID}"
                self.consecutive_read_failures = 0
                logger.info(f"Successfully opened camera stream: {self.source}")
                return True
            else:
                self.status = CameraStatus.ERROR
                self.status_message = f"Failed opening camera source: {self.source}"
                logger.warning(self.status_message)
                return False
        except Exception as e:
            self.status = CameraStatus.ERROR
            self.status_message = f"Exception opening camera {self.source}: {e}"
            logger.error(self.status_message)
            return False

    def _capture_loop(self):
        """Main camera acquisition and detection pipeline loop."""
        if not self._open_camera():
            # If initial open fails, wait briefly before entering retry loop
            time.sleep(1.0)

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self.status = CameraStatus.CONNECTING
                self.status_message = f"Reconnecting to camera {self.source}..."
                time.sleep(2.0)
                if not self._open_camera():
                    continue

            ret, frame = self._cap.read()

            if not ret or frame is None or frame.size == 0:
                self.consecutive_read_failures += 1
                if self.consecutive_read_failures > 5:
                    self.status = CameraStatus.DISCONNECTED
                    self.status_message = "Camera stream disconnected or no frames received"
                    logger.warning(self.status_message)
                    self._cleanup_camera()
                    time.sleep(1.5)
                else:
                    time.sleep(0.05)
                continue

            self.consecutive_read_failures = 0
            self.status = CameraStatus.CONNECTED
            self.status_message = f"Monitoring active on {self.config.CAMERA_ID}"
            self.last_frame_timestamp = time.time()

            # Execute full detection pipeline
            try:
                result = self.pipeline.process_frame(frame)
                ret_enc, jpeg = cv2.imencode(".jpg", result.annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret_enc:
                    with self._lock:
                        self.latest_result = result
                        self.latest_frame_jpeg = jpeg.tobytes()
            except Exception as e:
                logger.error(f"Error during frame processing: {e}")

            # Regulate capture rate to target FPS
            target_delay = 1.0 / max(1, self.config.CAMERA_FPS_TARGET)
            time.sleep(max(0.005, target_delay * 0.5))

        self._cleanup_camera()

    def get_jpeg_frame(self) -> bytes:
        """Returns the latest annotated JPEG frame or a status placeholder canvas."""
        with self._lock:
            if self.latest_frame_jpeg is not None and self.status == CameraStatus.CONNECTED:
                return self.latest_frame_jpeg

        # Generate status placeholder canvas
        w, h = 640, 360
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        placeholder[:] = (20, 24, 33)  # Dark slate background

        # Status text
        msg = f"CAMERA STATUS: {self.status.value}"
        sub = self.status_message

        cv2.putText(placeholder, msg, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
        cv2.putText(placeholder, sub, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if self.status == CameraStatus.STOPPED:
            cv2.putText(placeholder, "Click 'Start Monitoring' to initiate live feed.", (40, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1)

        ret, jpeg = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return jpeg.tobytes() if ret else b""

    def generate_mjpeg_stream(self):
        """Generator yielding multipart MJPEG frames for HTTP stream response."""
        while True:
            frame_bytes = self.get_jpeg_frame()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)  # ~25 FPS streaming rate

    def get_status(self) -> Dict[str, Any]:
        """Telemetry and state dictionary for UI polling."""
        with self._lock:
            latest = self.latest_result
            alert_mgr = getattr(self.pipeline, "alert_manager", None)
            active_alert = alert_mgr.get_active_alert_summary() if alert_mgr else None
            return {
                "camera_id": self.config.CAMERA_ID,
                "camera_location": self.config.CAMERA_LOCATION,
                "camera_source": str(self.source),
                "status": self.status.value,
                "status_message": self.status_message,
                "is_monitoring": (self.status == CameraStatus.CONNECTED),
                "enable_fire_detection": getattr(self.pipeline, "enable_fire_detection", getattr(self.config, "ENABLE_FIRE_DETECTION", True)),
                "enable_smoke_detection": getattr(self.pipeline, "enable_smoke_detection", getattr(self.config, "ENABLE_SMOKE_DETECTION", False)),
                "fps": latest.fps if latest else 0.0,
                "frame_index": latest.frame_index if latest else 0,
                "last_frame_timestamp": self.last_frame_timestamp,
                "environmental_fire_count": latest.environmental_fire_count if latest else 0,
                "environmental_smoke_count": latest.environmental_smoke_count if latest else 0,
                "screen_rejected_count": latest.screen_rejected_count if latest else 0,
                "screen_uncertain_count": latest.screen_uncertain_count if latest else 0,
                "face_rejected_count": getattr(latest, "face_rejected_count", 0) if latest else 0,
                "face_uncertain_count": getattr(latest, "face_uncertain_count", 0) if latest else 0,
                "white_rejected_count": getattr(latest, "white_rejected_count", 0) if latest else 0,
                "white_uncertain_count": getattr(latest, "white_uncertain_count", 0) if latest else 0,
                "total_screen_false_positives_blocked": getattr(self.pipeline, "total_screen_false_positives_blocked", 0),
                "total_face_false_positives_blocked": getattr(self.pipeline, "total_face_false_positives_blocked", 0),
                "total_white_false_positives_blocked": getattr(self.pipeline, "total_white_false_positives_blocked", 0),
                "total_false_positives_blocked": (
                    getattr(self.pipeline, "total_screen_false_positives_blocked", 0)
                    + getattr(self.pipeline, "total_face_false_positives_blocked", 0)
                    + getattr(self.pipeline, "total_white_false_positives_blocked", 0)
                ),
                "fire_state": latest.fire_state.value if latest else "normal",
                "smoke_state": latest.smoke_state.value if latest else "normal",
                "alert_state": latest.alert_state.value if latest else (alert_mgr.current_state.value if alert_mgr else "normal"),
                "audio_alarm_active": latest.audio_alarm_active if latest else (alert_mgr.audio_trigger_active if alert_mgr else False),
                "active_alert": active_alert,
            }
