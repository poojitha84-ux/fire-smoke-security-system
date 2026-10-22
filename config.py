"""
Application Configuration Module for Fire and Smoke Security Monitoring System.
Handles loading, parsing, and validating environment variables and defaults.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AppConfig:
    """Central configuration class."""
    # Project paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "models")
    STATIC_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "static")

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    SECRET_KEY: str = "fire-smoke-security-monitor-secret-key-2026"

    # Camera settings
    CAMERA_SOURCE: Union[int, str] = 0
    CAMERA_ID: str = "CAM-MAIN-01"
    CAMERA_LOCATION: str = "Main Building - Floor 1 Central"
    CAMERA_FRAME_WIDTH: int = 1280
    CAMERA_FRAME_HEIGHT: int = 720
    CAMERA_FPS_TARGET: int = 30

    # Detection Models
    FIRE_SMOKE_MODEL_PATH: str = str(Path(__file__).resolve().parent / "models" / "fire_smoke_yolov8n.pt")
    SCREEN_DETECTOR_MODEL_PATH: str = str(Path(__file__).resolve().parent / "models" / "yolov8n.pt")

    # Hazard Class Detection Toggles
    ENABLE_FIRE_DETECTION: bool = True                  # Fire flame detection actively monitored
    ENABLE_SMOKE_DETECTION: bool = False                # Smoke detection disabled to eliminate false alarms on white objects, clothing, glare

    # Confidence thresholds
    FIRE_CONFIDENCE_THRESHOLD: float = 0.45
    SMOKE_CONFIDENCE_THRESHOLD: float = 0.40
    DETECTION_IOU_THRESHOLD: float = 0.45

    # Screen context & mobile phone rejection
    ENABLE_SCREEN_REJECTION: bool = True
    SCREEN_DETECTION_CONFIDENCE: float = 0.35
    SCREEN_TILTED_MIN_CONFIDENCE: float = 0.20          # Lower confidence floor to catch tilted / partially occluded phones
    SCREEN_OVERLAP_REJECT_THRESHOLD: float = 0.65       # >= 65% overlap with screen means reject (if fully enclosed)
    SCREEN_OVERLAP_UNCERTAIN_THRESHOLD: float = 0.25    # 25%-65% overlap means SCREEN_UNCERTAIN
    PROTRUSION_MARGIN_RATIO: float = 0.15               # Max protrusion past screen bezel before flagging as fire behind/around screen
    FIRE_BEHIND_SCREEN_AREA_RATIO: float = 0.85         # Fire area relative to screen above which fire cannot be physically inside phone
    REJECT_SMOKE_ON_SCREENS: bool = False               # Smoke is atmospheric; never reject smoke overlapping screens as fake

    # Face & Person Context Filtering (False-Positive Smoke Defense)
    ENABLE_FACE_SMOKE_FILTER: bool = True
    PERSON_DETECTION_CONFIDENCE: float = 0.35            # Confidence threshold to detect persons in COCO model
    FACE_OVERLAP_REJECT_THRESHOLD: float = 0.65         # >= 65% overlap with head region means face artifact
    FACE_OVERLAP_UNCERTAIN_THRESHOLD: float = 0.30      # 30%-65% overlap means FACE_UNCERTAIN
    FACE_SMOKE_CALIBRATED_THRESHOLD: float = 0.72       # Required confidence to confirm smoke strictly inside head box
    FACE_UNCERTAIN_CONF_FLOOR: float = 0.55             # Below this confidence, confined face smoke is rejected
    SMOKE_PLUME_DISPERSION_THRESHOLD: float = 0.35      # Fraction of smoke box extending outside person body for environmental plume

    # White-Object, Clothing & Bright-Area Smoke Defense
    ENABLE_WHITE_OBJECT_FILTER: bool = True             # Reject white walls, paper, glare, and bright rigid surfaces
    ENABLE_CLOTHING_SMOKE_FILTER: bool = True           # Reject white shirt / clothing artifacts on detected persons
    WHITE_OBJECT_REJECT_CONFIDENCE: float = 0.65        # Rejection ceiling for rigid/flat white objects and optical glare
    WHITE_OBJECT_UNCERTAIN_CONFIDENCE: float = 0.75     # Intermediate confidence threshold for white-object review
    GLARE_SATURATION_THRESHOLD: float = 0.70            # Fraction of blown-out highlight pixels (V>=248, Sat<=25)
    UNIFORM_WALL_STD_THRESHOLD: float = 4.5             # Max std dev of pixel values to qualify as flat uniform wall/board

    # Temporal confirmation
    TEMPORAL_WINDOW_SIZE: int = 8                       # Window of recent frames to analyze
    TEMPORAL_CONFIRM_COUNT: int = 5                     # Required positive detections within window
    ALERT_COOLDOWN_SECONDS: float = 30.0               # Suppression period for duplicate incident alerts
    CLEAR_CONSECUTIVE_FRAMES: int = 15                 # Consecutive clear frames to reset hazard status

    # Security Notification Settings
    ENABLE_NOTIFICATIONS: bool = False
    NOTIFICATION_TYPE: str = "smtp"                     # "smtp" or "webhook"
    SMTP_SERVER: str = ""
    SMTP_PORT: int = 587
    SMTP_USE_TLS: bool = True
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_EMAIL_FROM: str = ""
    ALERT_EMAIL_TO: str = ""

    WEBHOOK_URL: str = ""
    WEBHOOK_TIMEOUT_SECONDS: float = 5.0
    WEBHOOK_MAX_RETRIES: int = 2

    # Audio alert
    ENABLE_AUDIO_ALARM: bool = True
    AUDIO_COOLDOWN_SECONDS: float = 10.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build configuration instance from environment variables with fallbacks."""
        config = cls()

        # Parse camera source (int for webcam index, str for RTSP/HTTP)
        cam_source_raw = os.getenv("CAMERA_SOURCE", "0").strip()
        if cam_source_raw.isdigit():
            config.CAMERA_SOURCE = int(cam_source_raw)
        else:
            config.CAMERA_SOURCE = cam_source_raw

        config.CAMERA_ID = os.getenv("CAMERA_ID", config.CAMERA_ID)
        config.CAMERA_LOCATION = os.getenv("CAMERA_LOCATION", config.CAMERA_LOCATION)
        config.CAMERA_FRAME_WIDTH = int(os.getenv("CAMERA_FRAME_WIDTH", str(config.CAMERA_FRAME_WIDTH)))
        config.CAMERA_FRAME_HEIGHT = int(os.getenv("CAMERA_FRAME_HEIGHT", str(config.CAMERA_FRAME_HEIGHT)))
        config.CAMERA_FPS_TARGET = int(os.getenv("CAMERA_FPS_TARGET", str(config.CAMERA_FPS_TARGET)))

        config.PORT = int(os.getenv("PORT", str(config.PORT)))
        config.DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        config.SECRET_KEY = os.getenv("SECRET_KEY", config.SECRET_KEY)

        if os.getenv("FIRE_SMOKE_MODEL_PATH"):
            config.FIRE_SMOKE_MODEL_PATH = os.getenv("FIRE_SMOKE_MODEL_PATH")
        if os.getenv("SCREEN_DETECTOR_MODEL_PATH"):
            config.SCREEN_DETECTOR_MODEL_PATH = os.getenv("SCREEN_DETECTOR_MODEL_PATH")

        config.ENABLE_FIRE_DETECTION = os.getenv("ENABLE_FIRE_DETECTION", "True").lower() in ("true", "1", "t")
        config.ENABLE_SMOKE_DETECTION = os.getenv("ENABLE_SMOKE_DETECTION", "False").lower() in ("true", "1", "t")

        config.FIRE_CONFIDENCE_THRESHOLD = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", str(config.FIRE_CONFIDENCE_THRESHOLD)))
        config.SMOKE_CONFIDENCE_THRESHOLD = float(os.getenv("SMOKE_CONFIDENCE_THRESHOLD", str(config.SMOKE_CONFIDENCE_THRESHOLD)))
        config.DETECTION_IOU_THRESHOLD = float(os.getenv("DETECTION_IOU_THRESHOLD", str(config.DETECTION_IOU_THRESHOLD)))

        config.ENABLE_SCREEN_REJECTION = os.getenv("ENABLE_SCREEN_REJECTION", "True").lower() in ("true", "1", "t")
        config.SCREEN_DETECTION_CONFIDENCE = float(os.getenv("SCREEN_DETECTION_CONFIDENCE", str(config.SCREEN_DETECTION_CONFIDENCE)))
        config.SCREEN_TILTED_MIN_CONFIDENCE = float(os.getenv("SCREEN_TILTED_MIN_CONFIDENCE", str(config.SCREEN_TILTED_MIN_CONFIDENCE)))
        config.SCREEN_OVERLAP_REJECT_THRESHOLD = float(os.getenv("SCREEN_OVERLAP_REJECT_THRESHOLD", str(config.SCREEN_OVERLAP_REJECT_THRESHOLD)))
        config.SCREEN_OVERLAP_UNCERTAIN_THRESHOLD = float(os.getenv("SCREEN_OVERLAP_UNCERTAIN_THRESHOLD", str(config.SCREEN_OVERLAP_UNCERTAIN_THRESHOLD)))
        config.PROTRUSION_MARGIN_RATIO = float(os.getenv("PROTRUSION_MARGIN_RATIO", str(config.PROTRUSION_MARGIN_RATIO)))
        config.FIRE_BEHIND_SCREEN_AREA_RATIO = float(os.getenv("FIRE_BEHIND_SCREEN_AREA_RATIO", str(config.FIRE_BEHIND_SCREEN_AREA_RATIO)))
        config.REJECT_SMOKE_ON_SCREENS = os.getenv("REJECT_SMOKE_ON_SCREENS", "False").lower() in ("true", "1", "t")

        config.ENABLE_FACE_SMOKE_FILTER = os.getenv("ENABLE_FACE_SMOKE_FILTER", "True").lower() in ("true", "1", "t")
        config.PERSON_DETECTION_CONFIDENCE = float(os.getenv("PERSON_DETECTION_CONFIDENCE", str(config.PERSON_DETECTION_CONFIDENCE)))
        config.FACE_OVERLAP_REJECT_THRESHOLD = float(os.getenv("FACE_OVERLAP_REJECT_THRESHOLD", str(config.FACE_OVERLAP_REJECT_THRESHOLD)))
        config.FACE_OVERLAP_UNCERTAIN_THRESHOLD = float(os.getenv("FACE_OVERLAP_UNCERTAIN_THRESHOLD", str(config.FACE_OVERLAP_UNCERTAIN_THRESHOLD)))
        config.FACE_SMOKE_CALIBRATED_THRESHOLD = float(os.getenv("FACE_SMOKE_CALIBRATED_THRESHOLD", str(config.FACE_SMOKE_CALIBRATED_THRESHOLD)))
        config.FACE_UNCERTAIN_CONF_FLOOR = float(os.getenv("FACE_UNCERTAIN_CONF_FLOOR", str(config.FACE_UNCERTAIN_CONF_FLOOR)))
        config.SMOKE_PLUME_DISPERSION_THRESHOLD = float(os.getenv("SMOKE_PLUME_DISPERSION_THRESHOLD", str(config.SMOKE_PLUME_DISPERSION_THRESHOLD)))

        config.ENABLE_WHITE_OBJECT_FILTER = os.getenv("ENABLE_WHITE_OBJECT_FILTER", "True").lower() in ("true", "1", "t")
        config.ENABLE_CLOTHING_SMOKE_FILTER = os.getenv("ENABLE_CLOTHING_SMOKE_FILTER", "True").lower() in ("true", "1", "t")
        config.WHITE_OBJECT_REJECT_CONFIDENCE = float(os.getenv("WHITE_OBJECT_REJECT_CONFIDENCE", str(config.WHITE_OBJECT_REJECT_CONFIDENCE)))
        config.WHITE_OBJECT_UNCERTAIN_CONFIDENCE = float(os.getenv("WHITE_OBJECT_UNCERTAIN_CONFIDENCE", str(config.WHITE_OBJECT_UNCERTAIN_CONFIDENCE)))
        config.GLARE_SATURATION_THRESHOLD = float(os.getenv("GLARE_SATURATION_THRESHOLD", str(config.GLARE_SATURATION_THRESHOLD)))
        config.UNIFORM_WALL_STD_THRESHOLD = float(os.getenv("UNIFORM_WALL_STD_THRESHOLD", str(config.UNIFORM_WALL_STD_THRESHOLD)))

        config.TEMPORAL_WINDOW_SIZE = int(os.getenv("TEMPORAL_WINDOW_SIZE", str(config.TEMPORAL_WINDOW_SIZE)))
        config.TEMPORAL_CONFIRM_COUNT = int(os.getenv("TEMPORAL_CONFIRM_COUNT", str(config.TEMPORAL_CONFIRM_COUNT)))
        config.ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", str(config.ALERT_COOLDOWN_SECONDS)))
        config.CLEAR_CONSECUTIVE_FRAMES = int(os.getenv("CLEAR_CONSECUTIVE_FRAMES", str(config.CLEAR_CONSECUTIVE_FRAMES)))

        config.ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "False").lower() in ("true", "1", "t")
        config.NOTIFICATION_TYPE = os.getenv("NOTIFICATION_TYPE", config.NOTIFICATION_TYPE).lower()
        config.SMTP_SERVER = os.getenv("SMTP_SERVER", "")
        config.SMTP_PORT = int(os.getenv("SMTP_PORT", str(config.SMTP_PORT)))
        config.SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "t")
        config.SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
        config.SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
        config.ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
        config.ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

        config.WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
        config.WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", str(config.WEBHOOK_TIMEOUT_SECONDS)))
        config.WEBHOOK_MAX_RETRIES = int(os.getenv("WEBHOOK_MAX_RETRIES", str(config.WEBHOOK_MAX_RETRIES)))

        config.ENABLE_AUDIO_ALARM = os.getenv("ENABLE_AUDIO_ALARM", "True").lower() in ("true", "1", "t")
        config.AUDIO_COOLDOWN_SECONDS = float(os.getenv("AUDIO_COOLDOWN_SECONDS", str(config.AUDIO_COOLDOWN_SECONDS)))

        return config

    def validate(self) -> list[str]:
        """Validate config values and return list of validation errors."""
        errors = []
        if self.FIRE_CONFIDENCE_THRESHOLD <= 0.0 or self.FIRE_CONFIDENCE_THRESHOLD >= 1.0:
            errors.append("FIRE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        if self.SMOKE_CONFIDENCE_THRESHOLD <= 0.0 or self.SMOKE_CONFIDENCE_THRESHOLD >= 1.0:
            errors.append("SMOKE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        if self.SCREEN_OVERLAP_UNCERTAIN_THRESHOLD >= self.SCREEN_OVERLAP_REJECT_THRESHOLD:
            errors.append("SCREEN_OVERLAP_UNCERTAIN_THRESHOLD must be strictly less than SCREEN_OVERLAP_REJECT_THRESHOLD")
        if self.SCREEN_TILTED_MIN_CONFIDENCE <= 0.0 or self.SCREEN_TILTED_MIN_CONFIDENCE >= self.SCREEN_DETECTION_CONFIDENCE:
            errors.append("SCREEN_TILTED_MIN_CONFIDENCE must be between 0.0 and SCREEN_DETECTION_CONFIDENCE")
        if self.PROTRUSION_MARGIN_RATIO <= 0.0 or self.PROTRUSION_MARGIN_RATIO >= 1.0:
            errors.append("PROTRUSION_MARGIN_RATIO must be between 0.0 and 1.0")
        if self.FACE_OVERLAP_UNCERTAIN_THRESHOLD >= self.FACE_OVERLAP_REJECT_THRESHOLD:
            errors.append("FACE_OVERLAP_UNCERTAIN_THRESHOLD must be strictly less than FACE_OVERLAP_REJECT_THRESHOLD")
        if self.FACE_UNCERTAIN_CONF_FLOOR >= self.FACE_SMOKE_CALIBRATED_THRESHOLD:
            errors.append("FACE_UNCERTAIN_CONF_FLOOR must be strictly less than FACE_SMOKE_CALIBRATED_THRESHOLD")
        if self.TEMPORAL_CONFIRM_COUNT > self.TEMPORAL_WINDOW_SIZE:
            errors.append("TEMPORAL_CONFIRM_COUNT cannot exceed TEMPORAL_WINDOW_SIZE")
        if self.TEMPORAL_CONFIRM_COUNT <= 0:
            errors.append("TEMPORAL_CONFIRM_COUNT must be positive")
        if self.ALERT_COOLDOWN_SECONDS < 0:
            errors.append("ALERT_COOLDOWN_SECONDS cannot be negative")
        return errors
