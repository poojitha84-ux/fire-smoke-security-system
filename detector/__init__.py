"""
Detector package for real-time fire and smoke detection with screen-context rejection.
"""

from .model_loader import ModelLoader
from .screen_context import ScreenContextFilter, ScreenDetection, DetectionValidationResult, HazardStatus
from .temporal_filter import TemporalFilter, HazardState
from .alert_manager import AlertManager, AlertState
from .inference import DetectionPipeline, FrameResult

__all__ = [
    "ModelLoader",
    "ScreenContextFilter",
    "ScreenDetection",
    "DetectionValidationResult",
    "HazardStatus",
    "TemporalFilter",
    "HazardState",
    "AlertManager",
    "AlertState",
    "DetectionPipeline",
    "FrameResult",
]
