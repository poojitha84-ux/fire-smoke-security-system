"""
Model Loader and Verifier Module.
Handles downloading, caching, loading, and verifying models for:
1. Fire and Smoke detection (D-Fire fine-tuned YOLOv8n)
2. Screen and mobile phone context detection (COCO YOLOv8n)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Verified model sources and provenance
FIRE_SMOKE_HF_URL = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
COCO_YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"


class ModelLoadError(Exception):
    """Raised when a required model cannot be acquired or verified."""
    pass


class ModelLoader:
    """Manages loading, downloading, and verifying detection models."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.fire_smoke_model: Optional[Any] = None
        self.screen_model: Optional[Any] = None
        self.fire_class_ids: list[int] = []
        self.smoke_class_ids: list[int] = []
        self.screen_class_ids: list[int] = []
        self.person_class_ids: list[int] = []

    def download_file(self, url: str, destination: Path, label: str) -> Path:
        """Download a file with user-agent and basic verification."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 1024 * 100:
            logger.info(f"Model file already exists: {destination}")
            return destination

        logger.info(f"Downloading {label} from {url} to {destination}...")
        headers = {"User-Agent": "FireSmokeSecurityMonitor/1.0"}
        req = urllib.request.Request(url, headers=headers)

        temp_dest = destination.with_suffix(".download")
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_dest, "wb") as out_file:
                chunk_size = 1024 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            temp_dest.replace(destination)
            logger.info(f"Successfully downloaded {label} ({destination.stat().st_size} bytes)")
            return destination
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
            raise ModelLoadError(
                f"Failed to download {label} from {url}: {e}. "
                f"Please manually place the weights file at {destination}."
            ) from e

    def load_fire_smoke_model(self, model_path: Optional[str] = None) -> Any:
        """
        Load and verify the fire-and-smoke YOLO model.
        Verifies that 'fire' and 'smoke' classes are present.
        """
        target_path = Path(model_path) if model_path else self.models_dir / "fire_smoke_yolov8n.pt"

        if not target_path.exists():
            logger.info(f"Fire/Smoke weights not found at {target_path}. Attempting verified download...")
            try:
                self.download_file(FIRE_SMOKE_HF_URL, target_path, "Fire/Smoke YOLOv8n")
            except ModelLoadError as err:
                raise ModelLoadError(
                    f"Fire and Smoke model weights are unavailable at {target_path} and download failed. "
                    f"Provenance: rabahdev/fire-smoke-yolov8n (HuggingFace, D-Fire dataset fine-tune). "
                    f"Details: {err}"
                ) from err

        try:
            from ultralytics import YOLO
            model = YOLO(str(target_path))
            self.fire_smoke_model = model

            # Inspect and map classes
            names = model.names
            if isinstance(names, dict):
                class_dict = {int(k): str(v).lower().strip() for k, v in names.items()}
            elif isinstance(names, list):
                class_dict = {i: str(v).lower().strip() for i, v in enumerate(names)}
            else:
                class_dict = {}

            logger.info(f"Fire/Smoke model loaded classes: {class_dict}")

            # Identify fire and smoke indices
            self.fire_class_ids = [k for k, v in class_dict.items() if "fire" in v or "flame" in v]
            self.smoke_class_ids = [k for k, v in class_dict.items() if "smoke" in v]

            if not self.fire_class_ids and not self.smoke_class_ids:
                raise ModelLoadError(
                    f"Model at {target_path} does not contain 'fire' or 'smoke' classes. "
                    f"Discovered classes: {class_dict}."
                )

            return model
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Error initializing YOLO model from {target_path}: {e}") from e

    def load_screen_detector_model(self, model_path: Optional[str] = None) -> Any:
        """
        Load the screen & mobile phone detector (standard COCO YOLOv8n).
        Detects cell phones, laptops, and TV monitors.
        """
        target_path = Path(model_path) if model_path else self.models_dir / "yolov8n.pt"

        if not target_path.exists():
            logger.info(f"Screen detector weights not found at {target_path}. Attempting verified download...")
            try:
                self.download_file(COCO_YOLO_URL, target_path, "COCO YOLOv8n (Screen Detector)")
            except ModelLoadError as err:
                logger.warning(f"Could not download COCO screen detector: {err}. Screen rejection may use contour fallback.")
                return None

        try:
            from ultralytics import YOLO
            model = YOLO(str(target_path))
            self.screen_model = model

            names = model.names
            if isinstance(names, dict):
                class_dict = {int(k): str(v).lower().strip() for k, v in names.items()}
            elif isinstance(names, list):
                class_dict = {i: str(v).lower().strip() for i, v in enumerate(names)}
            else:
                class_dict = {}

            # COCO screen classes: cell phone (67), tv (62), laptop (63)
            self.screen_class_ids = [
                k for k, v in class_dict.items()
                if any(x in v for x in ["cell phone", "phone", "tv", "laptop", "monitor", "screen"])
            ]
            # COCO person class: person (0)
            self.person_class_ids = [
                k for k, v in class_dict.items()
                if v == "person"
            ]
            logger.info(
                f"Context detector loaded. Screen IDs: {self.screen_class_ids}, Person IDs: {self.person_class_ids}"
            )
            return model
        except Exception as e:
            logger.warning(f"Could not load screen detector from {target_path}: {e}")
            return None
