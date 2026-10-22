"""
Live Frame Detection Pipeline.

Orchestrates:
1. Deep learning fire and smoke inference.
2. Deep learning & optical screen / mobile phone detection.
3. Screen context analysis (filtering phone false positives while monitoring environmental scene).
4. Temporal persistence confirmation.
5. Alert manager state evaluation.
6. Professional visual HUD and bounding box rendering.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .model_loader import ModelLoader
from .screen_context import (
    ScreenContextFilter,
    ScreenDetection,
    PersonDetection,
    DetectionValidationResult,
    HazardStatus,
)
from .temporal_filter import TemporalFilter, HazardState
from .alert_manager import AlertManager, AlertState

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Output summary of processing a single video frame."""
    frame_index: int
    timestamp: float
    fps: float
    annotated_frame: np.ndarray
    raw_hazard_count: int
    environmental_fire_count: int
    environmental_smoke_count: int
    screen_rejected_count: int
    screen_uncertain_count: int
    fire_state: HazardState
    smoke_state: HazardState
    alert_state: AlertState
    audio_alarm_active: bool
    face_rejected_count: int = 0
    face_uncertain_count: int = 0
    white_rejected_count: int = 0
    white_uncertain_count: int = 0
    validation_results: List[DetectionValidationResult] = field(default_factory=list)
    screen_detections: List[ScreenDetection] = field(default_factory=list)
    person_detections: List[PersonDetection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "fps": round(self.fps, 1),
            "raw_hazard_count": self.raw_hazard_count,
            "environmental_fire_count": self.environmental_fire_count,
            "environmental_smoke_count": self.environmental_smoke_count,
            "screen_rejected_count": self.screen_rejected_count,
            "screen_uncertain_count": self.screen_uncertain_count,
            "face_rejected_count": self.face_rejected_count,
            "face_uncertain_count": self.face_uncertain_count,
            "white_rejected_count": getattr(self, "white_rejected_count", 0),
            "white_uncertain_count": getattr(self, "white_uncertain_count", 0),
            "fire_state": self.fire_state.value,
            "smoke_state": self.smoke_state.value,
            "alert_state": self.alert_state.value,
            "audio_alarm_active": self.audio_alarm_active,
            "validation_results": [r.to_dict() for r in self.validation_results],
        }


class DetectionPipeline:
    """
    Core detection engine processing live camera frames.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        screen_filter: ScreenContextFilter,
        temporal_filter: TemporalFilter,
        alert_manager: AlertManager,
        fire_threshold: float = 0.45,
        smoke_threshold: float = 0.40,
        enable_fire_detection: bool = True,
        enable_smoke_detection: bool = True,
    ):
        self.loader = model_loader
        self.screen_filter = screen_filter
        self.temporal_filter = temporal_filter
        self.alert_manager = alert_manager

        self.fire_threshold = fire_threshold
        self.smoke_threshold = smoke_threshold
        self.enable_fire_detection = enable_fire_detection
        self.enable_smoke_detection = enable_smoke_detection

        self.frame_index: int = 0
        self.prev_frame_time: float = time.time()
        self.fps_rolling: float = 0.0
        self.total_screen_false_positives_blocked: int = 0
        self.total_face_false_positives_blocked: int = 0
        self.total_white_false_positives_blocked: int = 0

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single BGR OpenCV frame from the camera stream.
        """
        self.frame_index += 1
        now = time.time()
        elapsed = now - self.prev_frame_time
        self.prev_frame_time = now
        current_fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        self.fps_rolling = (0.9 * self.fps_rolling + 0.1 * current_fps) if self.fps_rolling > 0 else current_fps

        if frame is None or frame.size == 0:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No Video Frame", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return FrameResult(
                frame_index=self.frame_index,
                timestamp=now,
                fps=0.0,
                annotated_frame=blank,
                raw_hazard_count=0,
                environmental_fire_count=0,
                environmental_smoke_count=0,
                screen_rejected_count=0,
                screen_uncertain_count=0,
                fire_state=HazardState.NORMAL,
                smoke_state=HazardState.NORMAL,
                alert_state=AlertState.NORMAL,
                audio_alarm_active=False,
                face_rejected_count=0,
                face_uncertain_count=0,
            )

        # 1. Run Fire and Smoke Detection
        hazard_candidates = self._run_fire_smoke_inference(frame)

        # Enforce detection toggles at pipeline entry
        if not self.enable_smoke_detection:
            hazard_candidates = [c for c in hazard_candidates if c.get("class_name") != "smoke"]
        if not self.enable_fire_detection:
            hazard_candidates = [c for c in hazard_candidates if c.get("class_name") != "fire"]

        # 2. Run Screen & Person Context Detection
        screen_detections, person_detections = self._run_context_detection(
            frame, [c["bbox"] for c in hazard_candidates]
        )

        # 3. Context & Containment Evaluation (Screens + Faces + White Objects)
        validation_results = self.screen_filter.evaluate_detections(
            hazard_candidates,
            screen_detections,
            detected_persons=person_detections,
            frame=frame,
        )

        # 4. Separate Environmental Hazards vs Screen/Face/White-Rejected / Uncertain
        env_fire_boxes: List[List[int]] = []
        env_smoke_boxes: List[List[int]] = []
        max_fire_conf: float = 0.0
        max_smoke_conf: float = 0.0
        screen_rejected_count: int = 0
        face_rejected_count: int = 0
        white_rejected_count: int = 0
        screen_uncertain_count: int = 0
        face_uncertain_count: int = 0
        white_uncertain_count: int = 0

        for res in validation_results:
            if res.status == HazardStatus.CONFIRMED_ENVIRONMENTAL:
                if res.class_name == "fire":
                    env_fire_boxes.append(res.bbox)
                    max_fire_conf = max(max_fire_conf, res.confidence)
                elif res.class_name == "smoke":
                    env_smoke_boxes.append(res.bbox)
                    max_smoke_conf = max(max_smoke_conf, res.confidence)
            elif res.status == HazardStatus.SCREEN_REJECTED:
                screen_rejected_count += 1
                self.total_screen_false_positives_blocked += 1
            elif res.status == HazardStatus.FACE_REJECTED:
                face_rejected_count += 1
                self.total_face_false_positives_blocked += 1
            elif res.status == HazardStatus.WHITE_OBJECT_REJECTED:
                white_rejected_count += 1
                self.total_white_false_positives_blocked += 1
            elif res.status == HazardStatus.FACE_UNCERTAIN:
                face_uncertain_count += 1
            elif res.status == HazardStatus.WHITE_OBJECT_UNCERTAIN:
                white_uncertain_count += 1
            elif res.status == HazardStatus.SCREEN_UNCERTAIN:
                screen_uncertain_count += 1

        # Total combined uncertain count for reporting
        total_uncertain_count = screen_uncertain_count + face_uncertain_count + white_uncertain_count
        total_rejections = screen_rejected_count + face_rejected_count + white_rejected_count

        if not self.enable_smoke_detection:
            env_smoke_boxes = []
            max_smoke_conf = 0.0

        # 5. Temporal Persistence Filtering (only environmental hazards)
        states = self.temporal_filter.update(
            environmental_fire_bboxes=env_fire_boxes,
            environmental_fire_conf=max_fire_conf,
            environmental_smoke_bboxes=env_smoke_boxes,
            environmental_smoke_conf=max_smoke_conf,
            has_uncertain_detection=(screen_uncertain_count > 0),
        )
        fire_state = states["fire"]
        smoke_state = states["smoke"] if self.enable_smoke_detection else HazardState.NORMAL

        # 6. Alert Manager State Machine
        alert_state = self.alert_manager.evaluate_states(
            fire_state=fire_state,
            smoke_state=smoke_state,
            fire_conf=max_fire_conf,
            smoke_conf=max_smoke_conf if self.enable_smoke_detection else 0.0,
            uncertain_active=(screen_uncertain_count > 0),
            screen_rejections_count=total_rejections,
        )

        # 7. Render Professional Overlays and HUD
        annotated_frame = self._render_overlays(
            frame.copy(),
            validation_results=validation_results,
            screen_detections=screen_detections,
            person_detections=person_detections,
            fire_state=fire_state,
            smoke_state=smoke_state,
            alert_state=alert_state,
            fps=self.fps_rolling,
        )

        return FrameResult(
            frame_index=self.frame_index,
            timestamp=now,
            fps=self.fps_rolling,
            annotated_frame=annotated_frame,
            raw_hazard_count=len(hazard_candidates),
            environmental_fire_count=len(env_fire_boxes),
            environmental_smoke_count=len(env_smoke_boxes),
            screen_rejected_count=screen_rejected_count,
            screen_uncertain_count=total_uncertain_count,
            fire_state=fire_state,
            smoke_state=smoke_state,
            alert_state=alert_state,
            audio_alarm_active=self.alert_manager.audio_trigger_active,
            face_rejected_count=face_rejected_count,
            face_uncertain_count=face_uncertain_count,
            white_rejected_count=white_rejected_count,
            white_uncertain_count=white_uncertain_count,
            validation_results=validation_results,
            screen_detections=screen_detections,
            person_detections=person_detections,
        )

    def _run_fire_smoke_inference(self, frame: np.ndarray) -> List[dict]:
        """Run deep learning model for fire and smoke candidates."""
        candidates = []
        model = self.loader.fire_smoke_model
        if model is None:
            return candidates

        min_conf = self.fire_threshold if not self.enable_smoke_detection else min(self.fire_threshold, self.smoke_threshold)
        try:
            results = model.predict(frame, conf=min_conf, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    class_name = None
                    if cls_id in self.loader.fire_class_ids:
                        if not self.enable_fire_detection:
                            continue
                        class_name = "fire"
                        if conf < self.fire_threshold:
                            continue
                    elif cls_id in self.loader.smoke_class_ids:
                        if not self.enable_smoke_detection:
                            continue
                        class_name = "smoke"
                        if conf < self.smoke_threshold:
                            continue
                    else:
                        # Fallback: check raw label text
                        raw_name = r.names.get(cls_id, "").lower()
                        if "fire" in raw_name or "flame" in raw_name:
                            if not self.enable_fire_detection:
                                continue
                            class_name = "fire"
                            if conf < self.fire_threshold:
                                continue
                        elif "smoke" in raw_name:
                            if not self.enable_smoke_detection:
                                continue
                            class_name = "smoke"
                            if conf < self.smoke_threshold:
                                continue

                    if class_name:
                        candidates.append({
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": xyxy,
                        })
        except Exception as e:
            logger.error(f"Inference error in fire/smoke detection: {e}")

        return candidates

    def _run_context_detection(
        self,
        frame: np.ndarray,
        hazard_bboxes: List[List[int]],
    ) -> Tuple[List[ScreenDetection], List[PersonDetection]]:
        """Run COCO model for screens and persons, augmented by contour analysis."""
        screens: List[ScreenDetection] = []
        persons: List[PersonDetection] = []

        # 1. Primary: Unified context detector (screens & persons)
        if self.loader.screen_model is not None and self.screen_filter.enabled:
            screens, persons = self.screen_filter.detect_context_from_model(
                self.loader.screen_model,
                frame,
                screen_class_ids=getattr(self.loader, "screen_class_ids", None),
                person_class_ids=getattr(self.loader, "person_class_ids", None),
            )

        # 2. Contour analysis fallback if screens might exist around candidates
        if len(screens) == 0 and len(hazard_bboxes) > 0 and self.screen_filter.enabled:
            contour_screens = self.screen_filter.detect_display_contours_fallback(frame, hazard_bboxes)
            screens.extend(contour_screens)

        return screens, persons

    def _run_screen_detection(
        self,
        frame: np.ndarray,
        hazard_bboxes: List[List[int]],
    ) -> List[ScreenDetection]:
        """Backward-compatible screen detection wrapper."""
        screens, _ = self._run_context_detection(frame, hazard_bboxes)
        return screens

    def _render_overlays(
        self,
        img: np.ndarray,
        validation_results: List[DetectionValidationResult],
        screen_detections: List[ScreenDetection],
        person_detections: Optional[List[PersonDetection]] = None,
        fire_state: HazardState = HazardState.NORMAL,
        smoke_state: HazardState = HazardState.NORMAL,
        alert_state: AlertState = AlertState.NORMAL,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw bounding boxes and security HUD on the frame."""
        h, w = img.shape[:2]

        # Draw detected screens (phones / monitors)
        for s in screen_detections:
            x1, y1, x2, y2 = s.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 30), 2)
            label = f"[DISPLAY] {s.label.upper()} {int(s.confidence*100)}%"
            self._draw_label(img, label, x1, max(18, y1 - 6), (255, 120, 30), (255, 255, 255))

        # Note: Person detections and face bounding boxes / labels are intentionally NOT drawn
        # on the camera feed to keep the live security feed clean and avoid drawing boxes on human faces/persons.

        # Draw hazard boxes with status-dependent styling
        for res in validation_results:
            x1, y1, x2, y2 = res.bbox
            conf_pct = int(res.confidence * 100)

            if res.status == HazardStatus.CONFIRMED_ENVIRONMENTAL:
                if res.class_name == "fire":
                    color = (0, 0, 255)  # Red
                    tag = f"FIRE {conf_pct}% [ENVIRONMENTAL]"
                else:
                    color = (200, 180, 50)  # Cyan-yellow / smoke
                    tag = f"SMOKE {conf_pct}% [ENVIRONMENTAL]"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                self._draw_label(img, tag, x1, max(18, y1 - 6), color, (255, 255, 255))

            elif res.status == HazardStatus.SCREEN_REJECTED:
                # Cyan / Orange: False positive prevented!
                color = (0, 220, 255)  # Bright amber/yellow-cyan
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                tag = f"REJECTED: {res.class_name.upper()} ON SCREEN ({conf_pct}%)"
                self._draw_label(img, tag, x1, max(18, y1 - 6), color, (0, 0, 0))

            elif res.status == HazardStatus.SCREEN_UNCERTAIN:
                # Magenta / Yellow: Ambiguous proximity or fire behind screen
                color = (0, 255, 255)  # Yellow
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if "behind" in res.reason.lower():
                    tag = f"UNCERTAIN: POSSIBLE FIRE BEHIND SCREEN ({conf_pct}%)"
                elif "smoke" in res.reason.lower():
                    tag = f"UNCERTAIN: SMOKE OVER SCREEN ({conf_pct}%)"
                else:
                    tag = f"UNCERTAIN: SCREEN PROXIMITY ({conf_pct}%)"
                self._draw_label(img, tag, x1, max(18, y1 - 6), color, (0, 0, 0))

            # FACE_REJECTED and FACE_UNCERTAIN: Never drawn on camera feed per privacy & false-box prevention requirements.

            elif res.status == HazardStatus.WHITE_OBJECT_REJECTED:
                # Slate-cyan: White object or glare artifact prevented!
                color = (220, 180, 140)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                tag = f"REJECTED: WHITE OBJECT / GLARE ({conf_pct}%)"
                self._draw_label(img, tag, x1, max(18, y1 - 6), color, (0, 0, 0))

            elif res.status == HazardStatus.WHITE_OBJECT_UNCERTAIN:
                # Amber: White object or glare ambiguous
                color = (0, 215, 255)  # Amber
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                tag = f"UNCERTAIN: BRIGHT/WHITE REGION ({conf_pct}%)"
                self._draw_label(img, tag, x1, max(18, y1 - 6), color, (0, 0, 0))

        # Top Security HUD Banner
        hud_bg = img[:50, :].copy()
        cv2.rectangle(img, (0, 0), (w, 50), (15, 23, 42), -1)  # Dark slate header
        cv2.addWeighted(img[:50, :], 0.85, hud_bg, 0.15, 0, img[:50, :])

        # HUD Text items
        cam_info = f"{self.alert_manager.camera_id} | {self.alert_manager.camera_location} | LIVE SURVEILLANCE"
        cv2.putText(img, cam_info, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        fps_text = f"FPS: {fps:.1f} | Frame: {self.frame_index}"
        cv2.putText(img, fps_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Status badge on right side of HUD
        status_color = (0, 200, 0)
        status_text = "STATUS: NORMAL (SECURE)"

        if alert_state == AlertState.CONFIRMED_ALERT:
            status_color = (0, 0, 255)
            status_text = "ALERT: HAZARD CONFIRMED"
        elif alert_state == AlertState.COOLDOWN:
            status_color = (0, 140, 255)
            status_text = "STATUS: ACTIVE HAZARD (COOLDOWN)"
        elif alert_state == AlertState.SUSPECTED:
            status_color = (0, 215, 255)
            status_text = "STATUS: SUSPECTED HAZARD"

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        badge_x = w - text_size[0] - 25
        cv2.rectangle(img, (badge_x - 8, 8), (w - 10, 42), status_color, -1)
        cv2.putText(img, status_text, (badge_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return img

    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        size = cv2.getTextSize(text, font, scale, thick)[0]
        cv2.rectangle(img, (x, y - size[1] - 4), (x + size[0] + 6, y + 4), bg_color, -1)
        cv2.putText(img, text, (x + 3, y - 1), font, scale, text_color, thick, cv2.LINE_AA)
