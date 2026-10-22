"""
Mobile Phone and Screen False-Positive Prevention Component.

Distinguishes genuine environmental fire and smoke from fire/smoke displayed on:
- Mobile phones held in front of the camera
- Television monitors and laptops
- Digital display surfaces

Key capabilities:
- Analyzes spatial containment / intersection between hazard detections and screen boundaries.
- Separates individual detections in the frame: phone-screen fire is rejected while genuine
  environmental fire in other parts of the scene continues to be actively monitored and confirmed.
- Avoids over-rejection: fire near a screen is NOT discarded unless significant containment is proven.
- Distinguishes CONFIRMED_ENVIRONMENTAL, SCREEN_REJECTED, and SCREEN_UNCERTAIN states.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HazardStatus(str, Enum):
    """Classification status after screen, person, and white-object context evaluation."""
    CONFIRMED_ENVIRONMENTAL = "confirmed_environmental"
    SCREEN_REJECTED = "screen_rejected"
    SCREEN_UNCERTAIN = "screen_uncertain"
    FACE_REJECTED = "face_rejected"
    FACE_UNCERTAIN = "face_uncertain"
    WHITE_OBJECT_REJECTED = "white_object_rejected"
    WHITE_OBJECT_UNCERTAIN = "white_object_uncertain"


@dataclass
class ScreenDetection:
    """Represents a detected display surface, phone, or monitor."""
    bbox: List[int]             # [x1, y1, x2, y2]
    label: str                  # "cell phone", "tv", "laptop", "display"
    confidence: float
    detector_source: str = "model"  # "model" or "contour_analysis"
    is_low_confidence: bool = False # True if between tilted_min_conf and min_screen_conf


@dataclass
class PersonDetection:
    """Represents a detected person and their estimated head/face and torso/clothing regions."""
    bbox: List[int]                         # Full person bbox [x1, y1, x2, y2]
    confidence: float
    head_bbox: List[int]                    # Estimated head/face region [x1, y1, x2, y2]
    torso_bbox: Optional[List[int]] = None  # Estimated torso/clothing region [x1, y1, x2, y2]
    label: str = "person"

    def __post_init__(self):
        if self.torso_bbox is None and self.bbox:
            self.torso_bbox = estimate_torso_bbox(self.bbox)


@dataclass
class DetectionValidationResult:
    """Result of context evaluation on a single hazard detection."""
    class_name: str                         # "fire" or "smoke"
    confidence: float
    bbox: List[int]                         # [x1, y1, x2, y2]
    status: HazardStatus
    overlap_ratio: float = 0.0
    associated_screen: Optional[ScreenDetection] = None
    associated_person: Optional[PersonDetection] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "status": self.status.value,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "screen_label": self.associated_screen.label if self.associated_screen else None,
            "screen_confidence": round(self.associated_screen.confidence, 4) if self.associated_screen else None,
            "person_confidence": round(self.associated_person.confidence, 4) if self.associated_person else None,
            "reason": self.reason,
        }


def estimate_head_bbox(person_bbox: List[int]) -> List[int]:
    """
    Estimates the head/face bounding box from a detected person bounding box.
    In webcam/surveillance poses, the head is located in the top portion
    of the bounding box, centered horizontally.
    """
    px1, py1, px2, py2 = person_bbox
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    # Head occupies top ~42% of person height, with 10% horizontal inset
    hx1 = int(px1 + 0.10 * pw)
    hx2 = int(px2 - 0.10 * pw)
    hy1 = int(py1)
    hy2 = int(py1 + 0.42 * ph)

    return [hx1, hy1, max(hx1 + 1, hx2), max(hy1 + 1, hy2)]


def estimate_torso_bbox(person_bbox: List[int]) -> List[int]:
    """
    Estimates the torso / clothing bounding box from a detected person bounding box.
    Typically spans from ~28% below top down to ~88% of full height.
    """
    px1, py1, px2, py2 = person_bbox
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    tx1 = int(px1 + 0.05 * pw)
    tx2 = int(px2 - 0.05 * pw)
    ty1 = int(py1 + 0.28 * ph)
    ty2 = int(py1 + 0.88 * ph)

    return [tx1, ty1, max(tx1 + 1, tx2), max(ty1 + 1, ty2)]


def is_skin_face_patch(frame: Optional[np.ndarray], bbox: List[int]) -> bool:
    """
    Checks if a bounding box region consists predominantly of normal human skin tones
    rather than high-intensity flame.
    """
    if frame is None:
        return False
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size < 50:
        return False

    try:
        # Check for skin tone in YCrCb color space
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        # Standard skin tone range: Cr in [133, 173], Cb in [77, 127]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.mean(skin_mask))

        # Check for flame core: high brightness, high red, high R-B difference
        b, g, r = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
        flame_mask = (r > 200) & (r > g) & (g > b) & ((r.astype(int) - b.astype(int)) > 50)
        flame_ratio = float(np.mean(flame_mask))

        # If predominantly skin and very low flame pixels
        return (skin_ratio >= 0.40) and (flame_ratio < 0.10)
    except Exception:
        return False


def compute_intersection_ratio(box_a: List[int], box_b: List[int]) -> float:
    """
    Computes what fraction of box_a is contained inside box_b.
    Ratio = Area(Intersection(A, B)) / Area(A)
    Range: 0.0 to 1.0
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection coordinates
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection_area = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return float(intersection_area) / float(area_a)


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Standard Intersection over Union (IoU)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection_area = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0
    return float(intersection_area) / float(union_area)


def compute_protrusion_ratio(hazard_box: List[int], screen_box: List[int]) -> float:
    """
    Computes what fraction of the hazard box protrudes outside the screen boundaries.
    Flames physically displayed on a phone cannot bleed past the physical chassis/bezel.
    Protrusion indicates flame extending around or behind the phone.
    Returns: maximum ratio of boundary protrusion (0.0 = completely inside screen).
    """
    hx1, hy1, hx2, hy2 = hazard_box
    sx1, sy1, sx2, sy2 = screen_box

    hw = max(1, hx2 - hx1)
    hh = max(1, hy2 - hy1)

    left_out = max(0, sx1 - hx1)
    right_out = max(0, hx2 - sx2)
    top_out = max(0, sy1 - hy1)
    bottom_out = max(0, hy2 - sy2)

    horiz_ratio = float(left_out + right_out) / float(hw)
    vert_ratio = float(top_out + bottom_out) / float(hh)

    return max(horiz_ratio, vert_ratio)


def compute_area_ratio(hazard_box: List[int], screen_box: List[int]) -> float:
    """Computes Area(hazard) / Area(screen)."""
    hw = max(1, hazard_box[2] - hazard_box[0])
    hh = max(1, hazard_box[3] - hazard_box[1])
    sw = max(1, screen_box[2] - screen_box[0])
    sh = max(1, screen_box[3] - screen_box[1])

    return float(hw * hh) / float(sw * sh)


class ScreenContextFilter:
    """
    Evaluates visual hazard detections in the context of detected screens.
    """

    def __init__(
        self,
        reject_threshold: float = 0.65,
        uncertain_threshold: float = 0.25,
        enabled: bool = True,
        min_screen_conf: float = 0.35,
        tilted_min_conf: float = 0.20,
        protrusion_margin_ratio: float = 0.15,
        fire_behind_screen_area_ratio: float = 0.85,
        reject_smoke_on_screens: bool = False,
        enable_face_smoke_filter: bool = True,
        min_person_conf: float = 0.35,
        face_overlap_reject_threshold: float = 0.65,
        face_overlap_uncertain_threshold: float = 0.30,
        face_smoke_calibrated_threshold: float = 0.72,
        face_uncertain_conf_floor: float = 0.55,
        smoke_plume_dispersion_threshold: float = 0.35,
        enable_white_object_filter: bool = True,
        enable_clothing_smoke_filter: bool = True,
        white_object_reject_confidence: float = 0.65,
        white_object_uncertain_confidence: float = 0.75,
        glare_saturation_threshold: float = 0.70,
        uniform_wall_std_threshold: float = 4.5,
    ):
        self.reject_threshold = reject_threshold
        self.uncertain_threshold = uncertain_threshold
        self.enabled = enabled
        self.min_screen_conf = min_screen_conf
        self.tilted_min_conf = tilted_min_conf
        self.protrusion_margin_ratio = protrusion_margin_ratio
        self.fire_behind_screen_area_ratio = fire_behind_screen_area_ratio
        self.reject_smoke_on_screens = reject_smoke_on_screens
        self.enable_face_smoke_filter = enable_face_smoke_filter
        self.min_person_conf = min_person_conf
        self.face_overlap_reject_threshold = face_overlap_reject_threshold
        self.face_overlap_uncertain_threshold = face_overlap_uncertain_threshold
        self.face_smoke_calibrated_threshold = face_smoke_calibrated_threshold
        self.face_uncertain_conf_floor = face_uncertain_conf_floor
        self.smoke_plume_dispersion_threshold = smoke_plume_dispersion_threshold
        self.enable_white_object_filter = enable_white_object_filter
        self.enable_clothing_smoke_filter = enable_clothing_smoke_filter
        self.white_object_reject_confidence = white_object_reject_confidence
        self.white_object_uncertain_confidence = white_object_uncertain_confidence
        self.glare_saturation_threshold = glare_saturation_threshold
        self.uniform_wall_std_threshold = uniform_wall_std_threshold

    def detect_persons_from_model(
        self,
        screen_model: Any,
        frame: np.ndarray,
        person_class_ids: Optional[List[int]] = None,
    ) -> List[PersonDetection]:
        """
        Detects persons and their estimated head/face regions from COCO model.
        """
        if screen_model is None or not self.enabled or not self.enable_face_smoke_filter:
            return []

        persons: List[PersonDetection] = []
        try:
            results = screen_model.predict(
                frame,
                conf=self.min_person_conf,
                verbose=False,
                classes=person_class_ids if person_class_ids else None,
            )
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    name = r.names.get(cls_id, f"class_{cls_id}").lower()
                    if name == "person" or (person_class_ids and cls_id in person_class_ids):
                        head_box = estimate_head_bbox(xyxy)
                        persons.append(
                            PersonDetection(
                                bbox=xyxy,
                                confidence=conf,
                                head_bbox=head_box,
                                label=name,
                            )
                        )
        except Exception as e:
            logger.error(f"Error detecting persons from model: {e}")

        return persons

    def detect_context_from_model(
        self,
        screen_model: Any,
        frame: np.ndarray,
        screen_class_ids: Optional[List[int]] = None,
        person_class_ids: Optional[List[int]] = None,
    ) -> Tuple[List[ScreenDetection], List[PersonDetection]]:
        """
        Unified context detection in a single forward pass:
        Detects both screens (phones, monitors, laptops) and persons/faces.
        """
        if screen_model is None or not self.enabled:
            return [], []

        screens: List[ScreenDetection] = []
        persons: List[PersonDetection] = []

        target_classes = []
        if screen_class_ids:
            target_classes.extend(screen_class_ids)
        if person_class_ids and self.enable_face_smoke_filter:
            target_classes.extend(person_class_ids)

        try:
            query_conf = min(self.min_screen_conf, self.tilted_min_conf, self.min_person_conf)
            results = screen_model.predict(
                frame,
                conf=query_conf,
                verbose=False,
                classes=target_classes if target_classes else None,
            )
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    name = r.names.get(cls_id, f"class_{cls_id}").lower()

                    if person_class_ids and cls_id in person_class_ids:
                        if conf >= self.min_person_conf and self.enable_face_smoke_filter:
                            head_box = estimate_head_bbox(xyxy)
                            persons.append(
                                PersonDetection(
                                    bbox=xyxy,
                                    confidence=conf,
                                    head_bbox=head_box,
                                    label=name,
                                )
                            )
                    elif (screen_class_ids and cls_id in screen_class_ids) or (not screen_class_ids and name != "person"):
                        is_low = (conf < self.min_screen_conf)
                        screens.append(
                            ScreenDetection(
                                bbox=xyxy,
                                label=name,
                                confidence=conf,
                                detector_source="model",
                                is_low_confidence=is_low,
                            )
                        )
        except Exception as e:
            logger.error(f"Error executing unified context detector model: {e}")

        return screens, persons

    def detect_screens_from_model(
        self,
        screen_model: Any,
        frame: np.ndarray,
        target_class_ids: Optional[List[int]] = None,
    ) -> List[ScreenDetection]:
        """
        Run inference using the screen detector model (e.g. COCO YOLOv8) to find
        phones, laptops, and TV monitors in the current frame.
        Captures both confident screens and tilted / occluded screens with lower confidence.
        """
        if screen_model is None or not self.enabled:
            return []

        screens: List[ScreenDetection] = []
        try:
            # Query down to tilted_min_conf floor to catch angled or occluded devices
            query_conf = min(self.min_screen_conf, self.tilted_min_conf)
            results = screen_model.predict(
                frame,
                conf=query_conf,
                verbose=False,
                classes=target_class_ids if target_class_ids else None,
            )

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    name = r.names.get(cls_id, f"class_{cls_id}").lower()

                    is_low = (conf < self.min_screen_conf)
                    screens.append(
                        ScreenDetection(
                            bbox=xyxy,
                            label=name,
                            confidence=conf,
                            detector_source="model",
                            is_low_confidence=is_low,
                        )
                    )
        except Exception as e:
            logger.error(f"Error executing screen detector model: {e}")

        return screens

    def detect_display_contours_fallback(
        self,
        frame: np.ndarray,
        hazard_boxes: List[List[int]],
    ) -> List[ScreenDetection]:
        """
        Geometric display contour fallback:
        Locates prominent quadrilateral bezels with mobile/screen aspect ratios
        directly surrounding suspected hazard boxes.
        Avoids rejecting arbitrary rectangles by only looking in candidate zones.
        """
        if frame is None or len(hazard_boxes) == 0:
            return []

        display_screens: List[ScreenDetection] = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Bilateral filter to preserve strong screen borders while smoothing noise
            blurred = cv2.bilateralFilter(gray, 7, 50, 50)
            edges = cv2.Canny(blurred, 50, 150)

            # Morphological close to bridge small gaps in phone border
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h, w = frame.shape[:2]
            frame_area = h * w

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore tiny or giant contours (phone screens are usually between 2% and 60% of frame)
                if area < (frame_area * 0.02) or area > (frame_area * 0.70):
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                # Check if quadrilateral
                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    aspect = float(bw) / float(bh) if bh > 0 else 0
                    # Standard phone / screen aspect ratios (portrait 0.4-0.7, landscape 1.3-2.3)
                    is_screen_aspect = (0.4 <= aspect <= 0.75) or (1.3 <= aspect <= 2.3)
                    if not is_screen_aspect:
                        continue

                    screen_box = [x, y, x + bw, y + bh]

                    # Only register if it encloses at least one hazard candidate
                    for h_box in hazard_boxes:
                        overlap = compute_intersection_ratio(h_box, screen_box)
                        if overlap >= self.uncertain_threshold:
                            display_screens.append(
                                ScreenDetection(
                                    bbox=screen_box,
                                    label="display_contour",
                                    confidence=0.55,
                                    detector_source="contour_analysis",
                                )
                            )
                            break
        except Exception as e:
            logger.warning(f"Display contour analysis error: {e}")

        return display_screens

    def evaluate_white_object_features(
        self,
        frame: Optional[np.ndarray],
        bbox: List[int],
        confidence: float,
    ) -> Tuple[bool, bool, str]:
        """
        Evaluates visual characteristics of a smoke candidate to prevent false detections
        from white walls, paper, bright window glare, and flat surfaces, without rejecting
        genuine amorphous, billowing smoke plumes.

        Returns: (is_rejected: bool, is_uncertain: bool, reason: str)
        """
        if frame is None or not self.enable_white_object_filter:
            return False, False, ""

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return False, False, ""

        patch = frame[y1:y2, x1:x2]
        patch_h, patch_w = patch.shape[:2]
        patch_area = patch_h * patch_w

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat))

        # 1. Optical Glare / Specular Saturation (Sunlight through window, bright lightbulb)
        # Pure blown-out white pixels (V >= 248, Sat <= 25)
        glare_mask = (hsv[:, :, 2] >= 248) & (sat <= 25)
        glare_ratio = float(np.sum(glare_mask)) / float(patch_area)

        if glare_ratio >= self.glare_saturation_threshold and std_val < 8.0:
            if confidence < self.white_object_reject_confidence:
                return True, False, "Rejected: optical glare / saturated overexposed region"
            elif confidence < self.white_object_uncertain_confidence:
                return False, True, "Uncertain: bright saturated glare area"

        # 2. Flat Uniform White / Light Surface (Plain painted wall, ceiling, whiteboard)
        # Near zero texture variance and high brightness with no saturation
        if mean_val >= 230 and mean_sat <= 15 and std_val < self.uniform_wall_std_threshold:
            if confidence < 0.60:
                return True, False, "Rejected: flat uniform white surface (wall/whiteboard)"
            elif confidence < 0.72:
                return False, True, "Uncertain: uniform white surface with low contrast"

        # 3. Rigid Rectilinear Objects (White paper, notebook, framed window, rigid furniture)
        # Rigid objects have straight linear edges forming 4-sided convex polygons or Hough lines
        edges = cv2.Canny(gray, 40, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_rigid_rect = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (0.18 * patch_area):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    has_rigid_rect = True
                    break

        if not has_rigid_rect:
            min_dim = min(patch_h, patch_w)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_dim // 3, maxLineGap=12)
            if lines is not None and len(lines) >= 4:
                hv_lines = 0
                for line in lines:
                    pts = np.asarray(line).reshape(-1)
                    if len(pts) >= 4:
                        lx1, ly1, lx2, ly2 = pts[:4]
                        dx = abs(int(lx2) - int(lx1))
                        dy = abs(int(ly2) - int(ly1))
                        angle = np.degrees(np.arctan2(dy, dx))
                        if angle <= 12 or angle >= 78:
                            hv_lines += 1
                if hv_lines >= 3:
                    has_rigid_rect = True

        if has_rigid_rect and (mean_val >= 160 or mean_sat <= 30):
            if confidence < self.white_object_reject_confidence:
                return True, False, "Rejected: rigid rectilinear object (paper/notebook/window frame)"
            elif confidence < self.white_object_uncertain_confidence:
                return False, True, "Uncertain: rigid white object boundary"

        return False, False, ""

    def evaluate_detections(
        self,
        hazard_candidates: List[dict],
        detected_screens: List[ScreenDetection],
        detected_persons: Optional[List[PersonDetection]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[DetectionValidationResult]:
        """
        Evaluate each hazard candidate against all detected screens and persons.

        hazard_candidates: list of dicts with keys:
            - 'class_name': 'fire' or 'smoke'
            - 'confidence': float
            - 'bbox': [x1, y1, x2, y2]

        Returns:
            List of DetectionValidationResult, one for each hazard candidate.
        """
        results: List[DetectionValidationResult] = []
        has_fire_in_frame = any(h.get("class_name") == "fire" for h in hazard_candidates)

        for candidate in hazard_candidates:
            c_name = candidate["class_name"]
            c_conf = float(candidate["confidence"])
            c_bbox = [int(v) for v in candidate["bbox"]]

            # --- A0. FACE CONTEXT EVALUATION (FOR FALSE FIRE ON HUMAN FACES) ---
            # Ensures human faces are never displayed as fire, while genuine fires
            # near persons are preserved.
            if (
                c_name == "fire"
                and self.enabled
                and detected_persons
                and frame is not None
            ):
                rejected_as_face = False
                for person in detected_persons:
                    h_overlap = compute_intersection_ratio(c_bbox, person.head_bbox)
                    area_ratio = compute_area_ratio(c_bbox, person.head_bbox)
                    if h_overlap >= 0.85 and area_ratio <= 1.05:
                        if is_skin_face_patch(frame, c_bbox):
                            results.append(
                                DetectionValidationResult(
                                    class_name=c_name,
                                    confidence=c_conf,
                                    bbox=c_bbox,
                                    status=HazardStatus.FACE_REJECTED,
                                    overlap_ratio=h_overlap,
                                    associated_person=person,
                                    reason="Fire detection rejected: strictly coincides with human face skin region without flame characteristics.",
                                )
                            )
                            rejected_as_face = True
                            break
                if rejected_as_face:
                    continue

            # --- A. FACE & PERSON CONTEXT EVALUATION (FOR SMOKE) ---
            # Smoke predicted directly on a human face is a common false-positive due to
            # model confusion on soft skin gradients. Evaluate evidence-based calibration.
            if (
                c_name == "smoke"
                and self.enabled
                and self.enable_face_smoke_filter
                and detected_persons
                and not has_fire_in_frame
            ):
                max_head_overlap = 0.0
                best_person: Optional[PersonDetection] = None

                for person in detected_persons:
                    h_overlap = compute_intersection_ratio(c_bbox, person.head_bbox)
                    if h_overlap > max_head_overlap:
                        max_head_overlap = h_overlap
                        best_person = person

                if best_person and max_head_overlap >= self.face_overlap_uncertain_threshold:
                    p_overlap = compute_intersection_ratio(c_bbox, best_person.bbox)
                    external_plume_ratio = max(0.0, 1.0 - p_overlap)
                    head_area = max(
                        1,
                        (best_person.head_bbox[2] - best_person.head_bbox[0])
                        * (best_person.head_bbox[3] - best_person.head_bbox[1]),
                    )
                    smoke_area = max(1, (c_bbox[2] - c_bbox[0]) * (c_bbox[3] - c_bbox[1]))
                    area_ratio = float(smoke_area) / float(head_area)

                    # 1. Genuine environmental plume or very high confidence
                    if (
                        c_conf >= self.face_smoke_calibrated_threshold
                        or external_plume_ratio >= self.smoke_plume_dispersion_threshold
                    ):
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.CONFIRMED_ENVIRONMENTAL,
                                overlap_ratio=max_head_overlap,
                                associated_person=best_person,
                                reason=(
                                    f"Environmental smoke confirmed: plume dispersion ({external_plume_ratio * 100:.1f}% external) "
                                    f"or high calibrated confidence ({c_conf:.2f} >= {self.face_smoke_calibrated_threshold:.2f})."
                                ),
                            )
                        )
                        continue

                    # 2. Confined Person/Face/Clothing Artifact: strictly inside person/head/clothing, low-to-moderate confidence
                    elif (
                        (max_head_overlap >= self.face_overlap_reject_threshold or p_overlap >= 0.75)
                        and c_conf < max(self.face_uncertain_conf_floor, 0.60)
                        and external_plume_ratio < 0.25
                    ):
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.FACE_REJECTED,
                                overlap_ratio=max_head_overlap,
                                associated_person=best_person,
                                reason=(
                                    f"Smoke detection rejected: strictly coincides with human face/head/clothing "
                                    f"({p_overlap * 100:.1f}% inside person, conf {c_conf:.2f} < 0.60) "
                                    f"without environmental plume dispersion."
                                ),
                            )
                        )
                        continue

                    # 3. Face-Proximity Uncertain: ambiguous overlap or intermediate confidence
                    else:
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.FACE_UNCERTAIN,
                                overlap_ratio=max_head_overlap,
                                associated_person=best_person,
                                reason=(
                                    f"Smoke detection overlaps human face/head ({max_head_overlap * 100:.1f}% overlap, conf {c_conf:.2f}). "
                                    f"Maintained as UNCERTAIN without false environmental alarm."
                                ),
                            )
                        )
                        continue

                elif best_person and self.enable_clothing_smoke_filter:
                    # Torso / clothing evaluation on person
                    torso_box = best_person.torso_bbox if best_person.torso_bbox else estimate_torso_bbox(best_person.bbox)
                    t_overlap = compute_intersection_ratio(c_bbox, torso_box)
                    b_overlap = compute_intersection_ratio(c_bbox, best_person.bbox)
                    max_body_overlap = max(t_overlap, b_overlap)

                    if max_body_overlap >= 0.35:
                        external_plume_ratio = max(0.0, 1.0 - b_overlap)
                        if (
                            c_conf >= self.face_smoke_calibrated_threshold
                            or external_plume_ratio >= self.smoke_plume_dispersion_threshold
                        ):
                            results.append(
                                DetectionValidationResult(
                                    class_name=c_name,
                                    confidence=c_conf,
                                    bbox=c_bbox,
                                    status=HazardStatus.CONFIRMED_ENVIRONMENTAL,
                                    overlap_ratio=max_body_overlap,
                                    associated_person=best_person,
                                    reason=(
                                        f"Environmental smoke confirmed near person: plume dispersion ({external_plume_ratio * 100:.1f}% external) "
                                        f"or high calibrated confidence ({c_conf:.2f} >= {self.face_smoke_calibrated_threshold:.2f})."
                                    ),
                                )
                            )
                            continue
                        elif max_body_overlap >= 0.55 and c_conf < 0.60 and external_plume_ratio < 0.25:
                            results.append(
                                DetectionValidationResult(
                                    class_name=c_name,
                                    confidence=c_conf,
                                    bbox=c_bbox,
                                    status=HazardStatus.FACE_REJECTED,
                                    overlap_ratio=max_body_overlap,
                                    associated_person=best_person,
                                    reason=(
                                        f"Smoke detection rejected: strictly coincides with clothing/torso on person "
                                        f"({max_body_overlap * 100:.1f}% overlap, conf {c_conf:.2f} < 0.60) "
                                        f"without environmental plume dispersion."
                                    ),
                                )
                            )
                            continue
                        else:
                            results.append(
                                DetectionValidationResult(
                                    class_name=c_name,
                                    confidence=c_conf,
                                    bbox=c_bbox,
                                    status=HazardStatus.FACE_UNCERTAIN,
                                    overlap_ratio=max_body_overlap,
                                    associated_person=best_person,
                                    reason=(
                                        f"Smoke detection overlaps clothing/torso on person ({max_body_overlap * 100:.1f}% overlap, conf {c_conf:.2f}). "
                                        f"Maintained as UNCERTAIN without false environmental alarm."
                                    ),
                                )
                            )
                            continue

            if not self.enabled or not detected_screens:
                # Screen rejection disabled or no screens detected in frame
                if c_name == "smoke" and self.enable_white_object_filter and frame is not None and not has_fire_in_frame:
                    w_rej, w_unc, w_reason = self.evaluate_white_object_features(frame, c_bbox, c_conf)
                    if w_rej:
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.WHITE_OBJECT_REJECTED,
                                reason=w_reason,
                            )
                        )
                        continue
                    elif w_unc:
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.WHITE_OBJECT_UNCERTAIN,
                                reason=w_reason,
                            )
                        )
                        continue

                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.CONFIRMED_ENVIRONMENTAL,
                        overlap_ratio=0.0,
                        associated_screen=None,
                        reason="No screen detected in frame; environmental hazard candidate",
                    )
                )
                continue

            # Find the screen with the highest containment / overlap ratio for this hazard
            max_overlap = 0.0
            best_screen: Optional[ScreenDetection] = None

            for screen in detected_screens:
                overlap = compute_intersection_ratio(c_bbox, screen.bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_screen = screen

            # Evaluate against thresholds
            closest_label = best_screen.label if best_screen else (detected_screens[0].label if detected_screens else "screen")

            # Calculate relative metrics if a screen was detected in frame
            protrusion = compute_protrusion_ratio(c_bbox, best_screen.bbox) if best_screen else 0.0
            area_ratio = compute_area_ratio(c_bbox, best_screen.bbox) if best_screen else 0.0
            screen_coverage = compute_intersection_ratio(best_screen.bbox, c_bbox) if best_screen else 0.0

            # 1. SPECIAL CASE: Smoke overlapping screens/monitors
            # Smoke is atmospheric, translucent, and voluminous. Never reject smoke overlapping a TV/screen as fake.
            if c_name == "smoke" and not self.reject_smoke_on_screens:
                if max_overlap >= self.uncertain_threshold:
                    reason = (
                        f"Atmospheric smoke detected overlapping digital display ({best_screen.label}, "
                        f"{max_overlap * 100:.1f}% overlap). Maintained as UNCERTAIN to prevent false dismissal of genuine smoke."
                    )
                    results.append(
                        DetectionValidationResult(
                            class_name=c_name,
                            confidence=c_conf,
                            bbox=c_bbox,
                            status=HazardStatus.SCREEN_UNCERTAIN,
                            overlap_ratio=max_overlap,
                            associated_screen=best_screen,
                            reason=reason,
                        )
                    )
                else:
                    reason = (
                        f"Atmospheric smoke is environmental. Overlap with closest screen ({closest_label}) "
                        f"is {max_overlap * 100:.1f}%, below uncertain threshold ({self.uncertain_threshold * 100:.1f}%)."
                    )
                    results.append(
                        DetectionValidationResult(
                            class_name=c_name,
                            confidence=c_conf,
                            bbox=c_bbox,
                            status=HazardStatus.CONFIRMED_ENVIRONMENTAL,
                            overlap_ratio=max_overlap,
                            associated_screen=best_screen,
                            reason=reason,
                        )
                    )
                continue

            # 2. SPECIAL CASE: Suspected tilted or occluded phone detected with weak confidence
            if best_screen and best_screen.is_low_confidence and max_overlap >= self.uncertain_threshold:
                reason = (
                    f"Hazard in proximity to suspected tilted/occluded {best_screen.label} "
                    f"(marginal detector confidence {best_screen.confidence:.2f}, overlap {max_overlap * 100:.1f}%). "
                    f"Marked UNCERTAIN for security verification."
                )
                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.SCREEN_UNCERTAIN,
                        overlap_ratio=max_overlap,
                        associated_screen=best_screen,
                        reason=reason,
                    )
                )
                continue

            # 3. SPECIAL CASE: Genuine fire behind or engulfing the screen
            # A fire displayed on a screen is strictly bounded by the physical display bezels.
            # If flames protrude noticeably outside the phone borders, or the flame area exceeds the phone,
            # genuine fire may be located behind or around the screen.
            is_fire_behind_screen = (
                best_screen is not None
                and (max_overlap >= self.uncertain_threshold or screen_coverage >= 0.60)
                and (protrusion > self.protrusion_margin_ratio or area_ratio > self.fire_behind_screen_area_ratio)
            )

            if is_fire_behind_screen:
                reason = (
                    f"Fire overlaps {best_screen.label} ({max_overlap * 100:.1f}%), but boundary protrudes "
                    f"{protrusion * 100:.1f}% beyond screen bezel (area ratio {area_ratio:.2f}). "
                    f"Possible genuine fire behind or around screen; marked UNCERTAIN for review."
                )
                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.SCREEN_UNCERTAIN,
                        overlap_ratio=max_overlap,
                        associated_screen=best_screen,
                        reason=reason,
                    )
                )
                continue

            # 4. HIGH OVERLAP EVALUATION (>= reject_threshold, default 65%)
            if max_overlap >= self.reject_threshold and best_screen:
                # Contained predominantly within a mobile phone or display bezel without outward protrusion
                reason = (
                    f"Hazard bbox is {max_overlap * 100:.1f}% contained within {best_screen.label} "
                    f"(conf {best_screen.confidence:.2f}) with enclosed bezels. Rejected as screen-displayed false positive."
                )
                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.SCREEN_REJECTED,
                        overlap_ratio=max_overlap,
                        associated_screen=best_screen,
                        reason=reason,
                    )
                )

            elif max_overlap >= self.uncertain_threshold and best_screen:
                # Ambiguous boundary: hazard partially intersects screen
                reason = (
                    f"Hazard bbox has {max_overlap * 100:.1f}% overlap with {best_screen.label}. "
                    f"Ambiguous boundary marked UNCERTAIN for security staff review."
                )
                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.SCREEN_UNCERTAIN,
                        overlap_ratio=max_overlap,
                        associated_screen=best_screen,
                        reason=reason,
                    )
                )
            else:
                # Hazard is outside the screen (fire near a screen is NOT discarded)
                if c_name == "smoke" and self.enable_white_object_filter and frame is not None and not has_fire_in_frame:
                    w_rej, w_unc, w_reason = self.evaluate_white_object_features(frame, c_bbox, c_conf)
                    if w_rej:
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.WHITE_OBJECT_REJECTED,
                                reason=w_reason,
                            )
                        )
                        continue
                    elif w_unc:
                        results.append(
                            DetectionValidationResult(
                                class_name=c_name,
                                confidence=c_conf,
                                bbox=c_bbox,
                                status=HazardStatus.WHITE_OBJECT_UNCERTAIN,
                                reason=w_reason,
                            )
                        )
                        continue

                reason = (
                    f"Hazard is environmental. Overlap with closest screen ({closest_label}) "
                    f"is {max_overlap * 100:.1f}%, below uncertain threshold ({self.uncertain_threshold * 100:.1f}%)."
                )
                results.append(
                    DetectionValidationResult(
                        class_name=c_name,
                        confidence=c_conf,
                        bbox=c_bbox,
                        status=HazardStatus.CONFIRMED_ENVIRONMENTAL,
                        overlap_ratio=max_overlap,
                        associated_screen=best_screen,
                        reason=reason,
                    )
                )

        return results
