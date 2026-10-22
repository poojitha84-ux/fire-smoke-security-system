"""
Temporal Confirmation and Persistence Filter.

Prevents single-frame transient noise, flicker, or optical glitches from triggering
false alarms by requiring consistent detection across a sliding temporal window of live frames.
Also handles smooth state recovery when a hazard extinguishes or clears.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class HazardState(str, Enum):
    """Hazard confirmation state."""
    NORMAL = "normal"          # No hazard detected
    SUSPECTED = "suspected"    # Intermittent detection; accumulating evidence
    CONFIRMED = "confirmed"    # Confirmed hazard meeting temporal threshold


@dataclass
class FrameHazardRecord:
    """Detection state for a single processed frame."""
    frame_index: int
    detected: bool
    max_confidence: float
    detection_count: int
    bboxes: List[List[int]] = field(default_factory=list)


class ClassTemporalTracker:
    """Tracks temporal persistence for a single hazard class (e.g. fire or smoke)."""

    def __init__(
        self,
        hazard_class: str,
        window_size: int = 8,
        confirm_count: int = 5,
        clear_consecutive_frames: int = 15,
    ):
        self.hazard_class = hazard_class
        self.window_size = window_size
        self.confirm_count = min(confirm_count, window_size)
        self.clear_consecutive_frames = clear_consecutive_frames

        self.history: Deque[FrameHazardRecord] = deque(maxlen=window_size)
        self.state: HazardState = HazardState.NORMAL
        self.consecutive_clear_count: int = 0
        self.total_frames_processed: int = 0
        self.last_confirmed_frame: int = -1

    def update(
        self,
        frame_index: int,
        detected: bool,
        max_confidence: float = 0.0,
        bboxes: Optional[List[List[int]]] = None,
    ) -> HazardState:
        """
        Record current frame's environmental detection and return updated state.
        """
        self.total_frames_processed += 1
        record = FrameHazardRecord(
            frame_index=frame_index,
            detected=detected,
            max_confidence=max_confidence if detected else 0.0,
            detection_count=len(bboxes) if bboxes else (1 if detected else 0),
            bboxes=bboxes or [],
        )
        self.history.append(record)

        # Count positive frames in sliding window
        positive_frames = sum(1 for r in self.history if r.detected)

        previous_state = self.state

        if detected:
            self.consecutive_clear_count = 0
        else:
            self.consecutive_clear_count += 1

        # State transition logic
        if positive_frames >= self.confirm_count:
            self.state = HazardState.CONFIRMED
            self.last_confirmed_frame = frame_index
        elif positive_frames > 0:
            if previous_state == HazardState.CONFIRMED:
                # In confirmed state: remain confirmed until clear threshold reached
                if self.consecutive_clear_count >= self.clear_consecutive_frames:
                    self.state = HazardState.NORMAL
                else:
                    self.state = HazardState.CONFIRMED
            else:
                self.state = HazardState.SUSPECTED
        else:
            # 0 detections in window
            if previous_state == HazardState.CONFIRMED and self.consecutive_clear_count < self.clear_consecutive_frames:
                self.state = HazardState.CONFIRMED
            else:
                self.state = HazardState.NORMAL

        if self.state != previous_state:
            logger.info(
                f"[{self.hazard_class.upper()}] State changed: {previous_state.value} -> {self.state.value} "
                f"(window: {positive_frames}/{len(self.history)}, clear_consecutive: {self.consecutive_clear_count})"
            )

        return self.state

    def reset(self):
        """Reset history and state to normal."""
        self.history.clear()
        self.state = HazardState.NORMAL
        self.consecutive_clear_count = 0

    @property
    def positive_ratio(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.detected) / len(self.history)

    @property
    def latest_confidence(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1].max_confidence


class TemporalFilter:
    """
    Manages temporal persistence across multiple hazard classes (e.g. fire, smoke)
    and tracks uncertain screen detections.
    """

    def __init__(
        self,
        window_size: int = 8,
        confirm_count: int = 5,
        clear_consecutive_frames: int = 15,
    ):
        self.trackers: Dict[str, ClassTemporalTracker] = {
            "fire": ClassTemporalTracker("fire", window_size, confirm_count, clear_consecutive_frames),
            "smoke": ClassTemporalTracker("smoke", window_size, confirm_count, clear_consecutive_frames),
        }
        self.uncertain_history: Deque[bool] = deque(maxlen=window_size)
        self.frame_counter: int = 0

    def update(
        self,
        environmental_fire_bboxes: List[List[int]],
        environmental_fire_conf: float,
        environmental_smoke_bboxes: List[List[int]],
        environmental_smoke_conf: float,
        has_uncertain_detection: bool = False,
    ) -> Dict[str, HazardState]:
        """
        Process current frame detections through class trackers.
        """
        self.frame_counter += 1

        fire_state = self.trackers["fire"].update(
            frame_index=self.frame_counter,
            detected=len(environmental_fire_bboxes) > 0,
            max_confidence=environmental_fire_conf,
            bboxes=environmental_fire_bboxes,
        )

        smoke_state = self.trackers["smoke"].update(
            frame_index=self.frame_counter,
            detected=len(environmental_smoke_bboxes) > 0,
            max_confidence=environmental_smoke_conf,
            bboxes=environmental_smoke_bboxes,
        )

        self.uncertain_history.append(has_uncertain_detection)

        return {
            "fire": fire_state,
            "smoke": smoke_state,
        }

    def is_fire_confirmed(self) -> bool:
        return self.trackers["fire"].state == HazardState.CONFIRMED

    def is_smoke_confirmed(self) -> bool:
        return self.trackers["smoke"].state == HazardState.CONFIRMED

    def is_any_confirmed(self) -> bool:
        return self.is_fire_confirmed() or self.is_smoke_confirmed()

    def has_active_uncertainty(self) -> bool:
        return any(self.uncertain_history)

    def get_summary(self) -> dict:
        return {
            "frame_counter": self.frame_counter,
            "fire": {
                "state": self.trackers["fire"].state.value,
                "positive_ratio": round(self.trackers["fire"].positive_ratio, 2),
                "latest_conf": round(self.trackers["fire"].latest_confidence, 2),
                "consecutive_clear": self.trackers["fire"].consecutive_clear_count,
            },
            "smoke": {
                "state": self.trackers["smoke"].state.value,
                "positive_ratio": round(self.trackers["smoke"].positive_ratio, 2),
                "latest_conf": round(self.trackers["smoke"].latest_confidence, 2),
                "consecutive_clear": self.trackers["smoke"].consecutive_clear_count,
            },
            "uncertain_active": self.has_active_uncertainty(),
        }

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.uncertain_history.clear()
        self.frame_counter = 0
