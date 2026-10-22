"""
Unit tests for Temporal Filter and Confirmation Logic.
"""

import pytest
from detector.temporal_filter import TemporalFilter, HazardState, ClassTemporalTracker


def test_single_frame_prediction_does_not_confirm():
    """A single frame prediction must NOT raise a confirmed alert."""
    tracker = ClassTemporalTracker(hazard_class="fire", window_size=8, confirm_count=5)

    # Frame 1: positive detection
    state = tracker.update(frame_index=1, detected=True, max_confidence=0.85)
    assert state == HazardState.SUSPECTED

    # Frame 2: no detection
    state = tracker.update(frame_index=2, detected=False)
    assert state in (HazardState.SUSPECTED, HazardState.NORMAL)


def test_temporal_confirmation_threshold():
    """Consistent detections reaching 5 of 8 frames must confirm hazard."""
    tracker = ClassTemporalTracker(hazard_class="fire", window_size=8, confirm_count=5)

    # 4 positive frames: still suspected
    for i in range(1, 5):
        state = tracker.update(frame_index=i, detected=True, max_confidence=0.8)
        assert state == HazardState.SUSPECTED

    # 5th positive frame: reaches threshold -> CONFIRMED
    state = tracker.update(frame_index=5, detected=True, max_confidence=0.82)
    assert state == HazardState.CONFIRMED


def test_intermittent_glitches_rejected():
    """Intermittent flicker (e.g. 1 on, 3 off) does not reach confirmation."""
    tracker = ClassTemporalTracker(hazard_class="fire", window_size=8, confirm_count=5)

    sequence = [True, False, False, True, False, False, False, True, False]
    for idx, detected in enumerate(sequence):
        state = tracker.update(frame_index=idx, detected=detected)
        assert state != HazardState.CONFIRMED


def test_hazard_clear_recovery():
    """When hazard disappears for clear_consecutive_frames, state resets to NORMAL."""
    clear_threshold = 10
    tracker = ClassTemporalTracker(
        hazard_class="fire",
        window_size=8,
        confirm_count=5,
        clear_consecutive_frames=clear_threshold,
    )

    # Confirm hazard
    for i in range(6):
        tracker.update(frame_index=i, detected=True, max_confidence=0.8)
    assert tracker.state == HazardState.CONFIRMED

    # Follow with consecutive clear frames
    for i in range(6, 6 + clear_threshold - 1):
        tracker.update(frame_index=i, detected=False)
        # Still confirmed during clear countdown
        assert tracker.state == HazardState.CONFIRMED

    # 10th consecutive clear frame triggers recovery
    tracker.update(frame_index=6 + clear_threshold, detected=False)
    assert tracker.state == HazardState.NORMAL


def test_multiclass_temporal_filter_independence():
    """Fire and smoke must be tracked independently."""
    filter_engine = TemporalFilter(window_size=8, confirm_count=5, clear_consecutive_frames=10)

    # Feed smoke detections only
    for _ in range(5):
        states = filter_engine.update(
            environmental_fire_bboxes=[],
            environmental_fire_conf=0.0,
            environmental_smoke_bboxes=[[10, 10, 100, 100]],
            environmental_smoke_conf=0.75,
        )

    assert states["smoke"] == HazardState.CONFIRMED
    assert states["fire"] == HazardState.NORMAL
    assert filter_engine.is_smoke_confirmed() is True
    assert filter_engine.is_fire_confirmed() is False
