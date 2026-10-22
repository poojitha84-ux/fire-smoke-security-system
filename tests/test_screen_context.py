"""
Unit and Integration Tests for Mobile Phone & Screen False-Positive Prevention.
Tests all scenarios:
- Fire displayed on a phone
- Genuine fire outside a phone
- Genuine fire behind a phone (protruding boundary / larger area)
- Genuine fire beside a monitor
- Smoke overlapping a television (must not be rejected)
- Tilted phone with weak screen-detector confidence
- Missing phone detection policy
- Multiple simultaneous hazards
"""

import pytest

from detector.screen_context import (
    ScreenContextFilter,
    ScreenDetection,
    HazardStatus,
    compute_intersection_ratio,
    compute_protrusion_ratio,
    compute_area_ratio,
    compute_iou,
)


def test_intersection_and_protrusion_calculations():
    # Box A completely inside Box B
    box_a = [100, 100, 200, 200]
    box_b = [50, 50, 300, 300]
    assert compute_intersection_ratio(box_a, box_b) == 1.0
    assert compute_protrusion_ratio(box_a, box_b) == 0.0

    # Fire protruding out of screen:
    # Screen is [100, 100, 200, 200], Fire is [80, 100, 220, 200]
    # Fire width = 140, left protrusion = 20, right protrusion = 20
    # Horiz protrusion ratio = 40/140 = 0.285 (28.5%)
    fire_protruding = [80, 100, 220, 200]
    screen_box = [100, 100, 200, 200]
    protrusion = compute_protrusion_ratio(fire_protruding, screen_box)
    assert pytest.approx(protrusion, 0.02) == 0.285


def test_fire_displayed_on_phone_rejected():
    """A phone showing fire enclosed inside its display must be rejected as a screen false positive."""
    filter_engine = ScreenContextFilter(
        reject_threshold=0.65,
        uncertain_threshold=0.25,
        protrusion_margin_ratio=0.15,
    )

    phone_screen = ScreenDetection(
        bbox=[100, 100, 300, 400],
        label="cell phone",
        confidence=0.88,
    )

    # Fire detection strictly inside the phone bezel
    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.82, "bbox": [120, 130, 280, 370]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [phone_screen])

    assert len(results) == 1
    res = results[0]
    assert res.status == HazardStatus.SCREEN_REJECTED
    assert res.overlap_ratio >= 0.90
    assert res.associated_screen.label == "cell phone"
    assert "rejected" in res.reason.lower()


def test_genuine_fire_outside_phone_confirmed():
    """Genuine fire outside any phone screen must remain environmental."""
    filter_engine = ScreenContextFilter(reject_threshold=0.65, uncertain_threshold=0.25)

    phone_screen = ScreenDetection(
        bbox=[50, 50, 150, 250],
        label="cell phone",
        confidence=0.85,
    )

    # Fire is across the room at [400, 300, 500, 400]
    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.78, "bbox": [400, 300, 500, 400]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [phone_screen])

    assert len(results) == 1
    res = results[0]
    assert res.status == HazardStatus.CONFIRMED_ENVIRONMENTAL
    assert res.overlap_ratio == 0.0


def test_genuine_fire_behind_phone_marked_uncertain():
    """
    When genuine fire is behind a phone, the 2D overlap might be high, but flames
    protrude past the phone edges or the fire area is larger than the phone chassis.
    Must be classified as SCREEN_UNCERTAIN rather than silently rejected!
    """
    filter_engine = ScreenContextFilter(
        reject_threshold=0.65,
        uncertain_threshold=0.25,
        protrusion_margin_ratio=0.15,
        fire_behind_screen_area_ratio=0.85,
    )

    phone_screen = ScreenDetection(
        bbox=[200, 200, 360, 450],  # Width: 160, Height: 250
        label="cell phone",
        confidence=0.85,
    )

    # Fire behind phone: visually overlaps screen box, but extends beyond phone's
    # top, left, and right borders: [150, 140, 410, 440]
    # Fire width: 260 (extends 50px left, 50px right)
    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.91, "bbox": [150, 140, 410, 440]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [phone_screen])

    assert len(results) == 1
    res = results[0]
    # MUST be UNCERTAIN, NOT REJECTED!
    assert res.status == HazardStatus.SCREEN_UNCERTAIN
    assert "behind" in res.reason.lower() or "protrudes" in res.reason.lower()


def test_genuine_fire_beside_monitor_confirmed():
    """Fire near a phone or TV should NOT be discarded when overlap is minimal."""
    filter_engine = ScreenContextFilter(reject_threshold=0.65, uncertain_threshold=0.25)

    tv_screen = ScreenDetection(
        bbox=[100, 100, 300, 300],
        label="tv",
        confidence=0.90,
    )

    # Wastebasket fire sitting right beside the monitor: [290, 200, 390, 300] (~10% overlap)
    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.85, "bbox": [290, 200, 390, 300]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [tv_screen])

    assert len(results) == 1
    res = results[0]
    assert res.status == HazardStatus.CONFIRMED_ENVIRONMENTAL
    assert res.overlap_ratio < 0.25


def test_smoke_overlapping_television_not_rejected():
    """
    CRITICAL SAFETY REQUIREMENT:
    Atmospheric smoke overlapping a TV or monitor must NOT be automatically rejected as fake.
    """
    filter_engine = ScreenContextFilter(
        reject_threshold=0.65,
        uncertain_threshold=0.25,
        reject_smoke_on_screens=False,  # Default safety policy
    )

    tv_screen = ScreenDetection(
        bbox=[100, 100, 500, 400],
        label="tv",
        confidence=0.92,
    )

    # Smoke drifting in front of the TV: [120, 120, 480, 380] (high overlap)
    hazard_candidates = [
        {"class_name": "smoke", "confidence": 0.76, "bbox": [120, 120, 480, 380]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [tv_screen])

    assert len(results) == 1
    res = results[0]
    # Smoke must NOT be rejected!
    assert res.status != HazardStatus.SCREEN_REJECTED
    assert res.status == HazardStatus.SCREEN_UNCERTAIN
    assert "smoke" in res.reason.lower()
    assert "dismissal" in res.reason.lower() or "prevent" in res.reason.lower()


def test_tilted_phone_weak_confidence_marked_uncertain():
    """
    When a phone is tilted or partially occluded, detector confidence may be low (e.g. 0.25).
    Overlapping hazard should be marked UNCERTAIN rather than discarded or confidently rejected.
    """
    filter_engine = ScreenContextFilter(
        reject_threshold=0.65,
        uncertain_threshold=0.25,
        min_screen_conf=0.35,
        tilted_min_conf=0.20,
    )

    # Tilted phone with marginal confidence 0.26
    tilted_phone = ScreenDetection(
        bbox=[150, 150, 320, 380],
        label="cell phone",
        confidence=0.26,
        is_low_confidence=True,
    )

    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.80, "bbox": [160, 160, 300, 360]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [tilted_phone])

    assert len(results) == 1
    res = results[0]
    assert res.status == HazardStatus.SCREEN_UNCERTAIN
    assert "tilted" in res.reason.lower() or "occluded" in res.reason.lower() or "marginal" in res.reason.lower()


def test_missing_phone_detection_policy():
    """
    If a phone is present but undetected (e.g. extreme angle, completely occluded bezels),
    the system treats hazard as environmental rather than crashing or inventing screens.
    """
    filter_engine = ScreenContextFilter(enabled=True)

    # Empty screen detections
    hazard_candidates = [
        {"class_name": "fire", "confidence": 0.88, "bbox": [200, 200, 350, 350]}
    ]

    results = filter_engine.evaluate_detections(hazard_candidates, [])

    assert len(results) == 1
    res = results[0]
    assert res.status == HazardStatus.CONFIRMED_ENVIRONMENTAL
    assert "no screen detected" in res.reason.lower()


def test_multiple_simultaneous_hazards_independent_classification():
    """
    Multi-Hazard Scene with 4 simultaneous distinct detections:
    1. Fire displayed inside a phone -> SCREEN_REJECTED
    2. Fire behind a phone (protruding) -> SCREEN_UNCERTAIN
    3. Real fire across the room -> CONFIRMED_ENVIRONMENTAL
    4. Smoke drifting over TV -> SCREEN_UNCERTAIN
    """
    filter_engine = ScreenContextFilter(
        reject_threshold=0.65,
        uncertain_threshold=0.25,
        protrusion_margin_ratio=0.15,
        reject_smoke_on_screens=False,
    )

    screens = [
        ScreenDetection(bbox=[50, 50, 200, 300], label="cell phone", confidence=0.90),
        ScreenDetection(bbox=[250, 50, 400, 300], label="cell phone", confidence=0.85),
        ScreenDetection(bbox=[600, 50, 900, 400], label="tv", confidence=0.92),
    ]

    candidates = [
        # 1. Fire strictly inside Phone 1
        {"class_name": "fire", "confidence": 0.85, "bbox": [60, 70, 190, 280]},
        # 2. Fire behind Phone 2 (protruding 40px left and right)
        {"class_name": "fire", "confidence": 0.88, "bbox": [210, 40, 440, 290]},
        # 3. Real fire on desk away from screens
        {"class_name": "fire", "confidence": 0.92, "bbox": [1000, 400, 1150, 550]},
        # 4. Smoke drifting over TV
        {"class_name": "smoke", "confidence": 0.70, "bbox": [650, 80, 850, 350]},
    ]

    results = filter_engine.evaluate_detections(candidates, screens)

    assert len(results) == 4

    # Candidate 1: Phone fire -> SCREEN_REJECTED
    assert results[0].status == HazardStatus.SCREEN_REJECTED

    # Candidate 2: Fire behind phone -> SCREEN_UNCERTAIN
    assert results[1].status == HazardStatus.SCREEN_UNCERTAIN

    # Candidate 3: Genuine desk fire -> CONFIRMED_ENVIRONMENTAL
    assert results[2].status == HazardStatus.CONFIRMED_ENVIRONMENTAL

    # Candidate 4: Smoke over TV -> SCREEN_UNCERTAIN
    assert results[3].status == HazardStatus.SCREEN_UNCERTAIN


def test_disabled_screen_filter():
    """When disabled, all candidates pass through as confirmed environmental."""
    filter_engine = ScreenContextFilter(enabled=False)

    phone = ScreenDetection(bbox=[100, 100, 300, 300], label="cell phone", confidence=0.9)
    candidates = [{"class_name": "fire", "confidence": 0.8, "bbox": [120, 120, 280, 280]}]

    results = filter_engine.evaluate_detections(candidates, [phone])
    assert len(results) == 1
    assert results[0].status == HazardStatus.CONFIRMED_ENVIRONMENTAL
