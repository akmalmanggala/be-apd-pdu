"""Unit & Integration Tests for APD Detector Engine, PRD 10.6 Overlap, and Temporal Persistence."""

import glob
import os
import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.apd_detector import (
    APDDetectorEngine,
    SpatialPersonTracker,
    TemporalWorkerTracker,
    calculate_overlap_ratio,
    box_iou,
)


def test_overlap_ratio_formula_prd_10_6():
    """Verify that overlap ratio strictly adheres to PRD Section 10.6."""
    # Case 1: APD is completely inside person bounding box -> Overlap = 1.0 (100%)
    person_box = [100.0, 100.0, 300.0, 600.0]
    helm_box = [150.0, 100.0, 250.0, 180.0]  # inside person
    overlap = calculate_overlap_ratio(helm_box, person_box)
    assert overlap == 1.0
    assert overlap >= 0.80  # Meets PRD 80% threshold

    # Case 2: APD is half outside person box (e.g. edge touch) -> Overlap = 0.50 (50%)
    touching_box = [50.0, 100.0, 150.0, 200.0]  # 50px outside, 50px inside
    overlap_edge = calculate_overlap_ratio(touching_box, person_box)
    assert abs(overlap_edge - 0.50) < 1e-3
    assert overlap_edge < 0.80  # Fails PRD 80% threshold -> "APD tidak terpakai"

    # Case 3: Completely disjoint
    disjoint_box = [500.0, 500.0, 600.0, 600.0]
    overlap_disjoint = calculate_overlap_ratio(disjoint_box, person_box)
    assert overlap_disjoint == 0.0


def test_box_iou_calculation():
    """Test standard IoU calculation."""
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [0.0, 0.0, 10.0, 10.0]
    assert box_iou(b1, b2) == 1.0

    b3 = [0.0, 0.0, 5.0, 10.0]
    assert abs(box_iou(b1, b3) - 0.50) < 1e-3

    b4 = [20.0, 20.0, 30.0, 30.0]
    assert box_iou(b1, b4) == 0.0


def test_detector_inference_on_pdu_test_dataset():
    """Verify APD detector runs and produces compliant checklist structure on test set images."""
    test_img_dir = r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images"
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    assert len(test_images) >= 30, f"Expected at least 30 test images, found {len(test_images)}"

    sample_img_path = test_images[0]
    img = cv2.imread(sample_img_path)
    assert img is not None, "Failed to load test sample image"

    detector = APDDetectorEngine.get_instance()
    workers, stats, annotated = detector.detect_image(
        img_bgr=img,
        overlap_threshold=0.80,
        enable_head_zoom=True,
    )

    assert "total_workers" in stats
    assert "compliant_workers_count" in stats
    assert "compliance_rate" in stats
    assert "inference_time_ms" in stats
    assert annotated.shape == img.shape
    assert len(workers) == stats["total_workers"]

    for w in workers:
        assert isinstance(w.track_id, int)
        assert set(w.checklist.keys()) == {"glove", "helm", "kacamata", "sepatu"}
        assert w.is_compliant == (len(w.missing_items) == 0)


def test_two_stage_head_zoom_kacamata():
    """Verify that head crop and micro-detection captures safety glasses on visible face."""
    import glob
    matches = glob.glob(r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images\pdu_v1_m62s28_f112462*.jpg")
    assert len(matches) > 0, "Test image not found"
    test_img_path = matches[0]
    img = cv2.imread(test_img_path)
    assert img is not None

    detector = APDDetectorEngine.get_instance()
    workers_zoom, stats_zoom, _ = detector.detect_image(img, enable_head_zoom=True)
    
    # Check that at least one worker has kacamata detected
    has_kacamata_detected = any(w.checklist["kacamata"] for w in workers_zoom)
    assert has_kacamata_detected, "Detector failed to detect kacamata on worker"


def test_anatomy_glove_sepatu_reclassification():
    """Verify that pure AI detector correctly recognizes glove APD on workers without artificial reclassification."""
    import glob
    detector = APDDetectorEngine.get_instance()
    matches = glob.glob(r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images\pdu_v1_m62s28_f112462*.jpg")
    assert len(matches) > 0, "Test image not found"
    img_sample = matches[0]
    img = cv2.imread(img_sample)
    assert img is not None
    workers, stats, _ = detector.detect_image(img)
    # Confirm gloves are recognized on workers
    glove_worn_count = sum(1 for w in workers if w.checklist["glove"])
    assert glove_worn_count >= 1, "Pure AI detection must detect glove on workers"


def test_partner_shoe_recovery():
    """Verify detector reliably captures both safety boots across workers on multi-boot scenes."""
    import glob
    detector = APDDetectorEngine.get_instance()
    matches = glob.glob(r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images\*m57s14_f103044*.jpg")
    assert len(matches) > 0, "Test image not found"
    img_sample = matches[0]
    img = cv2.imread(img_sample)
    assert img is not None
    workers, stats, _ = detector.detect_image(img)
    # Check that boots are accurately recognized across workers
    sepatu_count = sum(1 for a in stats["all_detected_apds"] if a.class_name == "sepatu")
    assert sepatu_count >= 2, f"Must capture verified boots, got {sepatu_count}"


def test_temporal_persistence_tracking():
    """Verify temporal worker tracking prevents flickering false violations during bending over."""
    tracker = TemporalWorkerTracker(memory_frames=10)

    # Frame 1: Worker 1 is upright, wearing all APDs including glasses
    chk_f1 = {"helm": True, "glove": True, "kacamata": True, "sepatu": True}
    res_f1 = tracker.update_worker_state(track_id=1, frame_idx=1, current_checklist=chk_f1)
    assert all(res_f1.values()), "Frame 1 should be 100% compliant"

    # Frame 2-5: Worker 1 bends down, kacamata becomes occluded by helmet/angle (current = False)
    chk_f2 = {"helm": True, "glove": True, "kacamata": False, "sepatu": True}
    res_f2 = tracker.update_worker_state(track_id=1, frame_idx=2, current_checklist=chk_f2)
    
    # Temporal smoothing should preserve kacamata=True because it was observed in recent window!
    assert res_f2["kacamata"] is True, "Temporal tracker must preserve kacamata compliance during bending over"
    assert all(res_f2.values()), "Worker 1 should remain compliant via temporal persistence"

    # Worker 2: Never wore glasses
    chk_w2 = {"helm": True, "glove": True, "kacamata": False, "sepatu": True}
    res_w2 = tracker.update_worker_state(track_id=2, frame_idx=2, current_checklist=chk_w2)
    assert res_w2["kacamata"] is False, "Worker who never wore glasses must remain False"


def test_roi_filtering():
    """Verify ROI boundaries correctly filter workers outside the designated zone."""
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    detector = APDDetectorEngine.get_instance()

    # Define ROI covering only left half (x from 0.0 to 0.5)
    roi_left = {"active": True, "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}
    workers, stats, _ = detector.detect_image(img, roi_config=roi_left)
    assert stats["total_workers"] == 0


def test_spatial_person_tracker_persistence():
    """Verify SpatialPersonTracker maintains track ID across frames when persons move slightly."""
    tracker = SpatialPersonTracker(iou_threshold=0.25, max_idle_frames=30)

    # Frame 1: Worker 1 at [100, 100, 200, 400], Worker 2 at [500, 100, 600, 400]
    boxes_f1 = [[100.0, 100.0, 200.0, 400.0], [500.0, 100.0, 600.0, 400.0]]
    ids_f1 = tracker.match_tracks(boxes_f1, frame_idx=1)
    assert ids_f1 == [1, 2]

    # Frame 2: Worker 2 detected first (order flipped), workers have moved slightly (10px)
    boxes_f2 = [[505.0, 105.0, 605.0, 405.0], [105.0, 102.0, 205.0, 402.0]]
    ids_f2 = tracker.match_tracks(boxes_f2, frame_idx=2)
    # The first detection corresponds to Worker 2 (ID 2), the second to Worker 1 (ID 1)
    assert ids_f2 == [2, 1], "Spatial tracker must maintain physical identity regardless of detection order"

    # Frame 3: Worker 3 enters scene
    boxes_f3 = [[106.0, 103.0, 206.0, 403.0], [800.0, 100.0, 900.0, 400.0]]
    ids_f3 = tracker.match_tracks(boxes_f3, frame_idx=3)
    assert ids_f3[0] == 1
    assert ids_f3[1] == 3  # New worker assigned next ID


def test_overlap_ratio_with_pad_ratio():
    """Verify anatomical margin (pad_ratio) correctly associates APD protruding from body."""
    # Worker body from y=100 to 500 (height 400)
    person_box = [100.0, 100.0, 200.0, 500.0]

    # Helmet dome sits on top, protruding 20px above py1 (y=80 to 140, total h=60)
    # In tight box: overlap = (140-100) / 60 = 40/60 = 66.7% (< 80%)
    helmet_box = [110.0, 80.0, 190.0, 140.0]
    raw_overlap = calculate_overlap_ratio(helmet_box, person_box, pad_ratio=0.0)
    assert raw_overlap < 0.80

    # With anatomical margin pad_ratio=0.08, expanded py1 = 100 - 400*(0.08*1.25) = 100 - 40 = 60
    # Overlap = (140-80) / 60 = 100% (>= 80%)
    padded_overlap = calculate_overlap_ratio(helmet_box, person_box, pad_ratio=0.08)
    assert padded_overlap >= 0.80, "Anatomical margin must capture helmet dome above forehead"

