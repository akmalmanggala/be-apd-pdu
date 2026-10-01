"""Unit tests for APDVideoStabilizer engine."""

import pytest
from app.services.video_stabilizer import APDVideoStabilizer, APDTracklet, box_iou, box_containment


def test_box_iou_and_containment():
    b1 = [100.0, 100.0, 200.0, 200.0]
    b2 = [100.0, 100.0, 200.0, 200.0]
    assert box_iou(b1, b2) == pytest.approx(1.0)
    assert box_containment(b1, b2) == pytest.approx(1.0)

    # Partial overlap
    b3 = [150.0, 100.0, 250.0, 200.0]
    iou = box_iou(b1, b3)
    assert 0.30 <= iou <= 0.40

    # No overlap
    b4 = [300.0, 300.0, 400.0, 400.0]
    assert box_iou(b1, b4) == 0.0
    assert box_containment(b1, b4) == 0.0


def test_k_of_n_confirmation_filter():
    stabilizer = APDVideoStabilizer(min_hits_to_confirm=2, high_conf_instant=0.65)

    # Frame 1: Single-frame transient spike (conf = 0.35, sensitive)
    dets_f1 = [{"bbox": [500.0, 500.0, 600.0, 600.0], "class_name": "glove", "confidence": 0.35}]
    out_f1 = stabilizer.process_frame(1, dets_f1, 1920, 1080)
    # Since hits == 1 and conf < 0.65, it MUST NOT be rendered
    assert len(out_f1) == 0

    # Frame 2: Same object detected again
    dets_f2 = [{"bbox": [502.0, 501.0, 602.0, 601.0], "class_name": "glove", "confidence": 0.40}]
    out_f2 = stabilizer.process_frame(2, dets_f2, 1920, 1080)
    # Now hits == 2, confirmed!
    assert len(out_f2) == 1
    assert out_f2[0]["class_name"] == "glove"


def test_high_conf_instant_confirmation():
    stabilizer = APDVideoStabilizer(min_hits_to_confirm=2, high_conf_instant=0.65)

    # High confidence item (conf = 0.88, clear APD)
    dets_f1 = [{"bbox": [400.0, 400.0, 500.0, 500.0], "class_name": "helm", "confidence": 0.88}]
    out_f1 = stabilizer.process_frame(1, dets_f1, 1920, 1080)
    # High confidence confirms immediately on frame 1
    assert len(out_f1) == 1
    assert out_f1[0]["class_name"] == "helm"


def test_majority_voting_kills_class_flip():
    stabilizer = APDVideoStabilizer(bbox_smooth_alpha=0.70)

    # Establish tracklet as glove over 3 frames
    for f in range(1, 4):
        dets = [{"bbox": [100.0, 200.0, 160.0, 260.0], "class_name": "glove", "confidence": 0.85}]
        out = stabilizer.process_frame(f, dets, 1920, 1080)
        assert len(out) == 1
        assert out[0]["class_name"] == "glove"

    # Frame 4: 1-frame glitch flips class to helm (e.g. glove on dial detected as helm 0.37)
    glitch_det = [{"bbox": [102.0, 201.0, 161.0, 259.0], "class_name": "helm", "confidence": 0.37}]
    out_glitch = stabilizer.process_frame(4, glitch_det, 1920, 1080)
    assert len(out_glitch) == 1
    # Majority voting over history enforces GLOVE!
    assert out_glitch[0]["class_name"] == "glove"


def test_horizontal_shoe_conflict_resolution():
    stabilizer = APDVideoStabilizer()
    img_w, img_h = 2560, 1440

    # Overlapping glove and sepatu on the same object (shoe sole is horizontal: w=150, h=90)
    dets = [
        {"bbox": [1000.0, 1100.0, 1150.0, 1190.0], "class_name": "glove", "confidence": 0.68},
        {"bbox": [1000.0, 1100.0, 1150.0, 1190.0], "class_name": "sepatu", "confidence": 0.62},
    ]
    # Frame 1 creates tracklet and resolves conflict to sepatu
    stabilizer.process_frame(1, dets, img_w, img_h)
    # Frame 2 confirms tracklet and outputs verified detection
    out = stabilizer.process_frame(2, dets, img_w, img_h)
    assert len(out) == 1
    # Horizontal footwear sole naturally resolves to sepatu
    assert out[0]["class_name"] == "sepatu"


def test_transient_single_frame_spike_filtering():
    stabilizer = APDVideoStabilizer()
    img_w, img_h = 2560, 1440

    # Single-frame low-confidence glare or reflection (conf < 0.65)
    noise_det = [{"bbox": [2050.0, 250.0, 2150.0, 350.0], "class_name": "helm", "confidence": 0.55}]
    out = stabilizer.process_frame(1, noise_det, img_w, img_h)
    # Filtered out by K-of-N confirmation filter!
    assert len(out) == 0


def test_head_stacking_deduplication():
    stabilizer = APDVideoStabilizer()
    img_w, img_h = 2560, 1440

    # Real helmet on head + duplicate sub-box on chin/collar directly underneath
    head_dets = [
        {"bbox": [1200.0, 300.0, 1300.0, 400.0], "class_name": "helm", "confidence": 0.92},
        {"bbox": [1205.0, 380.0, 1295.0, 460.0], "class_name": "helm", "confidence": 0.48},
    ]
    out = stabilizer.process_frame(1, head_dets, img_w, img_h)
    # Only the primary higher confidence helmet is kept
    assert len(out) == 1
    assert out[0]["confidence"] == pytest.approx(0.92, abs=0.01)


def test_linear_velocity_extrapolation_and_dropout_bridging():
    stabilizer = APDVideoStabilizer(min_hits_to_confirm=2)
    img_w, img_h = 2560, 1440

    # Frame 1: worker moving foot
    d1 = [{"bbox": [500.0, 800.0, 560.0, 840.0], "class_name": "sepatu", "confidence": 0.75}]
    out1 = stabilizer.process_frame(1, d1, img_w, img_h)
    assert len(out1) == 1

    # Frame 2: foot moves +10px in X and +5px in Y
    d2 = [{"bbox": [510.0, 805.0, 570.0, 845.0], "class_name": "sepatu", "confidence": 0.78}]
    out2 = stabilizer.process_frame(2, d2, img_w, img_h)
    assert len(out2) == 1

    # Frame 3: momentary 1-frame dropout (empty detections)
    out3 = stabilizer.process_frame(3, [], img_w, img_h)
    # Velocity extrapolation bridges the gap smoothly (NO flickering/dropout)!
    assert len(out3) == 1
    assert out3[0]["class_name"] == "sepatu"
    # Centroid has advanced along motion vector (around 540 -> 550)
    assert out3[0]["bbox"][0] > 510.0

    # Frame 4: second consecutive missed frame stops displaying to prevent runaway drift
    out4 = stabilizer.process_frame(4, [], img_w, img_h)
    assert len(out4) == 0


def test_sticky_class_locking_prevents_flipping():
    stabilizer = APDVideoStabilizer(min_hits_to_confirm=2)
    img_w, img_h = 2560, 1440

    # Establish tracklet as sepatu across 3 frames
    for f in range(1, 4):
        d = [{"bbox": [600.0, 900.0, 680.0, 950.0], "class_name": "sepatu", "confidence": 0.80}]
        out = stabilizer.process_frame(f, d, img_w, img_h)
        assert out[0]["class_name"] == "sepatu"

    # Frame 4: momentary glitch predicts glove with 0.55 confidence on the boot
    glitch_d = [{"bbox": [600.0, 900.0, 680.0, 950.0], "class_name": "glove", "confidence": 0.55}]
    out_glitch = stabilizer.process_frame(4, glitch_d, img_w, img_h)
    assert len(out_glitch) == 1
    # Sticky class locking guarantees the object stays SEPATU!
    assert out_glitch[0]["class_name"] == "sepatu"

