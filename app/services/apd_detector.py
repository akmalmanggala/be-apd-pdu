"""High-Precision APD Detection Engine with Two-Stage Head Zoom & Temporal Persistence.

Implements PRD Section 10.6 overlap evaluation and addresses micro-detection & occlusion
of glasses (kacamata) and personal protective equipment in oil and gas drilling sites.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import settings
from app.schemas.detection import (
    BoundingBox,
    APDItem,
    WorkerDetection,
    ImageDetectionResponse,
)

logger = logging.getLogger(__name__)


def calculate_overlap_ratio(
    bbox_apd: List[float],
    bbox_person: List[float],
    pad_ratio: float = 0.0,
    class_name: Optional[str] = None,
) -> float:
    """Calculate overlap ratio of APD bounding box relative to worker body.
    
    Formula per PRD Section 10.6:
        Rasio Overlap = Luas(BBox_APD ∩ BBox_Person) / Luas(BBox_APD)
        
    An optional pad_ratio expands the worker boundary proportionally to account for
    anatomical extremities (helmet dome above forehead, boots below ankles, gloves on hands).
    """
    pw = max(0.0, bbox_person[2] - bbox_person[0])
    ph = max(0.0, bbox_person[3] - bbox_person[1])
    apd_w = max(0.0, bbox_apd[2] - bbox_apd[0])
    apd_h = max(0.0, bbox_apd[3] - bbox_apd[1])
    
    if pad_ratio <= 0.0:
        pad_top = 0.0
        pad_bottom = 0.0
        pad_side = 0.0
    elif class_name == "helm":
        # Allow helmet to extend slightly above shoulders if needed
        max_up = min(apd_h * 1.5, ph * 0.25) if ph > 80 else apd_h
        pad_top = max(pad_ratio * 1.5, max_up / max(1.0, ph), 0.18)
        pad_bottom = pad_ratio
        pad_side = max(pad_ratio * 1.5, 0.15)
    elif class_name == "sepatu":
        pad_top = pad_ratio
        pad_bottom = max(pad_ratio * 2.0, 0.18)
        pad_side = max(pad_ratio * 1.5, 0.15)
    elif class_name == "glove":
        pad_top = max(pad_ratio, 0.10)
        pad_bottom = max(pad_ratio, 0.10)
        pad_side = max(pad_ratio * 2.5, 0.22)
    elif class_name == "kacamata":
        pad_top = max(pad_ratio * 1.25, 0.15)
        pad_bottom = pad_ratio
        pad_side = max(pad_ratio * 1.25, 0.12)
    else:
        pad_top = pad_ratio * 1.25
        pad_bottom = pad_ratio * 1.25
        pad_side = pad_ratio * 1.25

    px1 = bbox_person[0] - pw * pad_side
    py1 = bbox_person[1] - ph * pad_top
    px2 = bbox_person[2] + pw * pad_side
    py2 = bbox_person[3] + ph * pad_bottom

    xA = max(bbox_apd[0], px1)
    yA = max(bbox_apd[1], py1)
    xB = min(bbox_apd[2], px2)
    yB = min(bbox_apd[3], py2)
    
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    
    apd_area = apd_w * apd_h
    
    return float(inter_area / apd_area) if apd_area > 0.0 else 0.0



def box_iou(b1: List[float], b2: List[float]) -> float:
    """Calculate standard Intersection over Union (IoU) between two bounding boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0.0 else 0.0


class APDDetectorEngine:
    """Singleton Computer Vision Engine for APD Detection and Worker Compliance."""

    _instance: Optional["APDDetectorEngine"] = None

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing APDDetectorEngine on device: {self.device}")
        
        # Load pre-trained COCO Person detector (YOLO11n)
        self.person_model = YOLO(settings.PERSON_MODEL_PATH)
        
        # Load fine-tuned 4-class APD detector (YOLO11s)
        self.apd_model = YOLO(settings.APD_MODEL_PATH)
        
        self.class_names = settings.CLASS_NAMES  # ['glove', 'helm', 'kacamata', 'sepatu']
        logger.info(f"APD Detector Engine ready with classes: {self.class_names}")

    @classmethod
    def get_instance(cls) -> "APDDetectorEngine":
        if cls._instance is None:
            cls._instance = APDDetectorEngine()
        return cls._instance

    def detect_image(
        self,
        img_bgr: np.ndarray,
        overlap_threshold: float = settings.DEFAULT_OVERLAP_THRESHOLD,
        conf_thresholds: Optional[Dict[str, float]] = None,
        person_conf: float = settings.DEFAULT_PERSON_CONF,
        enable_head_zoom: bool = settings.ENABLE_HEAD_ZOOM,
        head_crop_ratio: float = settings.HEAD_CROP_RATIO,
        head_zoom_conf: float = settings.HEAD_ZOOM_CONF,
        roi_config: Optional[Dict[str, Any]] = None,
        spatial_tracker: Optional["SpatialPersonTracker"] = None,
        temporal_tracker: Optional["TemporalWorkerTracker"] = None,
        frame_idx: int = 1,
        eval_mode: str = "individual",
        show_workers: bool = False,
    ) -> Tuple[List[WorkerDetection], Dict[str, Any], np.ndarray]:
        """Execute end-to-end detection on a single image.
        
        Returns:
            workers: List of detected workers with checklist and APD items
            stats: Summary metrics dictionary
            annotated_img: Image rendered with bounding boxes, labels, and badges
        """
        start_time = time.perf_counter()
        h, w, _ = img_bgr.shape
        
        if conf_thresholds is None:
            conf_thresholds = settings.DEFAULT_CONF_THRESHOLDS.copy()

        # Step 1: Detect Workers (Person class 0 from COCO)
        person_results = self.person_model(
            img_bgr,
            classes=[0],
            conf=person_conf,
            imgsz=settings.PRIMARY_IMGSZ,
            device=self.device,
            verbose=False,
        )[0]
        
        raw_person_boxes = []
        for box in person_results.boxes:
            b_coords = box.xyxy[0].cpu().numpy().tolist()
            b_conf = float(box.conf[0].item())
            
            # ROI Filtering if active
            if roi_config and roi_config.get("active", False):
                rx1 = roi_config.get("x1", 0.0) * w
                ry1 = roi_config.get("y1", 0.0) * h
                rx2 = roi_config.get("x2", 1.0) * w
                ry2 = roi_config.get("y2", 1.0) * h
                cx = (b_coords[0] + b_coords[2]) / 2.0
                cy = (b_coords[1] + b_coords[3]) / 2.0
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue  # Worker is outside configured ROI
                    
            pw = b_coords[2] - b_coords[0]
            ph = b_coords[3] - b_coords[1]
            # Filter isolated fragments / spurious hand detections as full persons
            if ph < 80 or pw < 35:
                continue
            raw_person_boxes.append((b_coords, b_conf))

        # Person NMS Deduplication (standard confidence-descending NMS with giant-box suppression)
        filtered_person_boxes = []
        raw_person_boxes.sort(key=lambda x: x[1], reverse=True)
        for pb, pconf in raw_person_boxes:
            is_dup = False
            pb_w = pb[2] - pb[0]
            pb_h = pb[3] - pb[1]
            pb_area = max(1.0, pb_w * pb_h)

            for fb, fconf in filtered_person_boxes:
                iou = box_iou(pb, fb)
                inter_w = max(0.0, min(pb[2], fb[2]) - max(pb[0], fb[0]))
                inter_h = max(0.0, min(pb[3], fb[3]) - max(pb[1], fb[1]))
                inter_area = inter_w * inter_h
                fb_w = fb[2] - fb[0]
                fb_h = fb[3] - fb[1]
                fb_area = max(1.0, fb_w * fb_h)

                # Standard IoU suppression
                if iou > 0.40:
                    is_dup = True
                    break

                # Candidate pb is contained inside already confirmed higher-conf fb
                if inter_area / pb_area > 0.65:
                    is_dup = True
                    break

                # Candidate pb is a loose/merged giant box subsuming an already confirmed higher-conf fb
                if inter_area / fb_area > 0.65 and pb_area > fb_area * 1.35:
                    is_dup = True
                    break

            if not is_dup:
                filtered_person_boxes.append((pb, pconf))

        # Step 2: Global APD Detection
        # Run global pass with calibrated confidence threshold to eliminate background noise early
        min_conf = min(conf_thresholds.values()) if conf_thresholds else 0.15
        apd_results = self.apd_model(
            img_bgr,
            conf=max(0.08, min_conf - 0.05),
            imgsz=settings.PRIMARY_IMGSZ,
            device=self.device,
            verbose=False,
        )[0]
        
        raw_apds = []
        for box in apd_results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cname = self.class_names[cls_id]
            raw_apds.append({
                "class_name": cname,
                "confidence": conf,
                "bbox": coords,
            })

        # Natural person detections from Step 1 are preserved without artificial stretching or merging

        # Assign persistent track IDs if spatial tracker is active
        if spatial_tracker is not None:
            matched_tids = spatial_tracker.match_tracks([p[0] for p in filtered_person_boxes], frame_idx)
        else:
            matched_tids = list(range(1, len(filtered_person_boxes) + 1))

        # Step 2c: Pure AI Detections with Calibrated Class Confidence Thresholds
        # Trusts the trained AI model directly without artificial anatomical rules, pixel limits, or reclassifications
        pure_apds = []
        for apd in raw_apds:
            cname = apd["class_name"]
            cconf = apd["confidence"]
            if cconf >= conf_thresholds.get(cname, 0.20):
                pure_apds.append(apd)

        # Step 3: Standard Per-Class Non-Maximum Suppression (NMS)
        # Removes duplicate predictions of the same object without dropping objects at distinct coordinates
        dedup_apds = []
        for cname in self.class_names:
            c_items = [a for a in pure_apds if a["class_name"] == cname]
            c_items.sort(key=lambda x: x["confidence"], reverse=True)
            kept = []
            iou_thresh = 0.25 if cname == "kacamata" else 0.40
            for item in c_items:
                ib = item["bbox"]
                is_dup = False
                for k in kept:
                    kb = k["bbox"]
                    iou = box_iou(ib, kb)
                    if cname == "kacamata":
                        inter_w = max(0.0, min(ib[2], kb[2]) - max(ib[0], kb[0]))
                        inter_h = max(0.0, min(ib[3], kb[3]) - max(ib[1], kb[1]))
                        inter_area = inter_w * inter_h
                        min_area = min((ib[2] - ib[0]) * (ib[3] - ib[1]), (kb[2] - kb[0]) * (kb[3] - kb[1]))
                        containment = inter_area / max(1.0, min_area)
                        if iou > iou_thresh or containment > 0.40:
                            is_dup = True
                            break
                    else:
                        if iou > iou_thresh:
                            is_dup = True
                            break
                if not is_dup:
                    kept.append(item)
            dedup_apds.extend(kept)

        # Step 3-Cross: Cross-Class Conflict Resolution (Class-Agnostic Suppression)
        # Prevents conflicting classes from occupying the same physical coordinates
        # (e.g. eliminating 'ketuker-tuker' flipping between glove and sepatu)
        dedup_apds.sort(key=lambda x: x["confidence"], reverse=True)
        cross_clean_apds = []
        for cand in dedup_apds:
            cb = cand["bbox"]
            c_name = cand["class_name"]
            has_conflict = False
            for kept in cross_clean_apds:
                kb = kept["bbox"]
                k_name = kept["class_name"]
                iou = box_iou(cb, kb)

                inter_w = max(0.0, min(cb[2], kb[2]) - max(cb[0], kb[0]))
                inter_h = max(0.0, min(cb[3], kb[3]) - max(cb[1], kb[1]))
                inter_area = inter_w * inter_h
                cand_area = max(1.0, (cb[2] - cb[0]) * (cb[3] - cb[1]))
                kept_area = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                containment = inter_area / min(cand_area, kept_area)

                # Mutually exclusive pair 1: Glove vs Sepatu (hands vs feet)
                if (c_name == "glove" and k_name == "sepatu") or (c_name == "sepatu" and k_name == "glove"):
                    if iou > 0.25 or containment > 0.40:
                        has_conflict = True
                        break

                # Mutually exclusive pair 2: Helm vs Sepatu or Helm vs Glove (head vs limbs)
                elif (c_name == "helm" and k_name in ["sepatu", "glove"]) or (c_name in ["sepatu", "glove"] and k_name == "helm"):
                    if iou > 0.20 or containment > 0.35:
                        has_conflict = True
                        break

                # Extreme general overlap between different classes
                # (Excluding helm & kacamata which naturally coexist together on the head)
                elif c_name != k_name:
                    if {c_name, k_name} == {"helm", "kacamata"}:
                        continue
                    if iou > 0.45 or containment > 0.65:
                        has_conflict = True
                        break

            if not has_conflict:
                cross_clean_apds.append(cand)

        dedup_apds = cross_clean_apds
        all_global_apds = dedup_apds
        two_stage_zoom_applied = False

        # Step 3b: Anatomical Anchoring & Isolated Background Suppression
        # Protects all valid worker APDs (including close-up drillers without full body boxes)
        # while eliminating spurious artifacts on distant background machinery and pipes.
        human_reach_regions = []
        for pb, _ in filtered_person_boxes:
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            human_reach_regions.append([
                max(0.0, pb[0] - pw * 0.45),
                max(0.0, pb[1] - ph * 0.25),
                min(float(w), pb[2] + pw * 0.45),
                min(float(h), pb[3] + ph * 0.25),
            ])

        for a in dedup_apds:
            if a["class_name"] == "helm" and a["confidence"] >= 0.25:
                hb = a["bbox"]
                hw = hb[2] - hb[0]
                hh = hb[3] - hb[1]
                human_reach_regions.append([
                    max(0.0, hb[0] - hw * 1.5),
                    max(0.0, hb[1] - hh * 0.3),
                    min(float(w), hb[2] + hw * 1.5),
                    min(float(h), hb[3] + hh * 3.5),
                ])

        validated_apds = []
        for a in dedup_apds:
            cname = a["class_name"]
            conf = a["confidence"]
            ab = a["bbox"]

            if cname == "helm":
                validated_apds.append(a)
            elif cname == "kacamata":
                cx = (ab[0] + ab[2]) / 2.0
                cy = (ab[1] + ab[3]) / 2.0
                is_near_human = any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in human_reach_regions)
                if is_near_human or conf >= 0.40:
                    validated_apds.append(a)
            else:
                cx = (ab[0] + ab[2]) / 2.0
                cy = (ab[1] + ab[3]) / 2.0
                is_near_human = any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in human_reach_regions)
                if is_near_human or conf >= 0.65:
                    validated_apds.append(a)

        all_global_apds = validated_apds

        # Step 4: Associate APD with Persons using PRD 10.6 Overlap Ratio (>= 80%)
        workers: List[WorkerDetection] = []
        helm_violations = 0
        glove_violations = 0
        sepatu_violations = 0
        kacamata_violations = 0
        worn_apd_indices = set()

        # Mode A: Individual Association
        if eval_mode != "global_roi" or filtered_person_boxes:
            for idx, (p_coords, p_conf) in enumerate(filtered_person_boxes):
                track_id = matched_tids[idx]
                p_bbox = BoundingBox(
                    x1=round(p_coords[0], 2),
                    y1=round(p_coords[1], 2),
                    x2=round(p_coords[2], 2),
                    y2=round(p_coords[3], 2),
                )
                
                checklist: Dict[str, bool] = {c: False for c in self.class_names}
                detected_apds: List[APDItem] = []
                
                for item_idx, item in enumerate(all_global_apds):
                    overlap = calculate_overlap_ratio(
                        item["bbox"], p_coords, pad_ratio=0.08, class_name=item["class_name"]
                    )
                    is_worn = overlap >= overlap_threshold
                    
                    apd_obj = APDItem(
                        class_name=item["class_name"],
                        confidence=round(item["confidence"], 3),
                        bbox=BoundingBox(
                            x1=round(item["bbox"][0], 2),
                            y1=round(item["bbox"][1], 2),
                            x2=round(item["bbox"][2], 2),
                            y2=round(item["bbox"][3], 2),
                        ),
                        overlap_ratio=round(overlap, 3),
                        is_worn=is_worn,
                    )
                    
                    if is_worn:
                        checklist[item["class_name"]] = True
                        detected_apds.append(apd_obj)
                        worn_apd_indices.add(item_idx)

                # Apply temporal persistence smoothing if tracker is active
                if temporal_tracker is not None:
                    checklist = temporal_tracker.update_worker_state(
                        track_id=track_id,
                        frame_idx=frame_idx,
                        current_checklist=checklist,
                    )

                missing_items = [name for name, worn in checklist.items() if not worn]
                is_compliant = len(missing_items) == 0
                
                if not checklist["helm"]:
                    helm_violations += 1
                if not checklist["glove"]:
                    glove_violations += 1
                if not checklist["sepatu"]:
                    sepatu_violations += 1
                if not checklist["kacamata"]:
                    kacamata_violations += 1

                workers.append(
                    WorkerDetection(
                        track_id=track_id,
                        bbox=p_bbox,
                        person_conf=round(p_conf, 3),
                        checklist=checklist,
                        is_compliant=is_compliant,
                        missing_items=missing_items,
                        detected_apds=detected_apds,
                    )
                )
        else:
            # Mode B: Global ROI Fallback (PRD Section 10.6 Mode B)
            roi_checklist: Dict[str, bool] = {c: False for c in self.class_names}
            roi_apds: List[APDItem] = []
            for item_idx, item in enumerate(all_global_apds):
                roi_checklist[item["class_name"]] = True
                worn_apd_indices.add(item_idx)
                roi_apds.append(
                    APDItem(
                        class_name=item["class_name"],
                        confidence=round(item["confidence"], 3),
                        bbox=BoundingBox(
                            x1=round(item["bbox"][0], 2),
                            y1=round(item["bbox"][1], 2),
                            x2=round(item["bbox"][2], 2),
                            y2=round(item["bbox"][3], 2),
                        ),
                        overlap_ratio=1.0,
                        is_worn=True,
                    )
                )
            missing = [name for name, worn in roi_checklist.items() if not worn]
            is_comp = len(missing) == 0
            if not roi_checklist["helm"]:
                helm_violations += 1
            if not roi_checklist["glove"]:
                glove_violations += 1
            if not roi_checklist["sepatu"]:
                sepatu_violations += 1
            if not roi_checklist["kacamata"]:
                kacamata_violations += 1
            workers.append(
                WorkerDetection(
                    track_id=1,
                    bbox=BoundingBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h)),
                    person_conf=1.0,
                    checklist=roi_checklist,
                    is_compliant=is_comp,
                    missing_items=missing,
                    detected_apds=roi_apds,
                )
            )

        # Collect all detected APDs and unassigned APDs
        all_detected_apds: List[APDItem] = []
        unassigned_apds: List[APDItem] = []
        for item_idx, item in enumerate(all_global_apds):
            is_worn_by_any = item_idx in worn_apd_indices
            apd_obj = APDItem(
                class_name=item["class_name"],
                confidence=round(item["confidence"], 3),
                bbox=BoundingBox(
                    x1=round(item["bbox"][0], 2),
                    y1=round(item["bbox"][1], 2),
                    x2=round(item["bbox"][2], 2),
                    y2=round(item["bbox"][3], 2),
                ),
                overlap_ratio=1.0 if is_worn_by_any else 0.0,
                is_worn=is_worn_by_any,
            )
            all_detected_apds.append(apd_obj)
            if not is_worn_by_any:
                unassigned_apds.append(apd_obj)

        total_workers = len(workers)
        compliant_count = sum(1 for w_item in workers if w_item.is_compliant)
        non_compliant_count = total_workers - compliant_count
        compliance_rate = (compliant_count / total_workers * 100.0) if total_workers > 0 else 0.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "total_workers": total_workers,
            "compliant_workers_count": compliant_count,
            "non_compliant_workers_count": non_compliant_count,
            "compliance_rate": round(compliance_rate, 2),
            "helm_violations": helm_violations,
            "glove_violations": glove_violations,
            "sepatu_violations": sepatu_violations,
            "kacamata_violations": kacamata_violations,
            "inference_time_ms": round(elapsed_ms, 2),
            "two_stage_zoom_applied": two_stage_zoom_applied,
            "total_apds_detected": len(all_detected_apds),
            "all_detected_apds": all_detected_apds,
            "unassigned_apds": unassigned_apds,
        }

        # Step 5: Render Annotated Visual Output
        annotated_img = self.render_annotations(
            img_bgr=img_bgr,
            workers=workers,
            all_apds=all_global_apds,
            stats=stats,
            roi_config=roi_config,
            show_workers=show_workers,
        )

        return workers, stats, annotated_img

    def render_annotations(
        self,
        img_bgr: np.ndarray,
        workers: List[WorkerDetection],
        all_apds: List[Dict[str, Any]],
        stats: Dict[str, Any],
        roi_config: Optional[Dict[str, Any]] = None,
        show_workers: bool = False,
    ) -> np.ndarray:
        """Render high-clarity bounding boxes, labels, compliance badges, and HUD banner."""
        vis = img_bgr.copy()
        h, w, _ = vis.shape
        
        # Color palette (BGR)
        apd_colors = {
            "helm": (0, 230, 0),        # Vivid Green
            "glove": (0, 140, 255),      # Orange
            "kacamata": (255, 0, 180),   # Magenta / Deep Pink
            "sepatu": (255, 220, 0),     # Bright Cyan-Yellow
        }

        # Draw ROI overlay if active
        if roi_config and roi_config.get("active", False):
            rx1 = int(roi_config.get("x1", 0.0) * w)
            ry1 = int(roi_config.get("y1", 0.0) * h)
            rx2 = int(roi_config.get("x2", 1.0) * w)
            ry2 = int(roi_config.get("y2", 1.0) * h)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            cv2.putText(vis, "ZONA DETEKSI AKTIF (ROI)", (rx1 + 10, ry1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 1. Draw Worker Bounding Boxes & Compliance Badges (if show_workers is True)
        if show_workers:
            for worker in workers:
                px1 = int(worker.bbox.x1)
                py1 = int(worker.bbox.y1)
                px2 = int(worker.bbox.x2)
                py2 = int(worker.bbox.y2)

                if worker.is_compliant:
                    box_color = (0, 230, 0)  # Vivid Green for Compliant
                    if getattr(worker, "is_partial", False):
                        status_text = f"Pekerja #{worker.track_id}: PARSIAL (PATUH)"
                    else:
                        status_text = f"Pekerja #{worker.track_id}: LENGKAP (100%)"
                else:
                    box_color = (0, 0, 230)  # Red for Violation
                    missing_str = ", ".join(m.capitalize() for m in worker.missing_items)
                    status_text = f"Pekerja #{worker.track_id}: MELANGGAR [- {missing_str}]"

                # Draw worker body box
                cv2.rectangle(vis, (px1, py1), (px2, py2), box_color, 2)

                # Draw worker header badge (clamp to avoid HUD overlap)
                (bw, bh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                if py1 < 65:
                    badge_y1 = py1
                    badge_y2 = py1 + bh + 10
                    text_y = py1 + bh + 3
                else:
                    badge_y1 = py1 - bh - 10
                    badge_y2 = py1
                    text_y = py1 - 5

                cv2.rectangle(vis, (px1, badge_y1), (px1 + bw + 14, badge_y2), box_color, -1)
                text_color = (0, 0, 0) if worker.is_compliant else (255, 255, 255)
                cv2.putText(vis, status_text, (px1 + 7, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, text_color, 2, cv2.LINE_AA)

        # 2. Draw APD Bounding Boxes (Crisp overlays on top)
        for item in all_apds:
            cname = item["class_name"]
            conf = item["confidence"]
            color = apd_colors.get(cname, (200, 200, 200))
            bx1, by1, bx2, by2 = [int(v) for v in item["bbox"]]
            
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            label = f"{cname.upper()} {conf:.2f}"
            if item.get("zoomed_source", False):
                label += " [ZOOM]"
                
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (bx1, max(0, by1 - lh - 6)), (bx1 + lw + 6, by1), color, -1)
            cv2.putText(vis, label, (bx1 + 3, max(lh, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Draw Top HUD Banner (Summary Metrics)
        hud_h = 55
        hud_overlay = vis.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(hud_overlay, 0.75, vis, 0.25, 0, vis)
        
        if show_workers:
            hud_title = "SISTEM DETEKSI APD MIGAS PDU"
            cv2.putText(vis, hud_title, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            hud_stats = (
                f"Pekerja: {stats['total_workers']} | "
                f"Patuh: {stats['compliant_workers_count']} | "
                f"Melanggar: {stats['non_compliant_workers_count']} | "
                f"Kepatuhan: {stats['compliance_rate']:.1f}% | "
                f"Latency: {stats['inference_time_ms']:.1f}ms"
            )
            cv2.putText(vis, hud_stats, (380, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            hud_title = "PREDIKSI AI (MURNI DETEKSI APD)"
            cv2.putText(vis, hud_title, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2, cv2.LINE_AA)
            
            c_counts = {c: 0 for c in self.class_names}
            for a in all_apds:
                cn = a["class_name"]
                if cn in c_counts:
                    c_counts[cn] += 1
            hud_stats = (
                f"Helm: {c_counts['helm']} | "
                f"Kacamata: {c_counts['kacamata']} | "
                f"Glove: {c_counts['glove']} | "
                f"Sepatu: {c_counts['sepatu']} | "
                f"Total: {len(all_apds)} | "
                f"Latency: {stats['inference_time_ms']:.1f}ms"
            )
            cv2.putText(vis, hud_stats, (420, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

class SpatialPersonTracker:
    """Fast, dependency-free IoU tracker for persons across continuous video frames.
    
    Assigns persistent track IDs across continuous frames based on spatial proximity
    and bounding box IoU without requiring external C++ libraries.
    """

    def __init__(self, iou_threshold: float = 0.25, max_idle_frames: int = 45):
        self.iou_threshold = iou_threshold
        self.max_idle_frames = max_idle_frames
        self.next_id = 1
        # track_id -> {"bbox": [x1, y1, x2, y2], "last_frame": int}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def match_tracks(self, person_boxes: List[List[float]], frame_idx: int) -> List[int]:
        """Match incoming person bounding boxes to persistent track IDs."""
        if not person_boxes:
            # Purge idle tracks
            self.tracks = {
                tid: data for tid, data in self.tracks.items()
                if frame_idx - data["last_frame"] <= self.max_idle_frames
            }
            return []

        if not self.tracks:
            assigned_ids = []
            for b in person_boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": b, "last_frame": frame_idx}
                assigned_ids.append(tid)
            return assigned_ids

        # Active tracks within max_idle_frames
        active_tids = [
            tid for tid, data in self.tracks.items()
            if frame_idx - data["last_frame"] <= self.max_idle_frames
        ]
        assigned_ids: List[Optional[int]] = [None] * len(person_boxes)
        used_tids = set()

        if active_tids:
            pairs = []
            for det_idx, p_box in enumerate(person_boxes):
                for tid in active_tids:
                    t_box = self.tracks[tid]["bbox"]
                    iou = box_iou(p_box, t_box)
                    if iou >= self.iou_threshold:
                        pairs.append((iou, det_idx, tid))
            
            pairs.sort(reverse=True, key=lambda x: x[0])
            matched_dets = set()
            for iou, det_idx, tid in pairs:
                if det_idx not in matched_dets and tid not in used_tids:
                    assigned_ids[det_idx] = tid
                    matched_dets.add(det_idx)
                    used_tids.add(tid)
                    self.tracks[tid] = {"bbox": person_boxes[det_idx], "last_frame": frame_idx}

        # For remaining unmatched detections, assign new track IDs
        for det_idx in range(len(person_boxes)):
            if assigned_ids[det_idx] is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": person_boxes[det_idx], "last_frame": frame_idx}
                assigned_ids[det_idx] = tid

        # Purge dead tracks
        self.tracks = {
            tid: data for tid, data in self.tracks.items()
            if frame_idx - data["last_frame"] <= self.max_idle_frames
        }
        return [int(t) for t in assigned_ids]


class TemporalWorkerTracker:
    """Temporal Tracking & State Persistence across continuous frames/video streams.
    
    Prevents false violations caused by bending over, camera glare, or brief occlusions
    by maintaining a temporal compliance buffer per worker track ID.
    """

    def __init__(self, memory_frames: int = settings.TEMPORAL_MEMORY_FRAMES):
        self.memory_frames = memory_frames
        # track_id -> class_name -> deque of booleans
        self.history: Dict[int, Dict[str, deque]] = defaultdict(
            lambda: {c: deque(maxlen=self.memory_frames) for c in settings.CLASS_NAMES}
        )
        self.last_seen_frame: Dict[int, int] = {}

    def update_worker_state(
        self,
        track_id: int,
        frame_idx: int,
        current_checklist: Dict[str, bool],
    ) -> Dict[str, bool]:
        """Update worker's temporal history and return smoothed compliance checklist."""
        worker_hist = self.history[track_id]
        self.last_seen_frame[track_id] = frame_idx

        smoothed_checklist: Dict[str, bool] = {}
        for cname in settings.CLASS_NAMES:
            # Append current observation
            worker_hist[cname].append(current_checklist.get(cname, False))
            # If item was detected in recent memory window, maintain compliance
            smoothed_checklist[cname] = any(worker_hist[cname])

        return smoothed_checklist

    def purge_stale_tracks(self, current_frame_idx: int, max_idle: int = 150):
        """Remove tracks that haven't been observed in max_idle frames."""
        stale_ids = [
            tid for tid, last_f in self.last_seen_frame.items()
            if current_frame_idx - last_f > max_idle
        ]
        for tid in stale_ids:
            self.history.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
