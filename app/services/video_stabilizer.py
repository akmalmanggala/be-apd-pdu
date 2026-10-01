"""Temporal Video Stabilizer Engine for Industrial Rig APD Detections.

Solves transient flickering, single-frame spikes, class-flipping ("ketuker-tuker"),
and false machinery detections across continuous CCTV footage without raising
confidence thresholds or relying on brittle full-body person bounding boxes.
"""

from collections import deque, defaultdict
import math
from typing import List, Dict, Any, Tuple, Optional


def box_iou(b1: List[float], b2: List[float]) -> float:
    """Compute standard Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def box_containment(b1: List[float], b2: List[float]) -> float:
    """Compute fraction of the smaller box contained within the larger box."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return float(inter / min(a1, a2))


class APDTracklet:
    """Direct object-level tracklet representing a physical APD item across continuous frames."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        class_name: str,
        confidence: float,
        frame_idx: int,
    ):
        self.track_id = track_id
        self.bbox = [float(c) for c in bbox]
        self.class_name = class_name
        self.confidence = float(confidence)
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.age = 1
        self.idle_frames = 0

        # Confirmation flag (K-of-N Filter)
        self.is_confirmed = False

        # Observation history: (frame_idx, class_name, confidence, bbox)
        self.history: deque = deque(maxlen=25)
        self.history.append((frame_idx, class_name, float(confidence), list(self.bbox)))

        # Centroid motion history: (cx, cy)
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.positions: deque = deque(maxlen=35)
        self.positions.append((cx, cy))

        # Static machinery fixture detection
        self.is_static_fixture = False

    def update(
        self,
        bbox: List[float],
        class_name: str,
        confidence: float,
        frame_idx: int,
        alpha: float = 0.85,
    ):
        """Update tracklet with matched detection using adaptive motion smoothing."""
        # Calculate instantaneous centroid displacement
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        disp = math.hypot(new_cx - old_cx, new_cy - old_cy)

        # Adaptive alpha:
        # If object is moving (> 6 px), snap instantly (alpha = 0.95) to eliminate lagging.
        # If object is stationary (<= 6 px), use alpha = 0.75 to suppress visual jitter.
        effective_alpha = 0.95 if disp > 6.0 else 0.75
        self.bbox = [
            effective_alpha * float(nb) + (1.0 - effective_alpha) * float(ob)
            for nb, ob in zip(bbox, self.bbox)
        ]
        self.confidence = float(confidence)
        self.hits += 1
        self.age += max(1, frame_idx - self.last_frame)
        self.last_frame = frame_idx
        self.idle_frames = 0

        # Record observation
        self.history.append((frame_idx, class_name, float(confidence), list(bbox)))

        # Update centroid trajectory
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.positions.append((cx, cy))

        # Determine consensus class via time-decayed voting
        self._update_consensus_class(frame_idx)

        # Confirm tracklet if it meets criteria
        if self.hits >= 2 or confidence >= 0.70:
            self.is_confirmed = True

        # Check if this object is an inanimate static fixture
        self._check_static_fixture()

    def mark_missed(self, frame_idx: int, img_w: int = 2560, img_h: int = 1440):
        """Mark tracklet as unobserved and extrapolate position via linear velocity for 1-frame dropout."""
        self.idle_frames += 1
        self.age += 1

        # Linear Velocity Extrapolation:
        # If confirmed and previously in active motion, advance box by velocity vector to bridge 1-frame dropout
        # without lagging behind or flickering!
        if self.is_confirmed and self.idle_frames == 1 and len(self.positions) >= 2:
            p1 = self.positions[-1]
            p0 = self.positions[-2]
            vx = p1[0] - p0[0]
            vy = p1[1] - p0[1]
            speed = math.hypot(vx, vy)
            # Velocity clamping: max 18 px/frame to avoid runaway drift
            if 1.0 <= speed <= 30.0:
                scale = min(1.0, 18.0 / speed)
                vx *= scale
                vy *= scale
                new_b = [
                    max(0.0, min(float(img_w), self.bbox[0] + vx)),
                    max(0.0, min(float(img_h), self.bbox[1] + vy)),
                    max(0.0, min(float(img_w), self.bbox[2] + vx)),
                    max(0.0, min(float(img_h), self.bbox[3] + vy)),
                ]
                self.bbox = new_b
                new_cx = (new_b[0] + new_b[2]) / 2.0
                new_cy = (new_b[1] + new_b[3]) / 2.0
                self.positions.append((new_cx, new_cy))

    def _update_consensus_class(self, current_frame_idx: int):
        """Calculate weighted class score over observation history with Sticky Class Locking.
        
        Uses responsive decay (0.85) and an established-class hysteresis margin (25%) so that
        transient 1-frame misdetections cannot flip an established object's class.
        """
        scores = defaultdict(float)
        for f_i, c_name, conf, b in self.history:
            dt = max(0, current_frame_idx - f_i)
            weight = (0.85 ** dt) * conf
            bw = b[2] - b[0]
            bh = max(1.0, b[3] - b[1])
            # Natural geometric prior: footwear sole aspect ratio favors sepatu
            if c_name == "sepatu" and bw >= 0.75 * bh:
                weight *= 1.35
            scores[c_name] += weight

        if scores:
            challenger = max(scores.keys(), key=lambda c: scores[c])
            # Sticky Class Locking: established class requires 25% margin to be overridden
            if challenger != self.class_name and self.hits >= 3 and self.class_name in scores:
                if scores[challenger] < 1.25 * scores[self.class_name]:
                    challenger = self.class_name
            self.class_name = challenger

    def _check_static_fixture(self):
        """Detect if tracklet has stayed completely motionless across many frames (machinery glare)."""
        if len(self.positions) < 15:
            return

        pos_list = list(self.positions)
        cx_vals = [p[0] for p in pos_list]
        cy_vals = [p[1] for p in pos_list]

        dx = max(cx_vals) - min(cx_vals)
        dy = max(cy_vals) - min(cy_vals)
        total_disp = math.sqrt(dx * dx + dy * dy)

        # If total displacement is under 3.5 pixels across 15+ frames, it's a fixed rigid fixture
        if total_disp < 3.5 and self.hits >= 12:
            self.is_static_fixture = True
        else:
            self.is_static_fixture = False

    def to_detection_dict(self) -> Dict[str, Any]:
        """Convert tracklet state into standardized detection dictionary."""
        return {
            "bbox": [round(float(c), 2) for c in self.bbox],
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 3),
            "track_id": self.track_id,
            "is_stabilized": True,
        }


class APDVideoStabilizer:
    """Multi-frame video stabilization pipeline for pure APD detections.
    
    Features:
    - Object-level APD tracking across frames (independent of person detector).
    - K-of-N temporal confirmation: drops single-frame transient false positives.
    - Time-decayed class consensus: eliminates 1-frame class swaps (e.g. glove -> helm).
    - Bounding box EMA smoothing: eliminates box coordinate jitter.
    - Rig-specific physical & anatomical filters: suppress lamps, switch panels, and double stacked helmets.
    - Inanimate static fixture suppression: removes fixed machinery misdetections.
    - Graceful occlusion coasting: bridges 1-2 frame momentary detector dropouts.
    """

    def __init__(
        self,
        iou_match_thresh: float = 0.25,
        centroid_dist_ratio: float = 0.65,
        high_conf_instant: float = 0.65,
        min_hits_to_confirm: int = 2,
        max_idle_frames: int = 2,
        bbox_smooth_alpha: float = 0.70,
    ):
        self.iou_match_thresh = iou_match_thresh
        self.centroid_dist_ratio = centroid_dist_ratio
        self.high_conf_instant = high_conf_instant
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_idle_frames = max_idle_frames
        self.bbox_smooth_alpha = bbox_smooth_alpha

        self.next_track_id = 1
        self.tracklets: Dict[int, APDTracklet] = {}
        self.worker_container = WorkerAnatomicalContainer()
        self.current_workers: List[Dict[str, Any]] = []

    def reset(self):
        """Reset all active tracks (e.g. at the start of a new video clip)."""
        self.next_track_id = 1
        self.tracklets.clear()
        if self.worker_container is not None:
            self.worker_container.reset()
        self.current_workers.clear()

    def process_frame(
        self,
        frame_idx: int,
        raw_detections: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
        person_boxes: Optional[List[List[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming raw detections for the current video frame and return stabilized detections."""
        # Step 1: Pre-filter physically impossible single-frame rig anomalies
        valid_dets = self._filter_rig_physical_anomalies(raw_detections, img_w, img_h)

        # Step 2: Associate detections with active tracklets
        active_tids = list(self.tracklets.keys())
        matched_pairs, unmatched_dets, unmatched_tids = self._associate_detections(
            valid_dets, active_tids, img_w, img_h
        )

        # Step 3: Update matched tracklets
        for det_idx, tid in matched_pairs:
            det = valid_dets[det_idx]
            self.tracklets[tid].update(
                bbox=det["bbox"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                frame_idx=frame_idx,
                alpha=self.bbox_smooth_alpha,
            )

        # Step 4: Handle unmatched detections (spawn new tracklets)
        for det_idx in unmatched_dets:
            det = valid_dets[det_idx]
            tid = self.next_track_id
            self.next_track_id += 1
            trk = APDTracklet(
                track_id=tid,
                bbox=det["bbox"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                frame_idx=frame_idx,
            )
            # High confidence detections can be confirmed immediately
            if det["confidence"] >= self.high_conf_instant:
                trk.is_confirmed = True
            self.tracklets[tid] = trk

        # Step 5: Handle unmatched tracklets (mark missed / idle)
        for tid in unmatched_tids:
            self.tracklets[tid].mark_missed(frame_idx, img_w, img_h)

        # Step 6: Prune dead tracklets
        dead_tids = [
            tid for tid, trk in self.tracklets.items()
            if trk.idle_frames > self.max_idle_frames
        ]
        for tid in dead_tids:
            del self.tracklets[tid]

        # Step 7: Filter and collect output detections for this frame
        output_detections: List[Dict[str, Any]] = []
        for trk in self.tracklets.values():
            # Check confirmation status (K-of-N filter)
            if not trk.is_confirmed:
                if trk.hits >= self.min_hits_to_confirm:
                    trk.is_confirmed = True
                else:
                    continue  # Suppress unconfirmed transient spikes!

            # Suppress static machinery fixtures with 0 motion (ONLY for false machinery detections like glove on crane)
            # NEVER suppress helmets, glasses, or boots of stationary/seated workers!
            if trk.is_static_fixture and trk.class_name not in ["helm", "kacamata", "sepatu"]:
                continue

            # If coasting (missed this exact frame):
            # With linear velocity extrapolation, a confirmed tracklet smoothly bridges idle_frames == 1
            # along its motion vector (preventing both lag and flickering/kedip-kedip).
            # If missed for > 1 frame or unverified, do not display.
            if trk.idle_frames > 0:
                if trk.idle_frames > 1 or trk.hits < 2:
                    continue

            # Apply Rig Specific Post-Check (e.g. switch panel & head stacking)
            det_dict = trk.to_detection_dict()
            output_detections.append(det_dict)

        # Step 8: Post-Deduplication (Double helmets on same head, cross-class overlaps)
        final_clean = self._post_clean_frame(output_detections, img_w, img_h)

        # Step 9: Worker Container & Anatomical Quota Enforcement
        if self.worker_container is not None:
            clean_apds, workers = self.worker_container.process_frame(
                frame_idx, final_clean, person_boxes or [], img_w, img_h
            )
            self.current_workers = workers
            return clean_apds

        return final_clean

    def get_current_workers(self) -> List[Dict[str, Any]]:
        """Return the current tight worker bounding boxes and compliance states."""
        return self.current_workers

    def get_current_worker_detections(self) -> list:
        """Convert current workers to WorkerDetection schema objects for rendering."""
        from app.schemas.detection import WorkerDetection, BoundingBox, APDItem
        objs = []
        for w in self.current_workers:
            tb = w.get("tight_box", w["body_box"])
            missing = [k for k, v in w["checklist"].items() if not v]
            if w.get("is_feet_out_of_frame", False) and "sepatu" in missing:
                missing.remove("sepatu")
            if w["is_compliant"]:
                missing = []
            w_obj = WorkerDetection(
                track_id=w["worker_id"],
                bbox=BoundingBox(x1=tb[0], y1=tb[1], x2=tb[2], y2=tb[3]),
                person_conf=0.90,
                checklist=w["checklist"],
                is_compliant=w["is_compliant"],
                is_partial=w.get("is_partial", False),
                status_label=w.get("status_label", None),
                missing_items=missing,
                detected_apds=[
                    APDItem(
                        class_name=a["class_name"],
                        confidence=a["confidence"],
                        bbox=BoundingBox(
                            x1=a["bbox"][0], y1=a["bbox"][1], x2=a["bbox"][2], y2=a["bbox"][3]
                        ),
                        overlap_ratio=1.0,
                        is_worn=True,
                    )
                    for a in w["apds"]
                ],
            )
            objs.append(w_obj)
        return objs

    def _associate_detections(
        self,
        detections: List[Dict[str, Any]],
        active_tids: List[int],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform bipartite matching between incoming detections and active tracklets."""
        if not active_tids:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(active_tids)

        cost_matrix = []
        for det_idx, det in enumerate(detections):
            d_box = det["bbox"]
            d_cls = det["class_name"]
            d_cx = (d_box[0] + d_box[2]) / 2.0
            d_cy = (d_box[1] + d_box[3]) / 2.0
            d_diag = math.sqrt((d_box[2] - d_box[0]) ** 2 + (d_box[3] - d_box[1]) ** 2)

            for tid in active_tids:
                trk = self.tracklets[tid]
                t_box = trk.bbox
                t_cls = trk.class_name
                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0
                t_diag = math.sqrt((t_box[2] - t_box[0]) ** 2 + (t_box[3] - t_box[1]) ** 2)

                iou = box_iou(d_box, t_box)
                containment = box_containment(d_box, t_box)

                # Normalized centroid distance
                avg_diag = max(10.0, (d_diag + t_diag) / 2.0)
                dist = math.sqrt((d_cx - t_cx) ** 2 + (d_cy - t_cy) ** 2) / avg_diag

                # Match metric: high IoU or small centroid distance
                # Grant class affinity bonus if same class
                class_bonus = 0.20 if d_cls == t_cls else 0.0
                score = max(iou, containment * 0.70) + class_bonus - (dist * 0.35)

                # Matching eligibility condition:
                # 1. IoU >= thresh
                # 2. Or centroid distance within ratio and containment/class matches
                # 3. Or small moving objects (hands, shoes) with high class affinity
                is_eligible = (
                    iou >= self.iou_match_thresh
                    or (dist <= self.centroid_dist_ratio and (containment >= 0.30 or d_cls == t_cls))
                    or (d_diag < 85.0 and dist <= 0.85 and d_cls == t_cls)
                )

                if is_eligible:
                    cost_matrix.append((score, det_idx, tid))

        # Greedy assignment sorted by best score
        cost_matrix.sort(key=lambda x: x[0], reverse=True)
        matched_pairs: List[Tuple[int, int]] = []
        assigned_dets = set()
        assigned_tids = set()

        for score, det_idx, tid in cost_matrix:
            if det_idx not in assigned_dets and tid not in assigned_tids:
                matched_pairs.append((det_idx, tid))
                assigned_dets.add(det_idx)
                assigned_tids.add(tid)

        unmatched_dets = [i for i in range(len(detections)) if i not in assigned_dets]
        unmatched_tids = [t for t in active_tids if t not in assigned_tids]

        return matched_pairs, unmatched_dets, unmatched_tids

    def _filter_rig_physical_anomalies(
        self, detections: List[Dict[str, Any]], img_w: int, img_h: int
    ) -> List[Dict[str, Any]]:
        """Universal sanity validation and intra-frame deduplication (agnostic to camera coordinates)."""
        filtered = []
        for det in detections:
            b = det["bbox"]
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            # Discard degenerate zero-area or out-of-frame boxes
            if bw < 4.0 or bh < 4.0:
                continue
            if b[0] < 0 or b[1] < 0 or b[2] > img_w + 10 or b[3] > img_h + 10:
                continue
            filtered.append(det)

        if len(filtered) <= 1:
            return filtered

        # Intra-frame cross-class resolution:
        # Footwear soles naturally present a horizontal aspect ratio (bw >= 0.85 bh).
        # When glove and sepatu compete for the same physical footprint in raw detections, favor sepatu if horizontal.
        def det_prio(d):
            c = d["class_name"]
            b = d["bbox"]
            bw = b[2] - b[0]
            bh = max(1.0, b[3] - b[1])
            prio = float(d["confidence"])
            if c == "sepatu" and bw >= 0.75 * bh:
                prio += 0.30
            return prio

        filtered.sort(key=det_prio, reverse=True)
        kept = []
        for cand in filtered:
            cb = cand["bbox"]
            c_cls = cand["class_name"]
            has_conflict = False
            for k in kept:
                kb = k["bbox"]
                k_cls = k["class_name"]
                if box_iou(cb, kb) > 0.30 or box_containment(cb, kb) > 0.45:
                    if (c_cls == "glove" and k_cls == "sepatu") or (c_cls == "sepatu" and k_cls == "glove"):
                        has_conflict = True
                        break
                    elif c_cls == k_cls:
                        has_conflict = True
                        break
            if not has_conflict:
                kept.append(cand)

        return kept

    def _post_clean_frame(
        self, detections: List[Dict[str, Any]], img_w: int, img_h: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate stacked helmets on the same head and resolve conflicting overlapping boxes."""
        if len(detections) <= 1:
            return detections

        # 1. Vertical Head Stacking Deduplication (Universal: 1 head = 1 helmet)
        helms = [d for d in detections if d["class_name"] == "helm"]
        non_helms = [d for d in detections if d["class_name"] != "helm"]

        cleaned_helms = []
        helms.sort(key=lambda x: x["confidence"], reverse=True)
        for h_cand in helms:
            cb = h_cand["bbox"]
            ccx = (cb[0] + cb[2]) / 2.0
            ccy = (cb[1] + cb[3]) / 2.0
            ch = max(1.0, cb[3] - cb[1])

            is_stacked_subbox = False
            for kept in cleaned_helms:
                kb = kept["bbox"]
                kcx = (kb[0] + kb[2]) / 2.0
                kcy = (kb[1] + kb[3]) / 2.0
                kh = max(1.0, kb[3] - kb[1])

                horiz_dist = abs(ccx - kcx)
                vert_dist = abs(ccy - kcy)

                # Same head check: vertically aligned within 0.5 width and 1.25 height
                if horiz_dist < 0.50 * (cb[2] - cb[0]) and vert_dist < 1.25 * max(ch, kh):
                    is_stacked_subbox = True
                    break

                # General IoU overlap between helmets
                if box_iou(cb, kb) > 0.25 or box_containment(cb, kb) > 0.40:
                    is_stacked_subbox = True
                    break

            if not is_stacked_subbox:
                cleaned_helms.append(h_cand)

        all_dets = cleaned_helms + non_helms

        # 2. Universal Conflict Resolution (Glove vs Sepatu)
        # Footwear soles naturally present a horizontal aspect ratio (bw >= 0.85 bh).
        # When glove and sepatu compete for the same physical footprint, favor sepatu if horizontal.
        def det_priority(d):
            c = d["class_name"]
            b = d["bbox"]
            bw = b[2] - b[0]
            bh = max(1.0, b[3] - b[1])
            prio = float(d["confidence"])
            if c == "sepatu" and bw >= 0.75 * bh:
                prio += 0.30  # Geometric preference for horizontal footwear soles
            return prio

        all_dets.sort(key=det_priority, reverse=True)
        final_kept = []
        for cand in all_dets:
            cb = cand["bbox"]
            c_cls = cand["class_name"]
            has_conflict = False

            for kept in final_kept:
                kb = kept["bbox"]
                k_cls = kept["class_name"]
                iou = box_iou(cb, kb)
                containment = box_containment(cb, kb)

                # Mutually exclusive: Glove vs Sepatu
                if (c_cls == "glove" and k_cls == "sepatu") or (c_cls == "sepatu" and k_cls == "glove"):
                    if iou > 0.15 or containment > 0.30:
                        has_conflict = True
                        break

                # Mutually exclusive: Helm vs (Glove/Sepatu)
                elif (c_cls == "helm" and k_cls in ["sepatu", "glove"]) or (c_cls in ["sepatu", "glove"] and k_cls == "helm"):
                    if iou > 0.15 or containment > 0.30:
                        has_conflict = True
                        break

                # Duplicate same class
                elif c_cls == k_cls:
                    thresh = 0.25 if c_cls == "kacamata" else 0.40
                    if iou > thresh or containment > 0.50:
                        has_conflict = True
                        break

            if not has_conflict:
                final_kept.append(cand)

        return final_kept


class WorkerAnatomicalContainer:
    """Universal, Camera-Agnostic Worker Container with Anatomical APD Quotas.
    
    Features:
    1. Robust Human Anchoring: Uses confirmed helmets (and person boxes) to anchor each worker.
    2. Tight Worker Bounding Box: Wraps tightly around the worker's head, body, and limbs without swallowing machinery.
    3. Isolated Machinery Suppression: Drops spurious floating APD on pipes/fixtures far from any worker.
    4. Anatomical Quota per Worker:
       - Helm: Max 1 (head zone).
       - Kacamata: Max 1 (face zone).
       - Sepatu: Max 2 (feet zone).
       - Glove: Max 2 via Bilateral Pairing (drops false gloves on pants/trousers pockets).
    5. Grace Period: Compliance memory prevents false violations when workers bend over (menunduk).
    """

    def __init__(self, temporal_grace_frames: int = 90):
        self.temporal_grace_frames = temporal_grace_frames
        # worker_id -> class_name -> last_seen_frame
        self.worker_compliance_memory: Dict[int, Dict[str, int]] = {}
        self.next_worker_id = 1
        # track_id -> {"centroid": (cx, cy), "body_box": [...], "last_frame": int}
        self.active_worker_tracks: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        self.worker_compliance_memory.clear()
        self.next_worker_id = 1
        self.active_worker_tracks.clear()

    def _assign_worker_ids(
        self, worker_candidates: List[Dict[str, Any]], frame_idx: int
    ) -> List[Dict[str, Any]]:
        """Match incoming worker candidates to persistent worker track IDs via scale-invariant Hungarian matching."""
        # Purge stale tracks older than 75 frames (bridges momentary worker occlusions)
        self.active_worker_tracks = {
            tid: data for tid, data in self.active_worker_tracks.items()
            if frame_idx - data["last_frame"] <= 75
        }

        def get_cand_anchor(cand):
            if cand["helmet"] is not None:
                hb = cand["helmet"]["bbox"]
                return {
                    "hc": ((hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0),
                    "hw": max(10.0, hb[2] - hb[0]),
                    "hh": max(10.0, hb[3] - hb[1]),
                    "has_helmet": True,
                }
            bb = cand["body_box"]
            bw = max(10.0, bb[2] - bb[0])
            bh = max(10.0, bb[3] - bb[1])
            return {
                "hc": ((bb[0] + bb[2]) / 2.0, bb[1] + bh * 0.20),
                "hw": bw * 0.40,
                "hh": bh * 0.25,
                "has_helmet": False,
            }

        def get_next_id():
            used = set(self.active_worker_tracks.keys())
            cand_id = 1
            while cand_id in used:
                cand_id += 1
            return cand_id

        if not self.active_worker_tracks:
            assigned = []
            for cand in worker_candidates:
                tid = get_next_id()
                cand["worker_id"] = tid
                bb = cand["body_box"]
                anc = get_cand_anchor(cand)
                self.active_worker_tracks[tid] = {
                    "centroid": ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0),
                    "body_box": list(bb),
                    "anchor": anc,
                    "last_frame": frame_idx,
                }
                assigned.append(cand)
            assigned.sort(key=lambda x: x["worker_id"])
            return assigned

        active_tids = sorted(list(self.active_worker_tracks.keys()))
        n_cands = len(worker_candidates)
        n_tracks = len(active_tids)

        # Build scale-invariant cost matrix
        cost_matrix = []
        for cand in worker_candidates:
            c_anc = get_cand_anchor(cand)
            c_hc = c_anc["hc"]
            c_hw = c_anc["hw"]
            bb = cand["body_box"]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0

            row = []
            for tid in active_tids:
                t_data = self.active_worker_tracks[tid]
                t_anc = t_data.get("anchor")
                t_bb = t_data["body_box"]
                tcx, tcy = t_data["centroid"]

                if t_anc is not None:
                    t_hc = t_anc["hc"]
                    t_hw = t_anc["hw"]
                    dist = math.hypot(c_hc[0] - t_hc[0], c_hc[1] - t_hc[1])
                    ref_scale = max(25.0, c_hw, t_hw)
                    norm_dist = dist / ref_scale
                    # Spatial gate: helmet anchor cannot jump > 3.0 helmet widths or > 180px between frames
                    if norm_dist > 3.0 and dist > 180.0:
                        cost = 9999.0
                    else:
                        cost = norm_dist
                else:
                    iou = box_iou(bb, t_bb)
                    dist = math.hypot(cx - tcx, cy - tcy)
                    diag = max(20.0, math.hypot(bb[2] - bb[0], bb[3] - bb[1]))
                    norm_dist = dist / diag
                    if norm_dist > 0.85 and iou < 0.10:
                        cost = 9999.0
                    else:
                        cost = norm_dist + (1.0 - iou) * 1.5

                row.append(cost)
            cost_matrix.append(row)

        # Global optimal bipartite matching (guarantees minimum global distance, eliminating label flips)
        import itertools
        best_pairs = []
        if n_cands > 0 and n_tracks > 0:
            if n_cands <= n_tracks:
                best_cost = float("inf")
                best_p = None
                for p in itertools.permutations(range(n_tracks), n_cands):
                    c = sum(cost_matrix[i][p[i]] for i in range(n_cands))
                    if c < best_cost:
                        best_cost = c
                        best_p = p
                if best_p is not None:
                    for i in range(n_cands):
                        if cost_matrix[i][best_p[i]] < 4.0:
                            best_pairs.append((i, best_p[i]))
            else:
                best_cost = float("inf")
                best_p = None
                for p in itertools.permutations(range(n_cands), n_tracks):
                    c = sum(cost_matrix[p[j]][j] for j in range(n_tracks))
                    if c < best_cost:
                        best_cost = c
                        best_p = p
                if best_p is not None:
                    for j in range(n_tracks):
                        if cost_matrix[best_p[j]][j] < 4.0:
                            best_pairs.append((best_p[j], j))

        assigned_c_idxs = set()
        assigned_tids = set()
        assigned = []

        for c_idx, t_idx in best_pairs:
            tid = active_tids[t_idx]
            cand = worker_candidates[c_idx]
            cand["worker_id"] = tid
            bb = cand["body_box"]
            anc = get_cand_anchor(cand)
            self.active_worker_tracks[tid] = {
                "centroid": ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0),
                "body_box": list(bb),
                "anchor": anc,
                "last_frame": frame_idx,
            }
            assigned_c_idxs.add(c_idx)
            assigned_tids.add(tid)
            assigned.append(cand)

        # Handle genuinely new unmatched candidates by assigning lowest unused integer ID
        for c_idx, cand in enumerate(worker_candidates):
            if c_idx not in assigned_c_idxs:
                tid = get_next_id()
                cand["worker_id"] = tid
                bb = cand["body_box"]
                anc = get_cand_anchor(cand)
                self.active_worker_tracks[tid] = {
                    "centroid": ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0),
                    "body_box": list(bb),
                    "anchor": anc,
                    "last_frame": frame_idx,
                }
                assigned.append(cand)

        assigned.sort(key=lambda x: x["worker_id"])
        return assigned

    def process_frame(
        self,
        frame_idx: int,
        apds: List[Dict[str, Any]],
        person_boxes: List[List[float]],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not apds and not person_boxes:
            return [], []

        # Find helmets with conf >= 0.22 (robust to seated/partially occluded workers)
        helmets = [a for a in apds if a["class_name"] == "helm" and a["confidence"] >= 0.22]
        helmets.sort(key=lambda x: x["confidence"], reverse=True)

        distinct_helmets = []
        for h in helmets:
            hb = h["bbox"]
            is_dup = False
            for dh in distinct_helmets:
                if box_iou(hb, dh["bbox"]) > 0.25:
                    is_dup = True
                    break
            if not is_dup:
                distinct_helmets.append(h)

        # Fallback if no helmets and no person boxes exist (e.g. isolated test case)
        if not distinct_helmets and not person_boxes:
            return apds, []

        worker_candidates = []
        matched_pb_indices = set()

        if distinct_helmets:
            for h_det in distinct_helmets:
                hb = h_det["bbox"]
                hw = hb[2] - hb[0]
                hh = hb[3] - hb[1]
                hcx = (hb[0] + hb[2]) / 2.0
                hcy = (hb[1] + hb[3]) / 2.0

                matched_pb = None
                best_match_score = -1.0
                best_p_idx = None
                for p_idx, pb in enumerate(person_boxes):
                    pw = pb[2] - pb[0]
                    ph = pb[3] - pb[1]
                    is_x_aligned = pb[0] - hw * 0.5 <= hcx <= pb[2] + hw * 0.5
                    is_y_aligned = pb[1] - hh * 0.5 <= hcy <= pb[3] + hh * 0.25
                    iou = box_iou(hb, pb)
                    containment = box_containment(hb, pb)

                    if (is_x_aligned and is_y_aligned) or iou >= 0.20 or containment >= 0.40:
                        # Favor higher IoU and containment
                        score = max(iou, containment * 0.8) + (0.5 if (is_x_aligned and is_y_aligned) else 0.0)
                        if score > best_match_score:
                            best_match_score = score
                            matched_pb = pb
                            best_p_idx = p_idx

                if matched_pb is not None:
                    matched_pb_indices.add(best_p_idx)
                    body_box = list(matched_pb)
                else:
                    body_box = [
                        max(0.0, hcx - hw * 1.4),
                        max(0.0, hb[1] - hh * 0.1),
                        min(float(img_w), hcx + hw * 1.4),
                        min(float(img_h), hb[1] + hh * 4.2),
                    ]

                worker_candidates.append({
                    "helmet": h_det,
                    "body_box": body_box,
                    "apds": [],
                })

        # Add remaining person detections that have no helmet (merging lower-body boxes into existing workers)
        for p_idx, pb in enumerate(person_boxes):
            if p_idx in matched_pb_indices:
                continue

            pb_cx = (pb[0] + pb[2]) / 2.0
            pb_w = pb[2] - pb[0]
            pb_h = pb[3] - pb[1]

            is_absorbed = False
            for w_cand in worker_candidates:
                wb = w_cand["body_box"]
                ww = max(1.0, wb[2] - wb[0])
                wh = max(1.0, wb[3] - wb[1])
                w_cx = (wb[0] + wb[2]) / 2.0

                # 1. Overlap / Containment check
                iou = box_iou(pb, wb)
                cont = box_containment(pb, wb)

                # 2. Vertical Column Alignment (human body is an upright anatomical column)
                # pb is horizontally aligned if center difference is within 65% of worker width
                is_x_col = abs(pb_cx - w_cx) <= max(ww * 0.65, pb_w * 0.65)
                # pb is vertically overlapping or attached directly below worker (e.g. legs/boots)
                is_y_connected = (pb[1] <= wb[3] + 25.0) and (pb[3] >= wb[1] + wh * 0.20)

                if iou > 0.15 or cont > 0.30 or (is_x_col and is_y_connected):
                    # Absorb / merge pb into existing worker's body box (prevents shoe splitting!)
                    wb[0] = min(wb[0], pb[0])
                    wb[1] = min(wb[1], pb[1])
                    wb[2] = max(wb[2], pb[2])
                    wb[3] = max(wb[3], pb[3])
                    is_absorbed = True
                    break

            if not is_absorbed:
                # Discard isolated person detections without helmets unless large and full-body
                # (Prevents background pipes, drill machinery, and floating boot boxes from spawning phantom workers)
                if pb_w >= 45.0 and pb_h >= 90.0:
                    worker_candidates.append({
                        "helmet": None,
                        "body_box": list(pb),
                        "apds": [],
                    })

        # Assign persistent worker IDs across frames
        workers = self._assign_worker_ids(worker_candidates, frame_idx)

        # Associate non-helmet APDs with the nearest Worker adhering to Physical & Anatomical Laws
        non_helm_apds = [a for a in apds if a["class_name"] != "helm"]
        for apd in non_helm_apds:
            ab = apd["bbox"]
            acx = (ab[0] + ab[2]) / 2.0
            acy = (ab[1] + ab[3]) / 2.0
            aw = max(1.0, ab[2] - ab[0])
            ah = max(1.0, ab[3] - ab[1])
            cname = apd["class_name"]

            best_worker = None
            best_dist = float("inf")

            for w in workers:
                wb = w["body_box"]
                ww = max(20.0, wb[2] - wb[0])
                wh = max(20.0, wb[3] - wb[1])
                wcx = (wb[0] + wb[2]) / 2.0
                wcy = (wb[1] + wb[3]) / 2.0

                # Check if this worker's lower body is truncated by the bottom camera edge
                is_worker_bottom_truncated = (wb[3] >= img_h - 35.0) and (wb[1] > img_h * 0.25)
                # Physical Rule 1: A bottom-truncated worker has NO feet in camera frame!
                # NEVER assign background shoes to a foreground worker whose feet are off-screen!
                if is_worker_bottom_truncated and cname == "sepatu":
                    continue

                # Physical Rule 2: Anatomical Bounds & Perspective Scale Consistency
                if w["helmet"] is not None:
                    hb = w["helmet"]["bbox"]
                    hw = max(1.0, hb[2] - hb[0])
                    hh = max(1.0, hb[3] - hb[1])
                    hcx = (hb[0] + hb[2]) / 2.0
                    hcy = (hb[1] + hb[3]) / 2.0

                    # Perspective Scale Check: In CCTV, an APD on the same worker must have compatible scale!
                    # A 30px shoe cannot belong to a 380px foreground worker!
                    scale_ratio = aw / hw
                    if scale_ratio < 0.18:
                        continue  # Background APD far in the distance; belongs to another worker!

                    if cname == "sepatu":
                        # Sepatu MUST be physically BELOW the helmet dome!
                        if acy < hb[3] + hh * 0.20:
                            continue
                        # Sepatu must be aligned within the vertical anatomical column
                        if abs(acx - hcx) > hw * 1.75:
                            continue

                    elif cname == "kacamata":
                        # Kacamata MUST be situated in the facial zone directly under helmet
                        if acy < hb[1] - hh * 0.15 or acy > hb[3] + hh * 0.65:
                            continue
                        if abs(acx - hcx) > hw * 0.75:
                            continue

                    elif cname == "glove":
                        # Human arm reach: gloves can reach forward, up, down, or sideways
                        # In high-angle / overhead CCTV, forward reach is projected higher up (lower Y) in 2D image
                        arm_reach = max(hw * 2.5, 120.0)
                        dist_to_helmet = math.hypot(acx - hcx, acy - hcy)
                        if dist_to_helmet > arm_reach * 1.8:
                            continue
                        if abs(acx - hcx) > max(hw * 2.4, 95.0):
                            continue

                # Proximity inside worker's reach zone
                norm_dx = abs(acx - wcx) / ww
                norm_dy = abs(acy - wcy) / wh
                dist = math.hypot(norm_dx, norm_dy)

                # Confirmed helmet workers get preference
                if w["helmet"] is not None:
                    dist *= 0.70

                if dist < best_dist and dist <= 1.25:
                    best_dist = dist
                    best_worker = w

            if best_worker is not None:
                best_worker["apds"].append(apd)

        # Apply Quota and Bilateral Pairing per Worker
        final_clean_apds = []
        final_workers = []

        for w in workers:
            w_apds = w["apds"]
            kept_for_worker = []
            if w["helmet"] is not None:
                kept_for_worker.append(w["helmet"])

            # 1. Kacamata: Max 1 in face region
            kacamata_cands = [a for a in w_apds if a["class_name"] == "kacamata"]
            if kacamata_cands:
                kacamata_cands.sort(key=lambda x: x["confidence"], reverse=True)
                kept_for_worker.append(kacamata_cands[0])

            # 2. Sepatu: Max 2 in feet region
            sepatu_cands = [a for a in w_apds if a["class_name"] == "sepatu"]
            if sepatu_cands:
                sepatu_cands.sort(key=lambda x: x["confidence"], reverse=True)
                kept_for_worker.extend(sepatu_cands[:2])

            # 3. Glove: Bilateral Hand Pairing (Max 2)
            glove_cands = [a for a in w_apds if a["class_name"] == "glove"]
            if len(glove_cands) <= 2:
                kept_for_worker.extend(glove_cands)
            else:
                wb = w["body_box"]
                wcx = (wb[0] + wb[2]) / 2.0
                ww = max(1.0, wb[2] - wb[0])

                best_pair = None
                best_score = -1.0
                for i in range(len(glove_cands)):
                    for j in range(i + 1, len(glove_cands)):
                        g1 = glove_cands[i]
                        g2 = glove_cands[j]
                        g1_cx = (g1["bbox"][0] + g1["bbox"][2]) / 2.0
                        g2_cx = (g2["bbox"][0] + g2["bbox"][2]) / 2.0

                        lat1 = (g1_cx - wcx) / ww
                        lat2 = (g2_cx - wcx) / ww
                        is_bilateral = (lat1 * lat2 < 0) or (abs(lat1 - lat2) > 0.30)
                        bilateral_bonus = 0.50 if is_bilateral else 0.0

                        sep_dist = abs(g1_cx - g2_cx) / ww
                        pair_score = g1["confidence"] + g2["confidence"] + bilateral_bonus + (sep_dist * 0.3)

                        if pair_score > best_score:
                            best_score = pair_score
                            best_pair = [g1, g2]

                if best_pair is not None:
                    kept_for_worker.extend(best_pair)
                else:
                    kept_for_worker.extend(glove_cands[:2])

            # CRITICAL RULE 1: Suppress phantom / ghost person detections!
            # An inanimate drill pipe, machinery pillar, or background shadow with NO helmet and NO confirmed APDs
            # MUST be discarded immediately!
            if w["helmet"] is None and len(kept_for_worker) == 0:
                continue

            # CRITICAL RULE 2: If a candidate has no helmet, require at least 2 distinct APDs (e.g. glove + sepatu)
            # with solid confidence (>= 0.35) to prevent faint noise reflections on machinery from being deemed a worker!
            if w["helmet"] is None:
                if len(kept_for_worker) < 2:
                    continue
                max_conf = max((float(a["confidence"]) for a in kept_for_worker), default=0.0)
                if max_conf < 0.35:
                    continue

            # Construct Tight Worker Bounding Box
            all_b = [a["bbox"] for a in kept_for_worker]
            if all_b:
                tight_x1 = max(0.0, min(b[0] for b in all_b) - 20.0)
                tight_y1 = max(0.0, min(b[1] for b in all_b) - 15.0)
                tight_x2 = min(float(img_w), max(b[2] for b in all_b) + 20.0)
                shoes_present = [a for a in kept_for_worker if a["class_name"] == "sepatu"]
                if shoes_present:
                    tight_y2 = min(float(img_h), max(s["bbox"][3] for s in shoes_present) + 15.0)
                else:
                    tight_y2 = min(float(img_h), max(w["body_box"][3], tight_y1 + (tight_x2 - tight_x1) * 1.8))
            else:
                tight_x1, tight_y1, tight_x2, tight_y2 = w["body_box"]

            wb = w["body_box"]
            # Check if feet are physically truncated by the camera's bottom frame boundary
            # (e.g. driller sitting at console where camera view is cropped above waist)
            is_feet_out_of_frame = (wb[3] >= img_h - 35.0) and (wb[1] > img_h * 0.25)
            w["is_feet_out_of_frame"] = is_feet_out_of_frame

            # EMA Bounding Box Smoothing: Eliminates coordinate jitter & box flickering!
            # Smoothly tracks movement without sudden size jumps when an APD drops out momentarily
            track_info = self.active_worker_tracks.get(w["worker_id"])
            if track_info is not None:
                prev_smooth = track_info.get("smoothed_box")
                if prev_smooth is not None:
                    alpha_smooth = 0.75  # 75% previous smoothed state, 25% new measurement
                    smoothed_tight = [
                        alpha_smooth * prev_smooth[0] + (1.0 - alpha_smooth) * tight_x1,
                        alpha_smooth * prev_smooth[1] + (1.0 - alpha_smooth) * tight_y1,
                        alpha_smooth * prev_smooth[2] + (1.0 - alpha_smooth) * tight_x2,
                        alpha_smooth * prev_smooth[3] + (1.0 - alpha_smooth) * tight_y2,
                    ]
                else:
                    smoothed_tight = [tight_x1, tight_y1, tight_x2, tight_y2]
                track_info["smoothed_box"] = smoothed_tight
                w["tight_box"] = smoothed_tight
            else:
                w["tight_box"] = [tight_x1, tight_y1, tight_x2, tight_y2]

            w["apds"] = kept_for_worker

            # Compliance Memory (Grace periods for menunduk, squatting, and occlusions)
            has_helm = any(a["class_name"] == "helm" for a in kept_for_worker)
            has_kacamata = any(a["class_name"] == "kacamata" for a in kept_for_worker)
            has_glove = any(a["class_name"] == "glove" for a in kept_for_worker)
            has_sepatu = any(a["class_name"] == "sepatu" for a in kept_for_worker)

            mem = self.worker_compliance_memory.setdefault(w["worker_id"], {})
            if has_helm: mem["helm"] = frame_idx
            if has_kacamata: mem["kacamata"] = frame_idx
            if has_glove: mem["glove"] = frame_idx
            if has_sepatu: mem["sepatu"] = frame_idx

            # Context-Aware Industrial Compliance Logic:
            # 1. Helm: Mandatory on rig
            # 2. Glove: MANDATORY on visible hands! NEVER excused by feet being out of frame!
            # 3. Sepatu: Mandatory IF feet are in frame. If feet are out of camera frame, exempt from penalty.
            # 4. Kacamata: Accommodates looking down / helmet brim occlusion.
            comp_helm = (frame_idx - mem.get("helm", -999)) <= 180
            comp_glove = (frame_idx - mem.get("glove", -999)) <= 150
            comp_sepatu = ((frame_idx - mem.get("sepatu", -999)) <= 300) or is_feet_out_of_frame
            comp_kacamata = ((frame_idx - mem.get("kacamata", -999)) <= 180) or comp_helm

            # Strict verification: Gloves are ALWAYS checked!
            is_compliant = comp_helm and comp_glove and comp_sepatu

            w["is_feet_out_of_frame"] = is_feet_out_of_frame
            w["is_partial"] = is_feet_out_of_frame
            w["is_compliant"] = is_compliant
            if is_compliant:
                w["status_label"] = "PARSIAL (PATUH)" if is_feet_out_of_frame else "LENGKAP (100%)"
            else:
                w["status_label"] = "MELANGGAR"

            w["checklist"] = {
                "helm": comp_helm,
                "kacamata": comp_kacamata,
                "glove": comp_glove,
                "sepatu": comp_sepatu,
            }

            final_clean_apds.extend(kept_for_worker)
            final_workers.append(w)

        return final_clean_apds, final_workers
