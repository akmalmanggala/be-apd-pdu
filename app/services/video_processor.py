"""Video Batch Processor Service with Worker Tracking and Temporal Compliance."""

import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from app.config import settings
from app.services.apd_detector import (
    APDDetectorEngine,
    SpatialPersonTracker,
    TemporalWorkerTracker,
)

logger = logging.getLogger(__name__)


class VideoBatchProcessor:
    """Processes uploaded video files frame-by-frame with worker tracking and temporal smoothing."""

    def __init__(self, detector: Optional[APDDetectorEngine] = None):
        self.detector = detector or APDDetectorEngine.get_instance()

    def process_video_file(
        self,
        video_path: str,
        output_dir: str = str(settings.OUTPUTS_DIR),
        sample_stride: int = 5,  # Process every 5th frame for balanced throughput and accuracy
        overlap_threshold: float = settings.DEFAULT_OVERLAP_THRESHOLD,
        conf_thresholds: Optional[Dict[str, float]] = None,
        person_conf: float = settings.DEFAULT_PERSON_CONF,
        enable_head_zoom: bool = settings.ENABLE_HEAD_ZOOM,
        roi_config: Optional[Dict[str, Any]] = None,
        max_frames: Optional[int] = 500,
    ) -> Dict[str, Any]:
        """Process video file, track workers, apply temporal persistence, and generate output artifacts."""
        start_time = time.perf_counter()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        session_code = f"VID-{uuid.uuid4().hex[:8].upper()}"
        out_video_filename = f"{session_code}_annotated.mp4"
        out_video_path = os.path.join(output_dir, out_video_filename)

        # Video writer (MP4V codec with fallback)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = max(1.0, orig_fps / sample_stride)
        writer = cv2.VideoWriter(out_video_path, fourcc, effective_fps, (orig_w, orig_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_video_path, fourcc, effective_fps, (orig_w, orig_h))

        spatial_tracker = SpatialPersonTracker(iou_threshold=0.25, max_idle_frames=45)
        temporal_tracker = TemporalWorkerTracker(memory_frames=settings.TEMPORAL_MEMORY_FRAMES)
        
        processed_frames_count = 0
        frame_idx = 0
        sample_images_saved: List[str] = []

        # Aggregated worker stats: track_id -> {last_seen, checklist, compliant_count, total_count}
        workers_summary: Dict[int, Dict[str, Any]] = {}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                if frame_idx % sample_stride != 0:
                    continue

                if max_frames and processed_frames_count >= max_frames:
                    break

                processed_frames_count += 1

                # Detect workers and APD with spatial track persistence and temporal smoothing
                workers, stats, annotated_frame = self.detector.detect_image(
                    img_bgr=frame,
                    overlap_threshold=overlap_threshold,
                    conf_thresholds=conf_thresholds,
                    person_conf=person_conf,
                    enable_head_zoom=enable_head_zoom,
                    roi_config=roi_config,
                    spatial_tracker=spatial_tracker,
                    temporal_tracker=temporal_tracker,
                    frame_idx=processed_frames_count,
                )

                # Record worker statistics
                for worker in workers:
                    tid = worker.track_id
                    if tid not in workers_summary:
                        workers_summary[tid] = {
                            "track_id": tid,
                            "first_seen_frame": processed_frames_count,
                            "last_seen_frame": processed_frames_count,
                            "last_bbox": [worker.bbox.x1, worker.bbox.y1, worker.bbox.x2, worker.bbox.y2],
                            "cumulative_checklist": {c: False for c in settings.CLASS_NAMES},
                            "frames_observed": 0,
                        }
                    
                    w_entry = workers_summary[tid]
                    w_entry["last_seen_frame"] = processed_frames_count
                    w_entry["last_bbox"] = [worker.bbox.x1, worker.bbox.y1, worker.bbox.x2, worker.bbox.y2]
                    w_entry["frames_observed"] += 1
                    for cname, worn in worker.checklist.items():
                        if worn:
                            w_entry["cumulative_checklist"][cname] = True

                # Write annotated frame (which is already rendered with smoothed badges!)
                writer.write(annotated_frame)

                # Save up to 5 sample keyframes
                if len(sample_images_saved) < 5 and processed_frames_count % 15 == 1:
                    sample_name = f"{session_code}_frame_{processed_frames_count}.jpg"
                    sample_path = os.path.join(output_dir, sample_name)
                    cv2.imwrite(sample_path, annotated_frame)
                    sample_images_saved.append(sample_name)

        finally:
            cap.release()
            writer.release()

        # Compute overall session compliance
        unique_workers_count = len(workers_summary)
        compliant_unique_workers = 0
        helm_violations = 0
        glove_violations = 0
        sepatu_violations = 0
        kacamata_violations = 0

        for w_data in workers_summary.values():
            cum_chk = w_data["cumulative_checklist"]
            is_worker_compliant = all(cum_chk.values())
            if is_worker_compliant:
                compliant_unique_workers += 1
            else:
                if not cum_chk["helm"]:
                    helm_violations += 1
                if not cum_chk["glove"]:
                    glove_violations += 1
                if not cum_chk["sepatu"]:
                    sepatu_violations += 1
                if not cum_chk["kacamata"]:
                    kacamata_violations += 1

        non_compliant_count = unique_workers_count - compliant_unique_workers
        overall_compliance = (
            (compliant_unique_workers / unique_workers_count * 100.0)
            if unique_workers_count > 0 else 0.0
        )

        total_sec = time.perf_counter() - start_time

        return {
            "session_code": session_code,
            "total_input_frames": total_input_frames,
            "total_frames_processed": processed_frames_count,
            "total_unique_workers": unique_workers_count,
            "compliant_workers_count": compliant_unique_workers,
            "non_compliant_workers_count": non_compliant_count,
            "overall_compliance_rate": round(overall_compliance, 2),
            "helm_violations": helm_violations,
            "glove_violations": glove_violations,
            "sepatu_violations": sepatu_violations,
            "kacamata_violations": kacamata_violations,
            "annotated_video_filename": out_video_filename,
            "annotated_sample_frames": sample_images_saved,
            "processing_time_sec": round(total_sec, 2),
            "workers_summary": list(workers_summary.values()),
        }
