"""Live Stream Frame Manager for Webcam / Real-Time Client Feeds."""

import base64
import logging
import time
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np

from app.config import settings
from app.schemas.detection import StreamFrameResponse, WorkerDetection
from app.services.apd_detector import (
    APDDetectorEngine,
    SpatialPersonTracker,
    TemporalWorkerTracker,
)

logger = logging.getLogger(__name__)


class StreamSession:
    """Represents an active streaming session from a webcam or video client."""

    def __init__(self, client_id: str, session_code: str):
        self.client_id = client_id
        self.session_code = session_code
        self.spatial_tracker = SpatialPersonTracker(iou_threshold=0.25, max_idle_frames=45)
        self.tracker = TemporalWorkerTracker(memory_frames=settings.TEMPORAL_MEMORY_FRAMES)
        self.frame_count = 0
        self.last_active = time.time()
        self.fps_tracker: list[float] = []

    def update_fps(self, delta_sec: float) -> float:
        if delta_sec > 0:
            current_fps = 1.0 / delta_sec
            self.fps_tracker.append(current_fps)
            if len(self.fps_tracker) > 20:
                self.fps_tracker.pop(0)
            return float(np.mean(self.fps_tracker))
        return 0.0


class StreamManager:
    """Manages active live-stream clients and maintains persistent temporal worker state."""

    _instance: Optional["StreamManager"] = None

    def __init__(self, detector: Optional[APDDetectorEngine] = None):
        self.detector = detector or APDDetectorEngine.get_instance()
        self.sessions: Dict[str, StreamSession] = {}

    @classmethod
    def get_instance(cls) -> "StreamManager":
        if cls._instance is None:
            cls._instance = StreamManager()
        return cls._instance

    def get_or_create_session(self, client_id: str, session_code: Optional[str] = None) -> StreamSession:
        if client_id not in self.sessions:
            code = session_code or f"STREAM-{int(time.time())}"
            self.sessions[client_id] = StreamSession(client_id, code)
        return self.sessions[client_id]

    def process_stream_frame(
        self,
        client_id: str,
        frame_base64: str,
        session_code: Optional[str] = None,
        overlap_threshold: float = settings.DEFAULT_OVERLAP_THRESHOLD,
        conf_thresholds: Optional[Dict[str, float]] = None,
        roi_config: Optional[Dict[str, Any]] = None,
    ) -> StreamFrameResponse:
        """Decode incoming base64 frame, run detector, apply client temporal tracking, and encode result."""
        session = self.get_or_create_session(client_id, session_code)
        now = time.time()
        delta = now - session.last_active
        session.last_active = now
        fps = session.update_fps(delta)
        session.frame_count += 1

        # Decode base64 image
        try:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image from base64")
        except Exception as e:
            raise ValueError(f"Invalid frame base64 encoding: {e}")

        # Run detection with client's persistent spatial tracker and temporal smoothing
        workers, stats, annotated_img = self.detector.detect_image(
            img_bgr=img_bgr,
            overlap_threshold=overlap_threshold,
            conf_thresholds=conf_thresholds,
            roi_config=roi_config,
            spatial_tracker=session.spatial_tracker,
            temporal_tracker=session.tracker,
            frame_idx=session.frame_count,
        )

        compliant_count = stats["compliant_workers_count"]
        non_compliant_count = stats["non_compliant_workers_count"]

        # Encode annotated image back to base64
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return StreamFrameResponse(
            client_id=client_id,
            session_code=session.session_code,
            frame_idx=session.frame_count,
            total_workers=len(workers),
            compliant_workers=compliant_count,
            non_compliant_workers=non_compliant_count,
            workers=workers,
            annotated_frame_base64=annotated_b64,
            fps=round(fps, 1),
        )
