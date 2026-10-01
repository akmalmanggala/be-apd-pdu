"""Detection Request and Response Schemas."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class APDItem(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox
    overlap_ratio: float = Field(..., description="Intersection over Area APD against worker body")
    is_worn: bool = Field(..., description="True if overlap_ratio >= threshold (default 80%)")


class WorkerDetection(BaseModel):
    track_id: int
    bbox: BoundingBox
    person_conf: float
    checklist: Dict[str, bool] = Field(
        ...,
        description="Compliance status per APD class: {'helm': bool, 'glove': bool, 'kacamata': bool, 'sepatu': bool}"
    )
    is_compliant: bool = Field(..., description="True if required APD items are worn")
    is_partial: bool = Field(default=False, description="True if worker is partially visible (e.g. feet out of camera frame)")
    status_label: Optional[str] = Field(default=None, description="Status label e.g. LENGKAP (100%), PARSIAL (PATUH), or MELANGGAR")
    missing_items: List[str] = Field(default_factory=list, description="List of APD items not worn")
    detected_apds: List[APDItem] = Field(default_factory=list)



class ImageDetectionResponse(BaseModel):
    session_id: Optional[int] = None
    session_code: str
    source_filename: Optional[str] = None
    total_workers: int
    compliant_workers_count: int
    non_compliant_workers_count: int
    compliance_rate: float
    workers: List[WorkerDetection]
    all_detected_apds: List[APDItem] = Field(default_factory=list, description="All APD items detected in the frame")
    unassigned_apds: List[APDItem] = Field(default_factory=list, description="APD items detected but not worn by any worker")
    annotated_image_url: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    inference_time_ms: float
    two_stage_zoom_applied: bool = True


class VideoDetectionResponse(BaseModel):
    session_id: int
    session_code: str
    source_filename: str
    total_frames_processed: int
    total_unique_workers: int
    compliant_workers_count: int
    non_compliant_workers_count: int
    overall_compliance_rate: float
    annotated_video_url: Optional[str] = None
    annotated_sample_frames: List[str] = Field(default_factory=list)
    processing_time_sec: float
    status: str = "completed"


class StreamFrameRequest(BaseModel):
    client_id: str = "default_client"
    session_code: Optional[str] = None
    frame_base64: str
    timestamp_ms: Optional[float] = None


class StreamFrameResponse(BaseModel):
    client_id: str
    session_code: str
    frame_idx: int
    total_workers: int
    compliant_workers: int
    non_compliant_workers: int
    workers: List[WorkerDetection]
    annotated_frame_base64: Optional[str] = None
    fps: float


class WorkerRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    frame_idx: int
    track_id: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    person_conf: float
    helm_worn: bool
    glove_worn: bool
    sepatu_worn: bool
    kacamata_worn: bool
    is_compliant: bool
    detected_at: datetime


class SessionDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_code: str
    source_type: str
    filename: Optional[str] = None
    annotated_output_path: Optional[str] = None
    total_frames: int
    total_workers_detected: int
    compliant_workers_count: int
    non_compliant_workers_count: int
    compliance_rate: float
    helm_violations: int
    glove_violations: int
    sepatu_violations: int
    kacamata_violations: int
    created_at: datetime
    worker_records: List[WorkerRecordResponse] = Field(default_factory=list)


class SessionListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_code: str
    source_type: str
    filename: Optional[str] = None
    total_workers_detected: int
    compliant_workers_count: int
    non_compliant_workers_count: int
    compliance_rate: float
    created_at: datetime

