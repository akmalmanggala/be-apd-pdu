"""Settings Schemas for ROI and Detection Thresholds."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ROIConfig(BaseModel):
    active: bool = False
    x1: float = Field(0.0, ge=0.0, le=1.0)
    y1: float = Field(0.0, ge=0.0, le=1.0)
    x2: float = Field(1.0, ge=0.0, le=1.0)
    y2: float = Field(1.0, ge=0.0, le=1.0)


class ThresholdsConfig(BaseModel):
    overlap_threshold: float = Field(0.80, ge=0.1, le=1.0, description="PRD Section 10.6 overlap ratio threshold (default 0.80)")
    person_conf: float = Field(0.20, ge=0.05, le=1.0)
    helm_conf: float = Field(0.25, ge=0.05, le=1.0)
    glove_conf: float = Field(0.15, ge=0.05, le=1.0)
    sepatu_conf: float = Field(0.18, ge=0.05, le=1.0)
    kacamata_conf: float = Field(0.10, ge=0.01, le=1.0)
    enable_head_zoom: bool = True
    head_crop_ratio: float = Field(0.40, ge=0.2, le=0.6)
    head_zoom_conf: float = Field(0.05, ge=0.01, le=1.0)
    enable_temporal_persistence: bool = True
    temporal_memory_frames: int = Field(30, ge=1, le=120)


class SettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    roi: ROIConfig
    thresholds: ThresholdsConfig
    updated_at: datetime



class SettingsUpdateRequest(BaseModel):
    roi: Optional[ROIConfig] = None
    thresholds: Optional[ThresholdsConfig] = None
