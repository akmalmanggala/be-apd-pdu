"""System Settings Model for Configurable ROI and Confidence Thresholds."""

from datetime import datetime
from sqlalchemy import Integer, Float, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class SystemSettingsModel(Base):
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    
    # Overlap and person thresholds
    overlap_threshold: Mapped[float] = mapped_column(Float, default=0.70, nullable=False)
    person_conf: Mapped[float] = mapped_column(Float, default=0.10, nullable=False)
    
    # APD class confidence thresholds (synced with config.py DEFAULT_CONF_THRESHOLDS)
    helm_conf: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)
    glove_conf: Mapped[float] = mapped_column(Float, default=0.20, nullable=False)
    sepatu_conf: Mapped[float] = mapped_column(Float, default=0.22, nullable=False)
    kacamata_conf: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)
    
    # Two-stage zoom for micro-detection
    enable_head_zoom: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    head_crop_ratio: Mapped[float] = mapped_column(Float, default=0.38, nullable=False)
    head_zoom_conf: Mapped[float] = mapped_column(Float, default=0.08, nullable=False)

    # Temporal persistence
    enable_temporal_persistence: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    temporal_memory_frames: Mapped[int] = mapped_column(Integer, default=30, nullable=False)

    # Region of Interest (ROI) normalized bounds [0.0 - 1.0]
    roi_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    roi_x1: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    roi_y1: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    roi_x2: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    roi_y2: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
