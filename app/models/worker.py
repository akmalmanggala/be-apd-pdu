"""Worker Compliance Record Model."""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class WorkerRecord(Base):
    __tablename__ = "worker_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_idx: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Bounding box of the worker: [x1, y1, x2, y2]
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    person_conf: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # 4 APD compliance flags
    helm_worn: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    glove_worn: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sepatu_worn: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    kacamata_worn: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Overall compliance (True if all 4 are worn)
    is_compliant: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # JSON text storing detected item details and overlap ratios
    detection_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    session: Mapped["DetectionSession"] = relationship("DetectionSession", back_populates="worker_records")
