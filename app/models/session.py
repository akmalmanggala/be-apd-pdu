"""Detection Session Model for History and Dashboard Recording."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    session_code: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'image', 'video', 'stream'
    filename: Mapped[str] = mapped_column(String(255), nullable=True)
    annotated_output_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    total_frames: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    total_workers_detected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compliant_workers_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    non_compliant_workers_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compliance_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0.0 to 100.0%
    
    # Missing items distribution count in this session
    helm_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    glove_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sepatu_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    kacamata_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    summary_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    worker_records: Mapped[List["WorkerRecord"]] = relationship(
        "WorkerRecord", back_populates="session", cascade="all, delete-orphan"
    )
