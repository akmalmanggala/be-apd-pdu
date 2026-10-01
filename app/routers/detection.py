"""Detection Endpoints for Images, Videos, Live Streams, and Sessions."""

import base64
import json
import logging
import os
import shutil
import uuid
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.session import DetectionSession
from app.models.worker import WorkerRecord
from app.models.settings import SystemSettingsModel
from app.schemas.detection import (
    ImageDetectionResponse,
    VideoDetectionResponse,
    StreamFrameRequest,
    StreamFrameResponse,
    SessionListItem,
    SessionDetailResponse,
)
from app.services.apd_detector import APDDetectorEngine
from app.services.video_processor import VideoBatchProcessor
from app.services.stream_manager import StreamManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["Detection & Inference"])


async def get_active_system_settings(db: AsyncSession) -> SystemSettingsModel:
    """Fetch stored system configuration or return defaults."""
    res = await db.execute(select(SystemSettingsModel).where(SystemSettingsModel.id == 1))
    s = res.scalars().first()
    if not s:
        # Create and return default
        s = SystemSettingsModel(id=1)
        db.add(s)
        await db.commit()
        await db.refresh(s)
    return s


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image_file(
    file: UploadFile = File(...),
    overlap_threshold: Optional[float] = Query(None, ge=0.1, le=1.0, description="PRD 10.6 overlap threshold"),
    enable_head_zoom: Optional[bool] = Query(None, description="Enable two-stage head crop for glasses"),
    db: AsyncSession = Depends(get_db),
):
    """Run APD compliance detection on an uploaded image file."""
    # Validate extension
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {list(allowed_exts)}",
        )

    # Read image contents
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted image file")

    # Fetch dynamic settings from DB
    sys_cfg = await get_active_system_settings(db)
    ov_thresh = overlap_threshold if overlap_threshold is not None else sys_cfg.overlap_threshold
    zoom_active = enable_head_zoom if enable_head_zoom is not None else sys_cfg.enable_head_zoom

    conf_dict = {
        "helm": sys_cfg.helm_conf,
        "glove": sys_cfg.glove_conf,
        "sepatu": sys_cfg.sepatu_conf,
        "kacamata": sys_cfg.kacamata_conf,
    }

    roi_dict = {
        "active": sys_cfg.roi_active,
        "x1": sys_cfg.roi_x1,
        "y1": sys_cfg.roi_y1,
        "x2": sys_cfg.roi_x2,
        "y2": sys_cfg.roi_y2,
    }

    detector = APDDetectorEngine.get_instance()
    workers, stats, annotated_img = detector.detect_image(
        img_bgr=img_bgr,
        overlap_threshold=ov_thresh,
        conf_thresholds=conf_dict,
        person_conf=sys_cfg.person_conf,
        enable_head_zoom=zoom_active,
        head_crop_ratio=sys_cfg.head_crop_ratio,
        head_zoom_conf=sys_cfg.head_zoom_conf,
        roi_config=roi_dict,
    )

    session_code = f"IMG-{uuid.uuid4().hex[:8].upper()}"
    output_filename = f"{session_code}_annotated.jpg"
    output_path = os.path.join(str(settings.OUTPUTS_DIR), output_filename)
    cv2.imwrite(output_path, annotated_img)

    # Encode annotated image to base64 for direct frontend display
    _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    # Persist session to database (PRD FR-5)
    session_record = DetectionSession(
        session_code=session_code,
        source_type="image",
        filename=file.filename,
        annotated_output_path=output_filename,
        total_frames=1,
        total_workers_detected=stats["total_workers"],
        compliant_workers_count=stats["compliant_workers_count"],
        non_compliant_workers_count=stats["non_compliant_workers_count"],
        compliance_rate=stats["compliance_rate"],
        helm_violations=stats["helm_violations"],
        glove_violations=stats["glove_violations"],
        sepatu_violations=stats["sepatu_violations"],
        kacamata_violations=stats["kacamata_violations"],
        summary_notes=f"Inference latency: {stats['inference_time_ms']}ms. Zoom applied: {stats['two_stage_zoom_applied']}.",
    )
    db.add(session_record)
    await db.flush()

    # Persist worker records
    for w in workers:
        w_rec = WorkerRecord(
            session_id=session_record.id,
            frame_idx=0,
            track_id=w.track_id,
            bbox_x1=w.bbox.x1,
            bbox_y1=w.bbox.y1,
            bbox_x2=w.bbox.x2,
            bbox_y2=w.bbox.y2,
            person_conf=w.person_conf,
            helm_worn=w.checklist["helm"],
            glove_worn=w.checklist["glove"],
            sepatu_worn=w.checklist["sepatu"],
            kacamata_worn=w.checklist["kacamata"],
            is_compliant=w.is_compliant,
            detection_details=json.dumps([d.model_dump() for d in w.detected_apds]),
        )
        db.add(w_rec)

    await db.commit()

    return ImageDetectionResponse(
        session_id=session_record.id,
        session_code=session_code,
        source_filename=file.filename,
        total_workers=stats["total_workers"],
        compliant_workers_count=stats["compliant_workers_count"],
        non_compliant_workers_count=stats["non_compliant_workers_count"],
        compliance_rate=stats["compliance_rate"],
        workers=workers,
        all_detected_apds=stats.get("all_detected_apds", []),
        unassigned_apds=stats.get("unassigned_apds", []),
        annotated_image_url=f"/outputs/{output_filename}",
        annotated_image_base64=annotated_b64,
        inference_time_ms=stats["inference_time_ms"],
        two_stage_zoom_applied=stats["two_stage_zoom_applied"],
    )


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video_file(
    file: UploadFile = File(...),
    sample_stride: int = Query(5, ge=1, le=30, description="Process every Nth frame"),
    overlap_threshold: Optional[float] = Query(None, ge=0.1, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    """Run batch APD detection on an uploaded video file with temporal tracking."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: {list(allowed_exts)}",
        )

    # Save uploaded video to uploads directory
    temp_video_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    temp_video_path = os.path.join(str(settings.UPLOADS_DIR), temp_video_name)
    with open(temp_video_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    sys_cfg = await get_active_system_settings(db)
    ov_thresh = overlap_threshold if overlap_threshold is not None else sys_cfg.overlap_threshold

    conf_dict = {
        "helm": sys_cfg.helm_conf,
        "glove": sys_cfg.glove_conf,
        "sepatu": sys_cfg.sepatu_conf,
        "kacamata": sys_cfg.kacamata_conf,
    }

    processor = VideoBatchProcessor()
    try:
        results = processor.process_video_file(
            video_path=temp_video_path,
            output_dir=str(settings.OUTPUTS_DIR),
            sample_stride=sample_stride,
            overlap_threshold=ov_thresh,
            conf_thresholds=conf_dict,
            person_conf=sys_cfg.person_conf,
            enable_head_zoom=sys_cfg.enable_head_zoom,
        )
    finally:
        # Clean up raw upload if desired, or keep for audit
        pass

    # Save session to DB
    session_record = DetectionSession(
        session_code=results["session_code"],
        source_type="video",
        filename=file.filename,
        annotated_output_path=results["annotated_video_filename"],
        total_frames=results["total_frames_processed"],
        total_workers_detected=results["total_unique_workers"],
        compliant_workers_count=results["compliant_workers_count"],
        non_compliant_workers_count=results["non_compliant_workers_count"],
        compliance_rate=results["overall_compliance_rate"],
        helm_violations=results["helm_violations"],
        glove_violations=results["glove_violations"],
        sepatu_violations=results["sepatu_violations"],
        kacamata_violations=results["kacamata_violations"],
        summary_notes=f"Video processed in {results['processing_time_sec']}s ({results['total_frames_processed']} frames).",
    )
    db.add(session_record)
    await db.flush()

    # Save worker summaries as worker records
    for w in results["workers_summary"]:
        chk = w["cumulative_checklist"]
        is_comp = all(chk.values())
        w_rec = WorkerRecord(
            session_id=session_record.id,
            frame_idx=w["last_seen_frame"],
            track_id=w["track_id"],
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=0.0,
            bbox_y2=0.0,
            person_conf=sys_cfg.person_conf,
            helm_worn=chk["helm"],
            glove_worn=chk["glove"],
            sepatu_worn=chk["sepatu"],
            kacamata_worn=chk["kacamata"],
            is_compliant=is_comp,
            detection_details=json.dumps(w),
        )
        db.add(w_rec)

    await db.commit()

    return VideoDetectionResponse(
        session_id=session_record.id,
        session_code=results["session_code"],
        source_filename=file.filename,
        total_frames_processed=results["total_frames_processed"],
        total_unique_workers=results["total_unique_workers"],
        compliant_workers_count=results["compliant_workers_count"],
        non_compliant_workers_count=results["non_compliant_workers_count"],
        overall_compliance_rate=results["overall_compliance_rate"],
        annotated_video_url=f"/outputs/{results['annotated_video_filename']}",
        annotated_sample_frames=[f"/outputs/{s}" for s in results["annotated_sample_frames"]],
        processing_time_sec=results["processing_time_sec"],
        status="completed",
    )


@router.post("/stream-frame", response_model=StreamFrameResponse)
async def process_stream_frame(payload: StreamFrameRequest, db: AsyncSession = Depends(get_db)):
    """Process a single live webcam frame while maintaining client-side temporal tracking."""
    sys_cfg = await get_active_system_settings(db)
    conf_dict = {
        "helm": sys_cfg.helm_conf,
        "glove": sys_cfg.glove_conf,
        "sepatu": sys_cfg.sepatu_conf,
        "kacamata": sys_cfg.kacamata_conf,
    }
    roi_dict = {
        "active": sys_cfg.roi_active,
        "x1": sys_cfg.roi_x1,
        "y1": sys_cfg.roi_y1,
        "x2": sys_cfg.roi_x2,
        "y2": sys_cfg.roi_y2,
    }

    stream_mgr = StreamManager.get_instance()
    try:
        response = stream_mgr.process_stream_frame(
            client_id=payload.client_id,
            frame_base64=payload.frame_base64,
            session_code=payload.session_code,
            overlap_threshold=sys_cfg.overlap_threshold,
            conf_thresholds=conf_dict,
            roi_config=roi_dict,
        )
        return response
    except Exception as e:
        logger.error(f"Stream processing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/sessions", response_model=List[SessionListItem])
async def list_detection_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    source_type: Optional[str] = Query(None, pattern="^(image|video|stream)$"),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve history of detection sessions (PRD FR-6)."""
    query = select(DetectionSession).order_by(desc(DetectionSession.created_at))
    if source_type:
        query = query.where(DetectionSession.source_type == source_type)
    query = query.offset(skip).limit(limit)
    res = await db.execute(query)
    return res.scalars().all()


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: int, db: AsyncSession = Depends(get_db)):
    """Retrieve full details of a detection session including worker checklists (PRD FR-7)."""
    query = (
        select(DetectionSession)
        .where(DetectionSession.id == session_id)
        .options(selectinload(DetectionSession.worker_records))
    )
    res = await db.execute(query)
    session_obj = res.scalars().first()
    if not session_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session_obj


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a detection session and its worker records."""
    res = await db.execute(select(DetectionSession).where(DetectionSession.id == session_id))
    session_obj = res.scalars().first()
    if not session_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    await db.delete(session_obj)
    await db.commit()
    return None
