"""System Settings Endpoints (ROI and Thresholds Configuration)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.settings import SystemSettingsModel
from app.schemas.settings import (
    ROIConfig,
    ThresholdsConfig,
    SettingsResponse,
    SettingsUpdateRequest,
)
from app.core.deps import get_current_user_optional

router = APIRouter(prefix="/settings", tags=["Settings & ROI Configuration"])


@router.get("", response_model=SettingsResponse)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Retrieve current ROI and detection threshold configuration (PRD FR-10)."""
    res = await db.execute(select(SystemSettingsModel).where(SystemSettingsModel.id == 1))
    s = res.scalars().first()
    if not s:
        s = SystemSettingsModel(id=1)
        db.add(s)
        await db.commit()
        await db.refresh(s)

    roi = ROIConfig(
        active=s.roi_active,
        x1=s.roi_x1,
        y1=s.roi_y1,
        x2=s.roi_x2,
        y2=s.roi_y2,
    )
    thresholds = ThresholdsConfig(
        overlap_threshold=s.overlap_threshold,
        person_conf=s.person_conf,
        helm_conf=s.helm_conf,
        glove_conf=s.glove_conf,
        sepatu_conf=s.sepatu_conf,
        kacamata_conf=s.kacamata_conf,
        enable_head_zoom=s.enable_head_zoom,
        head_crop_ratio=s.head_crop_ratio,
        head_zoom_conf=s.head_zoom_conf,
        enable_temporal_persistence=s.enable_temporal_persistence,
        temporal_memory_frames=s.temporal_memory_frames,
    )
    return SettingsResponse(id=s.id, roi=roi, thresholds=thresholds, updated_at=s.updated_at)


@router.put("", response_model=SettingsResponse)
async def update_settings(
    update_data: SettingsUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    """Update ROI zone and detection thresholds dynamically (PRD FR-10)."""
    res = await db.execute(select(SystemSettingsModel).where(SystemSettingsModel.id == 1))
    s = res.scalars().first()
    if not s:
        s = SystemSettingsModel(id=1)
        db.add(s)

    if update_data.roi:
        s.roi_active = update_data.roi.active
        s.roi_x1 = update_data.roi.x1
        s.roi_y1 = update_data.roi.y1
        s.roi_x2 = update_data.roi.x2
        s.roi_y2 = update_data.roi.y2

    if update_data.thresholds:
        t = update_data.thresholds
        s.overlap_threshold = t.overlap_threshold
        s.person_conf = t.person_conf
        s.helm_conf = t.helm_conf
        s.glove_conf = t.glove_conf
        s.sepatu_conf = t.sepatu_conf
        s.kacamata_conf = t.kacamata_conf
        s.enable_head_zoom = t.enable_head_zoom
        s.head_crop_ratio = t.head_crop_ratio
        s.head_zoom_conf = t.head_zoom_conf
        s.enable_temporal_persistence = t.enable_temporal_persistence
        s.temporal_memory_frames = t.temporal_memory_frames

    await db.commit()
    await db.refresh(s)

    roi = ROIConfig(
        active=s.roi_active,
        x1=s.roi_x1,
        y1=s.roi_y1,
        x2=s.roi_x2,
        y2=s.roi_y2,
    )
    thresholds = ThresholdsConfig(
        overlap_threshold=s.overlap_threshold,
        person_conf=s.person_conf,
        helm_conf=s.helm_conf,
        glove_conf=s.glove_conf,
        sepatu_conf=s.sepatu_conf,
        kacamata_conf=s.kacamata_conf,
        enable_head_zoom=s.enable_head_zoom,
        head_crop_ratio=s.head_crop_ratio,
        head_zoom_conf=s.head_zoom_conf,
        enable_temporal_persistence=s.enable_temporal_persistence,
        temporal_memory_frames=s.temporal_memory_frames,
    )
    return SettingsResponse(id=s.id, roi=roi, thresholds=thresholds, updated_at=s.updated_at)
