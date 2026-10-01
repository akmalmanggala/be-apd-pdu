"""Compliance Dashboard Endpoints for Statistics, Trends, and Audit Exports."""

import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.session import DetectionSession
from app.models.worker import WorkerRecord
from app.schemas.dashboard import (
    DashboardSummary,
    ComplianceTrendPoint,
    APDViolationStat,
    WorkerRecordFilter,
)
from app.schemas.detection import WorkerRecordResponse

router = APIRouter(prefix="/dashboard", tags=["Compliance Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve overall compliance statistics and APD violation distribution (PRD FR-8)."""
    query = select(DetectionSession)
    if start_date:
        query = query.where(DetectionSession.created_at >= start_date)
    if end_date:
        query = query.where(DetectionSession.created_at <= end_date)

    res = await db.execute(query)
    sessions = res.scalars().all()

    total_sessions = len(sessions)
    total_workers = sum(s.total_workers_detected for s in sessions)
    compliant_workers = sum(s.compliant_workers_count for s in sessions)
    non_compliant_workers = sum(s.non_compliant_workers_count for s in sessions)
    
    overall_compliance = (
        round((compliant_workers / total_workers * 100.0), 2) if total_workers > 0 else 100.0
    )

    helm_viol = sum(s.helm_violations for s in sessions)
    glove_viol = sum(s.glove_violations for s in sessions)
    sepatu_viol = sum(s.sepatu_violations for s in sessions)
    kacamata_viol = sum(s.kacamata_violations for s in sessions)

    violations_by_apd = {
        "helm": helm_viol,
        "glove": glove_viol,
        "sepatu": sepatu_viol,
        "kacamata": kacamata_viol,
    }

    breakdown: List[APDViolationStat] = []
    for apd_name, v_count in violations_by_apd.items():
        comp_pct = round(((total_workers - v_count) / total_workers * 100.0), 2) if total_workers > 0 else 100.0
        breakdown.append(
            APDViolationStat(
                apd_type=apd_name,
                violation_count=v_count,
                compliance_percentage=comp_pct,
            )
        )

    return DashboardSummary(
        total_sessions=total_sessions,
        total_workers_screened=total_workers,
        compliant_workers_total=compliant_workers,
        non_compliant_workers_total=non_compliant_workers,
        overall_compliance_rate=overall_compliance,
        violations_by_apd=violations_by_apd,
        apd_compliance_breakdown=breakdown,
    )


@router.get("/compliance-trends", response_model=List[ComplianceTrendPoint])
async def get_compliance_trends(
    days: int = Query(7, ge=1, le=90, description="Number of past days to aggregate"),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve compliance trend points over past days (PRD FR-8)."""
    since = datetime.utcnow() - timedelta(days=days)
    query = (
        select(DetectionSession)
        .where(DetectionSession.created_at >= since)
        .order_by(DetectionSession.created_at)
    )
    res = await db.execute(query)
    sessions = res.scalars().all()

    # Group by date YYYY-MM-DD
    grouped: Dict[str, Dict[str, int]] = {}
    for s in sessions:
        date_str = s.created_at.strftime("%Y-%m-%d")
        if date_str not in grouped:
            grouped[date_str] = {"total": 0, "compliant": 0, "violations": 0}
        grouped[date_str]["total"] += s.total_workers_detected
        grouped[date_str]["compliant"] += s.compliant_workers_count
        grouped[date_str]["violations"] += s.non_compliant_workers_count

    trends: List[ComplianceTrendPoint] = []
    for d_str, counts in sorted(grouped.items()):
        rate = round((counts["compliant"] / counts["total"] * 100.0), 2) if counts["total"] > 0 else 100.0
        trends.append(
            ComplianceTrendPoint(
                period=d_str,
                total_workers=counts["total"],
                compliant_workers=counts["compliant"],
                violations_count=counts["violations"],
                compliance_rate=rate,
            )
        )
    return trends


@router.get("/worker-records", response_model=List[WorkerRecordResponse])
async def filter_worker_records(
    is_compliant: Optional[bool] = Query(None),
    missing_apd: Optional[str] = Query(None, pattern="^(helm|glove|sepatu|kacamata)$"),
    session_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Search and filter individual worker records (PRD FR-9)."""
    query = select(WorkerRecord).order_by(desc(WorkerRecord.detected_at))
    if is_compliant is not None:
        query = query.where(WorkerRecord.is_compliant == is_compliant)
    if missing_apd:
        if missing_apd == "helm":
            query = query.where(WorkerRecord.helm_worn == False)
        elif missing_apd == "glove":
            query = query.where(WorkerRecord.glove_worn == False)
        elif missing_apd == "sepatu":
            query = query.where(WorkerRecord.sepatu_worn == False)
        elif missing_apd == "kacamata":
            query = query.where(WorkerRecord.kacamata_worn == False)
    if session_id:
        query = query.where(WorkerRecord.session_id == session_id)

    query = query.offset(offset).limit(limit)
    res = await db.execute(query)
    return res.scalars().all()


@router.get("/export", response_class=Response)
async def export_compliance_csv(db: AsyncSession = Depends(get_db)):
    """Export compliance records as CSV report for safety audit (PRD FR-13)."""
    res = await db.execute(select(WorkerRecord).order_by(desc(WorkerRecord.detected_at)))
    workers = res.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID", "Session ID", "Track ID", "Waktu Deteksi",
        "Helm", "Glove", "Sepatu", "Kacamata",
        "Status Kepatuhan", "Confidence Pekerja"
    ])

    for w in workers:
        status_str = "PATUH (100%)" if w.is_compliant else "MELANGGAR"
        writer.writerow([
            w.id,
            w.session_id,
            w.track_id,
            w.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
            "YA" if w.helm_worn else "TIDAK",
            "YA" if w.glove_worn else "TIDAK",
            "YA" if w.sepatu_worn else "TIDAK",
            "YA" if w.kacamata_worn else "TIDAK",
            status_str,
            f"{w.person_conf:.2f}",
        ])

    csv_data = output.getvalue()
    filename = f"Laporan_Kepatuhan_APD_PDU_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
