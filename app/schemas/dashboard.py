"""Dashboard Analytics and Reporting Schemas."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class APDViolationStat(BaseModel):
    apd_type: str
    violation_count: int
    compliance_percentage: float


class DashboardSummary(BaseModel):
    total_sessions: int
    total_workers_screened: int
    compliant_workers_total: int
    non_compliant_workers_total: int
    overall_compliance_rate: float
    violations_by_apd: Dict[str, int]
    apd_compliance_breakdown: List[APDViolationStat]


class ComplianceTrendPoint(BaseModel):
    period: str  # e.g. "2026-09-23" or "2026-09-23 01:00"
    total_workers: int
    compliant_workers: int
    violations_count: int
    compliance_rate: float


class WorkerRecordFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_compliant: Optional[bool] = None
    missing_apd: Optional[str] = None  # 'helm', 'glove', 'sepatu', 'kacamata'
    session_id: Optional[int] = None
    track_id: Optional[int] = None
    limit: int = 100
    offset: int = 0
