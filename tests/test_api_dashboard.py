"""Tests for Compliance Dashboard Endpoints using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_dashboard_summary(client):
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_sessions" in data
    assert "total_workers_screened" in data
    assert "overall_compliance_rate" in data
    assert "violations_by_apd" in data
    assert "helm" in data["violations_by_apd"]
    assert "kacamata" in data["violations_by_apd"]


def test_dashboard_compliance_trends(client):
    response = client.get("/api/dashboard/compliance-trends?days=7")
    assert response.status_code == 200
    trends = response.json()
    assert isinstance(trends, list)


def test_dashboard_worker_records_filter(client):
    response = client.get("/api/dashboard/worker-records?is_compliant=false&limit=10")
    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)


def test_dashboard_csv_export(client):
    response = client.get("/api/dashboard/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Record ID,Session ID,Track ID" in response.text
