"""Tests for ROI and Threshold Settings Endpoints using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_get_settings(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert "roi" in data
    assert "thresholds" in data
    assert "overlap_threshold" in data["thresholds"]
    assert "kacamata_conf" in data["thresholds"]


def test_update_settings(client):
    update_payload = {
        "roi": {
            "active": True,
            "x1": 0.10,
            "y1": 0.10,
            "x2": 0.90,
            "y2": 0.90,
        },
        "thresholds": {
            "overlap_threshold": 0.85,
            "person_conf": 0.30,
            "helm_conf": 0.40,
            "glove_conf": 0.35,
            "sepatu_conf": 0.35,
            "kacamata_conf": 0.15,
            "enable_head_zoom": True,
            "head_crop_ratio": 0.40,
            "head_zoom_conf": 0.10,
            "enable_temporal_persistence": True,
            "temporal_memory_frames": 45,
        },
    }
    response = client.put("/api/settings", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["roi"]["active"] is True
    assert data["roi"]["x1"] == 0.10
    assert data["thresholds"]["overlap_threshold"] == 0.85
    assert data["thresholds"]["temporal_memory_frames"] == 45

    # Restore calibrated production settings so test pollution does not break live API inference
    restore_payload = {
        "roi": {
            "active": False,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
        },
        "thresholds": {
            "overlap_threshold": 0.80,
            "person_conf": 0.20,
            "helm_conf": 0.25,
            "glove_conf": 0.15,
            "sepatu_conf": 0.18,
            "kacamata_conf": 0.10,
            "enable_head_zoom": True,
            "head_crop_ratio": 0.40,
            "head_zoom_conf": 0.05,
            "enable_temporal_persistence": True,
            "temporal_memory_frames": 30,
        },
    }
    res_restore = client.put("/api/settings", json=restore_payload)
    assert res_restore.status_code == 200
