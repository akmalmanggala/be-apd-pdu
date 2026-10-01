"""Tests for Detection Endpoints using FastAPI TestClient."""

import base64
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_detect_image_upload(client):
    import glob
    matches = glob.glob(r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images\pdu_v1_m56s00_f100828*.jpg")
    assert len(matches) > 0, "Test image not found"
    test_img_path = matches[0]

    with open(test_img_path, "rb") as f:
        files = {"file": ("test_frame.jpg", f, "image/jpeg")}
        response = client.post("/api/detect/image?overlap_threshold=0.80", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "session_id" in data
    assert "session_code" in data
    assert data["total_workers"] >= 1
    assert "workers" in data
    assert len(data["workers"]) == data["total_workers"]

    first_worker = data["workers"][0]
    assert "checklist" in first_worker
    assert "helm" in first_worker["checklist"]
    assert "glove" in first_worker["checklist"]
    assert "kacamata" in first_worker["checklist"]
    assert "sepatu" in first_worker["checklist"]
    assert "is_compliant" in first_worker
    assert data["annotated_image_url"] is not None
    assert data["inference_time_ms"] > 0


def test_api_detect_stream_frame(client):
    import glob
    matches = glob.glob(r"C:\Users\zvwah\OneDrive\Dokumen\KULIAH\TRPL\Sem 5\PMLD\Dataset\Dataset-APD-PDU.yolov11\test\images\pdu_v1_m56s00_f100828*.jpg")
    assert len(matches) > 0, "Test image not found"
    test_img_path = matches[0]
    with open(test_img_path, "rb") as f:
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "client_id": "cctv_rig_alpha",
        "session_code": "LIVE-STREAM-001",
        "frame_base64": img_b64,
    }
    response = client.post("/api/detect/stream-frame", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["client_id"] == "cctv_rig_alpha"
    assert data["frame_idx"] >= 1
    assert data["annotated_frame_base64"] is not None
    assert isinstance(data["workers"], list)


def test_api_sessions_history_and_detail(client):
    list_resp = client.get("/api/detect/sessions")
    assert list_resp.status_code == 200
    sessions = list_resp.json()
    assert isinstance(sessions, list)
    if sessions:
        s_id = sessions[0]["id"]
        detail_resp = client.get(f"/api/detect/sessions/{s_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["id"] == s_id
        assert "worker_records" in detail_data
