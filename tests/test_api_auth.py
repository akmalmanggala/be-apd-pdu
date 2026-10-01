"""Tests for Authentication Endpoints using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_auth_login_default_admin(client):
    response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_auth_login_invalid_password(client):
    response = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401


def test_auth_register_and_login_new_user(client):
    reg_payload = {
        "username": "hse_officer_test",
        "email": "officertest@pdu.co.id",
        "full_name": "Petugas HSE Test",
        "password": "officerpassword123",
        "role": "supervisor",
    }
    reg_resp = client.post("/api/auth/register", json=reg_payload)
    assert reg_resp.status_code in [201, 400]

    # Now login
    login_resp = client.post("/api/auth/login", json={"username": "hse_officer_test", "password": "officerpassword123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    # Fetch profile
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "hse_officer_test"


def test_auth_me_unauthorized(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401
