"""Application Configuration for Sistem Deteksi APD PDU Migas Backend."""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    # Base paths
    BASE_DIR: Path = BASE_DIR
    WEIGHTS_DIR: Path = WEIGHTS_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    OUTPUTS_DIR: Path = OUTPUTS_DIR

    # App info
    PROJECT_NAME: str = "Sistem Deteksi APD PDU Migas"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api"

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'apd_compliance.db'}"

    # JWT Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "pmld-pdu-migas-safety-secret-key-2026-very-secure")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Model Weights Paths
    APD_MODEL_PATH: str = str(WEIGHTS_DIR / "best_yolo11s_pdu.pt")
    PERSON_MODEL_PATH: str = (
        str(WEIGHTS_DIR / "yolo11s.pt")
        if (WEIGHTS_DIR / "yolo11s.pt").exists()
        else str(WEIGHTS_DIR / "yolo11n.pt")
    )

    # APD Classes
    CLASS_NAMES: list[str] = ["glove", "helm", "kacamata", "sepatu"]

    # PRD Section 10.6 Default Business Logic Thresholds
    DEFAULT_OVERLAP_THRESHOLD: float = 0.70  # Natural anatomical overlap
    DEFAULT_PERSON_CONF: float = 0.10  # Reliably detects occluded/partial workers in industrial CCTV

    # Calibrated class-specific confidence thresholds for maximal balance
    DEFAULT_CONF_THRESHOLDS: Dict[str, float] = {
        "helm": 0.25,
        "glove": 0.20,  # Eliminates spurious sub-0.20 noise while capturing hand and lever gloves
        "sepatu": 0.22,  # Reliably captures boots without false triggers on dark oily floors
        "kacamata": 0.15,  # Balanced for micro-detection without collar fold hallucinations
    }

    # Two-stage zoom parameters for kacamata
    ENABLE_HEAD_ZOOM: bool = True
    HEAD_CROP_RATIO: float = 0.40
    HEAD_ZOOM_CONF: float = 0.10
    HEAD_ZOOM_IMGSZ: int = 640
    ENABLE_HEAD_CLAHE: bool = False  # Avoid contrast noise / skin tone hallucination

    # Partner shoe recovery (Disabled: strictly uphold pure AI detection output)
    ENABLE_PARTNER_SHOE_RECOVERY: bool = False
    PARTNER_SHOE_CONF: float = 0.10

    # Anatomy-guided reclassification (Disabled: strictly uphold pure AI detection output)
    ENABLE_ANATOMY_RECLASSIFICATION: bool = False

    # Temporal persistence parameters for video streams
    ENABLE_TEMPORAL_PERSISTENCE: bool = True
    TEMPORAL_MEMORY_FRAMES: int = 30  # ~1-2 seconds persistence to mitigate bending over / occlusion

    # Inference resolution (640x640: native training resolution of the YOLO11 model)
    PRIMARY_IMGSZ: int = 640


settings = Settings()
