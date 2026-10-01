"""Main FastAPI Application Entrypoint."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

from app.config import settings
from app.database import init_db
from app.routers import auth_router, detection_router, dashboard_router, settings_router
from app.services.apd_detector import APDDetectorEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("be_apd_pdu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup initialization and shutdown cleanup."""
    logger.info("Starting up Sistem Deteksi APD PDU Migas Backend...")
    # Initialize SQLite database and seed defaults
    await init_db()
    # Pre-heat detection models into memory / GPU
    detector = APDDetectorEngine.get_instance()
    logger.info(f"Detection engine pre-heated on device: {detector.device}")
    yield
    logger.info("Shutting down APD detection backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API Sistem Deteksi APD PDU Migas berbasis Computer Vision (YOLO11s + YOLO11n)",
    lifespan=lifespan,
)

# CORS Configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Outputs and Uploads
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")

# Include Routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(detection_router, prefix=settings.API_V1_PREFIX)
app.include_router(dashboard_router, prefix=settings.API_V1_PREFIX)
app.include_router(settings_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Health"])
async def root():
    """Root status endpoint."""
    return {
        "status": "online",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "models": {
            "apd_detector": "YOLO11s (4 classes: glove, helm, kacamata, sepatu)",
            "person_detector": "YOLO11n (COCO Person class 0)",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed system health check."""
    detector = APDDetectorEngine.get_instance()
    return {
        "status": "healthy",
        "device": detector.device,
        "cuda_available": torch.cuda.is_available(),
        "classes": detector.class_names,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
