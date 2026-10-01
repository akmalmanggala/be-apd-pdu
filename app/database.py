"""Database Setup and Async Session for APD Detection Backend."""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create tables and seed initial default users and settings."""
    from app.models.user import User
    from app.models.settings import SystemSettingsModel
    from app.core.security import get_password_hash
    from sqlalchemy import select

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # Seed default admin user
        result = await session.execute(select(User).where(User.username == "admin"))
        admin_user = result.scalars().first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@pdu-migas.com",
                full_name="Admin HSE Migas",
                hashed_password=get_password_hash("admin123"),
                role="admin",
            )
            session.add(admin_user)

        # Seed default supervisor user
        result_sup = await session.execute(select(User).where(User.username == "supervisor"))
        sup_user = result_sup.scalars().first()
        if not sup_user:
            sup_user = User(
                username="supervisor",
                email="supervisor@pdu-migas.com",
                full_name="Pengawas Lapangan PDU",
                hashed_password=get_password_hash("super123"),
                role="supervisor",
            )
            session.add(sup_user)

        # Seed default system settings
        result_settings = await session.execute(select(SystemSettingsModel).where(SystemSettingsModel.id == 1))
        sys_settings = result_settings.scalars().first()
        if not sys_settings:
            sys_settings = SystemSettingsModel(
                id=1,
                overlap_threshold=settings.DEFAULT_OVERLAP_THRESHOLD,
                person_conf=settings.DEFAULT_PERSON_CONF,
                helm_conf=settings.DEFAULT_CONF_THRESHOLDS["helm"],
                glove_conf=settings.DEFAULT_CONF_THRESHOLDS["glove"],
                sepatu_conf=settings.DEFAULT_CONF_THRESHOLDS["sepatu"],
                kacamata_conf=settings.DEFAULT_CONF_THRESHOLDS["kacamata"],
                enable_head_zoom=settings.ENABLE_HEAD_ZOOM,
                head_crop_ratio=settings.HEAD_CROP_RATIO,
                head_zoom_conf=settings.HEAD_ZOOM_CONF,
                enable_temporal_persistence=settings.ENABLE_TEMPORAL_PERSISTENCE,
                temporal_memory_frames=settings.TEMPORAL_MEMORY_FRAMES,
                roi_active=False,
                roi_x1=0.0,
                roi_y1=0.0,
                roi_x2=1.0,
                roi_y2=1.0,
            )
            session.add(sys_settings)

        await session.commit()
    logger.info("Database initialized successfully with default records.")
