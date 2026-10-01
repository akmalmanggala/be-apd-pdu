"""Authentication Endpoints (Register, Login, Me)."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin, UserResponse, Token
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister, db: AsyncSession = Depends(get_db)):
    """Register a new user (admin or supervisor)."""
    # Check if username or email already exists
    existing = await db.execute(
        select(User).where((User.username == user_in.username) | (User.email == user_in.email))
    )
    if existing.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role or "supervisor",
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login_json(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """JSON-based login endpoint returning JWT bearer token."""
    result = await db.execute(select(User).where(User.username == credentials.username))
    user = result.scalars().first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username, "role": user.role})
    return Token(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.post("/token", response_model=Token, include_in_schema=False)
async def login_form(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """OAuth2 compatible token endpoint for Swagger UI documentation."""
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalars().first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username, "role": user.role})
    return Token(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve profile of the currently logged-in user."""
    return current_user
