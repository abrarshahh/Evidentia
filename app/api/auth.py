import uuid
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.session import get_async_session
from app.db.models import User, Workspace, WorkspaceMember, WorkspaceRole, RefreshToken
from app.schemas.auth import UserCreate, UserLogin, UserResponse, Token
from app.auth.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
    decode_token,
)
from app.core.config import settings

from app.core.logging_config import set_log_context

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

security_scheme = HTTPBearer()


class RefreshTokenRequest(BaseModel):
    refresh_token: str


def generate_slug(text: str) -> str:
    """
    Generate a URL-safe slug from a string with a unique short suffix.
    """
    clean_text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug_base = re.sub(r"[-\s]+", "-", clean_text)
    unique_suffix = uuid.uuid4().hex[:6]
    return f"{slug_base}-{unique_suffix}"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Dependency to validate JWT access token from Bearer header and return current User instance.
    """
    token = credentials.credentials.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    try:
        token_data = decode_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_uuid = uuid.UUID(token_data.user_id)
    result = await session.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive or not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    set_log_context(user_id=user.email)
    return user


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: UserCreate,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Register a new user account and automatically provision a default personal workspace.
    """
    # 1. Check if email already exists
    existing = await session.execute(
        select(User).where(User.email == payload.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    # 2. Create User
    new_user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        is_active=True,
    )
    session.add(new_user)
    await session.flush()  # Populates new_user.id

    # 3. Auto-provision default Personal Workspace
    workspace_name = (
        f"{payload.full_name}'s Workspace" if payload.full_name else f"{payload.email.split('@')[0]}'s Workspace"
    )
    workspace_slug = generate_slug(workspace_name)

    default_workspace = Workspace(
        name=workspace_name,
        slug=workspace_slug,
        created_by=new_user.id,
    )
    session.add(default_workspace)
    await session.flush()

    # 4. Add User as Owner of default Workspace
    member = WorkspaceMember(
        workspace_id=default_workspace.id,
        user_id=new_user.id,
        role=WorkspaceRole.owner,
    )
    session.add(member)

    await session.commit()
    await session.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
async def login(
    payload: UserLogin,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Authenticate user credentials, issue access & refresh tokens, and store refresh token hash.
    """
    result = await session.execute(
        select(User).where(User.email == payload.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )

    # Issue tokens
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    refresh_token = create_refresh_token(data={"sub": str(user.id), "email": user.email})

    # Save hashed refresh token in DB
    token_hash = hash_refresh_token(refresh_token)
    expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    db_refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(db_refresh_token)
    await session.commit()

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


@router.post("/refresh", response_model=Token)
async def refresh_tokens(
    payload: RefreshTokenRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Rotate refresh tokens and issue a new access token pair.
    """
    try:
        token_data = decode_token(payload.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}",
        )

    incoming_hash = hash_refresh_token(payload.refresh_token)
    user_uuid = uuid.UUID(token_data.user_id)

    # Query matching token in DB
    result = await session.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == user_uuid,
            RefreshToken.token_hash == incoming_hash,
            RefreshToken.revoked_at.is_(None),
        )
    )
    db_token = result.scalar_one_or_none()

    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked or is invalid",
        )

    if db_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )

    # Revoke old refresh token
    db_token.revoked_at = datetime.utcnow()

    # Fetch User
    user_result = await session.execute(select(User).where(User.id == user_uuid))
    user = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive or not found",
        )

    # Issue new token pair
    new_access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id), "email": user.email})

    new_token_hash = hash_refresh_token(new_refresh_token)
    new_expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    new_db_token = RefreshToken(
        user_id=user.id,
        token_hash=new_token_hash,
        expires_at=new_expires_at,
    )
    session.add(new_db_token)
    await session.commit()

    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get profile information for the currently authenticated user.
    """
    return current_user
