import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import JWTError, jwt

from app.core.config import settings
from app.schemas.auth import TokenData


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its bcrypt hashed counterpart.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """
    Generate a bcrypt hash of a plain password.
    """
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate a short-lived JWT access token.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access", "jti": uuid.uuid4().hex})
    encoded_jwt = jwt.encode(
        to_encode, settings.get_secret_key(), algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate a long-lived JWT refresh token.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh", "jti": uuid.uuid4().hex})
    encoded_jwt = jwt.encode(
        to_encode, settings.get_secret_key(), algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def hash_refresh_token(token: str) -> str:
    """
    Compute SHA-256 hash of a refresh token for secure database storage.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT access or refresh token payload.
    Raises ValueError if token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token, settings.get_secret_key(), algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        exp: int = payload.get("exp")

        if user_id is None:
            raise ValueError("Token missing user identifier ('sub') claim")

        return TokenData(user_id=str(user_id), email=email, exp=exp)
    except JWTError as e:
        raise ValueError(f"Invalid or expired JWT token: {str(e)}")
