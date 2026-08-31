import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict


# User Signup Payload
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


# User Login Payload
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# Public User Response
class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# JWT Token Response (includes validity duration in minutes)
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


# JWT Payload Claims
class TokenData(BaseModel):
    user_id: str
    email: Optional[str] = None
    exp: Optional[int] = None
