import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.db.models import WorkspaceRole


class WorkspaceCreate(BaseModel):
    name: str
    slug: Optional[str] = None


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    created_by: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkspaceMemberResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    role: WorkspaceRole
    joined_at: datetime

    model_config = ConfigDict(from_attributes=True)
