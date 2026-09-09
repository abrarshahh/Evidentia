import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    document_id: str
    workspace_id: str
    status: str  # "pending" | "processing" | "completed" | "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
