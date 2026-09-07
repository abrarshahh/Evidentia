import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict
from app.schemas.document_index import DocumentSkeletonIndex


class DocumentResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    original_filename: Optional[str] = None
    minio_raw_path: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IndexingTriggerResponse(BaseModel):
    document_id: str
    status: str  # "completed" | "processing" | "failed"
    message: str
    nodes_count: int
    glossary_count: int
