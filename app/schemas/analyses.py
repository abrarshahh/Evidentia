import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from app.db.models import AnalysisStatus


class AnalysisCreate(BaseModel):
    base_document_id: uuid.UUID
    reference_document_ids: List[uuid.UUID] = []


class AnalysisResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    base_document_id: uuid.UUID
    reference_document_ids: List[uuid.UUID] = []
    status: AnalysisStatus
    created_by: uuid.UUID
    created_at: datetime
    completed_at: Optional[datetime] = None
    claims_count: int = 0

    model_config = ConfigDict(from_attributes=True)
