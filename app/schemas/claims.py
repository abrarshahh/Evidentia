import uuid
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ClaimType(str, Enum):
    FACTUAL = "factual"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"
    LEGAL = "legal"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    EFFICACY = "efficacy"
    STATISTICAL = "statistical"
    PROMOTIONAL = "promotional"
    COMPARATIVE = "comparative"
    QUALITY = "quality"
    OTHER = "other"


class ExtractedClaim(BaseModel):
    claim_id: str
    statement: str
    claim_type: ClaimType = ClaimType.FACTUAL
    source_node_ids: List[str] = Field(default_factory=list)
    section_path: Optional[str] = None
    page_range: List[int] = Field(default_factory=list)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)


class ClaimExtractionResult(BaseModel):
    document_id: str
    workspace_id: str
    extraction_mode: str  # "single_pass" | "map_reduce"
    total_claims: int
    claims: List[ExtractedClaim] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
