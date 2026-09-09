from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ClaimContextAssociation(BaseModel):
    claim_id: str
    statement: str
    local_node_ids: List[str] = Field(default_factory=list)
    structural_node_ids: List[str] = Field(default_factory=list)
    reference_node_ids: List[str] = Field(default_factory=list)
    visual_ids: List[str] = Field(default_factory=list)
    glossary_terms: List[str] = Field(default_factory=list)
    sufficiency_flag: bool = Field(default=True)
    sufficiency_rationale: str = Field(default="Sufficient local context and evidence retrieved.")


class ContextBuildingResult(BaseModel):
    document_id: str
    workspace_id: str
    total_claims: int
    associations: List[ClaimContextAssociation] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
