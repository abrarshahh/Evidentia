import uuid
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    title: Optional[str] = "Untitled Document"
    doc_type: str = "pdf"
    total_pages: int = 1
    language: str = "en"


class StructureSection(BaseModel):
    section_id: str
    number: Optional[str] = None
    name: str
    node_ids: List[str] = []


class DocumentNode(BaseModel):
    node_id: str
    node_type: str  # paragraph | table | figure | footnote | heading | list_item
    parent_section_id: Optional[str] = None
    page_range: List[int]
    char_offsets: List[int]
    text: str
    cross_refs: List[str] = []
    embedding_id: Optional[str] = None
    claim_ids: List[str] = []


class VisualElement(BaseModel):
    visual_id: str
    type: str  # table | figure
    original_caption: Optional[str] = None
    generated_caption: Optional[str] = None
    page_range: List[int]
    structured_data: Optional[Dict[str, Any]] = None
    owner_node_id: Optional[str] = None


class GlossaryEntry(BaseModel):
    term: str
    definition: str
    defined_at_node: str


class CoverageMetrics(BaseModel):
    total_nodes: int
    covered_nodes: int
    coverage_pct: float


class PageView(BaseModel):
    page_no: int
    node_ids: List[str] = []
    visual_ids: List[str] = []


class DocumentSkeletonIndex(BaseModel):
    document_id: str
    checksum: str
    metadata: DocumentMetadata
    structure_tree: List[StructureSection] = []
    nodes: List[DocumentNode] = []
    visuals: List[VisualElement] = []
    glossary: List[GlossaryEntry] = []
    coverage: CoverageMetrics
    page_view: List[PageView] = []

    model_config = ConfigDict(from_attributes=True)
