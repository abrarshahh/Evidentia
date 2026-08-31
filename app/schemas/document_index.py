from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentMetadata(BaseModel):
    title: str
    file_type: str = "pdf"
    total_pages: int = 1
    page_count: int = 1
    char_count: int = 0
    language: str = "en"


class StructureSection(BaseModel):
    section_id: str
    number: Optional[str] = None
    name: str
    title: Optional[str] = None
    level: int = 1
    parent_section_id: Optional[str] = None
    page_start: int = 1
    page_end: int = 1
    node_ids: List[str] = []


class DocumentNode(BaseModel):
    node_id: str
    parent_section_id: Optional[str] = None
    node_type: str  # "heading" | "paragraph" | "list_item" | "table"
    text: str
    page_range: List[int] = []
    char_offsets: List[int] = []
    embedding_id: Optional[str] = None


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class VisualElement(BaseModel):
    visual_id: str
    node_id: Optional[str] = None
    visual_type: str  # "table" | "figure" | "chart"
    page_no: int
    bbox: Optional[BoundingBox] = None
    caption: Optional[str] = None


class GlossaryEntry(BaseModel):
    term: str
    definition: str
    defined_at_node: str


class CoverageMetrics(BaseModel):
    total_nodes: int
    covered_nodes: int
    coverage_pct: float
    uncovered_node_ids: List[str] = []

    @property
    def node_coverage_percent(self) -> float:
        return self.coverage_pct


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
    page_views: List[PageView] = []
    coverage_metrics: Optional[CoverageMetrics] = None

    model_config = ConfigDict(from_attributes=True)
