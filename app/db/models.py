import enum
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Boolean,
    Enum, Text, Integer, Float, UniqueConstraint, MetaData
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=naming_convention)
Base = declarative_base(metadata=metadata)


# Enums
class WorkspaceRole(str, enum.Enum):
    owner = "owner"
    admin = "admin"
    member = "member"


class DocumentRole(str, enum.Enum):
    base = "base"
    reference = "reference"


class SourceType(str, enum.Enum):
    upload = "upload"
    url = "url"


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    parsed = "parsed"
    indexed = "indexed"
    failed = "failed"


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    indexing = "indexing"
    extracting_claims = "extracting_claims"
    building_context = "building_context"
    ready = "ready"
    failed = "failed"


class EnrichmentStatus(str, enum.Enum):
    skeleton_only = "skeleton_only"
    glossary_done = "glossary_done"
    visuals_captioned = "visuals_captioned"
    full = "full"


class ClaimType(str, enum.Enum):
    statistic = "statistic"
    causal = "causal"
    comparison = "comparison"
    quote = "quote"
    definition = "definition"
    prediction = "prediction"
    other = "other"


class ClaimStatus(str, enum.Enum):
    extracted = "extracted"
    context_built = "context_built"
    substantiated = "substantiated"
    needs_review = "needs_review"


class ConfidenceLevel(str, enum.Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ReferenceMappingStatus(str, enum.Enum):
    shortlisted = "shortlisted"
    checked = "checked"
    skipped = "skipped"


class SpanStatus(str, enum.Enum):
    running = "running"
    success = "success"
    failed = "failed"


# Models
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    memberships = relationship("WorkspaceMember", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="workspace", cascade="all, delete-orphan")


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(Enum(WorkspaceRole, native_enum=False), nullable=False, default=WorkspaceRole.member)
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
    )

    user = relationship("User", back_populates="memberships")
    workspace = relationship("Workspace", back_populates="members")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="refresh_tokens")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    role = Column(Enum(DocumentRole, native_enum=False), nullable=False, default=DocumentRole.base)
    source_type = Column(Enum(SourceType, native_enum=False), nullable=False, default=SourceType.upload)
    original_filename = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    checksum = Column(String(64), nullable=False, index=True)
    minio_raw_path = Column(String, nullable=False)
    status = Column(Enum(DocumentStatus, native_enum=False), nullable=False, default=DocumentStatus.pending)
    page_count = Column(Integer, nullable=True)
    doc_language = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    workspace = relationship("Workspace", back_populates="documents")
    index = relationship("DocumentIndex", uselist=False, back_populates="document", cascade="all, delete-orphan")


class DocumentIndex(Base):
    __tablename__ = "document_indexes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), unique=True, nullable=False)
    minio_index_path = Column(String, nullable=False)
    coverage_pct = Column(Float, nullable=False, default=0.0)
    node_count = Column(Integer, nullable=False, default=0)
    built_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    enrichment_status = Column(Enum(EnrichmentStatus, native_enum=False), nullable=False, default=EnrichmentStatus.skeleton_only)

    document = relationship("Document", back_populates="index")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    base_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    status = Column(Enum(AnalysisStatus, native_enum=False), nullable=False, default=AnalysisStatus.pending)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    workspace = relationship("Workspace", back_populates="analyses")
    claims = relationship("Claim", back_populates="analysis", cascade="all, delete-orphan")
    spans = relationship("TraceSpan", back_populates="analysis", cascade="all, delete-orphan")


class AnalysisReference(Base):
    __tablename__ = "analysis_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)

    __table_args__ = (
        UniqueConstraint("analysis_id", "document_id", name="uq_analysis_reference"),
    )


class Claim(Base):
    __tablename__ = "claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    text = Column(Text, nullable=False)
    claim_type = Column(Enum(ClaimType, native_enum=False), nullable=False, default=ClaimType.other)
    source_node_ids = Column(JSONB, nullable=False)
    section_path = Column(Text, nullable=True)
    status = Column(Enum(ClaimStatus, native_enum=False), nullable=False, default=ClaimStatus.extracted)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="claims")
    context = relationship("ClaimContext", uselist=False, back_populates="claim", cascade="all, delete-orphan")


class ClaimContext(Base):
    __tablename__ = "claim_contexts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), unique=True, nullable=False)
    local_node_ids = Column(JSONB, nullable=False)
    structural_node_ids = Column(JSONB, nullable=False)
    global_refs = Column(JSONB, nullable=False)
    agent_trace_id = Column(UUID(as_uuid=True), ForeignKey("trace_spans.id"), nullable=True)
    context_sufficient = Column(Boolean, nullable=False, default=False)
    built_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    claim = relationship("Claim", back_populates="context")


class ReferenceMapping(Base):
    __tablename__ = "reference_mappings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    status = Column(Enum(ReferenceMappingStatus, native_enum=False), nullable=False, default=ReferenceMappingStatus.shortlisted)


class Verification(Base):
    __tablename__ = "verifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    validated = Column(Boolean, nullable=True)
    issue = Column(Text, nullable=True)
    evidence_node_id = Column(String, nullable=True)
    confidence = Column(Enum(ConfidenceLevel, native_enum=False), nullable=False, default=ConfidenceLevel.medium)
    checked_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TraceSpan(Base):
    __tablename__ = "trace_spans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    parent_span_id = Column(UUID(as_uuid=True), ForeignKey("trace_spans.id"), nullable=True)
    step = Column(String, nullable=False)
    agent_name = Column(String, nullable=True)
    status = Column(Enum(SpanStatus, native_enum=False), nullable=False, default=SpanStatus.running)
    input_ref = Column(String, nullable=True)
    output_ref = Column(String, nullable=True)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    analysis = relationship("Analysis", back_populates="spans")
    tool_calls = relationship("ToolCall", back_populates="span", cascade="all, delete-orphan")


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    span_id = Column(UUID(as_uuid=True), ForeignKey("trace_spans.id"), nullable=False)
    tool_name = Column(String, nullable=False)
    args = Column(JSONB, nullable=False)
    result_size_bytes = Column(Integer, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    called_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    span = relationship("TraceSpan", back_populates="tool_calls")


class ModelPricing(Base):
    __tablename__ = "model_pricing"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    input_price_per_1k = Column(Float, nullable=False)
    output_price_per_1k = Column(Float, nullable=False)
    effective_from = Column(DateTime, default=datetime.utcnow, nullable=False)
