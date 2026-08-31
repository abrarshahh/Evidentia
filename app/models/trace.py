import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.models import Base


class TraceSpanModel(Base):
    __tablename__ = "trace_spans"

    span_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(String(64), nullable=False, index=True)
    parent_span_id = Column(UUID(as_uuid=True), nullable=True)
    workspace_id = Column(String(64), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    span_type = Column(String(32), nullable=False, default="agent")  # agent | llm | tool
    start_time = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="RUNNING")  # RUNNING | SUCCESS | ERROR
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    tool_calls = relationship("ToolCallRecordModel", back_populates="span", cascade="all, delete-orphan")


class ToolCallRecordModel(Base):
    __tablename__ = "tool_calls"

    call_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    span_id = Column(UUID(as_uuid=True), ForeignKey("trace_spans.span_id", ondelete="CASCADE"), nullable=False, index=True)
    tool_name = Column(String(128), nullable=False)
    input_args = Column(JSON, nullable=True)
    output_result = Column(JSON, nullable=True)
    duration_ms = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="SUCCESS")  # SUCCESS | ERROR

    span = relationship("TraceSpanModel", back_populates="tool_calls")
