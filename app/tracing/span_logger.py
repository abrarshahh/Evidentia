import uuid
import time
import json
import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from app.db.session import AsyncSessionLocal
from app.db.models import TraceSpan, ToolCall, SpanStatus

from app.core.logging_config import set_log_context

logger = logging.getLogger(__name__)

# Context variable to maintain active span ID across async calls
active_span_id: ContextVar[Optional[uuid.UUID]] = ContextVar("active_span_id", default=None)


@asynccontextmanager
async def span_tracer(
    step: str,
    agent_name: Optional[str] = None,
    analysis_id: Optional[uuid.UUID] = None,
    input_ref: Optional[str] = None,
):
    """
    Async context manager wrapping execution spans and persisting trace records to PostgreSQL.
    """
    if analysis_id:
        set_log_context(analysis_id=str(analysis_id))

    parent_span = active_span_id.get()
    span_id = uuid.uuid4()
    token = active_span_id.set(span_id)

    started_at = datetime.utcnow()
    t0 = time.perf_counter()

    async with AsyncSessionLocal() as session:
        span_record = TraceSpan(
            id=span_id,
            analysis_id=analysis_id,
            parent_span_id=parent_span,
            step=step,
            agent_name=agent_name or step,
            status=SpanStatus.running,
            input_ref=input_ref,
            started_at=started_at,
        )
        session.add(span_record)
        await session.commit()

    trace_id_str = f"tr_{uuid.uuid4().hex[:12]}"
    span_info = {"span_id": span_id, "trace_id": trace_id_str, "output_ref": None}
    final_status = SpanStatus.success

    logger.info(f"Agent Span Started: step='{step}', agent='{agent_name or step}', span_id={span_id}, trace_id={trace_id_str}")

    try:
        yield span_info
    except Exception as e:
        final_status = SpanStatus.failed
        logger.error(f"Agent Span Failed: step='{step}', span_id={span_id}, error={e}")
        raise e
    finally:
        ended_at = datetime.utcnow()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        output_ref = span_info.get("output_ref")

        logger.info(f"Agent Span Completed: step='{step}', status={final_status.value}, duration={duration_ms}ms")

        async with AsyncSessionLocal() as session:
            db_span = await session.get(TraceSpan, span_id)
            if db_span:
                db_span.ended_at = ended_at
                db_span.duration_ms = duration_ms
                db_span.status = final_status
                if output_ref:
                    db_span.output_ref = str(output_ref)
                await session.commit()

        active_span_id.reset(token)


def trace_tool(tool_name: Optional[str] = None):
    """
    Decorator to wrap custom tool invocations and persist execution metrics to PostgreSQL.
    """
    def decorator(func: Callable):
        t_name = tool_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_id = active_span_id.get()
            t0 = time.perf_counter()
            result = None

            input_data = {}
            if kwargs:
                input_data = {k: str(v)[:200] for k, v in kwargs.items()}
            elif args:
                input_data = {"args": [str(a)[:200] for a in args]}

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                result_bytes = len(json.dumps(input_data).encode("utf-8"))

                if span_id:
                    try:
                        async with AsyncSessionLocal() as session:
                            call_record = ToolCall(
                                span_id=span_id,
                                tool_name=t_name,
                                args=input_data,
                                result_size_bytes=result_bytes,
                                duration_ms=duration_ms,
                                called_at=datetime.utcnow(),
                            )
                            session.add(call_record)
                            await session.commit()
                    except Exception as ex:
                        logger.error(f"Failed to record tool call span: {ex}")

        return async_wrapper
    return decorator
