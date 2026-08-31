import io
import uuid
import json
import logging
from typing import Dict, Any, Optional

from app.storage.minio_client import minio_client
from app.db.session import AsyncSessionLocal
from app.db.models import TraceSpan

logger = logging.getLogger(__name__)

# Model Pricing Table (USD per 1,000 tokens)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-3.6-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.00030},
    "gemini-3.5-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.00030},
    "gemini-2.5-flash-lite": {"input_per_1k": 0.0000375, "output_per_1k": 0.00015},
    "gemini-2.5-pro": {"input_per_1k": 0.00125, "output_per_1k": 0.00500},
    "gemini-embedding-001": {"input_per_1k": 0.000010, "output_per_1k": 0.0},
    "gemini-embedding-2": {"input_per_1k": 0.000010, "output_per_1k": 0.0},
}


class CostMapper:
    """
    MinIO Log Blob Writer & Cost Mapper Engine.
    Archives large LLM input/output JSON payloads into MinIO object storage
    and calculates exact USD execution costs persisted to PostgreSQL trace_spans.
    """

    @staticmethod
    def calculate_cost(
        model_name: str,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        """
        Calculate execution cost in USD based on input/output token volume.
        """
        pricing = MODEL_PRICING.get(model_name.lower()) or MODEL_PRICING["gemini-3.6-flash"]

        cost_in = (tokens_in / 1000.0) * pricing["input_per_1k"]
        cost_out = (tokens_out / 1000.0) * pricing["output_per_1k"]

        total_cost = cost_in + cost_out
        return round(total_cost, 6)

    @staticmethod
    def upload_log_blob(
        workspace_id: str,
        trace_id: str,
        span_id: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        Upload detailed input/output JSON payload to MinIO under logs prefix.
        Returns MinIO object key string.
        """
        object_key = f"logs/{workspace_id}/{trace_id}/{span_id}.json"
        json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        data_stream = io.BytesIO(json_bytes)

        try:
            minio_client.init_buckets()
            minio_client.client.put_object(
                bucket_name=minio_client.LOGS_BUCKET,
                object_name=object_key,
                data=data_stream,
                length=len(json_bytes),
                content_type="application/json",
            )
            logger.info(f"Uploaded log blob payload ({len(json_bytes)} bytes) to MinIO: '{object_key}'")
            return object_key
        except Exception as e:
            logger.error(f"Failed to upload log blob to MinIO key '{object_key}': {e}")
            return None

    @classmethod
    async def record_span_usage(
        cls,
        span_id: uuid.UUID,
        workspace_id: str,
        trace_id: str,
        model_name: str = "gemini-3.6-flash",
        tokens_in: int = 0,
        tokens_out: int = 0,
        payload_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate cost, upload MinIO log blob, and update PostgreSQL trace_spans metrics.
        """
        cost_usd = cls.calculate_cost(model_name, tokens_in, tokens_out)
        minio_ref = None

        if payload_data:
            minio_ref = cls.upload_log_blob(
                workspace_id=workspace_id,
                trace_id=trace_id,
                span_id=str(span_id),
                payload=payload_data,
            )

        async with AsyncSessionLocal() as session:
            db_span = await session.get(TraceSpan, span_id)
            if db_span:
                db_span.tokens_in = tokens_in
                db_span.tokens_out = tokens_out
                db_span.cost_usd = cost_usd
                if minio_ref:
                    db_span.output_ref = minio_ref
                await session.commit()

        logger.info(f"Span [{span_id}] updated: tokens_in={tokens_in}, tokens_out={tokens_out}, cost=${cost_usd:.6f}")
        return cost_usd


cost_mapper = CostMapper()
