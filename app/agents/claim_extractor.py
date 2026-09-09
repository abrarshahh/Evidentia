import re
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from app.schemas.document_index import DocumentSkeletonIndex
from app.schemas.claims import ClaimType, ExtractedClaim, ClaimExtractionResult
from app.indexing.llm_client import llm_client
from app.tracing.span_logger import span_tracer
from app.tracing.cost_mapper import cost_mapper

logger = logging.getLogger(__name__)

# Single-pass threshold: 24,000 characters (~6,000 tokens)
MAX_SINGLE_PASS_CHARS = 24000


class ClaimExtractor:
    """
    Claim Extraction Router Engine.
    Evaluates document token size to route compact documents to single-pass extraction
    or larger documents to map-reduce chunked extraction using Gemini 3.6 Flash.
    """

    @staticmethod
    def calculate_confidence(
        base_score: float,
        statement: str,
        source_node_ids: List[str],
    ) -> float:
        """
        Calculate refined confidence score based on LLM base assessment,
        numerical precision, node attribution, and speculative phrasing.
        """
        score = base_score if (base_score and 0.0 <= base_score <= 1.0) else 0.8

        # 1. Boost for numerical/quantitative precision (+0.10)
        if re.search(r"\d+|\%|\$|\b(increase|decrease|growth)\b", statement, re.IGNORECASE):
            score += 0.10

        # 2. Boost for direct source node attribution (+0.05)
        if source_node_ids:
            score += 0.05

        # 3. Deduction for speculative/uncertain phrasing (-0.15)
        if re.search(r"\b(might|possibly|maybe|expected to|projected to|could)\b", statement, re.IGNORECASE):
            score -= 0.15

        # Clamp score between 0.0 and 1.0
        return round(max(0.0, min(1.0, score)), 2)

    @classmethod
    async def extract_single_pass(
        cls,
        index: DocumentSkeletonIndex,
    ) -> List[ExtractedClaim]:
        """
        Execute single-pass claim extraction over compact document nodes.
        """
        nodes_payload = [
            {
                "node_id": n.node_id,
                "node_type": n.node_type,
                "page_range": n.page_range,
                "text": n.text,
            }
            for n in index.nodes
            if n.text and n.text.strip()
        ]

        prompt = f"""You are an enterprise claim extraction engine.
Examine the following document nodes and extract ALL verifiable claims:

Document Nodes:
{nodes_payload}

Return JSON format:
{{
  "claims": [
    {{
      "statement": "Revenue grew by 24% year-over-year in Q3 2026 reaching $4.2 million.",
      "claim_type": "statistical",
      "source_node_ids": ["sec_01_n01"],
      "page_range": [1],
      "base_confidence": 0.95
    }}
  ]
}}
"""

        claims: List[ExtractedClaim] = []
        try:
            res = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You extract precise claims from document nodes into structured JSON.",
            )
            raw_claims = res.get("claims", [])
            seen_statements = set()

            for idx, item in enumerate(raw_claims, start=1):
                stmt = str(item.get("statement", "")).strip()
                if not stmt or stmt.lower() in seen_statements:
                    continue
                seen_statements.add(stmt.lower())

                raw_type = item.get("claim_type", "factual").lower()
                try:
                    claim_type = ClaimType(raw_type)
                except ValueError:
                    claim_type = ClaimType.FACTUAL

                nodes = item.get("source_node_ids", [])
                pages = item.get("page_range", [1])
                base_conf = float(item.get("base_confidence", 0.8))
                final_conf = cls.calculate_confidence(base_conf, stmt, nodes)

                claims.append(
                    ExtractedClaim(
                        claim_id=f"c_{idx:03d}",
                        statement=stmt,
                        claim_type=claim_type,
                        source_node_ids=nodes,
                        page_range=pages,
                        confidence_score=final_conf,
                    )
                )
        except Exception as e:
            logger.error(f"Single-pass claim extraction error: {e}")

        return claims

    OVERLAP_CHARS = 1500

    @classmethod
    async def extract_map_reduce(
        cls,
        index: DocumentSkeletonIndex,
        overlap_chars: int = 1500,
    ) -> List[ExtractedClaim]:
        """
        Execute map-reduce chunked claim extraction over large document nodes with
        trailing node overlap (1,500 chars) to prevent claim loss across chunk boundaries.
        """
        all_claims = []
        current_chunk = []
        current_char_count = 0

        for n in index.nodes:
            if not n.text:
                continue
            current_chunk.append(n)
            current_char_count += len(n.text)

            if current_char_count >= MAX_SINGLE_PASS_CHARS:
                sub_index = DocumentSkeletonIndex(
                    document_id=index.document_id,
                    checksum=index.checksum,
                    metadata=index.metadata,
                    structure_tree=index.structure_tree,
                    nodes=current_chunk,
                    visuals=index.visuals,
                    glossary=index.glossary,
                )
                chunk_claims = await cls.extract_single_pass(sub_index)
                all_claims.extend(chunk_claims)

                # Preserve trailing overlap nodes for the next chunk
                overlap_nodes = []
                overlap_len = 0
                for node in reversed(current_chunk):
                    if node.text:
                        overlap_len += len(node.text)
                        overlap_nodes.insert(0, node)
                        if overlap_len >= overlap_chars:
                            break
                current_chunk = overlap_nodes
                current_char_count = overlap_len

        if current_chunk:
            sub_index = DocumentSkeletonIndex(
                document_id=index.document_id,
                checksum=index.checksum,
                metadata=index.metadata,
                structure_tree=index.structure_tree,
                nodes=current_chunk,
                visuals=index.visuals,
                glossary=index.glossary,
            )
            chunk_claims = await cls.extract_single_pass(sub_index)
            all_claims.extend(chunk_claims)

        return all_claims

    @classmethod
    async def extract_claims(
        cls,
        workspace_id: str,
        index: DocumentSkeletonIndex,
        analysis_id: Optional[uuid.UUID] = None,
    ) -> ClaimExtractionResult:
        """
        Evaluate document token size and route to Single-Pass or Map-Reduce extraction with tracing.
        """
        total_text_len = sum(len(n.text) for n in index.nodes if n.text)
        mode = "single_pass" if total_text_len <= MAX_SINGLE_PASS_CHARS else "map_reduce"

        async with span_tracer("claim_extraction", agent_name="ClaimExtractor", analysis_id=analysis_id) as span_info:
            if mode == "single_pass":
                logger.info(f"Routing document {index.document_id} ({total_text_len} chars) to Single-Pass Extraction...")
                extracted = await cls.extract_single_pass(index)
            else:
                logger.info(f"Routing document {index.document_id} ({total_text_len} chars) to Map-Reduce Extraction...")
                extracted = await cls.extract_map_reduce(index)

            payload = {
                "document_id": str(index.document_id),
                "extraction_mode": mode,
                "total_claims": len(extracted),
                "claims": [c.model_dump() for c in extracted],
            }
            tokens_in = max(100, total_text_len // 4)
            tokens_out = max(50, len(extracted) * 40)
            await cost_mapper.record_span_usage(
                span_id=span_info["span_id"],
                workspace_id=str(workspace_id),
                trace_id=span_info["trace_id"],
                model_name="gemini-3.6-flash",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                payload_data=payload,
            )

        return ClaimExtractionResult(
            document_id=str(index.document_id),
            workspace_id=str(workspace_id),
            extraction_mode=mode,
            total_claims=len(extracted),
            claims=extracted,
        )


claim_extractor = ClaimExtractor()
