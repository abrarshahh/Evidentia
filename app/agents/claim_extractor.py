import re
import asyncio
import logging
from typing import List, Dict, Any
from app.schemas.document_index import DocumentSkeletonIndex
from app.schemas.claims import ClaimType, ExtractedClaim, ClaimExtractionResult
from app.indexing.llm_client import llm_client

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
                "page": n.page_range[0] if n.page_range else 1,
                "text": n.text,
            }
            for n in index.nodes
            if n.text and n.text.strip()
        ]

        prompt = f"""You are an expert enterprise claim extraction engine.
Examine the following document nodes and extract all explicit factual, financial, compliance, technical, legal, performance, safety, efficacy, statistical, promotional, comparative, and quality claims.

Document Nodes:
{nodes_payload}

Claim Types Available:
- factual, financial, compliance, technical, legal, performance, safety, efficacy, statistical, promotional, comparative, quality, other

Return a JSON object matching this exact structure:
{{
  "claims": [
    {{
      "statement": "Clear factual claim statement extracted from text.",
      "claim_type": "efficacy",
      "source_node_ids": ["n_001"],
      "page_range": [1],
      "base_confidence": 0.85
    }}
  ]
}}
"""

        claims: List[ExtractedClaim] = []
        try:
            result = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You are an enterprise claim extraction assistant outputting structured JSON.",
            )
            raw_claims = result.get("claims", [])
            for idx, item in enumerate(raw_claims, start=1):
                stmt = item.get("statement", "").strip()
                if not stmt:
                    continue

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

    @classmethod
    async def extract_map_reduce(
        cls,
        index: DocumentSkeletonIndex,
    ) -> List[ExtractedClaim]:
        """
        Execute map-reduce claim extraction for large multi-section documents.
        """
        # Map Phase: Partition nodes into 20,000-char chunks (~5,000 tokens) to minimize API call count
        chunks: List[List[Dict[str, Any]]] = []
        current_chunk: List[Dict[str, Any]] = []
        current_chars = 0

        for n in index.nodes:
            if not n.text or not n.text.strip():
                continue
            item = {
                "node_id": n.node_id,
                "node_type": n.node_type,
                "page": n.page_range[0] if n.page_range else 1,
                "text": n.text,
            }
            if current_chars + len(n.text) > 20000 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_chars = len(n.text)
            else:
                current_chunk.append(item)
                current_chars += len(n.text)

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Map-Reduce claim extraction: processing {len(chunks)} document chunks in parallel...")

        # Concurrency limit bound by Gemini pool size (minimum 2 concurrent workers)
        pool_size = max(2, llm_client.get_pool_size())
        semaphore = asyncio.Semaphore(pool_size)

        async def process_chunk(chunk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            async with semaphore:
                prompt = f"""Extract all explicit claims from this document chunk.
Chunk Nodes:
{chunk_data}

Claim Types:
- factual, financial, compliance, technical, legal, performance, safety, efficacy, statistical, promotional, comparative, quality, other

Return a JSON object:
{{
  "claims": [
    {{
      "statement": "Statement",
      "claim_type": "financial",
      "source_node_ids": ["n_001"],
      "page_range": [1],
      "base_confidence": 0.85
    }}
  ]
}}
"""
                try:
                    res = await llm_client.generate_json(prompt=prompt)
                    return res.get("claims", [])
                except Exception as e:
                    logger.warning(f"Chunk map extraction skipped on chunk error: {e}")
                    return []

        # Execute Map Phase on all chunks concurrently
        results = await asyncio.gather(*[process_chunk(c) for c in chunks])

        mapped_claims: List[Dict[str, Any]] = []
        for chunk_claims in results:
            mapped_claims.extend(chunk_claims)

        # Reduce Phase: Aggregate and format
        claims: List[ExtractedClaim] = []
        seen_statements = set()

        for idx, item in enumerate(mapped_claims, start=1):
            stmt = item.get("statement", "").strip()
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

        return claims

    @classmethod
    async def extract_claims(
        cls,
        workspace_id: str,
        index: DocumentSkeletonIndex,
    ) -> ClaimExtractionResult:
        """
        Evaluate document token size and route to Single-Pass or Map-Reduce extraction.
        """
        total_text_len = sum(len(n.text) for n in index.nodes if n.text)

        if total_text_len <= MAX_SINGLE_PASS_CHARS:
            mode = "single_pass"
            logger.info(f"Routing document {index.document_id} ({total_text_len} chars) to Single-Pass Extraction...")
            extracted = await cls.extract_single_pass(index)
        else:
            mode = "map_reduce"
            logger.info(f"Routing document {index.document_id} ({total_text_len} chars) to Map-Reduce Extraction...")
            extracted = await cls.extract_map_reduce(index)

        return ClaimExtractionResult(
            document_id=str(index.document_id),
            workspace_id=str(workspace_id),
            extraction_mode=mode,
            total_claims=len(extracted),
            claims=extracted,
        )


claim_extractor = ClaimExtractor()
