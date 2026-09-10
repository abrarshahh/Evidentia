import uuid
import logging
from typing import List, Tuple, Set, Optional
from app.schemas.document_index import DocumentSkeletonIndex, DocumentNode, CoverageMetrics
from app.schemas.claims import ExtractedClaim, ClaimType
from app.agents.claim_extractor import ClaimExtractor
from app.indexing.llm_client import llm_client
from app.tracing.span_logger import span_tracer
from app.tracing.cost_mapper import cost_mapper

logger = logging.getLogger(__name__)


class CoverageChecker:
    """
    Extraction Coverage Checker & Loop Manager.
    Evaluates claim coverage across document nodes and executes targeted re-extraction
    passes on uncovered nodes to achieve target coverage (>98%).
    """

    @classmethod
    def evaluate_coverage(
        cls,
        index: DocumentSkeletonIndex,
        claims: List[ExtractedClaim],
    ) -> Tuple[CoverageMetrics, List[DocumentNode]]:
        """
        Calculate node coverage metrics and identify uncovered document nodes.
        """
        total_nodes = len(index.nodes)
        if total_nodes == 0:
            return CoverageMetrics(total_nodes=0, covered_nodes=0, coverage_pct=100.0), []

        referenced_node_ids: Set[str] = set()
        for claim in claims:
            for nid in claim.source_node_ids:
                referenced_node_ids.add(nid)

        covered_nodes_count = 0
        uncovered_nodes: List[DocumentNode] = []

        for node in index.nodes:
            if node.node_id in referenced_node_ids:
                covered_nodes_count += 1
            else:
                uncovered_nodes.append(node)

        coverage_pct = round((covered_nodes_count / total_nodes) * 100.0, 2)
        metrics = CoverageMetrics(
            total_nodes=total_nodes,
            covered_nodes=covered_nodes_count,
            coverage_pct=coverage_pct,
            uncovered_node_ids=[n.node_id for n in uncovered_nodes],
        )

        return metrics, uncovered_nodes

    @classmethod
    async def run_coverage_loop(
        cls,
        workspace_id: str,
        index: DocumentSkeletonIndex,
        claims: List[ExtractedClaim],
        target_coverage: float = 98.0,
        max_passes: int = 1,
        analysis_id: Optional[uuid.UUID] = None,
    ) -> Tuple[List[ExtractedClaim], CoverageMetrics]:
        """
        Evaluate node claim coverage with tracing. If coverage falls below target_coverage (98%),
        runs targeted re-extraction passes on uncovered nodes.
        """
        async with span_tracer("coverage_check", agent_name="CoverageChecker", analysis_id=analysis_id) as span_info:
            current_claims = list(claims)
            metrics, uncovered_nodes = cls.evaluate_coverage(index, current_claims)

            pass_count = 0
            logger.info(f"Initial claim coverage for document {index.document_id}: {metrics.node_coverage_percent}% (Uncovered nodes: {len(uncovered_nodes)})")

            while metrics.node_coverage_percent < target_coverage and uncovered_nodes and pass_count < max_passes:
                pass_count += 1
                logger.info(f"Triggering Targeted Re-Extraction Pass {pass_count}/{max_passes} on {len(uncovered_nodes)} uncovered nodes...")

                nodes_payload = [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type,
                        "page_range": n.page_range,
                        "text": n.text,
                    }
                    for n in uncovered_nodes
                ]

                prompt = f"""You are performing a targeted re-extraction pass to ensure exhaustive coverage.
The following document nodes were NOT referenced in the initial claim extraction. Thoroughly examine these uncovered nodes and extract any valid claims:

Uncovered Nodes:
{nodes_payload}

Return JSON format:
{{
  "claims": [
    {{
      "statement": "Statement extracted from uncovered node.",
      "claim_type": "factual",
      "source_node_ids": ["n_004"],
      "page_range": [1],
      "base_confidence": 0.85
    }}
  ]
}}
"""
                try:
                    res = await llm_client.generate_json(
                        prompt=prompt,
                        system_instruction="You perform targeted re-extraction on uncovered document nodes.",
                    )
                    raw_new_claims = res.get("claims", [])
                    start_id = len(current_claims) + 1

                    for item in raw_new_claims:
                        stmt = str(item.get("statement", "")).strip()
                        if not stmt:
                            continue

                        if any(c.statement.lower() == stmt.lower() for c in current_claims):
                            continue

                        raw_type = item.get("claim_type", "factual").lower()
                        try:
                            claim_type = ClaimType(raw_type)
                        except ValueError:
                            claim_type = ClaimType.FACTUAL

                        nodes = item.get("source_node_ids", [])
                        pages = item.get("page_range", [1])
                        base_conf = float(item.get("base_confidence", 0.8))
                        final_conf = ClaimExtractor.calculate_confidence(base_conf, stmt, nodes)

                        current_claims.append(
                            ExtractedClaim(
                                claim_id=f"c_{start_id:03d}",
                                statement=stmt,
                                claim_type=claim_type,
                                source_node_ids=nodes,
                                page_range=pages,
                                confidence_score=final_conf,
                            )
                        )
                        start_id += 1
                except Exception as e:
                    logger.error(f"Targeted re-extraction pass error: {e}")
                    break

                # Re-evaluate coverage metrics
                metrics, uncovered_nodes = cls.evaluate_coverage(index, current_claims)
                logger.info(f"Coverage after pass {pass_count}: {metrics.node_coverage_percent}% (Uncovered nodes: {len(uncovered_nodes)})")

            index.coverage_metrics = metrics

            payload = {
                "initial_coverage_pct": metrics.node_coverage_percent,
                "re_extraction_passes": pass_count,
                "total_claims": len(current_claims),
            }
            await cost_mapper.record_span_usage(
                span_id=span_info["span_id"],
                workspace_id=str(workspace_id),
                trace_id=span_info["trace_id"],
                model_name="gemini-3.6-flash",
                tokens_in=pass_count * 500,
                tokens_out=pass_count * 150,
                payload_data=payload,
            )

            return current_claims, metrics


coverage_checker = CoverageChecker()
