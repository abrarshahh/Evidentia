import logging
from typing import List, Tuple, Set
from app.schemas.document_index import DocumentSkeletonIndex, DocumentNode, CoverageMetrics
from app.schemas.claims import ExtractedClaim, ClaimType
from app.agents.claim_extractor import ClaimExtractor
from app.indexing.llm_client import llm_client

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
        Compute coverage metrics and isolate uncovered text-bearing nodes.
        """
        text_nodes = [n for n in index.nodes if n.text and n.text.strip()]
        total_nodes = len(text_nodes)
        total_chars = sum(len(n.text) for n in text_nodes)

        # Collect all source node IDs referenced by claims
        referenced_node_ids: Set[str] = set()
        for claim in claims:
            referenced_node_ids.update(claim.source_node_ids)

        covered_nodes = [n for n in text_nodes if n.node_id in referenced_node_ids]
        uncovered_nodes = [n for n in text_nodes if n.node_id not in referenced_node_ids]

        covered_chars = sum(len(n.text) for n in covered_nodes)

        node_coverage_pct = round((len(covered_nodes) / total_nodes * 100.0), 2) if total_nodes > 0 else 100.0
        char_coverage_pct = round((covered_chars / total_chars * 100.0), 2) if total_chars > 0 else 100.0

        metrics = CoverageMetrics(
            total_nodes=total_nodes,
            covered_nodes=len(covered_nodes),
            coverage_pct=node_coverage_pct,
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
        max_passes: int = 2,
    ) -> Tuple[List[ExtractedClaim], CoverageMetrics]:
        """
        Evaluate node claim coverage. If coverage falls below target_coverage (98%),
        runs targeted re-extraction passes on uncovered nodes.
        """
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
                    "page": n.page_range[0] if n.page_range else 1,
                    "text": n.text,
                }
                for n in uncovered_nodes
            ]

            prompt = f"""You are performing a targeted re-extraction pass to ensure exhaustive coverage.
The following document nodes were NOT referenced in the initial claim extraction. Thoroughly examine these uncovered nodes and extract any valid claims (factual, financial, compliance, technical, legal, performance, safety, efficacy, statistical, promotional, comparative, quality):

Uncovered Nodes:
{nodes_payload}

Return a JSON object:
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
                    stmt = item.get("statement", "").strip()
                    if not stmt:
                        continue

                    # Avoid duplicate statements
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

        # Update index coverage metrics
        index.coverage_metrics = metrics
        return current_claims, metrics


coverage_checker = CoverageChecker()
