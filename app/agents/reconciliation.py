import math
import logging
from typing import List, Tuple, Optional, Set
from app.schemas.claims import ExtractedClaim, ClaimType
from app.indexing.embedder import node_embedder
from app.indexing.llm_client import llm_client

logger = logging.getLogger(__name__)


class ClaimReconciliationEngine:
    """
    Claim Reconciliation Engine.
    Identifies duplicate and overlapping claims using pairwise vector cosine similarity (>0.85)
    and verifies/merges claims with Gemini 3.6 Flash while combining source node attributions.
    """

    @staticmethod
    def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two float vectors.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @classmethod
    def find_candidate_duplicates(
        cls,
        claims: List[ExtractedClaim],
        embeddings: List[List[float]],
        threshold: float = 0.85,
    ) -> List[Tuple[int, int, float]]:
        """
        Identify candidate duplicate claim index pairs whose vector cosine similarity exceeds threshold.
        """
        candidates: List[Tuple[int, int, float]] = []
        n = len(claims)

        for i in range(n):
            for j in range(i + 1, n):
                sim = cls.compute_cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    candidates.append((i, j, round(sim, 4)))

        # Sort candidates descending by similarity score
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    @classmethod
    async def reconcile_pair_llm(
        cls,
        claim_a: ExtractedClaim,
        claim_b: ExtractedClaim,
    ) -> Optional[ExtractedClaim]:
        """
        Submit a candidate pair to Gemini 3.6 Flash to confirm whether they represent duplicate factual claims.
        If confirmed, returns a merged ExtractedClaim combining source node IDs and page ranges.
        """
        prompt = f"""You are a claim reconciliation system. Analyze these two extracted claims and determine if they express the same core factual assertion or duplicate information:

Claim A: "{claim_a.statement}" (Type: {claim_a.claim_type.value})
Claim B: "{claim_b.statement}" (Type: {claim_b.claim_type.value})

If they are duplicates or convey the same factual assertion, return a single merged statement that captures the complete details.

Return a JSON object:
{{
  "is_duplicate": true,
  "merged_statement": "Concise merged statement combining details.",
  "claim_type": "{claim_a.claim_type.value}"
}}
If NOT duplicates, return:
{{
  "is_duplicate": false
}}
"""
        try:
            res = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You determine if claim statements are duplicate assertions and merge them.",
            )
            if res.get("is_duplicate", False):
                merged_stmt = res.get("merged_statement", claim_a.statement).strip()
                raw_type = res.get("claim_type", claim_a.claim_type.value).lower()
                try:
                    merged_type = ClaimType(raw_type)
                except ValueError:
                    merged_type = claim_a.claim_type

                # Combine & deduplicate source node IDs
                combined_nodes = list(dict.fromkeys(claim_a.source_node_ids + claim_b.source_node_ids))

                # Combine & sort deduplicated page ranges
                combined_pages = sorted(list(set(claim_a.page_range + claim_b.page_range)))

                # Recalculate confidence
                avg_confidence = max(claim_a.confidence_score, claim_b.confidence_score)

                return ExtractedClaim(
                    claim_id=claim_a.claim_id,
                    statement=merged_stmt,
                    claim_type=merged_type,
                    source_node_ids=combined_nodes,
                    page_range=combined_pages,
                    confidence_score=avg_confidence,
                )
        except Exception as e:
            logger.warning(f"LLM pair reconciliation skipped on error: {e}")

        return None

    @classmethod
    async def reconcile_claims(
        cls,
        claims: List[ExtractedClaim],
        similarity_threshold: float = 0.85,
    ) -> List[ExtractedClaim]:
        """
        Reconcile duplicate claims using vector embeddings and Gemini 3.6 Flash verification.
        """
        if len(claims) <= 1:
            return claims

        logger.info(f"Reconciling {len(claims)} claims with cosine threshold {similarity_threshold}...")

        # 1. Generate embeddings for claim statements
        statements = [c.statement for c in claims]
        try:
            embeddings = node_embedder.generate_embeddings(statements)
        except Exception as e:
            logger.error(f"Embedding generation for claim statements failed: {e}. Skipping reconciliation.")
            return claims

        # 2. Find candidate pairs exceeding similarity threshold
        candidates = cls.find_candidate_duplicates(claims, embeddings, threshold=similarity_threshold)
        logger.info(f"Found {len(candidates)} candidate duplicate pair(s) above threshold {similarity_threshold}.")

        if not candidates:
            return claims

        # 3. Process candidate pairs and merge duplicates
        merged_indices: Set[int] = set()
        claim_map: Dict[int, ExtractedClaim] = {i: claim for i, claim in enumerate(claims)}

        for i, j, sim in candidates:
            if i in merged_indices or j in merged_indices:
                continue

            claim_a = claim_map[i]
            claim_b = claim_map[j]

            logger.info(f"Checking candidate pair [{i}] and [{j}] (Similarity: {sim:.4f})...")
            merged_claim = await cls.reconcile_pair_llm(claim_a, claim_b)

            if merged_claim:
                logger.info(f"Successfully merged claim [{claim_a.claim_id}] and [{claim_b.claim_id}].")
                claim_map[i] = merged_claim
                merged_indices.add(j)

        # 4. Consolidate final claim list and re-number IDs
        final_claims: List[ExtractedClaim] = []
        new_counter = 1

        for idx in range(len(claims)):
            if idx not in merged_indices:
                c = claim_map[idx]
                c.claim_id = f"c_{new_counter:03d}"
                new_counter += 1
                final_claims.append(c)

        logger.info(f"Claim reconciliation complete: reduced from {len(claims)} to {len(final_claims)} claims.")
        return final_claims


reconciliation_engine = ClaimReconciliationEngine()
