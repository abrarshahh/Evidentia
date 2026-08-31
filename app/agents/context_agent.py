import logging
from typing import List, Dict, Any
from app.schemas.document_index import DocumentSkeletonIndex
from app.schemas.claims import ExtractedClaim
from app.schemas.context import ClaimContextAssociation, ContextBuildingResult
from app.agents.tools.document_tools import (
    get_section,
    get_page,
    get_visual,
    get_glossary_term,
    search_document,
)
from app.indexing.llm_client import llm_client

logger = logging.getLogger(__name__)


class ContextAgent:
    """
    Agentic Context Builder.
    Associates extracted claims with local node contents, section hierarchies, visual captions,
    domain glossary definitions, and Qdrant semantic search hits to evaluate claim context sufficiency.
    """

    @classmethod
    async def build_context_for_claim(
        cls,
        workspace_id: str,
        index: DocumentSkeletonIndex,
        claim: ExtractedClaim,
    ) -> ClaimContextAssociation:
        """
        Assemble comprehensive context for a single claim and evaluate sufficiency.
        """
        # 1. Direct local node IDs
        local_node_ids = list(claim.source_node_ids)

        # 2. Gather surrounding section text
        section_nodes = []
        if local_node_ids:
            first_node = next((n for n in index.nodes if n.node_id == local_node_ids[0]), None)
            if first_node and first_node.parent_section_id:
                section_nodes = get_section(index, first_node.parent_section_id)

        # 3. Perform Qdrant Cloud semantic vector search
        search_hits = await search_document(
            workspace_id=workspace_id,
            document_id=index.document_id,
            query=claim.statement,
            limit=3,
        )

        reference_node_ids = [
            hit["node_id"] for hit in search_hits
            if hit.get("node_id") and hit["node_id"] not in local_node_ids
        ]

        # 4. Gather visual IDs and glossary terms
        visual_ids = [v.visual_id for v in index.visuals if v.node_id in local_node_ids or (first_node and v.page_no in first_node.page_range)]
        matched_glossary_terms = []

        for g in index.glossary:
            if g.term.lower() in claim.statement.lower():
                matched_glossary_terms.append(g.term)

        # 5. Formulate LLM sufficiency prompt
        context_payload = {
            "claim": claim.statement,
            "claim_type": claim.claim_type.value,
            "local_text": [n.text for n in index.nodes if n.node_id in local_node_ids],
            "section_context": [n.text for n in section_nodes[:3]],
            "semantic_search_hits": [h["text"] for h in search_hits],
            "glossary_definitions": [f"{g.term}: {g.definition}" for g in index.glossary if g.term in matched_glossary_terms],
        }

        prompt = f"""You are an enterprise claim context evaluation agent.
Examine the following extracted claim and its associated document context payloads:

Context Payload:
{context_payload}

Determine:
1. Is the retrieved document context sufficient to verify or evaluate this claim? (true / false)
2. Provide a clear, detailed rationale explaining why the context is sufficient, or identifying what specific evidence/data is missing.

Return a JSON object:
{{
  "sufficiency_flag": true,
  "sufficiency_rationale": "Direct evidence provided in node n_001 with exact percentages and figures."
}}
"""

        sufficiency_flag = True
        sufficiency_rationale = "Sufficient local context and evidence retrieved."

        try:
            res = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You evaluate claim context sufficiency and output structured JSON.",
            )
            sufficiency_flag = bool(res.get("sufficiency_flag", True))
            sufficiency_rationale = str(res.get("sufficiency_rationale", sufficiency_rationale)).strip()
        except Exception as e:
            logger.warning(f"Sufficiency LLM evaluation fallback for claim {claim.claim_id}: {e}")

        return ClaimContextAssociation(
            claim_id=claim.claim_id,
            statement=claim.statement,
            local_node_ids=local_node_ids,
            reference_node_ids=reference_node_ids,
            visual_ids=visual_ids,
            glossary_terms=matched_glossary_terms,
            sufficiency_flag=sufficiency_flag,
            sufficiency_rationale=sufficiency_rationale,
        )

    @classmethod
    async def build_document_context(
        cls,
        workspace_id: str,
        index: DocumentSkeletonIndex,
        claims: List[ExtractedClaim],
    ) -> ContextBuildingResult:
        """
        Build context associations for all extracted claims across a document.
        """
        logger.info(f"Building context associations for {len(claims)} claims in document {index.document_id}...")
        associations: List[ClaimContextAssociation] = []

        for claim in claims:
            assoc = await cls.build_context_for_claim(
                workspace_id=workspace_id,
                index=index,
                claim=claim,
            )
            associations.append(assoc)

        return ContextBuildingResult(
            document_id=str(index.document_id),
            workspace_id=str(workspace_id),
            total_claims=len(associations),
            associations=associations,
        )


context_agent = ContextAgent()
