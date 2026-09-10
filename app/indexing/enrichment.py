import logging
from typing import List, Optional
from app.schemas.document_index import DocumentSkeletonIndex, GlossaryEntry, VisualElement
from app.indexing.llm_client import llm_client

logger = logging.getLogger(__name__)


class EnrichmentEngine:
    """
    Enrichment Engine that enhances deterministic document skeleton indexes with
    LLM-extracted vocabulary glossaries and generated table/figure captions.
    """

    @classmethod
    async def enrich_glossary(cls, index: DocumentSkeletonIndex) -> List[GlossaryEntry]:
        """
        Identify technical, domain-specific terms and acronyms using Gemini.
        """
        if not llm_client.is_available() or not index.nodes:
            return index.glossary

        existing_terms = {g.term.lower() for g in index.glossary}
        nodes_sample = [
            {"node_id": n.node_id, "text": n.text}
            for n in index.nodes[:15]
            if n.text and len(n.text.strip()) > 10
        ]

        if not nodes_sample:
            return index.glossary

        prompt = f"""Identify domain-specific terminology, acronyms, financial metrics, or technical jargon from these document text nodes.
Nodes:
{nodes_sample}

Return a JSON object:
{{
  "glossary": [
    {{
      "term": "Term or Acronym",
      "definition": "Clear concise definition derived from context.",
      "defined_at_node": "n_001"
    }}
  ]
}}
"""
        try:
            res = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You extract domain terminology and definitions into a structured glossary JSON.",
            )
            raw_entries = res.get("glossary", [])
            for item in raw_entries:
                t = item.get("term", "").strip()
                d = item.get("definition", "").strip()
                n = item.get("defined_at_node", index.nodes[0].node_id)
                if t and d and t.lower() not in existing_terms:
                    existing_terms.add(t.lower())
                    index.glossary.append(
                        GlossaryEntry(
                            term=t,
                            definition=d,
                            defined_at_node=n,
                        )
                    )
        except Exception as e:
            logger.warning(f"Glossary enrichment skipped due to LLM error/fallback: {e}")

        return index.glossary

    @classmethod
    async def enrich_visual_captions(cls, index: DocumentSkeletonIndex) -> List[VisualElement]:
        """
        Generate concise descriptive captions for extracted visual elements (tables, figures)
        using Gemini LLM multimodal visual context analysis.
        """
        if not index.visuals:
            return index.visuals

        # First populate original_caption for any visual missing it
        for visual in index.visuals:
            if not visual.original_caption:
                visual.original_caption = visual.caption or f"{visual.visual_type.capitalize()} on page {visual.page_no}"

        if not llm_client.is_available():
            for visual in index.visuals:
                visual.caption = visual.generated_caption or visual.original_caption
            return index.visuals

        for visual in index.visuals:
            # Gather surrounding page text nodes for context
            page_nodes = [
                n.text for n in index.nodes
                if visual.page_no in n.page_range and n.text
            ][:5]
            context_text = "\n".join(page_nodes) if page_nodes else "No text context available."

            prompt = f"""Generate a detailed, authoritative caption for this document visual element:
Visual ID: {visual.visual_id}
Type: {visual.visual_type}
Page: {visual.page_no}
Original Caption: {visual.original_caption}
Bounding Box: {visual.bbox}
Surrounding Page Context:
{context_text[:1000]}

Return a JSON object:
{{
  "generated_caption": "Clear analytical summary of what this {visual.visual_type} depicts."
}}
"""
            try:
                res = await llm_client.generate_json(
                    prompt=prompt,
                    system_instruction="You generate accurate analytical captions for document figures and tables.",
                )
                gen_cap = res.get("generated_caption", "").strip()
                if gen_cap:
                    visual.generated_caption = gen_cap
                    visual.caption = gen_cap
                else:
                    visual.caption = visual.original_caption
            except Exception as e:
                logger.warning(f"Visual caption generation skipped for {visual.visual_id}: {e}")
                visual.caption = visual.original_caption

        return index.visuals

    @classmethod
    async def enrich(cls, index: DocumentSkeletonIndex) -> DocumentSkeletonIndex:
        """
        Execute full enrichment pass over document index.
        """
        await cls.enrich_glossary(index)
        await cls.enrich_visual_captions(index)
        return index


enrichment_engine = EnrichmentEngine()
