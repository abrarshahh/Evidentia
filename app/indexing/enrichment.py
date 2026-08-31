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
        Generate concise descriptive captions for extracted visual elements (tables, figures).
        """
        if not index.visuals or not llm_client.is_available():
            return index.visuals

        for visual in index.visuals:
            if not visual.caption:
                visual.caption = f"{visual.visual_type.capitalize()} on page {visual.page_no}"

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
