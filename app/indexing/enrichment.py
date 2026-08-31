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
        Identify technical, domain-specific terms and acronyms using Gemini 1.5 Flash.
        """
        if not llm_client.is_available() or not index.nodes:
            return index.glossary

        existing_terms = {g.term.lower() for g in index.glossary}

        # Build node text context sample (first 15 nodes)
        nodes_sample = [
            {"node_id": n.node_id, "text": n.text}
            for n in index.nodes[:15]
            if n.node_type in ("paragraph", "heading")
        ]

        prompt = f"""You are an enterprise document analyst. Given the following document text nodes, identify up to 5 domain-specific technical terms, financial metrics, or acronyms and their concise definitions.

Text Nodes:
{nodes_sample}

Return a JSON object matching this exact structure:
{{
  "glossary": [
    {{
      "term": "Term Name",
      "definition": "Clear concise definition based on the text context.",
      "defined_at_node": "n_001"
    }}
  ]
}}
"""

        try:
            result = await llm_client.generate_json(
                prompt=prompt,
                system_instruction="You extract glossary terms and definitions from documents as structured JSON.",
            )
            raw_glossary = result.get("glossary", [])
            for item in raw_glossary:
                term = item.get("term", "").strip()
                def_text = item.get("definition", "").strip()
                node_id = item.get("defined_at_node", index.nodes[0].node_id if index.nodes else "n_001")

                if term and term.lower() not in existing_terms:
                    existing_terms.add(term.lower())
                    index.glossary.append(
                        GlossaryEntry(
                            term=term,
                            definition=def_text,
                            defined_at_node=node_id,
                        )
                    )
        except Exception as e:
            logger.warning(f"Glossary enrichment skipped due to LLM error/fallback: {e}")

        return index.glossary

    @classmethod
    async def enrich_visual_captions(cls, index: DocumentSkeletonIndex) -> List[VisualElement]:
        """
        Generate descriptive captions for tables and visual assets using Gemini 1.5 Flash.
        """
        if not llm_client.is_available() or not index.visuals:
            return index.visuals

        for visual in index.visuals:
            if visual.generated_caption:
                continue

            structured_repr = visual.structured_data or {}
            prompt = f"""Provide a single concise, professional caption (1 sentence) summarizing the purpose and contents of this {visual.type}:

Structured Data:
{structured_repr}

Return a JSON object:
{{
  "caption": "Summary caption describing the table or visual."
}}
"""

            try:
                result = await llm_client.generate_json(
                    prompt=prompt,
                    system_instruction="You generate concise descriptive captions for tables and document visuals.",
                )
                caption = result.get("caption", f"Table on page {visual.page_range[0]}").strip()
                visual.generated_caption = caption
            except Exception as e:
                logger.warning(f"Visual caption generation skipped for {visual.visual_id}: {e}")
                if not visual.generated_caption:
                    visual.generated_caption = f"Structured {visual.type} on page {visual.page_range[0]}"

        return index.visuals

    @classmethod
    async def enrich(cls, index: DocumentSkeletonIndex) -> DocumentSkeletonIndex:
        """
        Orchestrate glossary enrichment and visual caption generation on a DocumentSkeletonIndex.
        """
        logger.info(f"Starting enrichment for document {index.document_id}...")
        await cls.enrich_glossary(index)
        await cls.enrich_visual_captions(index)
        logger.info(f"Enrichment completed for document {index.document_id}.")
        return index
