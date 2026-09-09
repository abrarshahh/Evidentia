import logging
from typing import List, Optional, Dict, Any
from app.schemas.document_index import (
    DocumentSkeletonIndex,
    DocumentNode,
    VisualElement,
    GlossaryEntry,
)
from app.indexing.embedder import node_embedder
from app.storage.qdrant_client import qdrant_client
from app.tracing.span_logger import trace_tool

logger = logging.getLogger(__name__)


@trace_tool(tool_name="get_section")
def get_section(index: DocumentSkeletonIndex, section_id: str) -> List[DocumentNode]:
    """
    Retrieve all document nodes belonging to a section hierarchy or section ID.
    """
    target_section_ids = {section_id.strip()}
    node_ids_in_section = set()

    for sec in index.structure_tree:
        if sec.section_id == section_id or sec.parent_section_id == section_id:
            target_section_ids.add(sec.section_id)
            if hasattr(sec, "node_ids") and sec.node_ids:
                node_ids_in_section.update(sec.node_ids)

    matching_nodes = [
        node for node in index.nodes
        if (node.parent_section_id in target_section_ids) or (node.node_id in node_ids_in_section)
    ]

    # Fallback to text nodes if section matches first entry
    if not matching_nodes and index.nodes:
        matching_nodes = [n for n in index.nodes if n.node_type == "paragraph" or n.node_type == "heading"]

    return matching_nodes


@trace_tool(tool_name="get_page")
def get_page(index: DocumentSkeletonIndex, page_no: int) -> List[DocumentNode]:
    """
    Retrieve all document nodes located on a specific page number.
    """
    return [
        node for node in index.nodes
        if page_no in node.page_range
    ]


@trace_tool(tool_name="get_adjacent_siblings")
def get_adjacent_siblings(index: DocumentSkeletonIndex, target_node_ids: List[str]) -> List[DocumentNode]:
    """
    Retrieve adjacent structural sibling nodes (previous/next nodes in sequence)
    for a list of target node IDs to form Tier 1 local context excerpts.
    """
    if not index.nodes or not target_node_ids:
        return []

    node_id_to_idx = {node.node_id: idx for idx, node in enumerate(index.nodes)}
    sibling_indices = set()

    for nid in target_node_ids:
        if nid in node_id_to_idx:
            idx = node_id_to_idx[nid]
            if idx > 0:
                sibling_indices.add(idx - 1)
            if idx + 1 < len(index.nodes):
                sibling_indices.add(idx + 1)

    target_set = set(target_node_ids)
    return [
        index.nodes[i] for i in sorted(sibling_indices)
        if index.nodes[i].node_id not in target_set
    ]


@trace_tool(tool_name="get_visual")
def get_visual(index: DocumentSkeletonIndex, visual_id: str) -> Optional[VisualElement]:
    """
    Retrieve a visual element (table/figure/chart caption and metadata) by visual ID.
    """
    for v in index.visuals:
        if v.visual_id == visual_id:
            return v
    return None


@trace_tool(tool_name="get_footnote")
def get_footnote(index: DocumentSkeletonIndex, footnote_id_or_text: str) -> Optional[DocumentNode]:
    """
    Retrieve a footnote or reference node definition by ID or keyword.
    """
    term_lower = footnote_id_or_text.lower()
    for n in index.nodes:
        if n.node_id == footnote_id_or_text or term_lower in n.text.lower():
            return n
    return None


@trace_tool(tool_name="get_glossary_term")
def get_glossary_term(index: DocumentSkeletonIndex, term: str) -> Optional[GlossaryEntry]:
    """
    Query domain glossary definitions from the index by term name.
    """
    term_lower = term.strip().lower()
    for g in index.glossary:
        if g.term.strip().lower() == term_lower or term_lower in g.term.strip().lower():
            return g
    return None


@trace_tool(tool_name="search_document")
async def search_document(
    workspace_id: str,
    document_id: str,
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Perform Qdrant Cloud semantic vector search over document node chunks filtered by workspace and document ID.
    """
    if not query or not query.strip():
        return []

    try:
        embeddings = node_embedder.generate_embeddings([query.strip()])
        if not embeddings:
            return []
        query_vector = embeddings[0]

        scored_points = await qdrant_client.search_chunks(
            workspace_id=workspace_id,
            query_vector=query_vector,
            document_id=document_id,
            limit=limit,
        )

        results = []
        for pt in scored_points:
            payload = pt.payload or {}
            results.append(
                {
                    "point_id": str(pt.id),
                    "score": round(pt.score, 4),
                    "node_id": payload.get("node_id"),
                    "node_type": payload.get("node_type"),
                    "page_range": payload.get("page_range"),
                    "text": payload.get("text"),
                }
            )

        return results
    except Exception as e:
        logger.error(f"Error performing semantic document search: {e}")
        return []
