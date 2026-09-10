import io
import re
import uuid
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import pdfplumber

from app.schemas.document_index import (
    DocumentSkeletonIndex,
    DocumentMetadata,
    StructureSection,
    DocumentNode,
    VisualElement,
    GlossaryEntry,
    CoverageMetrics,
    PageView,
    BoundingBox,
)


class StructuralDocumentParser:
    """
    Deterministic Document Skeleton Parser engine.
    Extracts text, headings, lists, tables, cross-references, glossary terms, and coordinates
    from raw PDF and text documents without LLM latency.
    """

    @staticmethod
    def compute_checksum(content_bytes: bytes) -> str:
        """
        Compute SHA-256 checksum prefixed with 'sha256:'.
        """
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        return f"sha256:{sha256_hash}"

    @classmethod
    def parse_pdf(
        cls,
        content_bytes: bytes,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> DocumentSkeletonIndex:
        """
        Parse raw PDF bytes into a deterministic DocumentSkeletonIndex.
        """
        doc_id = document_id or str(uuid.uuid4())
        checksum = cls.compute_checksum(content_bytes)

        nodes: List[DocumentNode] = []
        structure_tree: List[StructureSection] = []
        visuals: List[VisualElement] = []
        glossary: List[GlossaryEntry] = []
        page_views: List[PageView] = []

        current_char_offset = 0
        current_section: Optional[StructureSection] = None
        node_counter = 1
        visual_counter = 1
        section_counter = 1

        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            total_pages = len(pdf.pages)
            doc_title = title or (pdf.metadata.get("Title") if pdf.metadata else None) or "Untitled Document"

            for page_idx, page in enumerate(pdf.pages, start=1):
                page_nodes: List[str] = []
                page_visuals: List[str] = []

                # 1. Extract tables and their bounding boxes
                found_tables = page.find_tables()
                table_bbox_list = [t.bbox for t in found_tables]

                for table_obj in found_tables:
                    table_data = table_obj.extract()
                    if not table_data or len(table_data) == 0:
                        continue

                    visual_id = f"visual_p{page_idx}_{visual_counter}"
                    node_id = f"n_{node_counter:03d}"
                    visual_counter += 1
                    node_counter += 1

                    table_text = "\n".join([" | ".join([str(cell or "") for cell in row]) for row in table_data])
                    start_offset = current_char_offset
                    end_offset = current_char_offset + len(table_text)
                    current_char_offset = end_offset + 1

                    x0, top, x1, bottom = table_obj.bbox
                    tbl_bbox = BoundingBox(
                        x=round(float(x0), 2),
                        y=round(float(top), 2),
                        w=round(float(x1 - x0), 2),
                        h=round(float(bottom - top), 2),
                    )
                    orig_caption = f"Table on page {page_idx}"

                    visual_elem = VisualElement(
                        visual_id=visual_id,
                        node_id=node_id,
                        visual_type="table",
                        page_no=page_idx,
                        bbox=tbl_bbox,
                        original_caption=orig_caption,
                        caption=orig_caption,
                    )
                    visuals.append(visual_elem)
                    page_visuals.append(visual_id)

                    table_node = DocumentNode(
                        node_id=node_id,
                        node_type="table",
                        parent_section_id=current_section.section_id if current_section else None,
                        page_range=[page_idx],
                        char_offsets=[start_offset, end_offset],
                        text=table_text,
                        cross_refs=[visual_id],
                    )
                    nodes.append(table_node)
                    page_nodes.append(node_id)

                    if current_section:
                        current_section.node_ids.append(node_id)

                # 2. Extract figures/images and their bounding boxes
                page_images = getattr(page, "images", []) or []
                for img in page_images:
                    x0 = float(img.get("x0", 0))
                    top = float(img.get("top", 0))
                    x1 = float(img.get("x1", x0 + img.get("width", 0)))
                    bottom = float(img.get("bottom", top + img.get("height", 0)))
                    w = max(1.0, float(x1 - x0))
                    h = max(1.0, float(bottom - top))
                    if w < 20 or h < 20:
                        continue

                    visual_id = f"visual_p{page_idx}_{visual_counter}"
                    node_id = f"n_{node_counter:03d}"
                    visual_counter += 1
                    node_counter += 1

                    img_bbox = BoundingBox(
                        x=round(x0, 2),
                        y=round(top, 2),
                        w=round(w, 2),
                        h=round(h, 2),
                    )
                    orig_caption = f"Figure on page {page_idx} ({int(w)}x{int(h)})"

                    fig_visual = VisualElement(
                        visual_id=visual_id,
                        node_id=node_id,
                        visual_type="figure",
                        page_no=page_idx,
                        bbox=img_bbox,
                        original_caption=orig_caption,
                        caption=orig_caption,
                    )
                    visuals.append(fig_visual)
                    page_visuals.append(visual_id)

                    fig_text = f"[Figure {visual_id}: Page {page_idx} Bounding Box ({int(x0)}, {int(top)}, {int(w)}, {int(h)})]"
                    start_offset = current_char_offset
                    end_offset = current_char_offset + len(fig_text)
                    current_char_offset = end_offset + 1

                    fig_node = DocumentNode(
                        node_id=node_id,
                        node_type="figure",
                        parent_section_id=current_section.section_id if current_section else None,
                        page_range=[page_idx],
                        char_offsets=[start_offset, end_offset],
                        text=fig_text,
                        cross_refs=[visual_id],
                    )
                    nodes.append(fig_node)
                    page_nodes.append(node_id)

                    if current_section:
                        current_section.node_ids.append(node_id)

                # 3. Extract non-table text lines from page
                # Crop non-table regions or filter text lines outside table bboxes
                raw_text = page.extract_text(layout=False) or ""
                lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

                # Filter out lines that match raw table rows if tables exist
                if table_bbox_list:
                    table_texts = set()
                    for t in found_tables:
                        for row in t.extract():
                            for cell in row:
                                if cell and str(cell).strip():
                                    table_texts.add(str(cell).strip())

                    filtered_lines = []
                    for line in lines:
                        # Skip line if it consists purely of table cell values
                        words_in_line = set(line.split())
                        if words_in_line and words_in_line.issubset(table_texts):
                            continue
                        filtered_lines.append(line)
                    lines = filtered_lines

                for line in lines:
                    start_offset = current_char_offset
                    end_offset = current_char_offset + len(line)
                    current_char_offset = end_offset + 1

                    # Heading pattern check (e.g. "1.0 Introduction", "2.1 Risk Factors", "Section 3")
                    heading_match = re.match(
                        r"^(?P<number>(?:\d+\.){1,3}\d*|\d+\.\d+)\s+(?P<name>[A-Z][A-Za-z0-9\s\-_:\(\)]+)$",
                        line,
                    )
                    is_heading_styled = (
                        line.isupper()
                        and 3 < len(line) < 60
                        and not line.endswith(".")
                        and not line.startswith("-")
                        and not any(c in line for c in ["$", "%", "+", "|", "="])
                    )

                    if heading_match or is_heading_styled:
                        sec_num = heading_match.group("number") if heading_match else None
                        sec_name = heading_match.group("name") if heading_match else line
                        sec_id = f"s_{sec_num}" if sec_num else f"s_{section_counter}"
                        section_counter += 1

                        current_section = StructureSection(
                            section_id=sec_id,
                            number=sec_num,
                            name=sec_name,
                            node_ids=[],
                        )
                        structure_tree.append(current_section)

                        node_id = f"n_{node_counter:03d}"
                        node_counter += 1

                        h_node = DocumentNode(
                            node_id=node_id,
                            node_type="heading",
                            parent_section_id=current_section.section_id,
                            page_range=[page_idx],
                            char_offsets=[start_offset, end_offset],
                            text=line,
                        )
                        nodes.append(h_node)
                        page_nodes.append(node_id)
                        current_section.node_ids.append(node_id)
                        continue

                    # List item check (e.g. "- item", "* item", "1. item", "a) item")
                    is_list_item = bool(re.match(r"^([\-\*\•]|(?:\d+|[a-z])[\.\)])\s+", line))
                    node_type = "list_item" if is_list_item else "paragraph"

                    node_id = f"n_{node_counter:03d}"
                    node_counter += 1

                    node = DocumentNode(
                        node_id=node_id,
                        node_type=node_type,
                        parent_section_id=current_section.section_id if current_section else None,
                        page_range=[page_idx],
                        char_offsets=[start_offset, end_offset],
                        text=line,
                    )
                    nodes.append(node)
                    page_nodes.append(node_id)

                    if current_section:
                        current_section.node_ids.append(node_id)

                    # Glossary term check (e.g. "Term: Definition", "Term means Definition")
                    glossary_match = re.match(
                        r"^(?P<term>[A-Z][A-Za-z0-9\s]{2,30})\s*(?::|means|is defined as)\s*(?P<def>.+)$",
                        line,
                        re.IGNORECASE,
                    )
                    if glossary_match:
                        glossary.append(
                            GlossaryEntry(
                                term=glossary_match.group("term").strip(),
                                definition=glossary_match.group("def").strip(),
                                defined_at_node=node_id,
                            )
                        )

                page_views.append(
                    PageView(
                        page_no=page_idx,
                        node_ids=page_nodes,
                        visual_ids=page_visuals,
                    )
                )

        total_nodes = len(nodes)
        covered_nodes = total_nodes
        coverage_pct = round((covered_nodes / total_nodes * 100.0) if total_nodes > 0 else 100.0, 1)

        return DocumentSkeletonIndex(
            document_id=doc_id,
            checksum=checksum,
            metadata=DocumentMetadata(
                title=doc_title,
                doc_type="pdf",
                total_pages=total_pages,
                language="en",
            ),
            structure_tree=structure_tree,
            nodes=nodes,
            visuals=visuals,
            glossary=glossary,
            coverage=CoverageMetrics(
                total_nodes=total_nodes,
                covered_nodes=covered_nodes,
                coverage_pct=coverage_pct,
            ),
            page_view=page_views,
        )

    @classmethod
    def parse_text(
        cls,
        text_content: str,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> DocumentSkeletonIndex:
        """
        Parse raw text or Markdown string into a deterministic DocumentSkeletonIndex.
        """
        content_bytes = text_content.encode("utf-8")
        doc_id = document_id or str(uuid.uuid4())
        checksum = cls.compute_checksum(content_bytes)

        nodes: List[DocumentNode] = []
        structure_tree: List[StructureSection] = []
        glossary: List[GlossaryEntry] = []

        current_char_offset = 0
        current_section: Optional[StructureSection] = None
        node_counter = 1
        section_counter = 1

        lines = [line.strip() for line in text_content.split("\n") if line.strip()]

        for line in lines:
            start_offset = current_char_offset
            end_offset = current_char_offset + len(line)
            current_char_offset = end_offset + 1

            heading_match = re.match(
                r"^(?:#{1,6}\s+|(?P<number>(?:\d+\.){1,3}\d*|\d+\.\d+)\s+)(?P<name>.+)$",
                line,
            )
            if heading_match or line.isupper() and len(line) < 60:
                sec_num = heading_match.group("number") if heading_match and "number" in heading_match.groupdict() else None
                sec_name = heading_match.group("name") if heading_match else line
                sec_id = f"s_{sec_num}" if sec_num else f"s_{section_counter}"
                section_counter += 1

                current_section = StructureSection(
                    section_id=sec_id,
                    number=sec_num,
                    name=sec_name,
                    node_ids=[],
                )
                structure_tree.append(current_section)

                node_id = f"n_{node_counter:03d}"
                node_counter += 1

                h_node = DocumentNode(
                    node_id=node_id,
                    node_type="heading",
                    parent_section_id=current_section.section_id,
                    page_range=[1],
                    char_offsets=[start_offset, end_offset],
                    text=line,
                )
                nodes.append(h_node)
                current_section.node_ids.append(node_id)
                continue

            is_list_item = bool(re.match(r"^([\-\*\•]|(?:\d+|[a-z])[\.\)])\s+", line))
            node_type = "list_item" if is_list_item else "paragraph"

            node_id = f"n_{node_counter:03d}"
            node_counter += 1

            node = DocumentNode(
                node_id=node_id,
                node_type=node_type,
                parent_section_id=current_section.section_id if current_section else None,
                page_range=[1],
                char_offsets=[start_offset, end_offset],
                text=line,
            )
            nodes.append(node)

            if current_section:
                current_section.node_ids.append(node_id)

            glossary_match = re.match(
                r"^(?P<term>[A-Z][A-Za-z0-9\s]{2,30})\s*(?::|means|is defined as)\s*(?P<def>.+)$",
                line,
                re.IGNORECASE,
            )
            if glossary_match:
                glossary.append(
                    GlossaryEntry(
                        term=glossary_match.group("term").strip(),
                        definition=glossary_match.group("def").strip(),
                        defined_at_node=node_id,
                    )
                )

        total_nodes = len(nodes)
        page_views = [PageView(page_no=1, node_ids=[n.node_id for n in nodes], visual_ids=[])]

        return DocumentSkeletonIndex(
            document_id=doc_id,
            checksum=checksum,
            metadata=DocumentMetadata(
                title=title or "Text Document",
                doc_type="text",
                total_pages=1,
                language="en",
            ),
            structure_tree=structure_tree,
            nodes=nodes,
            visuals=[],
            glossary=glossary,
            coverage=CoverageMetrics(
                total_nodes=total_nodes,
                covered_nodes=total_nodes,
                coverage_pct=100.0,
            ),
            page_view=page_views,
        )
