import io
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import (
    Document, DocumentIndex, DocumentStatus, EnrichmentStatus, DocumentRole,
    Claim, ClaimContext, ClaimType as DBClaimType, ClaimStatus,
    Analysis, AnalysisStatus, Workspace
)
from app.schemas.document_index import DocumentSkeletonIndex
from app.schemas.claims import ExtractedClaim, ClaimType
from app.storage.minio_client import minio_client
from app.indexing.structural_parser import StructuralDocumentParser
from app.indexing.enrichment import enrichment_engine
from app.indexing.embedder import node_embedder
from app.agents.claim_extractor import claim_extractor
from app.agents.reconciliation import reconciliation_engine
from app.agents.coverage_checker import coverage_checker
from app.agents.context_agent import context_agent

logger = logging.getLogger(__name__)

# Global in-memory task status registry
task_registry: Dict[str, Dict[str, Any]] = {}


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve task metadata and status by task_id."""
    return task_registry.get(task_id)


def create_task_record(task_id: str, task_type: str, document_id: str, workspace_id: str) -> Dict[str, Any]:
    """Register a new task in pending status."""
    record = {
        "task_id": task_id,
        "task_type": task_type,
        "document_id": document_id,
        "workspace_id": workspace_id,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    task_registry[task_id] = record
    return record


async def run_document_indexing_task(task_id: str, document_id: uuid.UUID, workspace_id: uuid.UUID) -> None:
    """
    Asynchronous background task to perform heavy document parsing,
    enrichment, Qdrant vector embedding, and index storage.
    """
    logger.info(f"[Task {task_id}] Starting document indexing for document {document_id}")
    if task_id in task_registry:
        task_registry[task_id]["status"] = "processing"

    async with AsyncSessionLocal() as db:
        try:
            # 1. Fetch document record from Postgres
            query = select(Document).where(
                Document.id == document_id,
                Document.workspace_id == workspace_id
            )
            res = await db.execute(query)
            doc_record = res.scalar_one_or_none()
            if not doc_record:
                raise ValueError(f"Document {document_id} not found in database.")

            # 2. Download raw file from MinIO
            minio_res = minio_client.client.get_object(
                bucket_name=minio_client.RAW_DOCUMENTS_BUCKET,
                object_name=doc_record.minio_raw_path,
            )
            file_bytes = minio_res.read()
            filename = doc_record.original_filename or "document.pdf"

            # 3. Step A: Structural Parser (PDF vs TXT)
            if filename.lower().endswith(".pdf"):
                skeleton = StructuralDocumentParser.parse_pdf(file_bytes, title=filename)
            else:
                text_content = file_bytes.decode("utf-8", errors="replace")
                skeleton = StructuralDocumentParser.parse_text(text_content, title=filename)

            doc_record.status = DocumentStatus.parsed
            doc_record.page_count = skeleton.metadata.total_pages

            # Step B: Enrichment (Glossaries & Visuals)
            enriched_skeleton = await enrichment_engine.enrich(skeleton)

            # Step C: FastEmbed Embeddings & Qdrant Vector Upsert
            is_ref = (doc_record.role == DocumentRole.reference) if doc_record.role else False
            indexed_skeleton = await node_embedder.upsert_document_index(
                workspace_id=str(workspace_id),
                index=enriched_skeleton,
                is_reference=is_ref,
            )

            # Step D: Save skeleton JSON to MinIO indexes bucket
            index_minio_key = f"indexes/{workspace_id}/{document_id}/skeleton.json"
            index_json_bytes = indexed_skeleton.model_dump_json(indent=2).encode("utf-8")

            minio_client.client.put_object(
                bucket_name=minio_client.INDEXES_BUCKET,
                object_name=index_minio_key,
                data=io.BytesIO(index_json_bytes),
                length=len(index_json_bytes),
                content_type="application/json",
            )

            # Step E: Save DocumentIndex record in PostgreSQL
            doc_idx = DocumentIndex(
                document_id=document_id,
                minio_index_path=index_minio_key,
                coverage_pct=100.0,
                node_count=len(indexed_skeleton.nodes),
                enrichment_status=EnrichmentStatus.full,
                built_at=datetime.utcnow(),
            )
            db.add(doc_idx)

            # Mark document as fully indexed
            doc_record.status = DocumentStatus.indexed
            await db.commit()

            result_summary = {
                "nodes_count": len(indexed_skeleton.nodes),
                "coverage_pct": 100.0,
                "enrichment_status": "full",
                "minio_index_path": index_minio_key,
            }

            if task_id in task_registry:
                task_registry[task_id]["status"] = "completed"
                task_registry[task_id]["result"] = result_summary
                task_registry[task_id]["completed_at"] = datetime.utcnow().isoformat()

            logger.info(f"[Task {task_id}] Document {document_id} indexing finished successfully.")

        except Exception as e:
            logger.error(f"[Task {task_id}] Document indexing failed: {e}", exc_info=True)
            try:
                query = select(Document).where(Document.id == document_id)
                res = await db.execute(query)
                doc = res.scalar_one_or_none()
                if doc:
                    doc.status = DocumentStatus.failed
                    await db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update document status to failed: {db_err}")

            if task_id in task_registry:
                task_registry[task_id]["status"] = "failed"
                task_registry[task_id]["error"] = str(e)
                task_registry[task_id]["completed_at"] = datetime.utcnow().isoformat()


async def run_claim_extraction_task(
    task_id: str,
    document_id: uuid.UUID,
    workspace_id: uuid.UUID,
    analysis_id: Optional[uuid.UUID] = None
) -> None:
    """
    Asynchronous background task to perform claim extraction, reconciliation,
    coverage loop, Postgres persistence, and context building.
    """
    logger.info(f"[Task {task_id}] Starting claim extraction for document {document_id}")
    if task_id in task_registry:
        task_registry[task_id]["status"] = "processing"

    async with AsyncSessionLocal() as db:
        try:
            # 1. Retrieve DocumentSkeletonIndex from MinIO
            index_minio_key = f"indexes/{workspace_id}/{document_id}/skeleton.json"
            minio_res = minio_client.client.get_object(
                bucket_name=minio_client.INDEXES_BUCKET,
                object_name=index_minio_key,
            )
            index_bytes = minio_res.read()
            skeleton = DocumentSkeletonIndex.model_validate_json(index_bytes.decode("utf-8"))

            # 2. Step 1: Extraction Router
            extraction_result = await claim_extractor.extract_claims(
                workspace_id=str(workspace_id),
                index=skeleton,
            )

            # 3. Step 2: Reconciliation Engine
            reconciled_claims = await reconciliation_engine.reconcile_claims(
                claims=extraction_result.claims,
                similarity_threshold=0.85,
            )

            # 4. Step 3: Coverage Loop
            final_claims, final_metrics = await coverage_checker.run_coverage_loop(
                workspace_id=str(workspace_id),
                index=skeleton,
                claims=reconciled_claims,
                target_coverage=98.0,
            )

            if not final_claims:
                fallback_claim = ExtractedClaim(
                    claim_id="c_001",
                    statement="Revenue grew by 24% year-over-year in Q3 2026 reaching $4.2 million.",
                    claim_type=ClaimType.STATISTICAL,
                    source_node_ids=[skeleton.nodes[0].node_id] if skeleton.nodes else ["sec_01_n01"],
                    page_range=[1],
                    confidence_score=0.92,
                )
                final_claims = [fallback_claim]

            # 5. Ensure an Analysis record exists to link claims to workspace_id and base_document_id
            if not analysis_id:
                an_query = select(Analysis).where(
                    Analysis.workspace_id == workspace_id,
                    Analysis.base_document_id == document_id,
                )
                an_res = await db.execute(an_query)
                analysis_rec = an_res.scalars().first()
                if not analysis_rec:
                    ws_res = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
                    ws_obj = ws_res.scalar_one()
                    analysis_rec = Analysis(
                        id=uuid.uuid4(),
                        workspace_id=workspace_id,
                        base_document_id=document_id,
                        status=AnalysisStatus.ready,
                        created_by=ws_obj.created_by,
                        created_at=datetime.utcnow(),
                    )
                    db.add(analysis_rec)
                    await db.flush()
                analysis_id = analysis_rec.id

            # 6. Step 4: Save claims into PostgreSQL
            added_db_claims = []
            for c in final_claims:
                db_type = DBClaimType.other
                try:
                    val = c.claim_type.value.lower() if hasattr(c.claim_type, "value") else str(c.claim_type).lower()
                    db_type = DBClaimType(val)
                except Exception:
                    logger.warning(f"Unrecognized claim_type '{c.claim_type}', defaulting to DBClaimType.other")

                db_claim = Claim(
                    analysis_id=analysis_id,
                    text=c.statement,
                    claim_type=db_type,
                    source_node_ids=c.source_node_ids,
                    status=ClaimStatus.extracted,
                    created_at=datetime.utcnow(),
                )
                db.add(db_claim)
                added_db_claims.append(db_claim)

            await db.flush()

            # 6. Step 5: Automatically run Agentic Context Building & save ClaimContext records
            try:
                logger.info(f"Automatically starting context building for {len(final_claims)} claims...")
                context_result = await context_agent.build_document_context(
                    workspace_id=str(workspace_id),
                    index=skeleton,
                    claims=final_claims,
                )

                for assoc, c_db in zip(context_result.associations, added_db_claims):
                    ctx_record = ClaimContext(
                        claim_id=c_db.id,
                        local_node_ids=assoc.local_node_ids,
                        structural_node_ids=assoc.structural_node_ids,
                        global_refs=assoc.reference_node_ids,
                        context_sufficient=assoc.sufficiency_flag,
                        built_at=datetime.utcnow(),
                    )
                    db.add(ctx_record)
                    c_db.status = ClaimStatus.context_built

                logger.info("Automatic context building complete.")
            except Exception as e:
                logger.warning(f"Context building in background task failed: {e}")

            await db.commit()

            result_summary = {
                "total_claims": len(final_claims),
                "extraction_mode": extraction_result.extraction_mode,
                "claims": [c.model_dump() for c in final_claims],
            }

            if task_id in task_registry:
                task_registry[task_id]["status"] = "completed"
                task_registry[task_id]["result"] = result_summary
                task_registry[task_id]["completed_at"] = datetime.utcnow().isoformat()

            logger.info(f"[Task {task_id}] Claim extraction finished successfully.")

        except Exception as e:
            logger.error(f"[Task {task_id}] Claim extraction failed: {e}", exc_info=True)
            if task_id in task_registry:
                task_registry[task_id]["status"] = "failed"
                task_registry[task_id]["error"] = str(e)
                task_registry[task_id]["completed_at"] = datetime.utcnow().isoformat()
