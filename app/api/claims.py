import json
import uuid
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_async_session
from app.db.models import Document, Claim, ClaimContext, WorkspaceMember, ClaimType as DBClaimType, ClaimStatus, Analysis
from app.schemas.claims import ClaimExtractionResult, ExtractedClaim, ClaimType
from app.schemas.workspaces import WorkspaceRole
from app.schemas.document_index import DocumentSkeletonIndex
from app.auth.dependencies import verify_workspace_access, require_workspace_role
from app.storage.minio_client import minio_client
from app.agents.claim_extractor import claim_extractor
from app.agents.reconciliation import reconciliation_engine
from app.agents.coverage_checker import coverage_checker
from app.schemas.tasks import TaskResponse
from workers.tasks import create_task_record, run_claim_extraction_task, get_task_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/documents/{document_id}/claims", tags=["Claims"])


@router.post("/extract", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def extract_document_claims(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    analysis_id: Optional[uuid.UUID] = None,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN, WorkspaceRole.EDITOR])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Trigger the Phase 4 Claim Extraction & Automatic Phase 5 Context Building Pipeline asynchronously:
    1. Validate document index existence in MinIO
    2. Register and dispatch background extraction worker task
    3. Return task status (202 Accepted) immediately
    """
    index_minio_key = f"indexes/{workspace_id}/{document_id}/skeleton.json"

    # Verify DocumentSkeletonIndex exists in MinIO
    try:
        minio_client.client.stat_object(
            bucket_name=minio_client.INDEXES_BUCKET,
            object_name=index_minio_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Document index skeleton not found. Please trigger document index endpoint first.",
        )

    task_id = str(uuid.uuid4())
    task_record = create_task_record(task_id, "claim_extraction", str(document_id), str(workspace_id))
    background_tasks.add_task(run_claim_extraction_task, task_id, document_id, workspace_id, analysis_id)

    logger.info(f"Dispatched background claim extraction task {task_id} for document {document_id}.")
    return task_record


@router.get("/", response_model=List[ExtractedClaim])
async def list_document_claims(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all claims extracted for a specific document in a workspace.
    """
    query = (
        select(Claim)
        .join(Analysis, Claim.analysis_id == Analysis.id)
        .where(
            Analysis.workspace_id == workspace_id,
            Analysis.base_document_id == document_id,
        )
    )
    res = await db.execute(query)
    db_claims = res.scalars().all()

    extracted_list = []
    for idx, c in enumerate(db_claims, start=1):
        claim_type_enum = ClaimType.OTHER
        try:
            claim_type_enum = ClaimType(c.claim_type.value.lower())
        except Exception:
            pass

        extracted_list.append(
            ExtractedClaim(
                claim_id=f"c_{idx:03d}",
                statement=c.text,
                claim_type=claim_type_enum,
                source_node_ids=c.source_node_ids or [],
                page_range=[1],
                confidence_score=0.85,
            )
        )
    return extracted_list
