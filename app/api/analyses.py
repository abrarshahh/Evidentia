import uuid
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.db.session import get_async_session
from app.db.models import (
    Analysis, AnalysisReference, Document, Claim, User, WorkspaceMember, AnalysisStatus
)
from app.schemas.analyses import AnalysisCreate, AnalysisResponse
from app.auth.dependencies import verify_workspace_access
from workers.tasks import run_claim_extraction_task, task_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workspaces", tags=["Analyses"])


@router.post("/{workspace_id}/analyses", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    workspace_id: uuid.UUID,
    analysis_in: AnalysisCreate,
    background_tasks: BackgroundTasks,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new Analysis session binding a base document and optional reference documents.
    Triggers asynchronous claim extraction in the background.
    """
    # 1. Verify base document belongs to workspace
    doc_query = select(Document).where(
        Document.id == analysis_in.base_document_id,
        Document.workspace_id == workspace_id,
    )
    doc_res = await db.execute(doc_query)
    base_doc = doc_res.scalar_one_or_none()
    if not base_doc:
        raise HTTPException(
            status_code=404,
            detail=f"Base document {analysis_in.base_document_id} not found in workspace.",
        )

    # 2. Create Analysis record
    analysis = Analysis(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        base_document_id=analysis_in.base_document_id,
        status=AnalysisStatus.pending,
        created_by=membership.user_id,
        created_at=datetime.utcnow(),
    )
    db.add(analysis)
    await db.flush()

    # 3. Create AnalysisReference bindings
    validated_ref_ids: List[uuid.UUID] = []
    for ref_id in analysis_in.reference_document_ids:
        ref_doc_res = await db.execute(
            select(Document).where(
                Document.id == ref_id,
                Document.workspace_id == workspace_id,
            )
        )
        if ref_doc_res.scalar_one_or_none():
            ref_bind = AnalysisReference(
                id=uuid.uuid4(),
                analysis_id=analysis.id,
                document_id=ref_id,
            )
            db.add(ref_bind)
            validated_ref_ids.append(ref_id)

    await db.commit()
    await db.refresh(analysis)

    # 4. Trigger background claim extraction task
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    task_registry[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "task_type": "claim_extraction",
        "document_id": str(analysis_in.base_document_id),
        "workspace_id": str(workspace_id),
        "analysis_id": str(analysis.id),
        "created_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(
        run_claim_extraction_task,
        task_id=task_id,
        document_id=analysis_in.base_document_id,
        workspace_id=workspace_id,
        analysis_id=analysis.id,
    )

    return AnalysisResponse(
        id=analysis.id,
        workspace_id=analysis.workspace_id,
        base_document_id=analysis.base_document_id,
        reference_document_ids=validated_ref_ids,
        status=analysis.status,
        created_by=analysis.created_by,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
        claims_count=0,
    )


@router.get("/{workspace_id}/analyses", response_model=List[AnalysisResponse])
async def list_workspace_analyses(
    workspace_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all analysis sessions in a workspace with claim counts and reference bindings.
    """
    query = select(Analysis).where(Analysis.workspace_id == workspace_id)
    res = await db.execute(query)
    analyses = res.scalars().all()

    response: List[AnalysisResponse] = []
    for an in analyses:
        # Get bound reference document IDs
        ref_res = await db.execute(
            select(AnalysisReference.document_id).where(AnalysisReference.analysis_id == an.id)
        )
        ref_ids = ref_res.scalars().all()

        # Get total claims count
        claims_count_res = await db.execute(
            select(func.count(Claim.id)).where(Claim.analysis_id == an.id)
        )
        claims_count = claims_count_res.scalar_one() or 0

        response.append(
            AnalysisResponse(
                id=an.id,
                workspace_id=an.workspace_id,
                base_document_id=an.base_document_id,
                reference_document_ids=list(ref_ids),
                status=an.status,
                created_by=an.created_by,
                created_at=an.created_at,
                completed_at=an.completed_at,
                claims_count=claims_count,
            )
        )

    return response


@router.get("/{workspace_id}/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    workspace_id: uuid.UUID,
    analysis_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed information about a specific analysis session.
    """
    query = select(Analysis).where(
        Analysis.id == analysis_id,
        Analysis.workspace_id == workspace_id,
    )
    res = await db.execute(query)
    an = res.scalar_one_or_none()
    if not an:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    ref_res = await db.execute(
        select(AnalysisReference.document_id).where(AnalysisReference.analysis_id == an.id)
    )
    ref_ids = ref_res.scalars().all()

    claims_count_res = await db.execute(
        select(func.count(Claim.id)).where(Claim.analysis_id == an.id)
    )
    claims_count = claims_count_res.scalar_one() or 0

    return AnalysisResponse(
        id=an.id,
        workspace_id=an.workspace_id,
        base_document_id=an.base_document_id,
        reference_document_ids=list(ref_ids),
        status=an.status,
        created_by=an.created_by,
        created_at=an.created_at,
        completed_at=an.completed_at,
        claims_count=claims_count,
    )


@router.delete("/{workspace_id}/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    workspace_id: uuid.UUID,
    analysis_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete an analysis session (cascades to claims, spans, and reference mappings).
    """
    query = select(Analysis).where(
        Analysis.id == analysis_id,
        Analysis.workspace_id == workspace_id,
    )
    res = await db.execute(query)
    an = res.scalar_one_or_none()
    if not an:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    await db.execute(delete(AnalysisReference).where(AnalysisReference.analysis_id == analysis_id))
    await db.delete(an)
    await db.commit()
    return None
