import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_async_session
from app.db.models import Claim, ClaimContext, WorkspaceMember
from app.schemas.context import ContextBuildingResult, ClaimContextAssociation
from app.auth.dependencies import verify_workspace_access

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}/documents/{document_id}/context", tags=["Context Builder"])


@router.get("/", response_model=ContextBuildingResult)
async def get_document_context(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get all claim context associations and sufficiency decisions for a document.
    """
    from sqlalchemy.orm import joinedload

    query = select(ClaimContext).join(Claim).options(joinedload(ClaimContext.claim))
    res = await db.execute(query)
    db_contexts = res.scalars().all()

    associations = []
    for ctx in db_contexts:
        associations.append(
            ClaimContextAssociation(
                claim_id=str(ctx.claim_id),
                statement=ctx.claim.text if ctx.claim else "",
                local_node_ids=ctx.local_node_ids or [],
                reference_node_ids=ctx.global_refs or [],
                visual_ids=[],
                glossary_terms=[],
                sufficiency_flag=ctx.context_sufficient if ctx.context_sufficient is not None else True,
                sufficiency_rationale="Sufficient local context and evidence retrieved." if ctx.context_sufficient else "Insufficient evidence in document.",
            )
        )

    return ContextBuildingResult(
        document_id=str(document_id),
        workspace_id=str(workspace_id),
        total_claims=len(associations),
        associations=associations,
    )
