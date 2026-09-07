import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.session import get_async_session
from app.db.models import (
    Workspace, WorkspaceMember, User, WorkspaceRole as DBWorkspaceRole,
    Document, AnalysisReference, ReferenceMapping, Verification
)
from app.schemas.workspaces import (
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceMemberResponse,
    WorkspaceMemberAdd,
    WorkspaceMemberRoleUpdate,
    WorkspaceRole,
)
from app.auth.dependencies import get_current_user, verify_workspace_access, require_workspace_role
from app.api.auth import generate_slug
from app.storage.minio_client import minio_client
from app.storage.qdrant_client import qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workspaces", tags=["Workspaces"])


@router.post("/", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_in: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new workspace and assign the creator as OWNER.
    """
    workspace = Workspace(
        name=workspace_in.name,
        slug=generate_slug(workspace_in.name),
        created_by=current_user.id,
    )
    db.add(workspace)
    await db.flush()

    membership = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=current_user.id,
        role=DBWorkspaceRole.owner,
    )
    db.add(membership)
    await db.commit()
    await db.refresh(workspace)
    return workspace


@router.get("/", response_model=List[WorkspaceResponse])
async def list_user_workspaces(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all workspaces where the current user is a member.
    """
    query = (
        select(Workspace)
        .join(WorkspaceMember)
        .where(WorkspaceMember.user_id == current_user.id)
    )
    result = await db.execute(query)
    workspaces = result.scalars().all()
    return workspaces


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed information about a specific workspace.
    """
    query = select(Workspace).where(Workspace.id == workspace_id)
    result = await db.execute(query)
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return workspace


@router.get("/{workspace_id}/members", response_model=List[WorkspaceMemberResponse])
async def list_workspace_members(
    workspace_id: uuid.UUID,
    membership: WorkspaceMember = Depends(verify_workspace_access),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all members in a workspace and their assigned roles.
    """
    query = select(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace_id)
    result = await db.execute(query)
    members = result.scalars().all()
    return members


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_workspace_member(
    workspace_id: uuid.UUID,
    member_in: WorkspaceMemberAdd,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Add/invite a user to the workspace with a specific role.
    Requires ADMIN or OWNER role.
    """
    user_query = select(User).where(User.email == member_in.email)
    user_res = await db.execute(user_query)
    target_user = user_res.scalar_one_or_none()
    if not target_user:
        raise HTTPException(status_code=404, detail=f"User with email '{member_in.email}' not found.")

    existing_query = select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == target_user.id,
    )
    existing_res = await db.execute(existing_query)
    if existing_res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User is already a member of this workspace.")

    new_member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=target_user.id,
        role=DBWorkspaceRole(member_in.role.value),
    )
    db.add(new_member)
    await db.commit()
    await db.refresh(new_member)
    return new_member


@router.patch("/{workspace_id}/members/{user_id}", response_model=WorkspaceMemberResponse)
async def update_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    role_update: WorkspaceMemberRoleUpdate,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a member's role in the workspace.
    Requires ADMIN or OWNER role.
    """
    query = select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    )
    res = await db.execute(query)
    target_member = res.scalar_one_or_none()
    if not target_member:
        raise HTTPException(status_code=404, detail="Member not found in workspace.")

    target_member.role = DBWorkspaceRole(role_update.role.value)
    await db.commit()
    await db.refresh(target_member)
    return target_member


@router.delete("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_workspace_member(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER, WorkspaceRole.ADMIN])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Remove a member from the workspace.
    Requires ADMIN or OWNER role.
    """
    query = select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    )
    res = await db.execute(query)
    target_member = res.scalar_one_or_none()
    if not target_member:
        raise HTTPException(status_code=404, detail="Member not found in workspace.")

    await db.delete(target_member)
    await db.commit()
    return None


@router.delete("/{workspace_id}", status_code=status.HTTP_200_OK)
async def delete_workspace(
    workspace_id: uuid.UUID,
    membership: WorkspaceMember = Depends(require_workspace_role([WorkspaceRole.OWNER])),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Permanently delete a workspace and purge all associated documents, indexes, claims,
    analyses, vector embeddings, and object storage files. Requires OWNER role.
    """
    # 1. Retrieve Workspace Record
    query = select(Workspace).where(Workspace.id == workspace_id)
    res = await db.execute(query)
    workspace = res.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    # 2. Delete Vector Embeddings from Qdrant Cloud
    try:
        await qdrant_client.delete_workspace_points(workspace_id=str(workspace_id))
    except Exception as e:
        logger.error(f"Error purging Qdrant points for workspace {workspace_id}: {e}")

    # 3. Delete Objects from MinIO Storage
    try:
        minio_client.remove_workspace_objects(workspace_id=str(workspace_id))
    except Exception as e:
        logger.error(f"Error purging MinIO objects for workspace {workspace_id}: {e}")

    # 4. Purge Dependent Records & Workspace from PostgreSQL
    try:
        # Delete AnalysisReferences, ReferenceMappings, and Verifications linked to documents in this workspace
        doc_ids_select = select(Document.id).where(Document.workspace_id == workspace_id)
        await db.execute(delete(AnalysisReference).where(AnalysisReference.document_id.in_(doc_ids_select)))
        await db.execute(delete(ReferenceMapping).where(ReferenceMapping.document_id.in_(doc_ids_select)))
        await db.execute(delete(Verification).where(Verification.document_id.in_(doc_ids_select)))

        # Delete Workspace (cascades to WorkspaceMember, Document, DocumentIndex, Analysis, Claim, ClaimContext, TraceSpan)
        await db.delete(workspace)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete workspace {workspace_id} from PostgreSQL database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace from database: {e}")

    logger.info(f"Workspace {workspace_id} and all associated data permanently deleted.")
    return {
        "detail": "Workspace and all associated data permanently deleted.",
        "workspace_id": str(workspace_id),
    }
