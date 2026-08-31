import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.db.models import User, Workspace, WorkspaceMember, WorkspaceRole
from app.schemas.workspaces import WorkspaceCreate, WorkspaceResponse
from app.api.auth import get_current_user, generate_slug
from app.auth.dependencies import verify_workspace_access

router = APIRouter(prefix="/api/workspaces", tags=["Workspaces"])


@router.post("/", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Create a new workspace and assign the current authenticated user as owner.
    """
    slug = payload.slug if payload.slug else generate_slug(payload.name)

    # Check slug uniqueness
    existing_slug = await session.execute(
        select(Workspace).where(Workspace.slug == slug)
    )
    if existing_slug.scalar_one_or_none():
        slug = generate_slug(payload.name)

    workspace = Workspace(
        name=payload.name,
        slug=slug,
        created_by=current_user.id,
    )
    session.add(workspace)
    await session.flush()

    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=current_user.id,
        role=WorkspaceRole.owner,
    )
    session.add(member)

    await session.commit()
    await session.refresh(workspace)
    return workspace


@router.get("/", response_model=List[WorkspaceResponse])
async def list_workspaces(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    List all workspaces where the current user holds a membership.
    """
    query = (
        select(Workspace)
        .join(WorkspaceMember, Workspace.id == WorkspaceMember.workspace_id)
        .where(WorkspaceMember.user_id == current_user.id)
    )
    result = await session.execute(query)
    workspaces = result.scalars().all()
    return workspaces


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: uuid.UUID,
    member: WorkspaceMember = Depends(verify_workspace_access),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get details of a specific workspace if the user is an authorized member.
    """
    workspace_res = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = workspace_res.scalar_one_or_none()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    return workspace
