import uuid
from typing import List, Callable
from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.db.models import User, WorkspaceMember, WorkspaceRole
from app.api.auth import get_current_user


async def verify_workspace_access(
    workspace_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> WorkspaceMember:
    """
    Dependency guard that verifies if current_user is an active member of workspace_id.
    Raises 403 Forbidden if user is not a member of the workspace.
    """
    result = await session.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == current_user.id,
        )
    )
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this workspace",
        )

    return member


def require_workspace_role(allowed_roles: List[WorkspaceRole]) -> Callable:
    """
    Factory dependency ensuring member holds a role permitted by allowed_roles.
    """
    async def role_checker(
        member: WorkspaceMember = Depends(verify_workspace_access),
    ) -> WorkspaceMember:
        if member.role not in allowed_roles:
            role_names = [r.value for r in allowed_roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: Required role in {role_names}",
            )
        return member

    return role_checker
