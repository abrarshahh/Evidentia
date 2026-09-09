"""Make claim analysis_id nullable

Revision ID: 8ec349a1f290
Revises: 7db770ffe738
Create Date: 2026-09-09 12:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8ec349a1f290'
down_revision: Union[str, Sequence[str], None] = '7db770ffe738'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('claims', 'analysis_id',
               existing_type=sa.UUID(),
               nullable=True)


def downgrade() -> None:
    op.alter_column('claims', 'analysis_id',
               existing_type=sa.UUID(),
               nullable=False)
