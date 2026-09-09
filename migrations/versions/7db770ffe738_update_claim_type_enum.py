"""Update claim_type enum values

Revision ID: 7db770ffe738
Revises: 6ca660ffd627
Create Date: 2026-09-09 10:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7db770ffe738'
down_revision: Union[str, Sequence[str], None] = '6ca660ffd627'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_CLAIM_TYPES = (
    'safety', 'efficacy', 'statistical', 'promotional', 'financial',
    'compliance', 'technical', 'legal', 'performance', 'quality',
    'comparative', 'statistic', 'causal', 'comparison', 'quote',
    'definition', 'prediction', 'factual', 'other'
)

OLD_CLAIM_TYPES = (
    'statistic', 'causal', 'comparison', 'quote', 'definition', 'prediction', 'other'
)


def upgrade() -> None:
    op.execute("ALTER TABLE claims DROP CONSTRAINT IF EXISTS ck_claims_claim_type")
    op.execute("ALTER TABLE claims DROP CONSTRAINT IF EXISTS claims_claim_type_check")

    values_sql = ", ".join(f"'{val}'" for val in NEW_CLAIM_TYPES)
    op.execute(f"ALTER TABLE claims ADD CONSTRAINT ck_claims_claim_type CHECK (claim_type IN ({values_sql}))")


def downgrade() -> None:
    op.execute("ALTER TABLE claims DROP CONSTRAINT IF EXISTS ck_claims_claim_type")
    op.execute("ALTER TABLE claims DROP CONSTRAINT IF EXISTS claims_claim_type_check")

    values_sql = ", ".join(f"'{val}'" for val in OLD_CLAIM_TYPES)
    op.execute(f"ALTER TABLE claims ADD CONSTRAINT ck_claims_claim_type CHECK (claim_type IN ({values_sql}))")
