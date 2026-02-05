"""add image column to plates table

Revision ID: 0eacabac1d3f
Revises: 18033a11c714
Create Date: 2025-11-07 10:05:47.437403

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0eacabac1d3f'
down_revision: Union[str, Sequence[str], None] = '18033a11c714'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('plates', sa.Column('image', sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    pass
