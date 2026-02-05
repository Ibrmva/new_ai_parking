"""add image_id column to plates table

Revision ID: 5fd2b2dc80aa
Revises: 347246dd0b74
Create Date: 2025-12-23 08:42:40.250589

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5fd2b2dc80aa'
down_revision: Union[str, Sequence[str], None] = '347246dd0b74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('plates', sa.Column('image_id', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    pass
