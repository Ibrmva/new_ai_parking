"""Change image column to LONGBLOB

Revision ID: 88adcc9e5d80
Revises: 0eacabac1d3f
Create Date: 2025-11-10 15:29:26.141234

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '88adcc9e5d80'
down_revision: Union[str, Sequence[str], None] = '0eacabac1d3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('plates', 'image', type_=sa.dialects.mysql.LONGBLOB, existing_type=sa.LargeBinary())


def downgrade() -> None:
    """Downgrade schema."""
    pass
