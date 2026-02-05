"""drop bbox column

Revision ID: 347246dd0b74
Revises: cb9d4e9e1af5
Create Date: 2025-11-19 23:16:41.715812

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '347246dd0b74'
down_revision: Union[str, Sequence[str], None] = 'cb9d4e9e1af5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
