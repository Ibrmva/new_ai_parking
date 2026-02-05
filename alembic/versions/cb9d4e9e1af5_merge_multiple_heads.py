"""merge multiple heads

Revision ID: cb9d4e9e1af5
Revises: 77f32ff21128, a9398292fa7f
Create Date: 2025-11-19 23:16:37.687866

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cb9d4e9e1af5'
down_revision: Union[str, Sequence[str], None] = ('77f32ff21128', 'a9398292fa7f')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
