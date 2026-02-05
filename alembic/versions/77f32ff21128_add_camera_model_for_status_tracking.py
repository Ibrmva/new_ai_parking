"""Add Camera model for status tracking

Revision ID: 77f32ff21128
Revises: 62b004e4ec88
Create Date: 2025-11-10 22:18:10.367829

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '77f32ff21128'
down_revision: Union[str, Sequence[str], None] = '62b004e4ec88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
