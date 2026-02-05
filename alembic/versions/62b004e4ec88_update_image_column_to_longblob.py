"""Update image column to LONGBLOB

Revision ID: 62b004e4ec88
Revises: 88adcc9e5d80
Create Date: 2025-11-10 15:34:02.956893

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '62b004e4ec88'
down_revision: Union[str, Sequence[str], None] = '88adcc9e5d80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
