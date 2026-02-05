"""change bbox column to string

Revision ID: a9398292fa7f
Revises: 18033a11c714
Create Date: 2024-11-07 10:24:20.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
import json


# revision identifiers, used by Alembic.
revision = 'a9398292fa7f'
down_revision = '18033a11c714'
branch_labels = None
dependencies = None


def upgrade():
    # Convert existing JSON data to strings
    connection = op.get_bind()
    result = connection.execute(sa.text("SELECT id, bbox FROM plates WHERE bbox IS NOT NULL"))
    for row in result:
        plate_id, bbox_json = row
        if isinstance(bbox_json, str):
            # Already a string, skip
            continue
        # Convert JSON to string
        bbox_str = json.dumps(bbox_json)
        connection.execute(
            sa.text("UPDATE plates SET bbox = :bbox WHERE id = :id"),
            {"bbox": bbox_str, "id": plate_id}
        )

    # Change column type to String(500)
    op.alter_column('plates', 'bbox',
                    existing_type=mysql.JSON(),
                    type_=sa.String(500),
                    existing_nullable=True)


def downgrade():
    # Change column type back to JSON
    op.alter_column('plates', 'bbox',
                    existing_type=sa.String(500),
                    type_=mysql.JSON(),
                    existing_nullable=True)

    # Convert strings back to JSON
    connection = op.get_bind()
    result = connection.execute(sa.text("SELECT id, bbox FROM plates WHERE bbox IS NOT NULL"))
    for row in result:
        plate_id, bbox_str = row
        if not isinstance(bbox_str, str):
            continue
        # Convert string to JSON
        try:
            bbox_json = json.loads(bbox_str)
        except json.JSONDecodeError:
            bbox_json = None
        connection.execute(
            sa.text("UPDATE plates SET bbox = :bbox WHERE id = :id"),
            {"bbox": bbox_json, "id": plate_id}
        )
