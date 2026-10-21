"""Add task creation timestamp.

Revision ID: 002_add_created_at
Revises: 001_tasks_baseline
"""

from alembic import op
import sqlalchemy as sa


revision = "002_add_created_at"
down_revision = "001_tasks_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing rows receive the migration transaction's timestamp; their true
    # historical creation times are unavailable.
    op.add_column(
        "tasks",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )


def downgrade() -> None:
    # Dropping this column permanently loses all stored creation timestamps.
    op.drop_column("tasks", "created_at")
