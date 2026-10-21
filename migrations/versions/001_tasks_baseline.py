"""Create the initial tasks table for a new database.

Revision ID: 001_tasks_baseline
Revises:
"""

from alembic import context, op
import sqlalchemy as sa


revision = "001_tasks_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not context.is_offline_mode() and sa.inspect(op.get_bind()).has_table("tasks"):
        raise RuntimeError(
            "tasks already exists; verify its schema and stamp this baseline instead"
        )
    op.create_table(
        "tasks",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    raise RuntimeError("Baseline downgrade is disabled to protect task records")
