"""Alembic environment using the same Psycopg connection settings as the app."""

import os

from alembic import context
import psycopg
from sqlalchemy import create_engine, pool


def run_migrations_offline() -> None:
    context.configure(
        url="postgresql+psycopg://",
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Psycopg reads DATABASE_URL here or standard PG* variables via libpq.
    engine = create_engine(
        "postgresql+psycopg://",
        creator=lambda: psycopg.connect(os.environ.get("DATABASE_URL", "")),
        poolclass=pool.NullPool,
    )
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=None)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
