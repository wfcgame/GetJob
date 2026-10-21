"""Run with TEST_DATABASE_URL set to a disposable PostgreSQL database."""

import os
from datetime import datetime
from uuid import UUID, uuid4

import pytest
psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo

from app.repository import PostgresTaskRepository
from app.schemas import TaskCreate, TaskUpdate


@pytest.fixture
def repository() -> PostgresTaskRepository:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")

    schema = f"test_tasks_{uuid4().hex}"
    with psycopg.connect(database_url) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(
            f'CREATE TABLE "{schema}".tasks ('
            "id UUID PRIMARY KEY, name VARCHAR(200) NOT NULL, "
            "completed BOOLEAN NOT NULL DEFAULT FALSE, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )

    scoped_url = make_conninfo(database_url, options=f"-csearch_path={schema}")
    try:
        yield PostgresTaskRepository(scoped_url)
    finally:
        # Only the unique schema created by this test is removed.
        with psycopg.connect(database_url) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_postgres_crud_and_partial_updates(repository: PostgresTaskRepository) -> None:
    assert repository.list() == []
    task = repository.create(TaskCreate(name="First"))
    assert isinstance(task.id, UUID)
    assert task.completed is False
    assert isinstance(task.created_at, datetime)
    assert task.created_at.tzinfo is not None
    assert task.created_at.utcoffset() is not None
    assert repository.get(task.id) == task
    assert repository.list() == [task]

    renamed = repository.update(task.id, TaskUpdate(name="Second"))
    assert renamed.id == task.id
    assert renamed.name == "Second"
    assert renamed.completed is False
    assert renamed.created_at == task.created_at

    completed = repository.update(task.id, TaskUpdate(completed=True))
    assert completed.id == task.id
    assert completed.name == "Second"
    assert completed.completed is True
    assert completed.created_at == task.created_at
    assert repository.get(task.id) == completed

    assert repository.delete(task.id) is True
    assert repository.get(task.id) is None
    assert repository.list() == []


def test_postgres_missing_id_and_parameterized_name(repository: PostgresTaskRepository) -> None:
    missing = uuid4()
    assert repository.get(missing) is None
    assert repository.update(missing, TaskUpdate(completed=True)) is None
    assert repository.delete(missing) is False

    name = "Robert'); DROP TABLE tasks; --"
    task = repository.create(TaskCreate(name=name))
    assert repository.get(task.id).name == name
    updated = repository.update(task.id, TaskUpdate(name="O'Reilly"))
    assert updated.name == "O'Reilly"
    assert repository.list() == [updated]
