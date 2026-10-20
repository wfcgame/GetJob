from __future__ import annotations

from uuid import UUID, uuid4
import os

from app.schemas import Task, TaskCreate, TaskUpdate


class InMemoryTaskRepository:
    """In-process task storage. Data is lost when the application stops."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, Task] = {}

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def create(self, task_create: TaskCreate) -> Task:
        task = Task(id=uuid4(), name=task_create.name, completed=False)
        self._tasks[task.id] = task
        return task

    def get(self, task_id: UUID) -> Task | None:
        return self._tasks.get(task_id)

    def update(self, task_id: UUID, task_update: TaskUpdate) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None

        updated_task = task.model_copy(update=task_update.model_dump(exclude_unset=True))
        self._tasks[task_id] = updated_task
        return updated_task

    def delete(self, task_id: UUID) -> bool:
        return self._tasks.pop(task_id, None) is not None


class PostgresTaskRepository:
    """Task storage using one short lived PostgreSQL connection per operation."""

    def __init__(self, conninfo: str | None = None) -> None:
        self._conninfo = conninfo

    def _connect(self) -> psycopg.Connection:
        import psycopg
        from psycopg.rows import dict_row

        # Psycopg also reads PGHOST, PGPORT, PGDATABASE, PGUSER and PGPASSWORD.
        return psycopg.connect(
            self._conninfo or os.environ.get("DATABASE_URL", ""),
            row_factory=dict_row,
        )

    @staticmethod
    def _task(row: dict) -> Task:
        return Task.model_validate(row)

    def list(self) -> list[Task]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, completed FROM tasks ORDER BY id"
            ).fetchall()
            return [self._task(row) for row in rows]

    def create(self, task_create: TaskCreate) -> Task:
        with self._connect() as connection:
            row = connection.execute(
                "INSERT INTO tasks (id, name, completed) VALUES (%s, %s, FALSE) "
                "RETURNING id, name, completed",
                (uuid4(), task_create.name),
            ).fetchone()
            return self._task(row)

    def get(self, task_id: UUID) -> Task | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, completed FROM tasks WHERE id = %s",
                (task_id,),
            ).fetchone()
            return self._task(row) if row else None

    def update(self, task_id: UUID, task_update: TaskUpdate) -> Task | None:
        changes = task_update.model_dump(exclude_unset=True)
        # The schema admits only these fields, so SQL identifiers remain fixed.
        if not changes:
            raise ValueError("at least one field must be provided")
        assignments = ", ".join(f"{field} = %s" for field in changes)
        with self._connect() as connection:
            row = connection.execute(
                f"UPDATE tasks SET {assignments} WHERE id = %s "
                "RETURNING id, name, completed",
                (*changes.values(), task_id),
            ).fetchone()
            return self._task(row) if row else None

    def delete(self, task_id: UUID) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "DELETE FROM tasks WHERE id = %s RETURNING id", (task_id,)
            ).fetchone()
            return row is not None

