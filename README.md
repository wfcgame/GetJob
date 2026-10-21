# Knowledge & Task Assistant

The first increment of a production-oriented assistant backend. It provides a
small task API backed by process-local memory by default. PostgreSQL is available
through the injectable `PostgresTaskRepository`.

## Requirements

- Python 3.12 or newer

## Setup and run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated interactive API docs.

## Test

```bash
pytest
```

PostgreSQL integration tests run when `TEST_DATABASE_URL` points to a test
database. They create a unique schema for each test and remove only that schema
afterward. Set `TASK_STORAGE=postgres` to use PostgreSQL in the application;
omit it or set `TASK_STORAGE=memory` for in-memory storage. An explicit
`create_app(repository)` argument takes priority. The PostgreSQL repository
accepts a connection string directly, or reads `DATABASE_URL` or Psycopg's
`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` environment variables.

## Database migrations

Alembic manages the PostgreSQL schema. Repository reads and writes still use
Psycopg 3 directly. `DATABASE_URL` or the standard `PG*` variables supply the
connection settings; credentials are never stored in `alembic.ini`.

For a **new, empty database**, run `alembic upgrade head` before starting the
API with `TASK_STORAGE=postgres`. The Compose service no longer runs the old
`sql/schema.sql` initialization script for new volumes.

For an **existing database** whose `tasks` table was created by that script,
first compare its columns, primary key, nullability, and default with
`migrations/versions/001_tasks_baseline.py`. Check `alembic_version` is absent
and back up the database according to your normal procedure. Once the schema
is confirmed equivalent, run `alembic stamp 001_tasks_baseline`. Stamping only
records the migration version; it does not recreate or change task rows. Do not
run `alembic upgrade head` against an unstamped database that already has the
`tasks` table: the baseline will stop with an error. The baseline downgrade is
disabled to protect task records.

For later schema changes, create a revision with `alembic revision -m
"description"`, review its SQL and data impact, then apply it with `alembic
upgrade head`. `alembic current` reports the recorded version. The baseline
contains no new task columns.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service health |
| `GET` | `/tasks` | List tasks |
| `POST` | `/tasks` | Create a task |
| `GET` | `/tasks/{task_id}` | Retrieve a task |
| `PATCH` | `/tasks/{task_id}` | Change a task's name and/or completion state |

Create a task:

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"name": "Review notes"}'
```

Tasks have UUID identifiers and an ISO 8601 `created_at` timestamp with a time
zone. Names are trimmed, must contain 1–200 characters, and new tasks start
with `completed: false`. PATCH requests must contain at least one non-null
supported field; clients cannot set or change `created_at`.

## Project structure

```text
app/
  main.py        # Application factory and HTTP routes
  repository.py  # In-memory and PostgreSQL persistence operations
  schemas.py     # Pydantic request and response models
tests/
  test_api.py    # Isolated endpoint tests
pyproject.toml   # Package metadata and dependencies
```

## Request flow

FastAPI matches an incoming request to a route in `main.py`. Pydantic validates
and normalizes its path and JSON inputs. The route calls the injected task
repository, then FastAPI validates the returned response model and serializes it
as JSON. A missing repository record is translated by the route into HTTP 404.

## Design decisions

1. **Repository boundary:** storage behavior is kept out of route handlers, so a
   PostgreSQL implementation can later replace in-memory storage with minimal API
   changes.
2. **Application factory:** tests construct a new application and repository for
   every test, preventing state leakage and making results deterministic.
3. **Separate input models:** create, patch, and response payloads express
   different contracts. Clients cannot choose IDs or initial completion state,
   and an empty or null-only patch is rejected.

## Practice change

Add `DELETE /tasks/{task_id}`. Return HTTP 204 when deletion succeeds and HTTP
404 for an unknown UUID. Add tests for both cases and ensure one test verifies
that a deleted task no longer appears in `GET /tasks`.
