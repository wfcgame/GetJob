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

## PostgreSQL on WSL2

Run these commands from the project directory inside WSL2. The existing Compose
service uses PostgreSQL 18, binds only to `127.0.0.1:5432`, and persists data in
the `postgres_data` named volume mounted at `/var/lib/postgresql`.

Store credentials in the ignored `.env` file with permissions `0600`. Use
`.env.example` as a template and replace the password with a generated hex
password (for example, `openssl rand -hex 32`). Development uses
`knowledge_tasks`; tests must use the separate `knowledge_tasks_test` database.
Do not commit `.env` or print resolved Compose configuration containing secrets.

```bash
chmod 600 .env
docker compose up -d --wait postgres
```

For a fresh installation only, create the test database if it does not already
exist (obtain approval before creating databases in a shared environment):

```bash
docker compose exec -T postgres createdb -U taskuser knowledge_tasks_test
```

Compose creates the development database on first initialization. Apply migrations
only after verifying that this development database is new and empty; for existing
data, follow the migration guidance below. Load `.env` into the current WSL shell
before running Alembic, pytest, or the application; these commands do not load it
automatically. Only source your own trusted `.env` file.

```bash
set -a
source .env
set +a
uv run alembic upgrade head
uv run alembic current
uv run --extra test pytest -rs -q
uv run uvicorn app.main:app --reload
```

The PostgreSQL tests create and clean up unique schemas inside the test database.
LLM tests mock the API client and do not make live OpenRouter requests.
Use `docker compose stop` to stop PostgreSQL while keeping its data.
Do not use `docker compose down -v` when preserving the database.

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

## Internal task tool execution

Lesson 4.4 separates validation from execution:
`app.tool_execution.validate_tool_call(payload)` validates and normalizes a call
without repository access or approval. A successful validation does not authorize
execution; the executor checks approval and revalidates before writing.

`app.tool_execution.proposal_to_tool_call()` converts a `TaskProposal` into a
validated `CreateTaskToolCall` without writing data. Calls have this shape:

```json
{"tool": "create_task", "arguments": {"name": "Review Python notes"}}
```

`execute_tool_call(call, repository, approved=True)` executes the call using the
injected `TaskRepository`. The trusted caller must obtain explicit approval for
that specific call before passing `approved=True`; model output cannot grant
approval. Omitted approval, false, and truthy non-boolean values are rejected.
Tool arguments reuse `TaskCreate` validation, and additional fields and unsupported
tools are rejected before repository writes. Even prebuilt model instances are
revalidated at execution. Preparation and the LLM practice script do not execute
tools, and no public execution endpoint is provided. Repeated approved calls
create separate tasks; this helper does not provide deduplication.

## Lesson 4.5: OpenRouter tool calling

`scripts/practice_tool_calling.py` sends a native `create_task` function schema to
OpenRouter's Responses API using `nvidia/nemotron-3.5-super-vl:free`, whose model
page lists tool calling and the Responses endpoint. It parses exactly one returned
`function_call`, validates its JSON arguments through `CreateTaskToolCall`, and
prints a proposal for review. Missing, malformed, unsupported, and multiple calls
are rejected without printing a partial proposal.
It never calls `execute_tool_call` or writes to a repository. The trusted caller
must still obtain explicit approval for a specific proposal before separately
calling the executor with `approved=True`.

Set `OPENROUTER_API_KEY` in the project `.env` file, then explicitly opt in to a
real API request:

```bash
uv run python scripts/practice_tool_calling.py --run "Remind me to review notes"
```

Without `--run`, the script only prints help. The unit tests mock the API and
make no live requests. The model's tool schema describes the expected argument;
Pydantic still checks length and rejects extra fields before accepting a call.

## Lesson 4.6: Intent classification

`scripts/practice_intent_classification.py` asks OpenRouter for one JSON decision:
`create_task` with a task name, or `clarify` with a question. A Pydantic
discriminated union uses the existing `TaskProposal` validation for the create
branch and `ClarifyDecision` for the question branch. Requests such as "Create a
task to review my notes" should propose a task; "Review my notes" should ask
which notes or what kind of review is wanted. The model can make mistakes, so
invalid JSON or invalid decision fields are rejected.
Classification and evaluation both request the fixed free model
`nvidia/nemotron-3.5-super-vl:free` through the Responses API. The requested
model is also recorded in evaluation JSON, alongside the model ID returned for
each response.

```bash
uv run python scripts/practice_intent_classification.py --run "Create a task to review my notes"
```

The script prints a decision for review. It neither runs a tool nor creates a
database record. A `create_task` decision still needs explicit approval before
the separate executor may write anything. Without `--run`, the script only
prints help. Unit tests mock the API and block network connections.

## Lesson 4.7–4.8: Intent evaluation

`app.intent_evaluation.EXAMPLES` holds 20 labeled cases and three policy cases
across explicit task creation,
direct assistance, missing information, negation, and indirect task requests.
These categories test clear requests, requests that need more detail, and
boundary cases where words such as "task" do not settle the user's intent.
Each case has a stable ID and category. Three cases have no expected label:
directly answering a traceback, setting a reminder, and turning a statement
of intent into a task need another supported intent or an explicit product
policy. Their policy notes appear in the report; predictions for these cases
are observed but excluded from accuracy.

The evaluator reuses the Lesson 4.6 Pydantic parser. Overall and per-category
accuracy are correct valid predictions divided by scored cases. A scored case
has a fixed expected label and a completed response; valid wrong decisions and
invalid completed outputs remain in this denominator. API failures, incomplete
responses, and policy-unresolved cases have no scored decision and are reported
separately. If a category has no scored cases, its accuracy is N/A. Token usage
is summed only when all three counts are available.
All displayed failure and availability fractions use total evaluated cases as
their denominator, including policy-unresolved cases. Correct and incorrect
prediction counts refer only to labeled, completed responses. The corresponding
JSON summary records both total and scored case counts.

The evaluator also records response status, incomplete reason, any available
finish reason, output text length, and available reasoning-token usage. It
distinguishes missing output, malformed JSON, and Pydantic schema failures.
An incomplete response is detected before JSON parsing and counted separately
from invalid completed output. Unexpected response structure is another
distinct invalid-output reason. API timeouts are reported separately from
other API failures.
For API exceptions, it records the exception type and, when available, the
transport cause type without printing exception messages or credentials.
Reaching the configured output-token limit is a warning sign, not proof of
truncation. A response with `status=incomplete` and
`incomplete_details.reason=max_output_tokens` confirms that the output limit
stopped that response. The earlier Lesson 4.8 JSON export did not capture
these response fields or the raw output, so its five invalid cases cannot be
classified retrospectively by parse failure or completion status.

Offline mocked tests need no API key:

```bash
uv run --extra test pytest tests/test_intent_evaluation.py
```

Twenty-three live evaluation cases require an explicit flag. With one bounded
retry per case, this can make up to 46 API attempts. The optional JSON
file includes a summary and per-case request, labels, outcome, sanitized API
error, policy note, and available token usage:

```bash
uv run python -m scripts.practice_intent_evaluation --run --output-json intent-results.json
```

Without `--run`, the script only prints help. The offline mocked tests block
network connections. The evaluator only compares decisions; it never executes
tools or writes tasks.

To prepare a targeted retry of the five invalid Lesson 4.8 cases, select their
IDs and raise the output limit. This command makes **five live requests** when
run, so obtain approval first. Use a new result path; the script refuses to
overwrite an existing file. It records the requested model, selected IDs, and
output limit in `run_config` for comparison with the baseline:

```bash
uv run python -m scripts.practice_intent_evaluation --run \
  --case-id assist-02 --case-id assist-03 --case-id assist-04 \
  --case-id missing-04 --case-id negation-01 \
  --max-output-tokens 1200 --output-json intent-retry-2026-11-03.json
```

Without `--case-id`, all 23 cases are evaluated. The output limit defaults to
1200 and accepts values from 1 through 4096. Unknown or repeated case IDs and
invalid limits are rejected before an API client is created.

## Lesson 4.11: Classification reliability

The fixed model remains `nvidia/nemotron-3.5-super-vl:free`. The default
`max_output_tokens` is 1200. The existing two actions remain `create_task`
and `clarify`: a request such as "Create a task" expresses creation intent,
but lacks a task name, so the next safe response is a clarification question.
An unavailable reference such as "the report we discussed" also needs
clarification. A specific task request with "don't save it yet" can produce a
proposal; the approval-based executor still cannot write without trusted
`approved=True`. Explicit negation prevents a creation proposal.

`app.intent_heldout.HELD_OUT_EXAMPLES` contains ten cases separate from the
original 23. Each has a rationale explaining which boundary it tests. One
indirect statement remains policy-unresolved, with no forced expected label.
These cases are not included in the default live evaluation.

The SDK's own retry count is zero. The application retries a response request
at most once after a timeout, connection error, or HTTP 429/500/502/503/504.
The default delay is 0.5 seconds. A numeric `Retry-After` up to two seconds is
honored; a longer or unreadable value stops retries. Authentication errors,
invalid output, and incomplete responses are not retried. A second transient
failure ends the attempt and is recorded. A retry can consume an additional
free-model request even if the first attempt times out.

Offline verification, without API or database access:

```bash
uv run --extra test pytest -rs -q -o addopts=''
```

The earlier live runs used different settings and are historical evidence.
Under the new denominator, the earlier 400-token run's 13 correct, 2 valid
wrong, 1 incomplete, and 4 timeouts would be described as 13/15 scored,
with the incomplete response and timeouts listed separately. This is only a
recount of saved results, not a new model evaluation.
Mocked tests verify code paths and metric arithmetic; they do not establish
new model accuracy. A new live evaluation requires separate approval.

## Lesson 4.12: Intent policy and safety review

The two supported decisions remain `create_task` (a proposal only) and
`clarify` (a question only). Apply these rules in order:

1. If the user explicitly forbids creating a task, do not propose one. A
   restriction on *saving* a requested task still permits a proposal, but
   never permits execution.
2. Propose `create_task` only when the user asks to add, create, or put a
   specific actionable item on a task list, including a clear indirect request
   such as "Could you put renew registration on my list?".
3. If task creation is requested but the item is missing, unfinished, or
   depends on unavailable context, ask a `clarify` question. Questions about
   whether an item should become a task also get `clarify`.
4. For requests to review or summarize material, `clarify` is the safe
   supported response when the material or desired result is unavailable.
   A possible future task stated without a request does not authorize a task.

Four evaluation cases remain **policy-unresolved** and keep `expected=None`:

| Case | Request | Product decision still needed |
| --- | --- | --- |
| `assist-04` | Explain a Python traceback | Add a direct-assistance intent or define how direct answers fit the current interface. |
| `indirect-03` | Remind me to renew my license | Define reminder behavior; a reminder is not automatically a task. |
| `indirect-04` | I should probably follow up with Mia next week | Decide whether a statement alone should trigger any proposal. It cannot authorize a write. |
| `heldout-indirect-02` | Maybe I should review my budget sometime | Decide how tentative statements should be handled. It cannot authorize a write. |

These are observed but never scored as correct or incorrect. The review does
not assign new labels or add a third intent. Until product policy is settled,
ambiguous requests should not cause task creation.

The model's JSON is untrusted data. The discriminated union rejects unknown
actions, extra fields (including `approved`), malformed JSON, and multiple
decisions. Conversion to `CreateTaskToolCall` explicitly rejects
`ClarifyDecision` with `TypeError` and only validates a `TaskProposal`.
`execute_tool_call` accepts approval solely from a trusted caller's exact
boolean `approved=True`, then revalidates before writing. The classifier and
evaluator never call the executor. Offline adversarial tests cover these
boundaries in separate parsing and authorization test files. A recording fake
repository confirms one write for a valid approved call and zero writes for
denied or malformed calls.

## Lesson 5.3: Local text embeddings

`app.semantic_search.rank_candidates()` accepts an injected text encoder, embeds
the query and candidates in one batch, then ranks candidates by the existing
`cosine_similarity()` function. It reports each score and vector dimension.
Offline tests use fake vectors and never import or download a model.

The optional `embeddings` extra provides Sentence Transformers. The practice
script prints help without `--run`; only `--run` loads
`sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). The first run may
download model files, so run it only when ready:

```bash
uv run --extra embeddings python -m scripts.practice_embeddings --run
```

Use `--query "..."` and repeat `--candidate "..."` to try other texts.
The experiment only prints local rankings; it does not write to PostgreSQL.

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
