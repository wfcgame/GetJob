import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.repository import InMemoryTaskRepository


@pytest.mark.parametrize("setting", [None, "memory"])
def test_memory_selection_and_application_state(monkeypatch, setting: str | None) -> None:
    if setting is None:
        monkeypatch.delenv("TASK_STORAGE", raising=False)
    else:
        monkeypatch.setenv("TASK_STORAGE", setting)

    app = create_app()
    with TestClient(app) as client:
        created = client.post("/tasks", json={"name": "Kept in app"}).json()
        assert client.get("/tasks").json() == [created]
        assert client.get(f"/tasks/{created['id']}").json() == created

    with TestClient(create_app()) as other_client:
        assert other_client.get("/tasks").json() == []


def test_postgres_selection(monkeypatch) -> None:
    monkeypatch.setenv("TASK_STORAGE", "postgres")
    selected = []

    def create_postgres_repository() -> InMemoryTaskRepository:
        repository = InMemoryTaskRepository()
        selected.append(repository)
        return repository

    monkeypatch.setattr("app.main.PostgresTaskRepository", create_postgres_repository)

    app = create_app()
    with TestClient(app) as client:
        created = client.post("/tasks", json={"name": "Selected"}).json()
        assert client.get("/tasks").json() == [created]
    assert len(selected) == 1
    assert selected[0].list()[0].id.hex == created["id"].replace("-", "")


def test_explicit_repository_overrides_invalid_storage(monkeypatch) -> None:
    monkeypatch.setenv("TASK_STORAGE", "invalid")
    repository = InMemoryTaskRepository()
    with TestClient(create_app(repository)) as client:
        assert client.post("/tasks", json={"name": "Injected"}).status_code == 201
        assert len(client.get("/tasks").json()) == 1


def test_invalid_storage_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setenv("TASK_STORAGE", "unknown")
    with pytest.raises(ValueError, match="Invalid TASK_STORAGE value 'unknown'"):
        create_app()
