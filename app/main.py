import os
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status

from app.interfaces import TaskRepository
from app.repository import InMemoryTaskRepository, PostgresTaskRepository
from app.schemas import Task, TaskCreate, TaskUpdate


def create_app(repository: TaskRepository | None = None) -> FastAPI:
    app = FastAPI(title="Knowledge & Task Assistant", version="0.1.0")
    if repository is not None:
        task_repository = repository
    else:
        storage = os.environ.get("TASK_STORAGE", "memory")
        if storage == "memory":
            task_repository = InMemoryTaskRepository()
        elif storage == "postgres":
            task_repository = PostgresTaskRepository()
        else:
            raise ValueError(
                f"Invalid TASK_STORAGE value {storage!r}; expected 'memory' or 'postgres'"
            )

    def get_repository() -> TaskRepository:
        return task_repository

    RepositoryDependency = Depends(get_repository)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks", response_model=list[Task])
    def list_tasks(repo: TaskRepository = RepositoryDependency) -> list[Task]:
        return repo.list()

    @app.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED)
    def create_task(
        task_create: TaskCreate,
        repo: TaskRepository = RepositoryDependency,
    ) -> Task:
        return repo.create(task_create)

    @app.get("/tasks/{task_id}", response_model=Task)
    def get_task(
        task_id: UUID,
        repo: TaskRepository = RepositoryDependency,
    ) -> Task:
        task = repo.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.patch("/tasks/{task_id}", response_model=Task)
    def update_task(
        task_id: UUID,
        task_update: TaskUpdate,
        repo: TaskRepository = RepositoryDependency,
    ) -> Task:
        task = repo.update(task_id, task_update)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_task(
        task_id: UUID,
        repo: TaskRepository = RepositoryDependency,
    ) -> None:
        if not repo.delete(task_id):
            raise HTTPException(status_code=404, detail="Task not found")

    return app


app = create_app()
