from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class TaskCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)


class TaskProposal(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["create_task"]
    name: str = Field(min_length=1, max_length=200)


class CreateTaskArguments(TaskCreate):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


class CreateTaskToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    tool: Literal["create_task"]
    arguments: CreateTaskArguments


class TaskUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=200)
    completed: bool | None = None

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "TaskUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("updated fields cannot be null")
        return self


class Task(BaseModel):
    id: UUID
    name: str
    completed: bool
    created_at: AwareDatetime

