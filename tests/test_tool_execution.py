import pytest
from pydantic import ValidationError

from app.repository import InMemoryTaskRepository
from app.schemas import CreateTaskArguments, CreateTaskToolCall, TaskCreate, TaskProposal
from app.tool_execution import execute_tool_call, proposal_to_tool_call, validate_tool_call


@pytest.fixture
def repository():
    repo = InMemoryTaskRepository()
    repo.create(TaskCreate(name="Existing task"))
    return repo


def valid_call():
    return {"tool": "create_task", "arguments": {"name": "  Review notes  "}}


def test_validation_normalizes_without_mutating_input_or_stored_data(repository):
    before = repository.list()
    payload = valid_call()
    call = validate_tool_call(payload)
    assert call.arguments.name == "Review notes"
    assert payload["arguments"]["name"] == "  Review notes  "
    assert repository.list() == before
    with pytest.raises(PermissionError):
        execute_tool_call(call, repository)
    assert repository.list() == before


@pytest.mark.parametrize("payload", [
    {"tool": "delete_task", "arguments": {"name": "Review notes"}},
    {"tool": "create_task", "arguments": {"name": "   "}},
])
def test_validation_rejects_invalid_calls_without_execution(repository, payload):
    before = repository.list()
    with pytest.raises(ValidationError):
        validate_tool_call(payload)
    assert repository.list() == before


def test_proposal_preparation_does_not_execute(repository):
    before = repository.list()
    proposal = TaskProposal(action="create_task", name="  Review notes  ")
    call = proposal_to_tool_call(proposal)
    assert call.tool == "create_task"
    assert call.arguments.name == "Review notes"
    assert repository.list() == before


def test_approved_call_creates_one_task_in_injected_repository(repository):
    before = repository.list()
    task = execute_tool_call(valid_call(), repository, approved=True)
    assert task.name == "Review notes"
    assert task.completed is False
    assert repository.list() == [*before, task]
    assert InMemoryTaskRepository().list() == []


def test_validated_call_can_execute_after_approval(repository):
    call = proposal_to_tool_call(TaskProposal(action="create_task", name="Review notes"))
    task = execute_tool_call(call, repository, approved=True)
    assert repository.get(task.id) == task


def test_missing_approval_leaves_data_unchanged(repository):
    before = repository.list()
    with pytest.raises(PermissionError, match="Explicit approval"):
        execute_tool_call(valid_call(), repository)
    assert repository.list() == before


@pytest.mark.parametrize("approval", [False, None, 0, 1, "true", "false", {}, [True]])
def test_only_boolean_true_is_approval(repository, approval):
    before = repository.list()
    with pytest.raises(PermissionError):
        execute_tool_call(valid_call(), repository, approved=approval)
    assert repository.list() == before


@pytest.mark.parametrize("call", [
    {"tool": "delete_task", "arguments": {"name": "Review notes"}},
    {"arguments": {"name": "Review notes"}},
    {"tool": "create_task"},
    {"tool": "create_task", "arguments": {}},
    {"tool": "create_task", "arguments": {"name": "   "}},
    {"tool": "create_task", "arguments": {"name": "a" * 201}},
    {"tool": "create_task", "arguments": {"name": None}},
    {"tool": "create_task", "arguments": {"name": 123}},
    {"tool": "create_task", "arguments": {"name": "Review notes", "completed": True}},
    {"tool": "create_task", "arguments": {"name": "Review notes"}, "approved": True},
    {"action": "create_task", "name": "Review notes"},
    {"tool": "create_task", "arguments": []},
    [], None, "not JSON",
])
def test_invalid_or_unsupported_calls_leave_data_unchanged(repository, call):
    before = repository.list()
    with pytest.raises(ValidationError):
        execute_tool_call(call, repository, approved=True)
    assert repository.list() == before


def test_payload_cannot_grant_itself_approval(repository):
    before = repository.list()
    with pytest.raises(PermissionError):
        execute_tool_call({**valid_call(), "approved": True}, repository)
    assert repository.list() == before


@pytest.mark.parametrize("mutation", ["tool", "name"])
def test_mutated_models_are_revalidated_before_execution(repository, mutation):
    call = CreateTaskToolCall.model_validate(valid_call())
    if mutation == "tool":
        call.tool = "delete_task"
    else:
        call.arguments.name = "   "
    before = repository.list()
    with pytest.raises(ValidationError):
        execute_tool_call(call, repository, approved=True)
    assert repository.list() == before


def test_unvalidated_constructed_models_are_rejected(repository):
    call = CreateTaskToolCall.model_construct(
        tool="create_task", arguments=CreateTaskArguments.model_construct(name="")
    )
    before = repository.list()
    with pytest.raises(ValidationError):
        execute_tool_call(call, repository, approved=True)
    assert repository.list() == before
