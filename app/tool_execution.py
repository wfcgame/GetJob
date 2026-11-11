"""Internal tool execution; approval must come from the trusted caller, not the LLM."""

from app.interfaces import TaskRepository
from app.schemas import CreateTaskToolCall, Task, TaskCreate, TaskProposal


def validate_tool_call(call: object) -> CreateTaskToolCall:
    """Validate and normalize a call without repository access or approval."""
    return CreateTaskToolCall.model_validate(call)


def proposal_to_tool_call(proposal: TaskProposal) -> CreateTaskToolCall:
    """Convert only a task proposal; never convert a clarification or grant approval."""
    if not isinstance(proposal, TaskProposal):
        raise TypeError("Only TaskProposal can be converted to a tool call")
    validated = TaskProposal.model_validate(proposal.model_dump())
    return validate_tool_call(
        {"tool": validated.action, "arguments": {"name": validated.name}}
    )


def execute_tool_call(
    call: object,
    repository: TaskRepository,
    *,
    approved: bool = False,
) -> Task:
    """Execute only an explicitly approved, validated create_task call.

    The caller must obtain approval for this specific call. This function does
    not infer approval from a proposal or accept approval inside the payload.
    ValidationError or PermissionError leaves the repository untouched.
    """
    if approved is not True:
        raise PermissionError("Explicit approval is required to execute a tool call")

    validated = validate_tool_call(call)
    return repository.create(TaskCreate(name=validated.arguments.name))
