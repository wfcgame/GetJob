import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError
from pydantic import ValidationError

from app.intent_classification import (
    ClarifyDecision,
    IncompleteResponseError,
    parse_intent_decision,
    parse_response_decision,
)
from app.repository import InMemoryTaskRepository
from app.schemas import TaskProposal
from app.tool_execution import execute_tool_call, proposal_to_tool_call
from scripts import practice_intent_classification as lesson


@pytest.fixture(autouse=True)
def block_network(monkeypatch) -> None:
    def deny_connection(*args, **kwargs) -> None:
        raise AssertionError("intent tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)


def mocked_client(monkeypatch, output_text: str) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text=output_text
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))
    return client


def test_explicit_task_request_returns_validated_proposal_without_execution(
    monkeypatch, capsys
) -> None:
    repository = InMemoryTaskRepository()
    client = mocked_client(
        monkeypatch, '{"action":"create_task","name":"  Review Python notes  "}'
    )

    assert lesson.run_experiment("Create a task to review my Python notes") == 0
    request = client.responses.create.call_args.kwargs
    assert request["model"] == "nvidia/nemotron-3.5-super-vl:free"
    assert request["max_output_tokens"] == 1200
    assert request["input"] == "Create a task to review my Python notes"
    assert "Create a task to review my notes" in request["instructions"]
    assert request["store"] is False
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0]) == {
        "action": "create_task", "name": "Review Python notes"
    }
    assert lines[1] == "Approval required; no task was created."
    assert repository.list() == []


def test_note_review_request_returns_clarification_without_tool_execution(
    monkeypatch, capsys
) -> None:
    repository = InMemoryTaskRepository()
    client = mocked_client(
        monkeypatch,
        '{"action":"clarify","question":"Which notes should I review?"}',
    )

    assert lesson.run_experiment("Review my notes") == 0
    assert client.responses.create.call_args.kwargs["input"] == "Review my notes"
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0]) == {
        "action": "clarify", "question": "Which notes should I review?"
    }
    assert lines[1] == "Clarification needed; no tool was executed."
    assert repository.list() == []


@pytest.mark.parametrize(
    ("user_request", "response_text", "action"),
    [
        ("Create a task.", '{"action":"clarify","question":"What task?"}', "clarify"),
        ("Create a task to...", '{"action":"clarify","question":"What should I add?"}', "clarify"),
        ("Create a task for the report we discussed.", '{"action":"clarify","question":"Which report?"}', "clarify"),
        ("Which of these should go on my task list?", '{"action":"clarify","question":"Which items?"}', "clarify"),
        ("Do not create a task; just review my notes.", '{"action":"clarify","question":"Which notes?"}', "clarify"),
        ("Create a task to call Sam, but don't save it yet.", '{"action":"create_task","name":"Call Sam"}', "create_task"),
    ],
)
def test_boundary_decisions_are_validated_without_execution(
    monkeypatch, capsys, user_request, response_text, action
) -> None:
    repository = InMemoryTaskRepository()
    mocked_client(monkeypatch, response_text)

    assert lesson.run_experiment(user_request) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["action"] == action
    assert repository.list() == []


def test_prompt_distinguishes_creation_intent_from_missing_details() -> None:
    assert "task-creation intent" in lesson.INSTRUCTIONS
    assert "unavailable earlier context" in lesson.INSTRUCTIONS
    assert "not to create a task overrides" in lesson.INSTRUCTIONS
    assert "without saving it" in lesson.INSTRUCTIONS


def test_incomplete_response_is_rejected_before_json_parser(monkeypatch) -> None:
    def forbidden_parser(_text: str) -> None:
        raise AssertionError("partial JSON must not be parsed")

    monkeypatch.setattr("app.intent_classification.parse_intent_decision", forbidden_parser)
    response = SimpleNamespace(
        status="incomplete",
        output_text='{"action":"create_task","name":"Call Sam"}',
    )
    with pytest.raises(IncompleteResponseError):
        parse_response_decision(response)


def test_classifier_rejects_incomplete_or_unexpected_response(
    monkeypatch, capsys
) -> None:
    client = mocked_client(monkeypatch, '{"action":"create_task","name":"Call Sam"}')
    client.responses.create.return_value.status = "incomplete"
    assert lesson.run_experiment("Create a task to call Sam") == 1
    assert "incomplete" in capsys.readouterr().err

    client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text={"action": "create_task"}
    )
    assert lesson.run_experiment("Create a task to call Sam") == 1
    assert "unexpected format" in capsys.readouterr().err


def test_model_proposal_cannot_approve_its_own_execution() -> None:
    decision = parse_response_decision(
        SimpleNamespace(
            status="completed",
            output_text='{"action":"create_task","name":"Call Sam"}',
        )
    )
    repository = InMemoryTaskRepository()
    call = proposal_to_tool_call(decision)

    with pytest.raises(PermissionError, match="Explicit approval"):
        execute_tool_call(call, repository)
    assert repository.list() == []


def test_discriminated_union_reuses_task_validation() -> None:
    create = parse_intent_decision('{"action":"create_task","name":"  Review notes  "}')
    clarify = parse_intent_decision(
        '{"action":"clarify","question":"  Which notes?  "}'
    )

    assert isinstance(create, TaskProposal)
    assert create.name == "Review notes"
    assert isinstance(clarify, ClarifyDecision)
    assert clarify.question == "Which notes?"


@pytest.mark.parametrize(
    "output_text",
    [
        "not JSON",
        "{}",
        '{"action":"delete_task","name":"Review notes"}',
        '{"action":"create_task","name":"   "}',
        json.dumps({"action": "create_task", "name": "x" * 201}),
        '{"action":"create_task","name":"Review notes","approved":true}',
        '{"action":"clarify","question":"   "}',
        '{"action":"clarify","question":"Which notes?","name":"Review notes"}',
        '[{"action":"clarify","question":"Which notes?"}]',
    ],
)
def test_invalid_model_decisions_are_rejected(output_text: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_intent_decision(output_text)


def test_invalid_model_output_prints_no_decision(monkeypatch, capsys) -> None:
    mocked_client(monkeypatch, '{"action":"create_task","name":" "}')

    assert lesson.run_experiment("Create a task") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid intent decision" in captured.err
    assert "test-secret" not in captured.err


def test_no_run_flag_never_creates_client(monkeypatch, capsys) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_intent_classification.py", "Review notes"])

    assert lesson.main() == 0
    constructor.assert_not_called()
    assert "--run" in capsys.readouterr().out


def test_run_requires_request_before_creating_client(monkeypatch) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_intent_classification.py", "--run"])

    with pytest.raises(SystemExit) as error:
        lesson.main()
    assert error.value.code == 2
    constructor.assert_not_called()


def test_missing_key_never_creates_client(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(lesson, "ENV_FILE", tmp_path / ".env")
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    assert lesson.run_experiment("Review notes") == 2
    constructor.assert_not_called()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_api_error_is_sanitized(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = APIConnectionError(request=Mock())
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))

    assert lesson.run_experiment("Review notes") == 1
    error = capsys.readouterr().err
    assert "OpenRouter request failed" in error
    assert "test-secret" not in error
