import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError

from app.repository import InMemoryTaskRepository
from app.tool_execution import execute_tool_call
from scripts import practice_tool_calling as lesson


@pytest.fixture(autouse=True)
def block_network(monkeypatch) -> None:
    def deny_connection(*args, **kwargs) -> None:
        raise AssertionError("unit tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)


def function_call(
    arguments: str = '{"name":"  Review notes  "}',
    *,
    name: str = "create_task",
    call_id: str = "call_1",
    status: str = "completed",
) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id,
        status=status,
    )


def response(*items: object, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(output=list(items), status=status)


def test_tool_schema_matches_create_task_arguments() -> None:
    tool = lesson.CREATE_TASK_TOOL
    assert tool["type"] == "function"
    assert tool["name"] == "create_task"
    assert tool["parameters"]["required"] == ["name"]
    assert tool["parameters"]["additionalProperties"] is False


def test_valid_native_calls_are_normalized_without_execution() -> None:
    repository = InMemoryTaskRepository()
    result = lesson.parse_model_tool_calls(
        response(
            SimpleNamespace(type="reasoning"),
            function_call(),
        )
    )

    assert [item.call_id for item in result] == ["call_1"]
    assert [item.call.arguments.name for item in result] == ["Review notes"]
    with pytest.raises(PermissionError, match="Explicit approval"):
        execute_tool_call(result[0].call, repository)
    assert repository.list() == []


@pytest.mark.parametrize(
    "model_response",
    [
        response(SimpleNamespace(type="message")),
        response(function_call("not JSON")),
        response(function_call("[]")),
        response(function_call('{"name":"   "}')),
        response(function_call("{}")),
        response(function_call(json.dumps({"name": "x" * 201}))),
        response(function_call('{"name":"x"}', name="delete_task")),
        response(function_call('{"name":"x","approved":true}')),
        response(function_call('{"name":"x","extra":1}')),
        response(function_call('{"name":123}')),
        response(function_call(call_id="")),
        response(function_call(status="incomplete")),
        response(function_call(), status="incomplete"),
        response(function_call(), function_call(call_id="call_1")),
        response(function_call(), function_call(call_id="call_2")),
    ],
)
def test_invalid_native_calls_are_rejected(model_response: SimpleNamespace) -> None:
    with pytest.raises(lesson.ToolCallError):
        lesson.parse_model_tool_calls(model_response)


def test_no_run_flag_never_creates_client(monkeypatch, capsys) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_tool_calling.py", "Review notes"])

    assert lesson.main() == 0
    constructor.assert_not_called()
    assert "--run" in capsys.readouterr().out


def test_run_requires_request_before_creating_client(monkeypatch) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_tool_calling.py", "--run"])

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


def test_mocked_request_prepares_call_and_preserves_approval(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    repository = InMemoryTaskRepository()
    client = Mock()
    client.responses.create.return_value = response(function_call())
    constructor = Mock(return_value=client)
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    assert lesson.run_experiment("Please remind me to review notes") == 0
    constructor.assert_called_once_with(
        api_key="test-secret",
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    client.responses.create.assert_called_once_with(
        model="nvidia/nemotron-3.5-super-vl:free",
        input="Please remind me to review notes",
        tools=[lesson.CREATE_TASK_TOOL],
        tool_choice={"type": "function", "name": "create_task"},
        parallel_tool_calls=False,
        max_output_tokens=400,
        store=False,
    )
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0]) == {
        "call_id": "call_1",
        "tool": "create_task",
        "arguments": {"name": "Review notes"},
    }
    assert lines[1] == "Approval required; no tool was executed."
    assert repository.list() == []
    assert "test-secret" not in "\n".join(lines)


def test_invalid_second_call_prints_no_partial_proposal(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.return_value = response(
        function_call(), function_call("not JSON", call_id="call_2")
    )
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))

    assert lesson.run_experiment("Review notes") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No valid create_task" in captured.err


def test_api_error_is_sanitized(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = APIConnectionError(request=Mock())
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))

    assert lesson.run_experiment("Review notes") == 1
    error = capsys.readouterr().err
    assert "OpenRouter request failed" in error
    assert "test-secret" not in error
