from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError

from scripts import practice_llm_api


def test_no_run_flag_never_creates_client(monkeypatch, capsys) -> None:
    client = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(practice_llm_api, "OpenAI", client)
    monkeypatch.setattr("sys.argv", ["practice_llm_api.py"])

    assert practice_llm_api.main() == 0
    client.assert_not_called()
    assert "--run" in capsys.readouterr().out


def test_missing_key_never_creates_client(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "old-provider-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=old-provider-key\n")
    monkeypatch.setattr(practice_llm_api, "ENV_FILE", env_file)
    client = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(practice_llm_api, "OpenAI", client)

    assert practice_llm_api.run_experiment("Review notes") == 2
    client.assert_not_called()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_reads_key_from_project_env_file(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setattr(practice_llm_api, "ENV_FILE", env_file)
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"action":"create_task","name":"Review notes"}', usage=None
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr(practice_llm_api, "OpenAI", constructor)

    assert practice_llm_api.run_experiment("Review notes") == 0
    constructor.assert_called_once_with(
        api_key="from-file",
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    output = capsys.readouterr().out
    assert '{"action":"create_task","name":"Review notes"}' in output
    assert "from-file" not in output


def test_response_text_usage_and_request_limits(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    response = SimpleNamespace(
        output_text='{"action":"create_task","name":"  Review one note today  "}',
        usage=SimpleNamespace(input_tokens=12, output_tokens=6, total_tokens=18),
    )
    client = Mock()
    client.responses.create.return_value = response
    constructor = Mock(return_value=client)
    monkeypatch.setattr(practice_llm_api, "OpenAI", constructor)

    assert practice_llm_api.run_experiment("Please review one note today") == 0
    constructor.assert_called_once_with(
        api_key="test-secret",
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    client.responses.create.assert_called_once_with(
        model="openrouter/free",
        input=practice_llm_api.prompt_for("Please review one note today"),
        max_output_tokens=400,
        store=False,
    )
    output = capsys.readouterr().out
    assert '{"action":"create_task","name":"Review one note today"}' in output
    assert "input=12, output=6, total=18" in output
    assert "test-secret" not in output


def test_api_error_is_sanitized(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = APIConnectionError(request=Mock())
    monkeypatch.setattr(practice_llm_api, "OpenAI", Mock(return_value=client))

    assert practice_llm_api.run_experiment("Review notes") == 1
    error = capsys.readouterr().err
    assert "OpenRouter request failed" in error
    assert "test-secret" not in error


@pytest.mark.parametrize("output_text", ["", "not JSON", '{"action":'])
def test_malformed_json_is_rejected(monkeypatch, capsys, output_text) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=output_text, usage=None
    )
    monkeypatch.setattr(practice_llm_api, "OpenAI", Mock(return_value=client))

    assert practice_llm_api.run_experiment("Review notes") == 1
    captured = capsys.readouterr()
    assert "malformed JSON" in captured.err
    assert "create_task" not in captured.out


@pytest.mark.parametrize(
    "output_text",
    [
        '{"action":"delete_task","name":"Review notes"}',
        '{"action":"create_task","name":"   "}',
        '{"action":"create_task","name":"Review notes","extra":1}',
        '[{"action":"create_task","name":"Review notes"}]',
    ],
)
def test_invalid_proposal_is_rejected(monkeypatch, capsys, output_text) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=output_text,
        usage=SimpleNamespace(input_tokens=10, output_tokens=8, total_tokens=18),
    )
    monkeypatch.setattr(practice_llm_api, "OpenAI", Mock(return_value=client))

    assert practice_llm_api.run_experiment("Review notes") == 1
    captured = capsys.readouterr()
    assert "invalid TaskProposal" in captured.err
    assert "create_task" not in captured.out
    assert "total=18" in captured.out


def test_run_requires_a_request_before_creating_client(monkeypatch) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(practice_llm_api, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_llm_api.py", "--run"])

    with pytest.raises(SystemExit) as error:
        practice_llm_api.main()

    assert error.value.code == 2
    constructor.assert_not_called()
