from types import SimpleNamespace
from unittest.mock import Mock

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

    assert practice_llm_api.run_experiment() == 2
    client.assert_not_called()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_reads_key_from_project_env_file(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setattr(practice_llm_api, "ENV_FILE", env_file)
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text="Done", usage=None)
    constructor = Mock(return_value=client)
    monkeypatch.setattr(practice_llm_api, "OpenAI", constructor)

    assert practice_llm_api.run_experiment() == 0
    constructor.assert_called_once_with(
        api_key="from-file",
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    output = capsys.readouterr().out
    assert "Done" in output
    assert "from-file" not in output


def test_response_text_usage_and_request_limits(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    response = SimpleNamespace(
        output_text="Review one note today.",
        usage=SimpleNamespace(input_tokens=12, output_tokens=6, total_tokens=18),
    )
    client = Mock()
    client.responses.create.return_value = response
    constructor = Mock(return_value=client)
    monkeypatch.setattr(practice_llm_api, "OpenAI", constructor)

    assert practice_llm_api.run_experiment() == 0
    constructor.assert_called_once_with(
        api_key="test-secret",
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    client.responses.create.assert_called_once_with(
        model="openrouter/free",
        input=practice_llm_api.PROMPT,
        max_output_tokens=400,
        store=False,
    )
    output = capsys.readouterr().out
    assert "Review one note today." in output
    assert "input=12, output=6, total=18" in output
    assert "test-secret" not in output


def test_api_error_is_sanitized(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = APIConnectionError(request=Mock())
    monkeypatch.setattr(practice_llm_api, "OpenAI", Mock(return_value=client))

    assert practice_llm_api.run_experiment() == 1
    error = capsys.readouterr().err
    assert "OpenRouter request failed" in error
    assert "test-secret" not in error
