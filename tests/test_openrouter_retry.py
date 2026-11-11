import socket
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from scripts import openrouter_retry as retry


@pytest.fixture(autouse=True)
def block_network(monkeypatch) -> None:
    monkeypatch.setattr(
        socket.socket, "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retry tests must not use the network")
        ),
    )


def status_error(code: int, retry_after: str | None = None) -> APIStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        code,
        headers=headers,
        request=httpx.Request("POST", "https://example.invalid/responses"),
    )
    return APIStatusError("simulated", response=response, body=None)


@pytest.mark.parametrize(
    "error",
    [
        APITimeoutError(request=Mock()),
        APIConnectionError(request=Mock()),
        status_error(503),
        status_error(429),
    ],
)
def test_one_retry_recovers_transient_error(monkeypatch, error) -> None:
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    response = SimpleNamespace(output_text='{"action":"clarify","question":"Which?"}')
    client = Mock()
    client.responses.create.side_effect = [error, response]

    assert retry.create_response_with_retry(client, model="test-model") is response
    assert client.responses.create.call_count == 2
    assert sleeps == [0.5]


def test_retry_limit_reached_raises_original_error(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    failure = APITimeoutError(request=Mock())
    client = Mock()
    client.responses.create.side_effect = failure

    with pytest.raises(APITimeoutError):
        retry.create_response_with_retry(client, model="test-model")
    assert client.responses.create.call_count == 2
    assert sleeps == [0.5]


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_non_retryable_http_errors_stop_immediately(monkeypatch, code: int) -> None:
    sleep = Mock(side_effect=AssertionError("unexpected backoff"))
    monkeypatch.setattr(retry.time, "sleep", sleep)
    client = Mock()
    client.responses.create.side_effect = status_error(code)

    with pytest.raises(APIStatusError):
        retry.create_response_with_retry(client)
    client.responses.create.assert_called_once()
    sleep.assert_not_called()


def test_retry_after_is_respected_within_bound(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    response = object()
    client = Mock()
    client.responses.create.side_effect = [status_error(429, "1.5"), response]

    assert retry.create_response_with_retry(client) is response
    assert sleeps == [1.5]
    assert client.responses.create.call_count == 2


@pytest.mark.parametrize("retry_after", ["10", "not-a-number"])
def test_long_or_unreadable_retry_after_terminates(monkeypatch, retry_after: str) -> None:
    sleep = Mock(side_effect=AssertionError("unexpected backoff"))
    monkeypatch.setattr(retry.time, "sleep", sleep)
    client = Mock()
    client.responses.create.side_effect = status_error(429, retry_after)

    with pytest.raises(APIStatusError):
        retry.create_response_with_retry(client)
    client.responses.create.assert_called_once()
    sleep.assert_not_called()
