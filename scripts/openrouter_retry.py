"""Bounded retries for read-only OpenRouter response requests."""

import time

from openai import APIConnectionError, APIStatusError, OpenAIError


MAX_API_RETRIES = 1
RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRY_AFTER_SECONDS = 2.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _retry_delay(error: OpenAIError) -> float | None:
    if isinstance(error, APIConnectionError):
        return RETRY_BACKOFF_SECONDS
    if not isinstance(error, APIStatusError):
        return None
    if error.status_code not in RETRYABLE_STATUS_CODES:
        return None
    headers = getattr(getattr(error, "response", None), "headers", {})
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return RETRY_BACKOFF_SECONDS
    try:
        delay = float(retry_after)
    except (TypeError, ValueError):
        return None
    if not 0 <= delay <= MAX_RETRY_AFTER_SECONDS:
        return None
    return delay


def create_response_with_retry(client: object, **request: object) -> object:
    """Retry once for transient transport/HTTP errors, then propagate the error."""
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            return client.responses.create(**request)
        except OpenAIError as error:
            delay = _retry_delay(error)
            if attempt == MAX_API_RETRIES or delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("retry loop ended without a response")
