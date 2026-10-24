"""Generate a validated TaskProposal with the OpenRouter Responses API.

Set OPENROUTER_API_KEY in the project root .env file, then run:
    uv run python scripts/practice_llm_api.py --run "Remind me to review my notes"

The --run flag is required to make a real API request.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from app.schemas import TaskProposal


MODEL = "openrouter/free"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 400
TIMEOUT_SECONDS = 30.0
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def prompt_for(request: str) -> str:
    return (
        "Convert the following request into one task proposal. "
        'Return only a JSON object with exactly two fields: "action" and "name". '
        'Set "action" to "create_task". Make "name" a short, nonempty task name '
        "of at most 200 characters. Do not use a Markdown code block. "
        "Do not execute the task.\n\n"
        f"Request: {request}"
    )


def parse_proposal(output_text: str) -> TaskProposal:
    payload = json.loads(output_text)
    if isinstance(payload, dict) and set(payload) - {"action", "name"}:
        raise ValueError("unexpected fields")
    return TaskProposal.model_validate(payload)


def print_usage(usage: object) -> None:
    if usage is None:
        print("Token usage unavailable")
    else:
        print(
            f"Token usage: input={usage.input_tokens}, "
            f"output={usage.output_tokens}, total={usage.total_tokens}"
        )


def run_experiment(request: str) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(ENV_FILE).get(
        "OPENROUTER_API_KEY"
    )
    if not api_key:
        print("Set OPENROUTER_API_KEY in the project .env file before running.", file=sys.stderr)
        return 2

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
        )
        response = client.responses.create(
            model=MODEL,
            input=prompt_for(request),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else f" ({type(error).__name__})"
        print(f"OpenRouter request failed{detail}.", file=sys.stderr)
        return 1

    try:
        proposal = parse_proposal(response.output_text or "")
    except json.JSONDecodeError:
        print("Model returned malformed JSON; no proposal was accepted.", file=sys.stderr)
        print_usage(response.usage)
        return 1
    except (ValidationError, ValueError):
        print("Model returned an invalid TaskProposal; no proposal was accepted.", file=sys.stderr)
        print_usage(response.usage)
        return 1

    print(proposal.model_dump_json())
    print_usage(response.usage)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", help="Natural-language task request")
    parser.add_argument("--run", action="store_true", help="Make one free-model API request")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.request or not args.request.strip():
        parser.error("a nonempty request is required with --run")
    return run_experiment(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
