"""Lesson 4.5: inspect native OpenRouter tool calls without executing them.

Set OPENROUTER_API_KEY in the project .env file, then run:
    uv run python scripts/practice_tool_calling.py --run "Remind me to review notes"
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from app.schemas import CreateTaskToolCall
from app.tool_execution import validate_tool_call


MODEL = "nvidia/nemotron-3.5-super-vl:free"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 400
TIMEOUT_SECONDS = 30.0
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

CREATE_TASK_TOOL = {
    "type": "function",
    "name": "create_task",
    "description": "Propose a task for the user to approve. Do not create it yet.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "A nonempty task name of at most 200 characters.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


class ToolCallError(ValueError):
    """The model response has no acceptable create_task tool calls."""


@dataclass(frozen=True)
class ProposedCall:
    call_id: str
    call: CreateTaskToolCall


def parse_model_tool_calls(response: object) -> list[ProposedCall]:
    """Validate exactly one native function_call item; never execute a tool."""
    if getattr(response, "status", None) not in (None, "completed"):
        raise ToolCallError("response is incomplete")

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        raise ToolCallError("response output is missing")

    calls: list[ProposedCall] = []
    seen_ids: set[str] = set()
    for item in output:
        if getattr(item, "type", None) != "function_call":
            continue
        call_id = getattr(item, "call_id", None)
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            raise ToolCallError("invalid function call ID")
        if getattr(item, "status", None) not in (None, "completed"):
            raise ToolCallError("function call is incomplete")
        try:
            arguments = json.loads(getattr(item, "arguments", None))
            call = validate_tool_call(
                {"tool": getattr(item, "name", None), "arguments": arguments}
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise ToolCallError("invalid function call") from error
        seen_ids.add(call_id)
        calls.append(ProposedCall(call_id=call_id, call=call))

    if not calls:
        raise ToolCallError("no function call returned")
    if len(calls) != 1:
        raise ToolCallError("multiple function calls returned")
    return calls


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
            input=request,
            tools=[CREATE_TASK_TOOL],
            tool_choice={"type": "function", "name": "create_task"},
            parallel_tool_calls=False,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else f" ({type(error).__name__})"
        print(f"OpenRouter request failed{detail}.", file=sys.stderr)
        return 1

    try:
        calls = parse_model_tool_calls(response)
    except ToolCallError:
        print("No valid create_task tool call was returned.", file=sys.stderr)
        return 1

    for proposed in calls:
        print(json.dumps({"call_id": proposed.call_id, **proposed.call.model_dump()}))
    print("Approval required; no tool was executed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", help="Natural-language task request")
    parser.add_argument("--run", action="store_true", help="Make one API request")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.request or not args.request.strip():
        parser.error("a nonempty request is required with --run")
    return run_experiment(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
