"""Lesson 4.6: classify a request as create_task or clarify.

Set OPENROUTER_API_KEY in the project .env file, then run:
    uv run python scripts/practice_intent_classification.py --run "Add a task to review notes"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from app.intent_classification import (
    ClarifyDecision,
    IncompleteResponseError,
    MissingOutputError,
    UnexpectedResponseFormatError,
    parse_response_decision,
)
from scripts.openrouter_retry import create_response_with_retry


MODEL = "nvidia/nemotron-3.5-super-vl:free"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 1200
TIMEOUT_SECONDS = 30.0
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

INSTRUCTIONS = (
    "Classify the user's intent. Return only one JSON object, without Markdown. "
    'For an explicit request to add or create a task with a specific, actionable '
    'task description, return '
    '{"action":"create_task","name":"short task name"}. '
    "The name must be nonempty and at most 200 characters. "
    'If the user wants a task but omits what to do, leaves a sentence unfinished, '
    'or refers to unavailable earlier context, ask for the missing task details '
    'using {"action":"clarify","question":"what task should I add?"}. '
    "This still recognizes task-creation intent; clarify is the next safe action. "
    "A question about which items should become tasks does not authorize creation. "
    "An explicit instruction not to create a task overrides task-related words. "
    "A request to propose a specific task without saving it may return create_task, "
    "but never claim the task was saved or approved. "
    'For a request to review, search, or summarize existing notes, or when the '
    'user does not clearly ask to create a task, return '
    '{"action":"clarify","question":"one brief question about what the user wants"}. '
    'For example, "Create a task to review my notes" means create_task, '
    'while "Review my notes" means clarify. '
    "Do not claim to have read notes or created a task."
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
        response = create_response_with_retry(
            client,
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=request,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else f" ({type(error).__name__})"
        print(f"OpenRouter request failed{detail}.", file=sys.stderr)
        return 1

    try:
        decision = parse_response_decision(response)
    except IncompleteResponseError:
        print("Model response was incomplete; no decision was accepted.", file=sys.stderr)
        return 1
    except UnexpectedResponseFormatError:
        print("Model response had an unexpected format.", file=sys.stderr)
        return 1
    except MissingOutputError:
        print("Model returned no intent decision.", file=sys.stderr)
        return 1
    except (ValueError, ValidationError):
        print("Model returned an invalid intent decision.", file=sys.stderr)
        return 1

    print(decision.model_dump_json())
    if isinstance(decision, ClarifyDecision):
        print("Clarification needed; no tool was executed.")
    else:
        print("Approval required; no task was created.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", help="Natural-language request")
    parser.add_argument(
        "--run", action="store_true",
        help="Classify with up to two API attempts on transient failures",
    )
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.request or not args.request.strip():
        parser.error("a nonempty request is required with --run")
    return run_experiment(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
