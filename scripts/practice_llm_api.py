"""A small, opt-in OpenRouter Responses API experiment.

Set OPENROUTER_API_KEY in the project root .env file, then run:
    uv run python scripts/practice_llm_api.py --run

The --run flag is required to make a real API request.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError


MODEL = "openrouter/free"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 400
TIMEOUT_SECONDS = 30.0
PROMPT = "Suggest one small, practical task for organizing my notes. Reply in one sentence."
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def run_experiment() -> int:
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
            input=PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else f" ({type(error).__name__})"
        print(f"OpenRouter request failed{detail}.", file=sys.stderr)
        return 1

    print(response.output_text or "(No output text returned)")
    usage = response.usage
    if usage is None:
        print("Token usage unavailable")
    else:
        print(
            f"Token usage: input={usage.input_tokens}, "
            f"output={usage.output_tokens}, total={usage.total_tokens}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make one free-model API request")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    return run_experiment()


if __name__ == "__main__":
    raise SystemExit(main())
