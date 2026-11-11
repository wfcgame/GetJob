"""Lessons 4.8/4.11: evaluate twenty labeled requests and three policy cases.

Run tests offline with: uv run --extra test pytest tests/test_intent_evaluation.py
Make live OpenRouter requests only with:
    uv run python -m scripts.practice_intent_evaluation --run --output-json results.json
For a selected retry, repeat --case-id and set --max-output-tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, Sequence

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError

from app.intent_evaluation import (
    APIFailure, CATEGORIES, EXAMPLES, EvaluationReport, LabeledExample, evaluate_responses,
)
from app.intent_heldout import HELD_OUT_EXAMPLES
from scripts.practice_intent_classification import (
    BASE_URL,
    ENV_FILE,
    INSTRUCTIONS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    TIMEOUT_SECONDS,
)
from scripts.openrouter_retry import MAX_API_RETRIES, create_response_with_retry


def _print_report(report: EvaluationReport) -> None:
    for result in report.results:
        if result.api_error:
            predicted = "api_failure"
        elif result.incomplete_response:
            predicted = "incomplete"
        elif result.invalid_output:
            predicted = "invalid"
        else:
            predicted = result.predicted
        usage = (
            f"tokens input={result.usage.input_tokens}, output={result.usage.output_tokens}, total={result.usage.total_tokens}"
            if result.usage else "token usage unavailable"
        )
        print(
            f"{result.example.case_id} [{result.example.category}]: "
            f"request={result.example.request!r}; expected={result.example.expected or 'unresolved'}, "
            f"predicted={predicted}, correct={result.correct}; {usage}"
        )
        if result.diagnostics:
            diagnostics = result.diagnostics
            reasoning = result.usage.reasoning_tokens if result.usage else None
            print(
                f"  Response: model={diagnostics.model or 'unavailable'}, "
                f"status={diagnostics.completion_status or 'unavailable'}, "
                f"incomplete_reason={diagnostics.incomplete_reason or 'unavailable'}, "
                f"finish_reason={diagnostics.finish_reason or 'unavailable'}, "
                f"reasoning_tokens={reasoning if reasoning is not None else 'unavailable'}, "
                f"output_text_chars={diagnostics.output_text_chars}, "
                f"at_output_limit={diagnostics.at_output_limit}, "
                f"truncation_confirmed={diagnostics.truncation_confirmed}"
            )
        if result.invalid_reason:
            print(f"  Invalid output reason: {result.invalid_reason}")
        if result.example.policy_note:
            print(f"  Policy unresolved: {result.example.policy_note}")
        if result.api_error:
            status = f", HTTP {result.api_error.status_code}" if result.api_error.status_code is not None else ""
            cause = f", cause={result.api_error.cause_type}" if result.api_error.cause_type else ""
            print(f"  API failure: {result.api_error.error_type}{status}{cause}")

    print(f"Total evaluation cases: {len(report.results)}")
    print(f"Scored cases: {report.scored_count}")
    accuracy = f"{report.accuracy:.0%}" if report.accuracy is not None else "N/A"
    print(f"Accuracy: {report.correct_count}/{report.scored_count} ({accuracy})")
    for category in CATEGORIES:
        summary = report.category_summary(category)
        rate = f"{summary.accuracy:.0%}" if summary.accuracy is not None else "N/A"
        print(f"  {category}: {summary.correct_predictions}/{summary.scored_cases} ({rate})")
    print(f"Correct predictions: {report.correct_count}")
    print(f"Incorrect predictions: {report.incorrect_count}")
    print(f"Invalid outputs: {report.invalid_outputs}/{len(report.results)}")
    print(f"Incomplete responses: {report.incomplete_responses}/{len(report.results)}")
    print(f"Invalid reasons: {report.invalid_reasons}")
    print(
        f"API timeouts: {report.api_timeouts}/{len(report.results)}; "
        f"other API failures: {report.other_api_failures}/{len(report.results)}"
    )
    print(f"API failures total: {report.api_failures}/{len(report.results)}")
    print(
        f"At output limit: {report.at_output_limit_count}/{len(report.results)}; "
        f"confirmed truncations: {report.confirmed_truncations}"
    )
    print(f"Unresolved policy cases: {report.unresolved_policy_cases}/{len(report.results)}")
    totals = report.token_totals
    print(
        f"Token usage available: {report.usage_available}/{len(report.results)}; "
        f"input={totals.input_tokens}, output={totals.output_tokens}, total={totals.total_tokens}"
    )
    print(
        f"Reasoning token usage available: {report.reasoning_usage_available}/{len(report.results)}; "
        f"total={totals.reasoning_tokens if totals.reasoning_tokens is not None else 'unavailable'}"
    )


DatasetName = Literal["baseline", "heldout"]


def _select_examples(
    case_ids: Sequence[str] | None, dataset: DatasetName
) -> tuple[LabeledExample, ...]:
    if dataset == "baseline":
        available = EXAMPLES
    elif dataset == "heldout":
        available = HELD_OUT_EXAMPLES
    else:
        raise ValueError("unknown dataset")
    if case_ids is None:
        return available
    if not case_ids:
        raise ValueError("at least one case ID is required")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must not be repeated")
    known_ids = {example.case_id for example in available}
    unknown = set(case_ids) - known_ids
    if unknown:
        raise ValueError(f"unknown case ID: {', '.join(sorted(unknown))}")
    selected = set(case_ids)
    return tuple(example for example in available if example.case_id in selected)


def run_evaluation(
    output_json: Path | None = None, *,
    case_ids: Sequence[str] | None = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    dataset: DatasetName = "baseline",
) -> int:
    try:
        examples = _select_examples(case_ids, dataset)
    except ValueError as error:
        print(f"Invalid case selection: {error}.", file=sys.stderr)
        return 2
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 4096:
        print("Set --max-output-tokens between 1 and 4096.", file=sys.stderr)
        return 2
    if output_json is not None and output_json.exists():
        print("JSON results file already exists; choose a new path.", file=sys.stderr)
        return 2
    if output_json is not None and not output_json.parent.is_dir():
        print("JSON results directory does not exist.", file=sys.stderr)
        return 2

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
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status is not None else f" ({type(error).__name__})"
        print(f"OpenRouter client failed{detail}.", file=sys.stderr)
        return 1

    responses = []
    for example in examples:
        try:
            response = create_response_with_retry(
                client,
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=example.request,
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except OpenAIError as error:
            status = getattr(error, "status_code", None)
            cause = getattr(error, "__cause__", None)
            response = APIFailure(
                type(error).__name__,
                status if type(status) is int else None,
                type(cause).__name__ if cause is not None else None,
            )
        responses.append(response)

    report = evaluate_responses(examples, responses, max_output_tokens=max_output_tokens)
    _print_report(report)
    if output_json is not None:
        try:
            document = report.to_dict()
            document["run_config"] = {
                "requested_model": MODEL,
                "max_output_tokens": max_output_tokens,
                "case_ids": [example.case_id for example in examples],
                "max_api_retries": MAX_API_RETRIES,
                "dataset": dataset,
            }
            with output_json.open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        except OSError as error:
            print(f"Could not write JSON results ({type(error).__name__}).", file=sys.stderr)
            return 1
        print(f"JSON results saved to {output_json}")
    return 1 if report.api_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make live API requests")
    parser.add_argument("--output-json", type=Path, metavar="PATH", help="Export per-case results")
    parser.add_argument("--case-id", action="append", metavar="ID", help="Evaluate only this case; repeat as needed")
    parser.add_argument(
        "--dataset", choices=("baseline", "heldout"), default=None,
        help="Evaluation set (default: baseline)",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, metavar="N",
        help=f"Output-token limit (1–4096; default: {MAX_OUTPUT_TOKENS})",
    )
    args = parser.parse_args()
    if not args.run:
        if args.output_json or args.case_id or args.max_output_tokens is not None or args.dataset:
            parser.error("evaluation options require --run")
        parser.print_help()
        return 0
    return run_evaluation(
        args.output_json,
        case_ids=args.case_id,
        max_output_tokens=(
            args.max_output_tokens if args.max_output_tokens is not None else MAX_OUTPUT_TOKENS
        ),
        dataset=args.dataset or "baseline",
    )


if __name__ == "__main__":
    raise SystemExit(main())
