"""Score intent decisions without calling a model or executing tools."""

import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from pydantic import ValidationError

from app.intent_classification import (
    IncompleteResponseError,
    MissingOutputError,
    UnexpectedResponseFormatError,
    parse_response_decision,
)


IntentName = Literal["create_task", "clarify"]
Category = Literal[
    "explicit_task_creation",
    "direct_assistance",
    "missing_information",
    "negation",
    "indirect_task_request",
]
CATEGORIES: tuple[Category, ...] = (
    "explicit_task_creation",
    "direct_assistance",
    "missing_information",
    "negation",
    "indirect_task_request",
)


@dataclass(frozen=True)
class LabeledExample:
    case_id: str
    category: Category
    request: str
    expected: IntentName | None
    policy_note: str | None = None

    def __post_init__(self) -> None:
        if (self.expected is None) != (self.policy_note is not None):
            raise ValueError("unresolved cases require a policy note; labeled cases must not have one")


# None means there is no defensible label under the current two-intent policy.
EXAMPLES: tuple[LabeledExample, ...] = (
    LabeledExample("explicit-01", "explicit_task_creation", "Create a task to review the incident notes tomorrow.", "create_task"),
    LabeledExample("explicit-02", "explicit_task_creation", "Add a task: submit travel receipts by Friday.", "create_task"),
    LabeledExample("explicit-03", "explicit_task_creation", "Please create a task called Renew passport.", "create_task"),
    LabeledExample("explicit-04", "explicit_task_creation", "Can you add a task to call the dentist?", "create_task"),
    LabeledExample("assist-01", "direct_assistance", "Review my Python notes.", "clarify"),
    LabeledExample("assist-02", "direct_assistance", "Summarize my interview notes.", "clarify"),
    LabeledExample("assist-03", "direct_assistance", "Find the deadline in my project notes.", "clarify"),
    LabeledExample("assist-04", "direct_assistance", "Explain why this Python traceback occurs.", None, "A direct answer needs an assistance intent or an explicit policy for mapping it to clarify."),
    LabeledExample("missing-01", "missing_information", "Create a task.", "clarify"),
    LabeledExample("missing-02", "missing_information", "Make a task for the thing we discussed.", "clarify"),
    LabeledExample("missing-03", "missing_information", "Which of these should go on my task list?", "clarify"),
    LabeledExample("missing-04", "missing_information", "Create a task to...", "clarify"),
    LabeledExample("negation-01", "negation", "Do not create a task; just review my notes.", "clarify"),
    LabeledExample("negation-02", "negation", "Don't add a task for the interview; summarize the notes instead.", "clarify"),
    LabeledExample("negation-03", "negation", "I don't want a task created. What did we decide?", "clarify"),
    LabeledExample("negation-04", "negation", "Create a task to call Sam, but don't save it yet.", "create_task"),
    LabeledExample("indirect-01", "indirect_task_request", "Could you put send the invoice on my task list?", "create_task"),
    LabeledExample("indirect-02", "indirect_task_request", "Would it help to create a task for the launch?", "clarify"),
    LabeledExample("indirect-03", "indirect_task_request", "Please make sure I remember to renew my license.", None, "A reminder may need a reminder intent or an explicit policy to create a task."),
    LabeledExample("indirect-04", "indirect_task_request", "I should probably follow up with Mia next week.", None, "A statement of intent does not clearly authorize task creation; define a policy before labeling."),
    LabeledExample("assist-05", "direct_assistance", "Review the notes from yesterday's meeting.", "clarify"),
    LabeledExample("indirect-05", "indirect_task_request", "Can you make a to-do for returning the library books?", "create_task"),
    LabeledExample("indirect-06", "indirect_task_request", "Do you think I should track the launch checklist as a task?", "clarify"),
)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


InvalidReason = Literal[
    "missing_output", "malformed_json", "schema_validation",
    "unexpected_response_format",
]


@dataclass(frozen=True)
class ResponseDiagnostics:
    model: str | None
    completion_status: str | None
    incomplete_reason: str | None
    finish_reason: str | None
    output_text_chars: int
    at_output_limit: bool | None
    truncation_confirmed: bool


@dataclass(frozen=True)
class APIFailure:
    error_type: str
    status_code: int | None = None
    cause_type: str | None = None


@dataclass(frozen=True)
class CaseResult:
    example: LabeledExample
    predicted: IntentName | None
    usage: TokenUsage | None
    api_error: APIFailure | None = None
    invalid_reason: InvalidReason | None = None
    diagnostics: ResponseDiagnostics | None = None

    @property
    def incomplete_response(self) -> bool:
        return (
            self.diagnostics is not None
            and self.diagnostics.completion_status == "incomplete"
        )

    @property
    def invalid_output(self) -> bool:
        return (
            self.api_error is None
            and not self.incomplete_response
            and self.predicted is None
        )

    @property
    def correct(self) -> bool | None:
        if (
            self.example.expected is None
            or self.api_error is not None
            or self.incomplete_response
        ):
            return None
        return self.predicted == self.example.expected

    def to_dict(self) -> dict:
        return {
            "case_id": self.example.case_id,
            "category": self.example.category,
            "request": self.example.request,
            "expected_intent": self.example.expected,
            "predicted_intent": self.predicted,
            "correct": self.correct,
            "invalid_output": self.invalid_output,
            "incomplete_response": self.incomplete_response,
            "invalid_reason": self.invalid_reason,
            "api_error": asdict(self.api_error) if self.api_error else None,
            "policy_note": self.example.policy_note,
            "token_usage": asdict(self.usage) if self.usage else None,
            "diagnostics": asdict(self.diagnostics) if self.diagnostics else None,
        }


@dataclass(frozen=True)
class CategorySummary:
    total_cases: int
    scored_cases: int
    correct_predictions: int
    incorrect_predictions: int
    invalid_outputs: int
    incomplete_responses: int
    api_timeouts: int
    other_api_failures: int
    api_failures: int
    unresolved_policy_cases: int
    accuracy: float | None


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]

    @property
    def correct_count(self) -> int:
        return sum(result.correct is True for result in self.results)

    @property
    def incorrect_count(self) -> int:
        return sum(result.correct is False and not result.invalid_output for result in self.results)

    @property
    def scored_count(self) -> int:
        return sum(result.correct is not None for result in self.results)

    @property
    def accuracy(self) -> float | None:
        return self.correct_count / self.scored_count if self.scored_count else None

    @property
    def invalid_outputs(self) -> int:
        return sum(result.invalid_output for result in self.results)

    @property
    def incomplete_responses(self) -> int:
        return sum(result.incomplete_response for result in self.results)

    @property
    def invalid_reasons(self) -> dict[str, int]:
        return {
            reason: sum(result.invalid_reason == reason for result in self.results)
            for reason in (
                "missing_output", "malformed_json", "schema_validation",
                "unexpected_response_format",
            )
        }

    @property
    def api_failures(self) -> int:
        return sum(result.api_error is not None for result in self.results)

    @property
    def api_timeouts(self) -> int:
        return sum(
            result.api_error is not None
            and result.api_error.error_type == "APITimeoutError"
            for result in self.results
        )

    @property
    def other_api_failures(self) -> int:
        return self.api_failures - self.api_timeouts

    @property
    def unresolved_policy_cases(self) -> int:
        return sum(result.example.expected is None for result in self.results)

    @property
    def usage_available(self) -> int:
        return sum(result.usage is not None for result in self.results)

    @property
    def reasoning_usage_available(self) -> int:
        return sum(
            result.usage is not None and result.usage.reasoning_tokens is not None
            for result in self.results
        )

    @property
    def at_output_limit_count(self) -> int:
        return sum(
            result.diagnostics is not None and result.diagnostics.at_output_limit is True
            for result in self.results
        )

    @property
    def confirmed_truncations(self) -> int:
        return sum(
            result.diagnostics is not None and result.diagnostics.truncation_confirmed
            for result in self.results
        )

    @property
    def token_totals(self) -> TokenUsage:
        available = [result.usage for result in self.results if result.usage is not None]
        return TokenUsage(
            input_tokens=sum(usage.input_tokens for usage in available),
            output_tokens=sum(usage.output_tokens for usage in available),
            total_tokens=sum(usage.total_tokens for usage in available),
            reasoning_tokens=(
                sum(usage.reasoning_tokens for usage in available if usage.reasoning_tokens is not None)
                if self.reasoning_usage_available else None
            ),
        )

    def category_summary(self, category: Category) -> CategorySummary:
        subset = EvaluationReport(tuple(result for result in self.results if result.example.category == category))
        return CategorySummary(
            total_cases=len(subset.results),
            scored_cases=subset.scored_count,
            correct_predictions=subset.correct_count,
            incorrect_predictions=subset.incorrect_count,
            invalid_outputs=subset.invalid_outputs,
            incomplete_responses=subset.incomplete_responses,
            api_timeouts=subset.api_timeouts,
            other_api_failures=subset.other_api_failures,
            api_failures=subset.api_failures,
            unresolved_policy_cases=subset.unresolved_policy_cases,
            accuracy=subset.accuracy,
        )

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_cases": len(self.results),
                "scored_cases": self.scored_count,
                "accuracy": self.accuracy,
                "correct_predictions": self.correct_count,
                "incorrect_predictions": self.incorrect_count,
                "invalid_outputs": self.invalid_outputs,
                "incomplete_responses": self.incomplete_responses,
                "invalid_reasons": self.invalid_reasons,
                "api_failures": self.api_failures,
                "api_timeouts": self.api_timeouts,
                "other_api_failures": self.other_api_failures,
                "unresolved_policy_cases": self.unresolved_policy_cases,
                "usage_available": self.usage_available,
                "reasoning_usage_available": self.reasoning_usage_available,
                "at_output_limit_count": self.at_output_limit_count,
                "confirmed_truncations": self.confirmed_truncations,
                "token_totals": asdict(self.token_totals),
                "categories": {category: asdict(self.category_summary(category)) for category in CATEGORIES},
            },
            "results": [result.to_dict() for result in self.results],
        }


def _read_usage(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    counts = tuple(
        getattr(usage, field, None)
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    if type(reasoning_tokens) is not int or reasoning_tokens < 0:
        reasoning_tokens = None
    return TokenUsage(*counts, reasoning_tokens=reasoning_tokens)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _read_diagnostics(
    response: object, usage: TokenUsage | None, output_text: object,
    max_output_tokens: int | None,
) -> ResponseDiagnostics:
    details = getattr(response, "incomplete_details", None)
    reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
    status = _optional_text(getattr(response, "status", None))
    at_limit = (
        usage.output_tokens >= max_output_tokens
        if usage is not None and max_output_tokens is not None else None
    )
    return ResponseDiagnostics(
        model=_optional_text(getattr(response, "model", None)),
        completion_status=status,
        incomplete_reason=_optional_text(reason),
        finish_reason=_optional_text(getattr(response, "finish_reason", None)),
        output_text_chars=len(output_text) if isinstance(output_text, str) else 0,
        at_output_limit=at_limit,
        truncation_confirmed=status == "incomplete" and reason == "max_output_tokens",
    )


def evaluate_responses(
    examples: Sequence[LabeledExample], responses: Sequence[object | APIFailure],
    *, max_output_tokens: int | None = None,
) -> EvaluationReport:
    """Validate model output; score labeled responses and retain per-case failures."""
    if not examples or len(examples) != len(responses):
        raise ValueError("one response or API failure is required for each example")
    if len({example.case_id for example in examples}) != len(examples):
        raise ValueError("case IDs must be unique")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")

    results = []
    for example, response in zip(examples, responses, strict=True):
        if isinstance(response, APIFailure):
            results.append(CaseResult(example, None, None, response))
            continue
        text = getattr(response, "output_text", None)
        usage = _read_usage(response)
        diagnostics = _read_diagnostics(response, usage, text, max_output_tokens)
        if diagnostics.completion_status == "failed":
            results.append(CaseResult(example, None, usage, APIFailure("ResponseFailed"), diagnostics=diagnostics))
            continue
        try:
            decision = parse_response_decision(response)
        except IncompleteResponseError:
            results.append(CaseResult(example, None, usage, diagnostics=diagnostics))
            continue
        except MissingOutputError:
            predicted = None
            invalid_reason: InvalidReason | None = "missing_output"
        except UnexpectedResponseFormatError:
            predicted = None
            invalid_reason = "unexpected_response_format"
        except json.JSONDecodeError:
            predicted = None
            invalid_reason = "malformed_json"
        except (ValidationError, ValueError):
            predicted = None
            invalid_reason = "schema_validation"
        else:
            predicted = decision.action
            invalid_reason = None
        results.append(CaseResult(example, predicted, usage, invalid_reason=invalid_reason, diagnostics=diagnostics))
    return EvaluationReport(tuple(results))
