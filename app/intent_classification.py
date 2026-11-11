"""Validate model intent decisions without executing any task tools."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.schemas import TaskProposal


class ClarifyDecision(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["clarify"]
    question: str = Field(min_length=1, max_length=300)


IntentDecision = Annotated[
    TaskProposal | ClarifyDecision,
    Field(discriminator="action"),
]

_decision_adapter = TypeAdapter(IntentDecision)


def parse_intent_decision(output_text: str) -> TaskProposal | ClarifyDecision:
    """Parse JSON, then validate its action-specific fields with Pydantic."""
    payload = json.loads(output_text)
    return _decision_adapter.validate_python(payload)


class IncompleteResponseError(ValueError):
    """The API marked the response incomplete; its text is not a decision."""


class UnexpectedResponseFormatError(ValueError):
    """The response has no completed text decision to validate."""


class MissingOutputError(ValueError):
    """A completed response has an empty text field."""


def parse_response_decision(response: object) -> TaskProposal | ClarifyDecision:
    """Check completion metadata before parsing and validating model text."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        raise IncompleteResponseError("model response is incomplete")
    if status != "completed":
        raise UnexpectedResponseFormatError("model response is not completed")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str):
        raise UnexpectedResponseFormatError("model response has no text field")
    if not output_text.strip():
        raise MissingOutputError("model response text is empty")
    return parse_intent_decision(output_text)
