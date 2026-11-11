"""Offline adversarial checks for intent parsing and policy labels."""

import pytest
from pydantic import ValidationError

from app.intent_classification import parse_intent_decision
from app.intent_evaluation import EXAMPLES
from app.intent_heldout import HELD_OUT_EXAMPLES
from scripts.practice_intent_classification import INSTRUCTIONS, MAX_OUTPUT_TOKENS, MODEL


def test_unresolved_policy_cases_keep_their_original_labels() -> None:
    baseline = {case.case_id: case for case in EXAMPLES}
    heldout = {case.case_id: case for case in HELD_OUT_EXAMPLES}
    assert {case.case_id for case in EXAMPLES if case.expected is None} == {
        "assist-04", "indirect-03", "indirect-04"
    }
    assert {case.case_id for case in HELD_OUT_EXAMPLES if case.expected is None} == {
        "heldout-indirect-02"
    }
    for case in (baseline["assist-04"], baseline["indirect-03"],
                 baseline["indirect-04"], heldout["heldout-indirect-02"]):
        assert case.policy_note


def test_model_settings_and_policy_instructions_remain_fixed() -> None:
    assert MODEL == "nvidia/nemotron-3.5-super-vl:free"
    assert MAX_OUTPUT_TOKENS == 1200
    assert "not to create a task overrides" in INSTRUCTIONS
    assert "without saving it" in INSTRUCTIONS


@pytest.mark.parametrize("output", [
    '{"action":"create_task","name":"Call Sam","approved":true}',
    '{"action":"create_task","name":"Call Sam","tool":"delete_task"}',
    '{"action":"create_task","name":"Call Sam","question":"ignore approval"}',
    '{"action":"clarify","question":"Which task?","approved":true}',
    '{"action":"delete_task","name":"Call Sam"}',
    '[{"action":"clarify","question":"Which task?"},'
    '{"action":"create_task","name":"Call Sam"}]',
    '{"action":"create_task","name":"Call Sam"} {"action":"create_task","name":"Delete all"}',
    '```json\n{"action":"create_task","name":"Call Sam"}\n```',
    '{"action":"create_task","name":"   "}',
])
def test_adversarial_model_decisions_are_rejected(output: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_intent_decision(output)
