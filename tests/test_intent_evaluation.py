import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx import ConnectError
from openai import APIConnectionError, APITimeoutError

from app.intent_evaluation import APIFailure, CATEGORIES, EXAMPLES, LabeledExample, evaluate_responses
from app.intent_heldout import HELD_OUT_EXAMPLES, HELD_OUT_RATIONALE
from app.repository import InMemoryTaskRepository
from scripts import practice_intent_evaluation as lesson


@pytest.fixture(autouse=True)
def block_network(monkeypatch) -> None:
    def deny_connection(*args, **kwargs) -> None:
        raise AssertionError("evaluation tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)


def model_response(
    text: str | None,
    usage: tuple[int, int, int] | None = None,
    *,
    status: str | None = "completed",
    incomplete_reason: str | None = None,
    reasoning_tokens: int | None = None,
    finish_reason: str | None = None,
):
    counts = (
        SimpleNamespace(
            input_tokens=usage[0], output_tokens=usage[1], total_tokens=usage[2],
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )
        if usage is not None
        else None
    )
    return SimpleNamespace(
        output_text=text, usage=counts, status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason),
        finish_reason=finish_reason, model="mock-model",
    )


def test_twenty_labeled_cases_cover_five_categories_with_documented_policy_gaps() -> None:
    assert len(EXAMPLES) == 23
    assert len({example.case_id for example in EXAMPLES}) == 23
    assert {example.category for example in EXAMPLES} == set(CATEGORIES)
    assert [sum(example.category == category for example in EXAMPLES) for category in CATEGORIES] == [4, 5, 4, 4, 6]
    assert sum(example.expected is not None for example in EXAMPLES) == 20
    assert sum(example.expected is None for example in EXAMPLES) == 3
    assert all(example.policy_note for example in EXAMPLES if example.expected is None)
    assert EXAMPLES[0].expected == "create_task"
    assert EXAMPLES[4].expected == "clarify"


def test_accuracy_invalid_outputs_and_available_usage_are_separate() -> None:
    responses = [
        model_response('{"action":"create_task","name":"Review notes"}', (10, 5, 15)),
        model_response('{"action":"clarify","question":"Which resume?"}'),
        model_response('{"action":"clarify","question":"Which notes?"}', (8, 4, 12)),
        model_response("not JSON", (3, 2, 5)),
    ]

    examples = [EXAMPLES[0], EXAMPLES[1], EXAMPLES[4], EXAMPLES[8]]
    report = evaluate_responses(examples, responses)

    assert [result.predicted for result in report.results] == [
        "create_task", "clarify", "clarify", None
    ]
    assert report.correct_count == 2
    assert report.incorrect_count == 1
    assert report.scored_count == 4
    assert report.accuracy == 0.5
    assert report.invalid_outputs == 1
    assert report.usage_available == 3
    assert report.token_totals.input_tokens == 21
    assert report.token_totals.output_tokens == 11
    assert report.token_totals.total_tokens == 32


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"action":"create_task","name":" "}',
        '{"action":"clarify","question":" "}',
        '{"action":"delete_task","name":"Review notes"}',
    ],
)
def test_invalid_model_output_counts_as_invalid_and_incorrect(text: str) -> None:
    report = evaluate_responses([EXAMPLES[0]], [model_response(text)])
    assert report.correct_count == 0
    assert report.accuracy == 0
    assert report.invalid_outputs == 1


def test_incomplete_usage_is_unavailable() -> None:
    response = SimpleNamespace(
        status="completed",
        output_text='{"action":"clarify","question":"Which notes?"}',
        usage=SimpleNamespace(input_tokens=5, output_tokens=None, total_tokens=8),
    )
    report = evaluate_responses([EXAMPLES[4]], [response])
    assert report.correct_count == 1
    assert report.usage_available == 0
    assert report.token_totals.total_tokens == 0


def test_one_response_per_example_is_required() -> None:
    with pytest.raises(ValueError):
        evaluate_responses(EXAMPLES, [])


def test_unresolved_policy_cases_are_observed_but_not_scored() -> None:
    examples = [EXAMPLES[7], EXAMPLES[16], EXAMPLES[18]]
    responses = [
        model_response('{"action":"create_task","name":"Explain traceback"}'),
        model_response('{"action":"create_task","name":"Send invoice"}'),
        model_response('{"action":"clarify","question":"Do you want a reminder?"}'),
    ]

    report = evaluate_responses(examples, responses)

    assert report.unresolved_policy_cases == 2
    assert report.scored_count == 1
    assert report.correct_count == 1
    assert report.incorrect_count == 0
    assert report.accuracy == 1
    assert [result.correct for result in report.results] == [None, True, None]
    assert report.category_summary("direct_assistance").accuracy is None
    assert report.category_summary("indirect_task_request").accuracy == 1


def test_api_failures_are_separate_from_invalid_outputs_and_score_denominator() -> None:
    examples = [EXAMPLES[0], EXAMPLES[1], EXAMPLES[4], EXAMPLES[8]]
    report = evaluate_responses(
        examples,
        [
            model_response('{"action":"create_task","name":"Review notes"}', (10, 5, 15)),
            model_response("not JSON", (4, 2, 6)),
            APIFailure("APIConnectionError"),
            model_response('{"action":"create_task","name":"Wrong intent"}'),
        ],
    )

    assert report.scored_count == 3
    assert report.correct_count == 1
    assert report.incorrect_count == 1
    assert report.invalid_outputs == 1
    assert report.api_failures == 1
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.category_summary("explicit_task_creation").accuracy == 0.5
    assert report.category_summary("direct_assistance").accuracy is None
    assert report.results[2].correct is None
    assert report.results[2].invalid_output is False
    assert report.token_totals.total_tokens == 21


def test_completed_malformed_and_schema_outputs_are_distinct() -> None:
    report = evaluate_responses(
        [EXAMPLES[0], EXAMPLES[1]],
        [
            model_response('{"action":"create_task"', status="completed"),
            model_response('{"action":"create_task","name":" "}', status="completed"),
        ],
    )
    assert [result.invalid_reason for result in report.results] == [
        "malformed_json", "schema_validation",
    ]
    assert report.scored_count == 2
    assert report.correct_count == 0
    assert report.invalid_outputs == 2
    assert report.incomplete_responses == 0


def test_unexpected_response_structure_is_not_a_json_failure() -> None:
    response = SimpleNamespace(status="completed", output_text={"action": "clarify"})
    report = evaluate_responses([EXAMPLES[4]], [response])
    assert report.results[0].invalid_reason == "unexpected_response_format"
    assert report.invalid_reasons["malformed_json"] == 0
    assert report.scored_count == 1

    missing_status = SimpleNamespace(
        output_text='{"action":"clarify","question":"Which notes?"}'
    )
    missing_report = evaluate_responses([EXAMPLES[4]], [missing_status])
    assert missing_report.results[0].invalid_reason == "unexpected_response_format"


def test_timeout_other_api_failure_and_incomplete_have_separate_denominators() -> None:
    report = evaluate_responses(
        [EXAMPLES[0], EXAMPLES[1], EXAMPLES[4], EXAMPLES[8], EXAMPLES[12]],
        [
            APIFailure("APITimeoutError"),
            APIFailure("APIConnectionError"),
            model_response("partial", (100, 1200, 1300), status="incomplete",
                           incomplete_reason="max_output_tokens"),
            model_response('{"action":"clarify","question":"What task?"}', status="completed"),
            model_response("not JSON", status="completed"),
        ],
        max_output_tokens=1200,
    )
    assert len(report.results) == 5
    assert report.api_timeouts == 1
    assert report.other_api_failures == 1
    assert report.api_failures == 2
    assert report.incomplete_responses == 1
    assert report.invalid_outputs == 1
    assert report.scored_count == 2
    assert report.correct_count == 1
    assert report.incorrect_count == 0
    assert report.accuracy == 0.5
    assert report.to_dict()["summary"]["api_timeouts"] == 1
    assert report.to_dict()["summary"]["incomplete_responses"] == 1


def test_held_out_cases_remain_separate_and_documented() -> None:
    assert len(HELD_OUT_EXAMPLES) == 10
    assert {case.case_id for case in HELD_OUT_EXAMPLES}.isdisjoint(
        {case.case_id for case in EXAMPLES}
    )
    assert {case.request for case in HELD_OUT_EXAMPLES}.isdisjoint(
        {case.request for case in EXAMPLES}
    )
    assert set(HELD_OUT_RATIONALE) == {case.case_id for case in HELD_OUT_EXAMPLES}
    assert all(HELD_OUT_RATIONALE.values())
    assert any(case.expected is None for case in HELD_OUT_EXAMPLES)

    responses = [
        model_response(
            '{"action":"create_task","name":"Example task"}'
            if case.expected == "create_task"
            else '{"action":"clarify","question":"Please clarify?"}',
            status="completed",
        )
        for case in HELD_OUT_EXAMPLES
    ]
    report = evaluate_responses(HELD_OUT_EXAMPLES, responses)
    assert report.scored_count == 9
    assert report.correct_count == 9
    assert report.unresolved_policy_cases == 1


def test_incomplete_response_is_not_parsed_as_malformed_json() -> None:
    response = model_response(
        '{"action":"clarify","question":"Which notes?',
        (150, 400, 550),
        status="incomplete",
        incomplete_reason="max_output_tokens",
        reasoning_tokens=380,
    )
    report = evaluate_responses([EXAMPLES[4]], [response], max_output_tokens=400)
    result = report.results[0]

    assert result.invalid_reason is None
    assert result.invalid_output is False
    assert result.incomplete_response is True
    assert result.correct is None
    assert result.diagnostics.completion_status == "incomplete"
    assert result.diagnostics.incomplete_reason == "max_output_tokens"
    assert result.diagnostics.at_output_limit is True
    assert result.diagnostics.truncation_confirmed is True
    assert result.diagnostics.output_text_chars == len(response.output_text)
    assert result.usage.reasoning_tokens == 380
    assert report.confirmed_truncations == 1
    assert report.incomplete_responses == 1
    assert report.scored_count == 0
    assert report.invalid_reasons["malformed_json"] == 0
    assert report.reasoning_usage_available == 1
    assert report.token_totals.reasoning_tokens == 380


def test_token_cap_alone_does_not_confirm_truncation_or_schema_cause() -> None:
    response = model_response(
        '{"action":"create_task","name":" "}',
        (100, 400, 500),
        status="completed",
        finish_reason="stop",
        reasoning_tokens=0,
    )
    report = evaluate_responses([EXAMPLES[0]], [response], max_output_tokens=400)
    result = report.results[0]

    assert result.invalid_reason == "schema_validation"
    assert result.diagnostics.at_output_limit is True
    assert result.diagnostics.truncation_confirmed is False
    assert result.diagnostics.finish_reason == "stop"
    assert report.invalid_reasons == {
        "missing_output": 0, "malformed_json": 0, "schema_validation": 1,
        "unexpected_response_format": 0,
    }
    assert report.confirmed_truncations == 0


def test_missing_output_with_reasoning_tokens_is_distinct_from_malformed_json() -> None:
    response = model_response(
        None, (100, 400, 500), status="incomplete",
        incomplete_reason="max_output_tokens", reasoning_tokens=400,
    )
    report = evaluate_responses([EXAMPLES[0]], [response], max_output_tokens=400)

    assert report.results[0].invalid_reason is None
    assert report.results[0].incomplete_response is True
    assert report.results[0].diagnostics.output_text_chars == 0
    assert report.invalid_reasons["malformed_json"] == 0


def test_failed_response_status_is_api_failure_not_invalid_output() -> None:
    response = model_response(None, (100, 0, 100), status="failed")
    report = evaluate_responses([EXAMPLES[0]], [response], max_output_tokens=400)

    assert report.api_failures == 1
    assert report.invalid_outputs == 0
    assert report.scored_count == 0
    assert report.accuracy is None
    assert report.results[0].api_error.error_type == "ResponseFailed"
    assert report.results[0].diagnostics.completion_status == "failed"


def test_invalid_reason_and_diagnostics_are_exported_without_raw_text() -> None:
    response = model_response(
        "not JSON", (100, 400, 500), status="incomplete",
        incomplete_reason="max_output_tokens", reasoning_tokens=330,
    )
    document = evaluate_responses(
        [EXAMPLES[0]], [response], max_output_tokens=400
    ).to_dict()

    assert document["summary"]["invalid_reasons"]["malformed_json"] == 0
    assert document["summary"]["incomplete_responses"] == 1
    assert document["summary"]["confirmed_truncations"] == 1
    assert document["summary"]["at_output_limit_count"] == 1
    assert document["results"][0]["invalid_reason"] is None
    assert document["results"][0]["incomplete_response"] is True
    assert document["results"][0]["diagnostics"]["incomplete_reason"] == "max_output_tokens"
    assert document["results"][0]["token_usage"]["reasoning_tokens"] == 330
    assert "not JSON" not in json.dumps(document)


def test_all_api_failures_have_no_accuracy() -> None:
    report = evaluate_responses([EXAMPLES[0]], [APIFailure("APIConnectionError")])
    assert report.accuracy is None
    assert report.scored_count == 0
    assert report.api_failures == 1


def test_export_preserves_case_details_without_raw_model_output() -> None:
    report = evaluate_responses(
        [EXAMPLES[0], EXAMPLES[7], EXAMPLES[8]],
        [
            model_response('{"action":"create_task","name":"Review notes"}', (10, 5, 15)),
            model_response('{"action":"clarify","question":"What do you mean?"}'),
            APIFailure("RateLimitError", 429),
        ],
    )
    document = report.to_dict()
    assert document["summary"]["accuracy"] == 1
    assert document["summary"]["unresolved_policy_cases"] == 1
    assert document["summary"]["api_failures"] == 1
    assert document["results"][0]["case_id"] == "explicit-01"
    assert document["results"][0]["token_usage"]["total_tokens"] == 15
    assert document["results"][1]["expected_intent"] is None
    assert document["results"][1]["policy_note"]
    assert document["results"][2]["api_error"] == {
        "error_type": "RateLimitError", "status_code": 429, "cause_type": None
    }
    assert all("output_text" not in result for result in document["results"])


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="case IDs"):
        evaluate_responses([EXAMPLES[0], EXAMPLES[0]], [model_response("{}"), model_response("{}")])


def test_unresolved_case_requires_policy_note() -> None:
    with pytest.raises(ValueError, match="policy note"):
        LabeledExample("x", "indirect_task_request", "Remember this", None)


def test_mocked_live_path_reports_results_without_executing_tools(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    repository = InMemoryTaskRepository()
    client = Mock()
    responses = [
        model_response(
            '{"action":"create_task","name":"Review notes"}'
            if example.expected == "create_task"
            else '{"action":"clarify","question":"Which details?"}',
            (10, 5, 15),
        )
        for example in EXAMPLES
    ]
    responses[1] = model_response('{"action":"clarify","question":"Which task?"}', (10, 5, 15))
    responses[5] = model_response("invalid output", (10, 5, 15))
    responses[10:11] = [
        APIConnectionError(request=Mock()), APIConnectionError(request=Mock())
    ]
    client.responses.create.side_effect = responses
    monkeypatch.setattr("scripts.openrouter_retry.time.sleep", lambda _delay: None)
    constructor = Mock(return_value=client)
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    output_path = tmp_path / "evaluation.json"
    assert lesson.run_evaluation(output_path) == 1
    assert client.responses.create.call_count == 24
    calls = client.responses.create.call_args_list
    calls = [*calls[:11], *calls[12:]]
    for example, call in zip(EXAMPLES, calls, strict=True):
        assert call.kwargs["model"] == "nvidia/nemotron-3.5-super-vl:free"
        assert call.kwargs["input"] == example.request
        assert call.kwargs["instructions"] == lesson.INSTRUCTIONS
        assert call.kwargs["max_output_tokens"] == 1200
        assert call.kwargs["store"] is False
    output = capsys.readouterr().out
    assert "Accuracy: 17/19 (89%)" in output
    assert "Correct predictions: 17" in output
    assert "Incorrect predictions: 1" in output
    assert "Invalid outputs: 1/23" in output
    assert "API failures total: 1/23" in output
    assert "Unresolved policy cases: 3/23" in output
    assert "Token usage available: 22/23; input=220, output=110, total=330" in output
    saved = json.loads(output_path.read_text())
    assert saved["summary"]["scored_cases"] == 19
    assert len(saved["results"]) == 23
    assert saved["run_config"]["max_output_tokens"] == 1200
    assert saved["run_config"]["max_api_retries"] == 1
    assert saved["run_config"]["dataset"] == "baseline"
    assert len(saved["run_config"]["case_ids"]) == 23
    assert saved["results"][10]["api_error"]["error_type"] == "APIConnectionError"
    assert saved["results"][5]["invalid_output"] is True
    assert "test-secret" not in output
    assert "test-secret" not in output_path.read_text()
    assert repository.list() == []


def test_selective_retry_uses_only_five_cases_and_new_limit(
    monkeypatch, tmp_path, capsys
) -> None:
    case_ids = ["assist-02", "assist-03", "assist-04", "missing-04", "negation-01"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = [
        model_response('{"action":"clarify","question":"Which notes?"}', (100, 35, 135))
        for _ in case_ids
    ]
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))
    output_path = tmp_path / "retry.json"

    assert lesson.run_evaluation(
        output_path, case_ids=case_ids, max_output_tokens=1200
    ) == 0

    assert client.responses.create.call_count == 5
    assert [call.kwargs["input"] for call in client.responses.create.call_args_list] == [
        example.request for example in EXAMPLES if example.case_id in set(case_ids)
    ]
    assert all(call.kwargs["max_output_tokens"] == 1200 for call in client.responses.create.call_args_list)
    saved = json.loads(output_path.read_text())
    assert saved["run_config"] == {
        "requested_model": "nvidia/nemotron-3.5-super-vl:free",
        "max_output_tokens": 1200,
        "case_ids": case_ids,
        "max_api_retries": 1,
        "dataset": "baseline",
    }
    assert saved["summary"]["total_cases"] == 5
    assert saved["summary"]["scored_cases"] == 4
    assert saved["summary"]["unresolved_policy_cases"] == 1
    assert "test-secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("case_ids", "max_output_tokens"),
    [
        (["unknown-01"], 1200),
        (["assist-02", "assist-02"], 1200),
        ([], 1200),
        (["assist-02"], 0),
        (["assist-02"], 4097),
    ],
)
def test_invalid_retry_options_never_create_client(
    monkeypatch, case_ids, max_output_tokens
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    assert lesson.run_evaluation(
        case_ids=case_ids, max_output_tokens=max_output_tokens
    ) == 2
    constructor.assert_not_called()


def test_existing_result_file_is_preserved_before_client_creation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    output_path = tmp_path / "baseline.json"
    output_path.write_text("baseline")

    assert lesson.run_evaluation(output_path, case_ids=["assist-02"]) == 2
    assert output_path.read_text() == "baseline"
    constructor.assert_not_called()


def test_missing_result_directory_is_rejected_before_client_creation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    assert lesson.run_evaluation(
        tmp_path / "missing" / "retry.json", case_ids=["assist-02"]
    ) == 2
    constructor.assert_not_called()


def test_cli_passes_case_selection_and_limit_only_with_run(monkeypatch) -> None:
    runner = Mock(return_value=0)
    monkeypatch.setattr(lesson, "run_evaluation", runner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "practice_intent_evaluation.py", "--run",
            "--case-id", "assist-02", "--case-id", "negation-01",
            "--max-output-tokens", "1200",
        ],
    )

    assert lesson.main() == 0
    runner.assert_called_once_with(
        None, case_ids=["assist-02", "negation-01"],
        max_output_tokens=1200, dataset="baseline",
    )


def test_heldout_dataset_uses_its_own_cases_and_export(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    client.responses.create.side_effect = [
        model_response(
            '{"action":"create_task","name":"Sample task"}'
            if example.expected == "create_task"
            else '{"action":"clarify","question":"Which details?"}',
            status="completed",
        )
        for example in HELD_OUT_EXAMPLES
    ]
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))
    result_path = tmp_path / "heldout.json"

    assert lesson.run_evaluation(result_path, dataset="heldout") == 0
    assert client.responses.create.call_count == len(HELD_OUT_EXAMPLES)
    assert [call.kwargs["input"] for call in client.responses.create.call_args_list] == [
        example.request for example in HELD_OUT_EXAMPLES
    ]
    saved = json.loads(result_path.read_text())
    assert saved["run_config"]["dataset"] == "heldout"
    assert saved["summary"]["total_cases"] == 10
    assert saved["summary"]["scored_cases"] == 9
    assert saved["summary"]["unresolved_policy_cases"] == 1
    assert "test-secret" not in capsys.readouterr().out


def test_cli_selects_heldout_only_with_run(monkeypatch) -> None:
    runner = Mock(return_value=0)
    monkeypatch.setattr(lesson, "run_evaluation", runner)
    monkeypatch.setattr(
        "sys.argv", ["practice_intent_evaluation.py", "--run", "--dataset", "heldout"]
    )
    assert lesson.main() == 0
    runner.assert_called_once_with(
        None, case_ids=None, max_output_tokens=1200, dataset="heldout"
    )


@pytest.mark.parametrize(
    "option",
    [
        ["--case-id", "assist-02"],
        ["--max-output-tokens", "1200"],
        ["--dataset", "heldout"],
    ],
)
def test_retry_options_without_run_never_create_client(
    monkeypatch, option
) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_intent_evaluation.py", *option])

    with pytest.raises(SystemExit) as error:
        lesson.main()
    assert error.value.code == 2
    constructor.assert_not_called()


def test_without_run_no_client_or_network(monkeypatch, capsys) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    monkeypatch.setattr("sys.argv", ["practice_intent_evaluation.py"])

    assert lesson.main() == 0
    constructor.assert_not_called()
    assert "--run" in capsys.readouterr().out


def test_output_json_requires_run_without_creating_client(monkeypatch, tmp_path) -> None:
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)
    output_path = tmp_path / "evaluation.json"
    monkeypatch.setattr("sys.argv", ["practice_intent_evaluation.py", "--output-json", str(output_path)])

    with pytest.raises(SystemExit) as error:
        lesson.main()
    assert error.value.code == 2
    constructor.assert_not_called()
    assert not output_path.exists()


def test_missing_key_never_creates_client(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(lesson, "ENV_FILE", tmp_path / ".env")
    constructor = Mock(side_effect=AssertionError("unexpected API client"))
    monkeypatch.setattr(lesson, "OpenAI", constructor)

    assert lesson.run_evaluation() == 2
    constructor.assert_not_called()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_api_error_is_recorded_and_remaining_cases_are_evaluated(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    client = Mock()
    connection_error = APIConnectionError(request=Mock())
    connection_error.__cause__ = ConnectError("secret transport detail")
    client.responses.create.side_effect = [
        connection_error, connection_error,
        *[
            model_response(
                '{"action":"create_task","name":"Task"}'
                if example.expected == "create_task"
                else '{"action":"clarify","question":"Which details?"}'
            )
            for example in EXAMPLES[1:]
        ],
    ]
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))
    monkeypatch.setattr("scripts.openrouter_retry.time.sleep", lambda _delay: None)

    assert lesson.run_evaluation() == 1
    assert client.responses.create.call_count == 24
    captured = capsys.readouterr()
    assert "API failures total: 1/23" in captured.out
    assert "Accuracy: 19/19 (100%)" in captured.out
    assert "cause=ConnectError" in captured.out
    assert "secret transport detail" not in captured.out
    assert "test-secret" not in captured.err


def test_run_evaluation_records_timeout_after_bounded_retry(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr("scripts.openrouter_retry.time.sleep", lambda _delay: None)
    client = Mock()
    client.responses.create.side_effect = APITimeoutError(request=Mock())
    monkeypatch.setattr(lesson, "OpenAI", Mock(return_value=client))
    result_path = tmp_path / "timeout.json"

    assert lesson.run_evaluation(result_path, case_ids=["explicit-01"]) == 1
    assert client.responses.create.call_count == 2
    saved = json.loads(result_path.read_text())
    assert saved["summary"]["total_cases"] == 1
    assert saved["summary"]["scored_cases"] == 0
    assert saved["summary"]["api_timeouts"] == 1
    assert saved["summary"]["other_api_failures"] == 0
    assert saved["summary"]["incorrect_predictions"] == 0
    assert saved["results"][0]["api_error"]["error_type"] == "APITimeoutError"
    assert "API timeouts: 1/1" in capsys.readouterr().out
