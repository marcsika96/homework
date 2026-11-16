"""review_agent on the fixture repository with a scripted model: no key, no network."""

from pathlib import Path

import pytest
from conftest import REPLY_TOKENS, ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleAuthenticationError, GoogleRateLimitError

from reviewgraph.graph import build_graph, recursion_limit, review_agent
from reviewgraph.schemas import Finding, ReviewOutput, Verification

VIEWS = "app/studio/views.py"
MODELS = "app/studio/models.py"
ARCHITECTURE = Path(__file__).resolve().parent.parent / "docs" / "architecture.md"


def make_finding(line: int, severity: str = "medium") -> Finding:
    return Finding(
        file=VIEWS,
        line_start=line,
        line_end=line,
        category="project-rule",
        severity=severity,
        explanation=f"Line {line} reads from the database directly.",
        fix="Move the query into a database function and call that.",
    )


def no_tool_calls() -> AIMessage:
    return AIMessage(content="I have what I need.")


def verdict(supported_lines: set[int]):
    """A scripted self-check: supports the findings that start at one of these lines."""

    def reply(messages) -> Verification:
        request = messages[-1].content
        supported = any(f"- lines: {line}-{line}\n" in request for line in supported_lines)
        return Verification(supported=supported, reason="Scripted verdict.")

    return reply


def tool_messages(messages) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


@pytest.fixture
def no_waiting(monkeypatch):
    """The retry policy sleeps between attempts; the tests do not have to."""
    monkeypatch.setattr("langgraph.pregel._retry.time.sleep", lambda seconds: None)


def test_full_run_with_budget_fan_out_and_self_check(tiny_repo):
    three_calls = AIMessage(
        content="",
        tool_calls=[
            tool_call("read_file", "call-1", path=MODELS),
            tool_call("search_code", "call-2", pattern="DanceClass"),
            tool_call("list_files", "call-3", directory="app/studio"),
        ],
    )
    model = ScriptedChatModel(
        replies=[
            three_calls,
            ReviewOutput(findings=[make_finding(3, "low"), make_finding(7, "high")]),
            verdict({7}),
            verdict({7}),
        ]
    )

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, unit_id=f"I4:{VIEWS}", model=model, max_tool_calls=2)

    assert result.error is None
    assert (result.system, result.unit_id) == ("agent", f"I4:{VIEWS}")
    assert result.findings == [make_finding(7, "high")]
    assert result.dropped == [make_finding(3, "low")]
    # One look-up call, one for the findings, one self-check per draft finding.
    assert result.stats.llm_calls == 4
    assert (result.stats.input_tokens, result.stats.output_tokens) == (4 * REPLY_TOKENS[0], 4 * REPLY_TOKENS[1])
    assert result.stats.tool_calls == 2
    assert [entry["status"] for entry in result.trace] == ["executed", "executed", "budget_exceeded"]
    assert [entry["tool_name"] for entry in result.trace] == ["read_file", "search_code", "list_files"]
    assert result.stats.latency_s >= 0


def test_the_model_is_told_its_budget_and_gets_an_answer_for_every_tool_call(tiny_repo):
    three_calls = AIMessage(
        content="",
        tool_calls=[
            tool_call("read_file", "call-1", path=MODELS),
            tool_call("search_code", "call-2", pattern="DanceClass"),
            tool_call("list_files", "call-3", directory="app/studio"),
        ],
    )
    model = ScriptedChatModel(replies=[three_calls, ReviewOutput(findings=[])])

    review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model, max_tool_calls=2)

    first_call, findings_call = model.received
    assert "at most 2 tool calls" in first_call[0].content
    assert "Views never touch the database directly." in first_call[1].content
    answers = tool_messages(findings_call)
    assert [answer.tool_call_id for answer in answers] == ["call-1", "call-2", "call-3"]
    assert "PositiveIntegerField" in answers[0].content          # models.py was really read
    assert answers[2].content == "Tool budget used up."
    assert "look-up phase is over" in findings_call[-1].content


def test_unknown_tool_counts_and_is_answered_with_the_valid_names(tiny_repo):
    model = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[tool_call("open_file", "call-1", path=MODELS)]),
            no_tool_calls(),
            ReviewOutput(findings=[]),
        ]
    )

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error is None and result.findings == [] and result.dropped == []
    assert [entry["status"] for entry in result.trace] == ["unknown_tool"]
    assert result.stats.tool_calls == 1
    assert result.stats.llm_calls == 3
    answer = tool_messages(model.received[1])[0]
    for name in ("read_file", "search_code", "list_files"):
        assert name in answer.content


def test_no_findings_skips_the_self_check(tiny_repo):
    model = ScriptedChatModel(replies=[no_tool_calls(), ReviewOutput(findings=[])])

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error is None
    assert result.findings == [] and result.dropped == [] and result.trace == []
    assert (result.stats.llm_calls, result.stats.tool_calls) == (2, 0)


def test_findings_are_sorted_by_severity_then_line(tiny_repo):
    draft = [make_finding(6, "low"), make_finding(7, "high"), make_finding(5, "medium"), make_finding(2, "high")]
    model = ScriptedChatModel(
        replies=[no_tool_calls(), ReviewOutput(findings=draft)] + [verdict({2, 5, 6, 7})] * 4
    )

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert [(f.severity, f.line_start) for f in result.findings] == [("high", 2), ("high", 7), ("medium", 5), ("low", 6)]
    assert result.dropped == []


def test_invalid_self_check_keeps_the_finding(tiny_repo):
    not_json = lambda messages: AIMessage(content="I think it is fine.")
    model = ScriptedChatModel(
        replies=[no_tool_calls(), ReviewOutput(findings=[make_finding(7)]), not_json, not_json]
    )

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error is None
    assert result.findings == [make_finding(7)] and result.dropped == []
    assert result.stats.llm_calls == 4          # the self-check was tried twice


def test_unknown_commit_is_reported_without_an_llm_call(tiny_repo):
    model = ScriptedChatModel(replies=[])

    result = review_agent(tiny_repo.repo, "0123abc", VIEWS, model=model)

    assert "0123abc" in result.error
    assert result.stats.llm_calls == 0 and model.received == []


def test_file_not_changed_by_the_commit_is_reported_without_an_llm_call(tiny_repo):
    model = ScriptedChatModel(replies=[])

    result = review_agent(tiny_repo.repo, tiny_repo.first_sha, VIEWS, model=model)

    assert VIEWS in result.error
    assert result.stats.llm_calls == 0 and model.received == []


def test_daily_quota_is_reported_and_not_retried(tiny_repo):
    quota = GoogleRateLimitError("429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
    model = ScriptedChatModel(replies=[quota])

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "GoogleRateLimitError" in result.error and "RESOURCE_EXHAUSTED" in result.error
    assert result.findings == []
    assert result.stats.llm_calls == 1 and len(model.received) == 1


def test_server_error_is_retried(tiny_repo, no_waiting):
    overloaded = GoogleAPIError(code=503, response_json={"error": {"code": 503, "status": "UNAVAILABLE"}})
    model = ScriptedChatModel(replies=[overloaded, no_tool_calls(), ReviewOutput(findings=[])])

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error is None
    assert result.stats.llm_calls == 3          # the failed attempt is counted


def test_server_error_on_every_attempt_ends_with_a_readable_error(tiny_repo, no_waiting):
    overloaded = GoogleAPIError(code=503, response_json={"error": {"code": 503, "status": "UNAVAILABLE"}})
    model = ScriptedChatModel(replies=[overloaded] * 4)

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error.startswith("LLM call failed: GoogleAPIError")
    assert result.stats.llm_calls == 4          # max_attempts of the retry policy


def test_failure_after_the_look_up_keeps_trace_and_counts(tiny_repo):
    model = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[tool_call("read_file", "call-1", path=MODELS)]),
            GoogleAuthenticationError("UNAUTHENTICATED: API key not valid."),
        ]
    )

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "GoogleAuthenticationError" in result.error
    assert result.findings == []
    assert [entry["status"] for entry in result.trace] == ["executed"]
    assert (result.stats.llm_calls, result.stats.tool_calls) == (2, 1)


def test_invalid_findings_twice_is_reported_and_all_calls_are_counted(tiny_repo):
    not_json = AIMessage(content="Here are my thoughts.")
    model = ScriptedChatModel(replies=[no_tool_calls(), not_json, not_json.model_copy()])

    result = review_agent(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "ReviewOutput" in result.error
    assert result.findings == []
    assert result.stats.llm_calls == 3 and len(model.received) == 3


def test_recursion_limit_leaves_room_for_a_full_budget():
    # 8 rounds of lookup and run_tools, plus load_context, write_findings, verify_finding, finalize.
    assert recursion_limit(8) >= 2 * 8 + 4
    assert recursion_limit(0) >= 6


def test_architecture_diagram_matches_the_graph(tiny_repo):
    graph = build_graph(ScriptedChatModel(), tiny_repo.repo, tiny_repo.second_sha)

    assert graph.get_graph().draw_mermaid().strip() in ARCHITECTURE.read_text(encoding="utf-8"), (
        "docs/architecture.md is out of date: run python scripts/draw_graph.py"
    )
