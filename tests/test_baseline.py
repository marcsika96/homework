import pytest
from conftest import FakeModel, ScriptedChatModel, reply
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

from reviewgraph.baseline import review_baseline
from reviewgraph.graph import build_graph
from reviewgraph.llm import (
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_INTERVAL,
    RETRY_MAX_ATTEMPTS,
    is_transient_error,
)
from reviewgraph.prompts import REVIEW_SYSTEM_PROMPT
from reviewgraph.schemas import Finding, ReviewOutput

VIEWS = "app/studio/views.py"


def make_finding(line_start: int, line_end: int, file: str = VIEWS) -> Finding:
    return Finding(
        file=file,
        line_start=line_start,
        line_end=line_end,
        category="project-rule",
        severity="medium",
        explanation="The view reads from the database directly.",
        fix="Move the query into a database function and call that.",
    )


def test_baseline_returns_the_findings_of_one_call(tiny_repo):
    model = FakeModel([reply(ReviewOutput(findings=[make_finding(7, 7)]), tokens=(900, 150))])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, unit_id=f"I4:{VIEWS}", model=model)

    assert result.error is None
    assert result.system == "baseline"
    assert result.unit_id == f"I4:{VIEWS}"
    assert result.findings == [make_finding(7, 7)]
    assert (result.stats.llm_calls, result.stats.tool_calls) == (1, 0)
    assert (result.stats.input_tokens, result.stats.output_tokens) == (900, 150)
    assert result.stats.latency_s >= 0
    assert result.dropped == [] and result.trace == []


def test_baseline_sends_system_prompt_rules_and_numbered_diff(tiny_repo):
    model = FakeModel([reply(ReviewOutput(findings=[]))])

    review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    system_message, request = model.received[0]
    assert system_message.content == REVIEW_SYSTEM_PROMPT
    assert "Views never touch the database directly." in request.content
    assert "    5 + " in request.content and "def class_list(request):" in request.content
    # Only the reviewed file is sent, not the other files of the commit.
    assert "PositiveIntegerField" not in request.content


def test_baseline_repairs_lines_and_file_name(tiny_repo):
    model = FakeModel([reply(ReviewOutput(findings=[make_finding(40, 3, file="views.py")]))])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    finding = result.findings[0]
    assert finding.file == VIEWS
    assert (finding.line_start, finding.line_end) == (3, 7)      # views.py has 7 lines


def test_no_findings_is_a_valid_result(tiny_repo):
    model = FakeModel([reply(ReviewOutput(findings=[]))])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.findings == [] and result.error is None


def test_unknown_commit_is_reported_without_an_llm_call(tiny_repo):
    model = FakeModel([])

    result = review_baseline(tiny_repo.repo, "0123abc", VIEWS, model=model)

    assert "0123abc" in result.error
    assert result.stats.llm_calls == 0 and model.received == []


def test_file_not_changed_by_the_commit_is_reported_without_an_llm_call(tiny_repo):
    model = FakeModel([])

    result = review_baseline(tiny_repo.repo, tiny_repo.first_sha, VIEWS, model=model)

    assert VIEWS in result.error
    assert result.stats.llm_calls == 0 and model.received == []


def test_api_failure_is_reported_in_the_result(tiny_repo):
    model = FakeModel([ConnectionError("429 quota exceeded")])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "ConnectionError" in result.error and "429 quota exceeded" in result.error
    assert result.findings == []
    assert result.stats.llm_calls == 1


def test_invalid_output_twice_is_reported_and_both_calls_are_counted(tiny_repo):
    model = FakeModel(
        [reply(None, ValueError("not json"), tokens=(900, 10)), reply(None, ValueError("not json"), tokens=(950, 10))]
    )

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "ReviewOutput" in result.error
    assert result.findings == []
    assert (result.stats.llm_calls, result.stats.input_tokens) == (2, 1850)


# --- Retries: the same rule as the agent's RetryPolicy ----------------------


def overloaded() -> GoogleAPIError:
    return GoogleAPIError(code=503, response_json={"error": {"code": 503, "status": "UNAVAILABLE"}})


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """The seconds the baseline wanted to sleep, without sleeping."""
    slept = []
    # Only the baseline's own name: patching time.sleep would also catch the
    # sleeps of subprocess while git runs, which happen on Linux (Colab).
    monkeypatch.setattr("reviewgraph.baseline.sleep", slept.append)
    return slept


def test_server_error_is_retried(tiny_repo, waits):
    model = FakeModel([overloaded(), reply(ReviewOutput(findings=[make_finding(7, 7)]))])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error is None
    assert result.findings == [make_finding(7, 7)]
    assert result.stats.llm_calls == 2          # the failed attempt is counted
    assert waits == [2.0]


def test_server_error_on_every_attempt_ends_with_a_readable_error(tiny_repo, waits):
    model = FakeModel([overloaded()] * RETRY_MAX_ATTEMPTS)

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert result.error.startswith("LLM call failed: GoogleAPIError")
    assert result.stats.llm_calls == RETRY_MAX_ATTEMPTS == 4
    assert waits == [2.0, 4.0, 8.0]


def test_daily_quota_is_not_retried(tiny_repo, waits):
    quota = GoogleRateLimitError("429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
    model = FakeModel([quota])

    result = review_baseline(tiny_repo.repo, tiny_repo.second_sha, VIEWS, model=model)

    assert "GoogleRateLimitError" in result.error
    assert result.stats.llm_calls == 1 and waits == []


def test_agent_and_baseline_use_the_same_retry_rule(tiny_repo):
    graph = build_graph(ScriptedChatModel(), tiny_repo.repo, tiny_repo.second_sha)

    for node in ("lookup", "write_findings", "verify_finding"):
        policy = graph.builder.nodes[node].retry_policy
        assert (policy.max_attempts, policy.initial_interval, policy.backoff_factor) == (
            RETRY_MAX_ATTEMPTS,
            RETRY_INITIAL_INTERVAL,
            RETRY_BACKOFF_FACTOR,
        )
        assert policy.retry_on is is_transient_error
