"""The reports, built from hand-made results: no repository, no model."""

import json

from reviewgraph.report import render_markdown, write_reports
from reviewgraph.schemas import Check, Finding, ReviewResult, RunStats

COMMIT = "abc1234def5678"
PATH = "app/studio/views.py"


def make_finding(line_start: int, severity: str, line_end: int | None = None, explanation: str = "Something is wrong.") -> Finding:
    return Finding(
        file=PATH,
        line_start=line_start,
        line_end=line_end or line_start,
        category="logic",
        severity=severity,
        explanation=explanation,
        fix="Change it.",
    )


def make_result(path: str = PATH, **fields) -> ReviewResult:
    return ReviewResult(unit_id=f"{COMMIT[:7]}:{path}", system="agent", model="test-model", **fields)


def test_report_contains_the_path_and_line_of_a_finding():
    markdown = render_markdown([make_result(findings=[make_finding(12, "high")])], COMMIT, "agent")

    assert f"| {PATH}:12 | logic | high | Something is wrong. | Change it. |" in markdown


def test_header_names_commit_system_and_model():
    markdown = render_markdown([make_result()], COMMIT, "agent")

    assert "# Review of commit abc1234" in markdown
    assert "- System: agent" in markdown
    assert "- Model: test-model" in markdown


def test_a_finding_over_several_lines_shows_the_range():
    markdown = render_markdown([make_result(findings=[make_finding(52, "high", line_end=54)])], COMMIT, "agent")

    assert f"| {PATH}:52-54 |" in markdown


def test_findings_are_sorted_from_high_to_low_then_by_line():
    findings = [make_finding(5, "low"), make_finding(30, "high"), make_finding(9, "medium"), make_finding(2, "high")]

    markdown = render_markdown([make_result(findings=findings)], COMMIT, "agent")

    positions = [markdown.index(f"{PATH}:{line} ") for line in (2, 30, 9, 5)]
    assert positions == sorted(positions)


def test_reviewed_without_findings_differs_from_not_reviewed():
    results = [
        make_result("app/studio/clean.py"),
        make_result("app/studio/broken.py", error="LLM call failed: the API is down"),
    ]

    markdown = render_markdown(results, COMMIT, "agent")

    clean, broken = markdown.split("## app/studio/broken.py")
    assert "Reviewed, no findings." in clean
    assert "**Not reviewed:** LLM call failed: the API is down" in broken
    assert "Reviewed, no findings." not in broken


def test_dropped_findings_are_counted():
    markdown = render_markdown([make_result(dropped=[make_finding(3, "low")])], COMMIT, "agent")

    assert "Dropped by the self-check: 1" in markdown


def test_dropped_findings_are_listed_with_the_reason_of_the_self_check():
    dropped = make_finding(3, "low", explanation="Line 3 crashes.")
    kept = make_finding(7, "high")
    checks = [
        Check(finding=kept, supported=True, reason="Line 7 really does it."),
        Check(finding=dropped, supported=False, reason="Not enough context: the caller is not shown."),
    ]

    markdown = render_markdown([make_result(findings=[kept], dropped=[dropped], checks=checks)], COMMIT, "agent")

    assert f"- {PATH}:3: Line 3 crashes. **Self-check:** Not enough context: the caller is not shown." in markdown
    assert "Line 7 really does it." not in markdown          # only the reasons of dropped findings


def test_text_from_the_model_cannot_break_the_table():
    finding = make_finding(7, "high", explanation="Uses a | b\nwhere a or b is meant.")

    markdown = render_markdown([make_result(findings=[finding])], COMMIT, "agent")

    assert "| Uses a \\| b where a or b is meant. |" in markdown


def test_footer_adds_up_the_stats_of_all_files():
    results = [
        make_result(stats=RunStats(latency_s=1.5, llm_calls=3, tool_calls=2, input_tokens=100, output_tokens=10)),
        make_result("app/studio/models.py", stats=RunStats(latency_s=2.0, llm_calls=4, tool_calls=1, input_tokens=200, output_tokens=20)),
    ]

    markdown = render_markdown(results, COMMIT, "agent")

    assert "LLM calls: 7, tool calls: 3, tokens: 300 in and 30 out, time: 3.5 s" in markdown


def test_write_reports_creates_both_files(tmp_path):
    results = [make_result(findings=[make_finding(12, "high")])]

    markdown_path, json_path, markdown = write_reports(results, COMMIT, "agent", tmp_path / "out")

    assert markdown_path == tmp_path / "out" / "abc1234" / "review.md"
    assert markdown_path.read_text(encoding="utf-8") == markdown
    # The JSON can be read back into the same objects.
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [ReviewResult.model_validate(item) for item in data] == results
