"""The judge on the fixture repository with a scripted model: no key, no network."""

import httpx
from conftest import ScriptedChatModel

from reviewgraph import cli
from reviewgraph.evaluation import judge
from reviewgraph.evaluation.judge import (
    JudgeVerdict,
    finding_id,
    findings_to_judge,
    judge_finding,
    judge_runs,
    load_verdicts,
    select_for_judging,
)
from reviewgraph.evaluation.runner import RunRecord, record_path, save_record
from reviewgraph.schemas import Finding, ReviewResult, Unit

VIEWS = "app/studio/views.py"
QUOTA_ERROR = "429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"


def make_finding(line: int, severity: str = "high") -> Finding:
    return Finding(file=VIEWS, line_start=line, line_end=line, category="project-rule", severity=severity,
                   explanation=f"Line {line} reads from the database directly.", fix="Use a database function.")


def make_unit(tiny_repo) -> Unit:
    return Unit(id=f"I1:{VIEWS}", increment="I1", commit=tiny_repo.second_sha, path=VIEWS, added_lines=20, split="dev")


def store_run(runs_dir, unit: Unit, system: str, findings: list[Finding]) -> None:
    """Put a finished review into the runs folder, as runner.py would."""
    result = ReviewResult(unit_id=unit.id, system=system, model="test-model", findings=findings)
    record = RunRecord(result=result, prompt_version="1", max_tool_calls=None, code_version="abc", timestamp="2026-11-16T00:00:00+00:00")
    save_record(record_path(runs_dir, system, unit.id), record)


def valid(reason: str = "Line 7 calls objects.all() in a view.") -> JudgeVerdict:
    return JudgeVerdict(label="valid", reason=reason)


# --- Choosing and naming findings --------------------------------------------


def test_only_medium_and_high_are_chosen_high_first_then_reporting_order():
    findings = [make_finding(1, "medium"), make_finding(2, "low"), make_finding(3, "high"),
                make_finding(4, "medium"), make_finding(5, "high")]
    result = ReviewResult(unit_id="u", system="agent", model="m", findings=findings)

    chosen = select_for_judging(result)

    assert [finding.line_start for finding in chosen] == [3, 5, 1]       # at most three


def test_finding_id_is_stable_and_differs_between_systems():
    finding = make_finding(7)

    assert finding_id("I1:a.py", "agent", finding) == finding_id("I1:a.py", "agent", make_finding(7))
    assert finding_id("I1:a.py", "agent", finding) != finding_id("I1:a.py", "baseline", finding)
    assert len(finding_id("I1:a.py", "agent", finding)) == 12


def test_the_order_of_judging_is_shuffled_but_the_same_every_time(tiny_repo, tmp_path):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "baseline", [make_finding(line) for line in (1, 2, 3)])
    store_run(tmp_path, unit, "agent", [make_finding(line) for line in (4, 5, 6)])

    first = findings_to_judge([unit], ["baseline", "agent"], tmp_path)
    second = findings_to_judge([unit], ["baseline", "agent"], tmp_path)

    assert first == second
    assert len(first) == 6
    systems_in_order = [system for _, system, _ in first]
    assert systems_in_order != ["baseline"] * 3 + ["agent"] * 3      # mixed


# --- One judgement ------------------------------------------------------------


def test_the_judge_sees_finding_diff_file_and_rules_but_not_the_system(tiny_repo):
    model = ScriptedChatModel(replies=[valid()])

    verdict, raw_messages = judge_finding(model, tiny_repo.repo, make_unit(tiny_repo), make_finding(7))

    assert verdict.label == "valid"
    assert len(raw_messages) == 1
    request = model.received[0][-1].content
    assert "- lines: 7-7" in request                        # the finding
    assert "+ def class_list(request):" in request          # the diff
    assert "    6   def class_list(request):" in request    # the whole file, numbered
    assert "Views never touch the database directly." in request     # the rules
    assert "baseline" not in request and "agent" not in request


def test_a_transient_error_is_retried(tiny_repo, monkeypatch):
    monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)
    model = ScriptedChatModel(replies=[httpx.ConnectError("no connection"), valid()])

    verdict, _ = judge_finding(model, tiny_repo.repo, make_unit(tiny_repo), make_finding(7))

    assert verdict.label == "valid"
    assert len(model.received) == 2


# --- Judging the stored runs --------------------------------------------------


def test_verdicts_are_written_one_line_per_finding(tiny_repo, tmp_path):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "baseline", [make_finding(7), make_finding(3, "low")])
    store_run(tmp_path, unit, "agent", [make_finding(7)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    model = ScriptedChatModel(replies=[valid(), JudgeVerdict(label="invalid", reason="Line 7 does not do that.")])

    summary = judge_runs([unit], ["baseline", "agent"], tiny_repo.repo, model, runs_dir=tmp_path, verdicts_path=verdicts_path)

    assert summary == {"judged": 2, "skipped": 0, "failed": 0, "quota_used_up": False}      # the low finding is not judged
    assert len(verdicts_path.read_text(encoding="utf-8").splitlines()) == 2
    verdicts = load_verdicts(verdicts_path)
    record = verdicts[finding_id(unit.id, "agent", make_finding(7))]
    assert (record.unit_id, record.system, record.finding) == (unit.id, "agent", make_finding(7))
    assert record.input_tokens > 0 and record.judge_prompt_version == judge.JUDGE_PROMPT_VERSION


def test_a_second_run_skips_judged_findings(tiny_repo, tmp_path):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "agent", [make_finding(7)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    judge_runs([unit], ["agent"], tiny_repo.repo, ScriptedChatModel(replies=[valid()]), runs_dir=tmp_path, verdicts_path=verdicts_path)
    second_model = ScriptedChatModel(replies=[])

    summary = judge_runs([unit], ["agent"], tiny_repo.repo, second_model, runs_dir=tmp_path, verdicts_path=verdicts_path)

    assert summary == {"judged": 0, "skipped": 1, "failed": 0, "quota_used_up": False}
    assert second_model.received == []


def test_a_failed_judgement_is_not_stored_and_the_others_continue(tiny_repo, tmp_path):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "agent", [make_finding(7), make_finding(8)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    model = ScriptedChatModel(replies=[ValueError("the API is down"), valid()])

    summary = judge_runs([unit], ["agent"], tiny_repo.repo, model, runs_dir=tmp_path, verdicts_path=verdicts_path)

    assert summary == {"judged": 1, "skipped": 0, "failed": 1, "quota_used_up": False}
    assert len(load_verdicts(verdicts_path)) == 1


def test_the_daily_quota_stops_the_judging(tiny_repo, tmp_path):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "agent", [make_finding(7), make_finding(8)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    model = ScriptedChatModel(replies=[valid(), ValueError(QUOTA_ERROR)])

    summary = judge_runs([unit], ["agent"], tiny_repo.repo, model, runs_dir=tmp_path, verdicts_path=verdicts_path)

    assert summary == {"judged": 1, "skipped": 0, "failed": 0, "quota_used_up": True}


def test_a_verdict_of_another_judge_prompt_is_judged_again_and_the_new_one_counts(tiny_repo, tmp_path, monkeypatch):
    unit = make_unit(tiny_repo)
    store_run(tmp_path, unit, "agent", [make_finding(7)])
    verdicts_path = tmp_path / "verdicts.jsonl"
    judge_runs([unit], ["agent"], tiny_repo.repo, ScriptedChatModel(replies=[valid()]), runs_dir=tmp_path, verdicts_path=verdicts_path)
    monkeypatch.setattr(judge, "JUDGE_PROMPT_VERSION", "next")

    new_verdict = JudgeVerdict(label="minor", reason="True, but only style.")
    judge_runs([unit], ["agent"], tiny_repo.repo, ScriptedChatModel(replies=[new_verdict]), runs_dir=tmp_path, verdicts_path=verdicts_path)

    assert len(verdicts_path.read_text(encoding="utf-8").splitlines()) == 2      # the old line is kept
    assert load_verdicts(verdicts_path)[finding_id(unit.id, "agent", make_finding(7))].verdict.label == "minor"


# --- The eval command ---------------------------------------------------------


def test_eval_judge_rates_the_stored_findings_and_reviews_nothing(tiny_repo, tmp_path, monkeypatch, capsys):
    unit = make_unit(tiny_repo)
    runs = tmp_path / "runs"
    store_run(runs, unit, "baseline", [make_finding(7)])
    units_file = tmp_path / "units.jsonl"
    units_file.write_text(unit.model_dump_json() + "\n", encoding="utf-8")
    model = ScriptedChatModel(replies=[valid()])        # one reply: a review would need more
    monkeypatch.setattr(cli, "get_chat_model", lambda: model)
    monkeypatch.setattr(cli, "check_ollama", lambda needs_tools=False: None)
    monkeypatch.setattr(cli, "get_settings", lambda: type("Settings", (), {"target_repo_path": str(tiny_repo.path)})())

    exit_code = cli.main(["eval", "--judge", "--units", str(units_file), "--runs", str(runs),
                          "--verdicts", str(tmp_path / "verdicts.jsonl")])

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "judge: 1 of 1 selected findings have a verdict" in output
    assert "judge: 0 of 0 selected findings have a verdict" in output       # the agent has no stored run
    assert "valid" not in output                                            # the labels are not shown
