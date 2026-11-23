"""The resumable runner on the fixture repository with a scripted model: no key, no network."""

import pytest
from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage

from reviewgraph import cli
from reviewgraph.evaluation import runner
from reviewgraph.evaluation.runner import load_record, load_records, record_path, run_units
from reviewgraph.schemas import Finding, ReviewOutput, Unit

MODELS = "app/studio/models.py"
VIEWS = "app/studio/views.py"
QUOTA_ERROR = "429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"


def make_units(tiny_repo, split: str = "dev") -> list[Unit]:
    """Two units: the two Python files of the second commit."""
    return [
        Unit(id=f"I1:{path}", increment="I1", commit=tiny_repo.second_sha, path=path, added_lines=20, split=split)
        for path in (MODELS, VIEWS)
    ]


def one_finding() -> ReviewOutput:
    finding = Finding(file=VIEWS, line_start=7, line_end=7, category="project-rule", severity="high",
                      explanation="The view reads from the database directly.", fix="Use a database function.")
    return ReviewOutput(findings=[finding])


def test_file_name_of_a_unit(tmp_path):
    path = record_path(tmp_path, "agent", "I5:app/studio/usecases/booking.py")

    assert path == tmp_path / "agent" / "I5_app_studio_usecases_booking.py.json"


def test_every_unit_gets_a_file_with_the_result_and_where_it_came_from(tiny_repo, tmp_path):
    units = make_units(tiny_repo)
    model = ScriptedChatModel(replies=[ReviewOutput(findings=[]), one_finding()])

    summary = run_units(units, "baseline", tiny_repo.repo, model, runs_dir=tmp_path)

    assert summary == {"reviewed": 2, "skipped": 0, "quota_used_up": False, "overloaded": False}
    record = load_record(record_path(tmp_path, "baseline", units[1].id))
    assert record.result.unit_id == f"I1:{VIEWS}"
    assert record.result.findings[0].line_start == 7
    assert record.prompt_version == runner.PROMPT_VERSION
    assert record.max_tool_calls is None            # the baseline has no tool budget
    assert record.code_version and record.timestamp


def test_the_agent_record_holds_the_tool_budget(tiny_repo, tmp_path):
    units = make_units(tiny_repo)[:1]
    model = ScriptedChatModel(replies=[AIMessage(content="Nothing to look up."), ReviewOutput(findings=[])])

    run_units(units, "agent", tiny_repo.repo, model, runs_dir=tmp_path)

    record = load_record(record_path(tmp_path, "agent", units[0].id))
    assert record.result.system == "agent"
    assert record.max_tool_calls > 0


def test_a_second_run_skips_the_stored_units(tiny_repo, tmp_path):
    units = make_units(tiny_repo)
    run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[ReviewOutput(findings=[]), one_finding()]), runs_dir=tmp_path)
    second_model = ScriptedChatModel(replies=[])

    summary = run_units(units, "baseline", tiny_repo.repo, second_model, runs_dir=tmp_path)

    assert summary == {"reviewed": 0, "skipped": 2, "quota_used_up": False, "overloaded": False}
    assert second_model.received == []              # the model was not called


def test_an_interrupted_run_continues_where_it_stopped(tiny_repo, tmp_path):
    def ctrl_c(messages):
        raise KeyboardInterrupt

    units = make_units(tiny_repo)
    with pytest.raises(KeyboardInterrupt):
        run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[ReviewOutput(findings=[]), ctrl_c]), runs_dir=tmp_path)
    assert len(load_records(units, "baseline", runs_dir=tmp_path)) == 1     # the first unit was stored

    summary = run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[one_finding()]), runs_dir=tmp_path)

    assert summary == {"reviewed": 1, "skipped": 1, "quota_used_up": False, "overloaded": False}
    assert len(load_records(units, "baseline", runs_dir=tmp_path)) == 2


def test_a_failed_review_is_stored_and_not_repeated(tiny_repo, tmp_path):
    units = make_units(tiny_repo)[:1]
    run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[ValueError("the API is down")]), runs_dir=tmp_path)

    record = load_record(record_path(tmp_path, "baseline", units[0].id))
    assert "the API is down" in record.result.error
    assert record.result.findings == []

    summary = run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[]), runs_dir=tmp_path)
    assert summary["skipped"] == 1


def test_the_daily_quota_stops_the_run_and_stores_nothing_for_that_unit(tiny_repo, tmp_path):
    units = make_units(tiny_repo)
    model = ScriptedChatModel(replies=[ReviewOutput(findings=[]), ValueError(QUOTA_ERROR)])

    summary = run_units(units, "baseline", tiny_repo.repo, model, runs_dir=tmp_path)

    assert summary == {"reviewed": 1, "skipped": 0, "quota_used_up": True, "overloaded": False}
    assert not record_path(tmp_path, "baseline", units[1].id).exists()


@pytest.mark.parametrize("error", [
    "503 UNAVAILABLE. This model is currently experiencing high demand.",
    "504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.",
])
def test_an_overloaded_model_stops_the_run_and_stores_nothing_for_that_unit(tiny_repo, tmp_path, error):
    # The errors of 2026-11-18. A ValueError is not retried, so the scripted
    # model needs one reply per unit; in a real run this is what is left after the retries.
    units = make_units(tiny_repo)
    model = ScriptedChatModel(replies=[ReviewOutput(findings=[]), ValueError(error)])

    summary = run_units(units, "baseline", tiny_repo.repo, model, runs_dir=tmp_path)

    assert summary == {"reviewed": 1, "skipped": 0, "quota_used_up": False, "overloaded": True}
    assert not record_path(tmp_path, "baseline", units[1].id).exists()


def test_the_eval_command_says_when_the_model_is_overloaded(tiny_repo, tmp_path, monkeypatch, capsys):
    units_file = write_units_file(tmp_path, make_units(tiny_repo)[:1])
    model = ScriptedChatModel(replies=[ValueError("503 UNAVAILABLE. high demand")])
    monkeypatch.setattr(cli, "get_chat_model", lambda: model)
    monkeypatch.setattr(cli, "check_ollama", lambda needs_tools: None)
    monkeypatch.setattr(cli, "get_settings", lambda: type("Settings", (), {"target_repo_path": str(tiny_repo.path)})())

    exit_code = cli.main(["eval", "--units", str(units_file), "--system", "baseline", "--runs", str(tmp_path / "runs")])

    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert "Stopped: the model is overloaded right now." in capsys.readouterr().err


def test_a_record_of_another_prompt_version_is_run_again(tiny_repo, tmp_path, monkeypatch):
    units = make_units(tiny_repo)[:1]
    run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[ReviewOutput(findings=[])]), runs_dir=tmp_path)
    monkeypatch.setattr(runner, "PROMPT_VERSION", "next")

    summary = run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[one_finding()]), runs_dir=tmp_path)

    assert summary["reviewed"] == 1
    assert load_record(record_path(tmp_path, "baseline", units[0].id)).prompt_version == "next"


# --- The eval command ---------------------------------------------------------


def write_units_file(tmp_path, units: list[Unit]):
    path = tmp_path / "units.jsonl"
    path.write_text("\n".join(unit.model_dump_json() for unit in units) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def no_model(monkeypatch):
    """Fails the test if the command line creates a model."""

    def fail():
        raise AssertionError("A model was created.")

    monkeypatch.setattr(cli, "get_chat_model", fail)


def test_eval_refuses_the_test_set_without_its_flag(tiny_repo, tmp_path, no_model, capsys):
    units_file = write_units_file(tmp_path, make_units(tiny_repo, split="test"))

    exit_code = cli.main(["eval", "--units", str(units_file), "--runs", str(tmp_path / "runs")])

    assert exit_code == cli.EXIT_BAD_INPUT
    assert "--test-set" in capsys.readouterr().err


def test_eval_cached_shows_the_stored_runs_without_a_model(tiny_repo, tmp_path, no_model, capsys):
    units = make_units(tiny_repo)
    runs = tmp_path / "runs"
    run_units(units, "baseline", tiny_repo.repo, ScriptedChatModel(replies=[ReviewOutput(findings=[]), one_finding()]), runs_dir=runs)
    capsys.readouterr()

    exit_code = cli.main(["eval", "--units", str(write_units_file(tmp_path, units)), "--runs", str(runs), "--cached"])

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "baseline: 2 of 2 units stored, 1 findings, 0 failed, 2 LLM calls" in output
    assert "agent: 0 of 2 units stored" in output


def test_eval_runs_the_chosen_system_on_the_first_units(tiny_repo, tmp_path, monkeypatch, capsys):
    units = make_units(tiny_repo)
    model = ScriptedChatModel(replies=[ReviewOutput(findings=[])])
    monkeypatch.setattr(cli, "get_chat_model", lambda: model)
    monkeypatch.setattr(cli, "check_ollama", lambda needs_tools: None)
    monkeypatch.setattr(cli, "get_settings", lambda: type("Settings", (), {"target_repo_path": str(tiny_repo.path)})())

    exit_code = cli.main(["eval", "--units", str(write_units_file(tmp_path, units)), "--runs", str(tmp_path / "runs"),
                          "--system", "baseline", "--limit", "1"])

    assert exit_code == cli.EXIT_OK
    assert "baseline: 1 of 1 units stored" in capsys.readouterr().out
