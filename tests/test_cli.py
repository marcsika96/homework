"""The review command on the fixture repository with a scripted model: no key, no network."""

import json
from types import SimpleNamespace

import ollama
import pytest
from conftest import ScriptedChatModel, git, tool_call, write
from langchain_core.messages import AIMessage

from reviewgraph import cli, llm
from reviewgraph.config import get_settings
from reviewgraph.schemas import Finding, ReviewOutput, Verification

MODELS = "app/studio/models.py"
VIEWS = "app/studio/views.py"


def finding_in_views() -> Finding:
    return Finding(
        file=VIEWS,
        line_start=7,
        line_end=7,
        category="project-rule",
        severity="high",
        explanation="The view reads from the database directly.",
        fix="Move the query into a database function.",
    )


def use_model(monkeypatch, replies: list) -> ScriptedChatModel:
    """Make the command line use a scripted model instead of Gemini or Ollama."""
    model = ScriptedChatModel(replies=replies)
    monkeypatch.setattr(cli, "get_chat_model", lambda: model)
    # The scripted model needs no Ollama, whatever LLM_PROVIDER is in the local .env.
    monkeypatch.setattr(cli, "check_ollama", lambda needs_tools: None)
    return model


@pytest.fixture
def no_model(monkeypatch):
    """Fails the test if the command line creates a model: the proof for 'no LLM call'."""

    def fail():
        raise AssertionError("A model was created.")

    monkeypatch.setattr(cli, "get_chat_model", fail)


def review_args(tiny_repo, *extra: str) -> list[str]:
    # The reports go next to the fixture repository, not into the project's out/ folder.
    return ["review", "--repo", str(tiny_repo.path), "--commit", tiny_repo.second_sha,
            "--out", str(out_folder(tiny_repo)), *extra]


def out_folder(tiny_repo):
    return tiny_repo.path.parent / "out"


def test_parser_defaults():
    args = cli.build_parser().parse_args(["review", "--repo", "some/path", "--commit", "abc1234"])

    assert (args.command, args.system, args.out, args.verbose) == ("review", "agent", "out", False)


def test_agent_reviews_every_reviewable_file_of_the_commit(tiny_repo, monkeypatch, capsys):
    # The second commit changes models.py, views.py and logo.png. Per Python file:
    # one look-up answer, then the findings; views.py also gets one self-check.
    model = use_model(monkeypatch, [
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[]),
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[finding_in_views()]),
        Verification(supported=True, reason="The view calls objects.all()."),
    ])

    exit_code = cli.main(review_args(tiny_repo))

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert model.replies == []          # every prepared reply was used
    assert f"[1/2] {MODELS}" in output
    assert f"[2/2] {VIEWS}" in output
    assert "logo.png" not in output
    assert f"| {VIEWS}:7 | project-rule | high |" in output        # the report is printed


def test_review_writes_both_reports(tiny_repo, monkeypatch):
    use_model(monkeypatch, [ReviewOutput(findings=[]), ReviewOutput(findings=[finding_in_views()])])

    cli.main(review_args(tiny_repo, "--system", "baseline"))

    folder = out_folder(tiny_repo) / tiny_repo.second_sha[:7]
    assert f"{VIEWS}:7" in (folder / "review.md").read_text(encoding="utf-8")
    results = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    assert [result["system"] for result in results] == ["baseline", "baseline"]
    assert results[1]["findings"][0]["line_start"] == 7


def test_baseline_makes_one_call_per_file(tiny_repo, monkeypatch, capsys):
    model = use_model(monkeypatch, [ReviewOutput(findings=[]), ReviewOutput(findings=[finding_in_views()])])

    exit_code = cli.main(review_args(tiny_repo, "--system", "baseline"))

    assert exit_code == cli.EXIT_OK
    assert len(model.received) == 2
    assert f"| {VIEWS}:7 |" in capsys.readouterr().out


def test_a_failed_file_does_not_stop_the_other_files(tiny_repo, monkeypatch, capsys):
    model = use_model(monkeypatch, [
        ValueError("the API is down"),      # models.py fails in its first model call
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[]),
    ])

    exit_code = cli.main(review_args(tiny_repo))

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert model.replies == []          # views.py was still reviewed
    assert "FAILED: LLM call failed: ValueError: the API is down" in captured.out
    assert "1 of 2 files could not be reviewed." in captured.err


def test_verbose_prints_the_graph_steps_and_tool_calls(tiny_repo, monkeypatch, capsys):
    asks_for_a_file = AIMessage(content="", tool_calls=[tool_call("read_file", "call-1", path=MODELS)])
    use_model(monkeypatch, [
        asks_for_a_file,
        AIMessage(content="Nothing more to look up."),
        ReviewOutput(findings=[]),
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[]),
    ])

    exit_code = cli.main(review_args(tiny_repo, "--verbose"))

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    for node in ("load_context", "lookup", "run_tools", "write_findings", "finalize"):
        assert f"[{node}]" in output
    assert f'asks for read_file({{"path": "{MODELS}"}})' in output
    assert "executed: read_file" in output


def test_without_verbose_no_graph_step_is_printed(tiny_repo, monkeypatch, capsys):
    use_model(monkeypatch, [
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[]),
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[]),
    ])

    cli.main(review_args(tiny_repo))

    assert "[lookup]" not in capsys.readouterr().out


def test_commit_without_python_changes_has_nothing_to_review(tiny_repo, no_model, capsys):
    write(tiny_repo.path / "AGENTS.md", "# Project rules\n\n- A new rule.\n")
    git(tiny_repo.path, "commit", "--quiet", "-am", "rules only")
    sha = git(tiny_repo.path, "rev-parse", "HEAD")

    exit_code = cli.main(["review", "--repo", str(tiny_repo.path), "--commit", sha])

    assert exit_code == cli.EXIT_OK
    assert "Nothing to review" in capsys.readouterr().out


def test_deleted_file_is_not_reviewed(tiny_repo, no_model, capsys):
    git(tiny_repo.path, "rm", "--quiet", VIEWS)
    git(tiny_repo.path, "commit", "--quiet", "-m", "remove the views")
    sha = git(tiny_repo.path, "rev-parse", "HEAD")

    exit_code = cli.main(["review", "--repo", str(tiny_repo.path), "--commit", sha])

    assert exit_code == cli.EXIT_OK
    assert "Nothing to review" in capsys.readouterr().out


# --- Exercise 5.3: the error cases -------------------------------------------
# The rows of the table in the plan, in its order. `no_model` proves that a
# case is answered without an LLM call.


@pytest.fixture
def environment(monkeypatch):
    """Lets a test set environment variables: get_settings() is cached, so the cache is emptied."""
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.fixture
def no_model_classes(monkeypatch):
    """The real get_chat_model() runs, but building a Gemini model fails the test."""

    def fail(*args, **kwargs):
        raise AssertionError("A model was created.")

    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", fail)


def test_malformed_commit(tiny_repo, no_model, capsys):
    exit_code = cli.main(["review", "--repo", str(tiny_repo.path), "--commit", "HEAD~1"])

    assert exit_code == cli.EXIT_BAD_INPUT
    assert capsys.readouterr().err == "Error: Not a commit SHA: 'HEAD~1'\n"


def test_missing_key_names_the_variable_and_the_local_path(tiny_repo, environment, no_model_classes, capsys):
    environment.setenv("LLM_PROVIDER", "gemini")
    environment.setenv("GEMINI_API_KEY", "")

    exit_code = cli.main(review_args(tiny_repo))

    error = capsys.readouterr().err
    assert exit_code == cli.EXIT_BAD_INPUT
    assert "GEMINI_API_KEY" in error
    assert "LLM_PROVIDER=ollama" in error
    assert error.count("\n") == 1       # one line


def test_invalid_setting_names_the_variable(tiny_repo, environment, no_model_classes, capsys):
    environment.setenv("MAX_TOOL_CALLS", "eight")

    exit_code = cli.main(review_args(tiny_repo))

    error = capsys.readouterr().err
    assert exit_code == cli.EXIT_BAD_INPUT
    assert error.startswith("Error: MAX_TOOL_CALLS='eight'")


def use_ollama(environment, show) -> None:
    """Choose Ollama, and replace the one request check_ollama() sends to the server."""
    environment.setenv("LLM_PROVIDER", "ollama")
    environment.setenv("OLLAMA_MODEL", "some-model:7b")
    environment.setattr(ollama.Client, "show", show)


def test_ollama_not_running_says_how_to_start_it(tiny_repo, environment, capsys):
    def show(self, model):
        raise ConnectionError("Failed to connect to Ollama.")

    use_ollama(environment, show)

    exit_code = cli.main(review_args(tiny_repo))

    error = capsys.readouterr().err
    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert "Ollama is not running at http://localhost:11434" in error
    assert "ollama serve" in error


def test_ollama_model_not_pulled_says_which_pull_to_run(tiny_repo, environment, capsys):
    def show(self, model):
        raise ollama.ResponseError(f"model '{model}' not found", 404)

    use_ollama(environment, show)

    exit_code = cli.main(review_args(tiny_repo))

    error = capsys.readouterr().err
    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert "`ollama pull some-model:7b`" in error
    assert not (out_folder(tiny_repo)).exists()     # nothing was reviewed, so no report


def shows(*capabilities: str):
    """A stand-in for Client.show: the answer has only the one field check_ollama reads."""
    return lambda self, model: SimpleNamespace(capabilities=list(capabilities))


def test_agent_refuses_a_local_model_without_tool_support(tiny_repo, environment, capsys):
    # Exercise 5.4. What `ollama show` reports for a model that cannot call tools.
    use_ollama(environment, shows("completion"))

    exit_code = cli.main(review_args(tiny_repo))

    error = capsys.readouterr().err
    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert "some-model:7b does not support tool calling" in error
    assert "--system baseline" in error


def test_baseline_accepts_a_local_model_without_tool_support(environment):
    use_ollama(environment, shows("completion"))

    llm.check_ollama(needs_tools=False)         # does not raise


def test_a_local_model_with_tool_support_passes_the_check(environment):
    use_ollama(environment, shows("completion", "tools"))

    llm.check_ollama(needs_tools=True)          # does not raise


def test_invalid_structured_output_fails_one_file_only(tiny_repo, monkeypatch, capsys):
    model = use_model(monkeypatch, [
        AIMessage(content="This is not JSON."),         # models.py: invalid twice
        AIMessage(content="Still not JSON."),
        ReviewOutput(findings=[]),                      # views.py is fine
    ])

    exit_code = cli.main(review_args(tiny_repo, "--system", "baseline"))

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_REVIEW_FAILED
    assert model.replies == []
    assert "**Not reviewed:** The model did not return a valid ReviewOutput in two attempts" in output
    assert "Reviewed, no findings." in output


def test_a_long_api_error_is_one_line_on_the_screen_and_complete_in_the_json(tiny_repo, monkeypatch, capsys):
    long_error = "429 RESOURCE_EXHAUSTED\n" + "{'details': 'quota'}\n" * 100
    use_model(monkeypatch, [ValueError(long_error), ReviewOutput(findings=[])])

    cli.main(review_args(tiny_repo, "--system", "baseline"))

    output = capsys.readouterr().out
    failed_line = next(line for line in output.splitlines() if "FAILED" in line)
    assert "429 RESOURCE_EXHAUSTED {'details': 'quota'}" in failed_line
    assert failed_line.endswith(" ...")
    results = json.loads((out_folder(tiny_repo) / tiny_repo.second_sha[:7] / "review.json").read_text(encoding="utf-8"))
    assert long_error in results[0]["error"]


def test_unknown_commit_is_one_line_and_no_model(tiny_repo, no_model, capsys):
    exit_code = cli.main(["review", "--repo", str(tiny_repo.path), "--commit", "1234abcd"])

    assert exit_code == cli.EXIT_BAD_INPUT
    assert capsys.readouterr().err == "Error: Unknown commit: 1234abcd\n"


def test_folder_that_is_not_a_repository(tmp_path, no_model, capsys):
    exit_code = cli.main(["review", "--repo", str(tmp_path), "--commit", "1234abcd"])

    assert exit_code == cli.EXIT_BAD_INPUT
    assert "Not a git repository" in capsys.readouterr().err
