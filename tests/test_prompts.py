from conftest import git, tool_call, write
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from reviewgraph.gitrepo import GitRepo
from reviewgraph.prompts import (
    NO_LOOKUPS,
    NO_RULES,
    REVIEW_SYSTEM_PROMPT,
    build_agent_system_prompt,
    build_review_request,
    build_verification_request,
    load_rules,
    render_diff,
    render_lookups,
)
from reviewgraph.schemas import Category, Finding


def test_render_diff_numbers_lines_as_in_the_new_file(tiny_repo):
    rendered = render_diff(tiny_repo.repo, tiny_repo.second_sha, "app/studio/models.py")

    assert rendered.splitlines() == [
        "    1   from django.db import models",
        "    2   ",
        "    3   ",
        "    4   class DanceClass(models.Model):",
        "      -     title = models.CharField(max_length=100)",
        "    5 +     title = models.CharField(max_length=120)",
        "    6 +     capacity = models.PositiveIntegerField(default=10)",
    ]


def test_render_diff_of_a_new_file_marks_every_line_as_added(tiny_repo):
    rendered = render_diff(tiny_repo.repo, tiny_repo.second_sha, "app/studio/views.py")
    lines = rendered.splitlines()

    assert lines[0] == "    1 + from django.shortcuts import render"
    assert lines[-1].startswith("    7 +     return render(")
    assert all(line[6] == "+" for line in lines)


def test_render_diff_numbers_match_the_file(tiny_repo):
    file_lines = tiny_repo.repo.read(tiny_repo.second_sha, "app/studio/models.py").splitlines()

    for line in render_diff(tiny_repo.repo, tiny_repo.second_sha, "app/studio/models.py").splitlines():
        if line[:5].strip():                    # removed lines have no number
            assert file_lines[int(line[:5]) - 1] == line[8:]


def test_render_diff_separates_distant_hunks(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet", "-b", "main")
    old_lines = [f"value_{number} = {number}" for number in range(1, 41)]
    write(path / "app/studio/values.py", "\n".join(old_lines) + "\n")
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "first")
    new_lines = list(old_lines)
    new_lines[1] = "value_2 = 'changed'"
    new_lines[37] = "value_38 = 'changed'"
    write(path / "app/studio/values.py", "\n".join(new_lines) + "\n")
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "second")
    sha = git(path, "rev-parse", "HEAD")

    lines = render_diff(GitRepo(path), sha, "app/studio/values.py", context=2).splitlines()

    assert "    2 + value_2 = 'changed'" in lines
    assert "   38 + value_38 = 'changed'" in lines
    assert lines.count("   ...") == 1
    assert not any("value_20 " in line for line in lines)


def test_render_diff_of_an_unchanged_file_is_empty(tiny_repo):
    assert render_diff(tiny_repo.repo, tiny_repo.second_sha, "AGENTS.md") == ""


def test_load_rules_reads_agents_md_at_the_commit(tiny_repo):
    (tiny_repo.path / "AGENTS.md").write_text("changed after the commit", encoding="utf-8")

    rules = load_rules(tiny_repo.repo, tiny_repo.second_sha)

    assert rules == "# Project rules\n\n- Views never touch the database directly."


def test_load_rules_gives_a_placeholder_without_agents_md(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet", "-b", "main")
    write(path / "app/studio/models.py", "x = 1\n")
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "no rules yet")
    sha = git(path, "rev-parse", "HEAD")

    assert load_rules(GitRepo(path), sha) == NO_RULES


def test_review_request_contains_path_rules_and_diff():
    request = build_review_request("app/studio/forms.py", "RULES TEXT", "    1 + x = 1")

    assert "app/studio/forms.py" in request
    assert "RULES TEXT" in request
    assert "    1 + x = 1" in request


def test_system_prompt_names_every_category_and_severity():
    for category in Category.__args__:
        assert f"- {category}:" in REVIEW_SYSTEM_PROMPT
    for severity in ("high", "medium", "low"):
        assert f"- {severity}:" in REVIEW_SYSTEM_PROMPT
    assert "empty list" in REVIEW_SYSTEM_PROMPT


def test_agent_system_prompt_is_the_reviewer_prompt_plus_tools_and_budget():
    prompt = build_agent_system_prompt(5)
    assert prompt.startswith(REVIEW_SYSTEM_PROMPT)
    assert "at most 5 tool calls" in prompt
    for tool_name in ("read_file", "search_code", "list_files"):
        assert tool_name in prompt
    # The baseline has no tools, so its prompt must not mention them.
    assert "tool" not in REVIEW_SYSTEM_PROMPT


# --- The self-check's view of the look-ups ------------------------------------


def look_up_state() -> tuple[list, list[dict]]:
    """Messages and trace of a look-up phase: two executed calls, one refused by the budget."""
    messages = [
        HumanMessage("Review this change."),
        AIMessage(content="", tool_calls=[
            tool_call("read_file", "call-1", path="app/studio/backend/classes.py", start_line=40),
            tool_call("search_code", "call-2", pattern="def add_class_to_schedule"),
            tool_call("read_file", "call-3", path="app/studio/models.py"),
        ]),
        ToolMessage(content="   58       def __bool__(self):", tool_call_id="call-1"),
        ToolMessage(content="app/studio/system/transactions.py:66: def add_class_to_schedule(request, checked_details):",
                    tool_call_id="call-2"),
        ToolMessage(content="Tool budget used up.", tool_call_id="call-3"),
        AIMessage(content="I think line 35 is wrong."),
    ]
    trace = [
        {"tool_call_id": "call-1", "tool_name": "read_file",
         "tool_args": {"path": "app/studio/backend/classes.py", "start_line": 40}, "status": "executed"},
        {"tool_call_id": "call-2", "tool_name": "search_code",
         "tool_args": {"pattern": "def add_class_to_schedule"}, "status": "executed"},
        {"tool_call_id": "call-3", "tool_name": "read_file",
         "tool_args": {"path": "app/studio/models.py"}, "status": "budget_exceeded"},
    ]
    return messages, trace


def test_render_lookups_shows_each_executed_call_with_its_answer():
    messages, trace = look_up_state()

    rendered = render_lookups(messages, trace)

    assert rendered == (
        "### read_file(path='app/studio/backend/classes.py', start_line=40)\n\n"
        "   58       def __bool__(self):\n\n"
        "### search_code(pattern='def add_class_to_schedule')\n\n"
        "app/studio/system/transactions.py:66: def add_class_to_schedule(request, checked_details):"
    )


def test_render_lookups_leaves_out_refused_calls_and_the_reviewers_own_words():
    messages, trace = look_up_state()

    rendered = render_lookups(messages, trace)

    assert "Tool budget used up." not in rendered
    assert "app/studio/models.py" not in rendered
    assert "I think line 35 is wrong." not in rendered
    assert "Review this change." not in rendered


def test_render_lookups_without_tool_calls_says_so():
    assert render_lookups([HumanMessage("Review this change.")], []) == NO_LOOKUPS


def test_verification_request_contains_the_looked_up_code():
    finding = Finding(file="app/studio/usecases/add_class.py", line_start=35, line_end=35, category="logic",
                      severity="high", explanation="Passes problems instead of checked details.", fix="Pass the details.")
    messages, trace = look_up_state()

    request = build_verification_request(finding, "rules", "   35   return add_class_to_schedule(request, problems)",
                                         render_lookups(messages, trace))
    without = build_verification_request(finding, "rules", "   35   return add_class_to_schedule(request, problems)")

    assert "## Code the reviewer looked up" in request
    assert "def __bool__(self):" in request
    assert without.endswith(NO_LOOKUPS + "\n")

