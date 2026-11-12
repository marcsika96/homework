from conftest import git, write

from reviewgraph.gitrepo import GitRepo
from reviewgraph.prompts import (
    NO_RULES,
    REVIEW_SYSTEM_PROMPT,
    build_review_request,
    load_rules,
    render_diff,
)
from reviewgraph.schemas import Category


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
