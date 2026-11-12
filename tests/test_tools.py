import pytest

from reviewgraph.gitrepo import UnknownCommit
from reviewgraph.tools import make_tools


@pytest.fixture
def tools(tiny_repo) -> dict:
    """The three tools at the second commit of the fixture repository, by name."""
    return {tool.name: tool for tool in make_tools(tiny_repo.repo, tiny_repo.second_sha)}


def test_three_tools_with_descriptions_and_without_a_commit_argument(tools):
    assert list(tools) == ["read_file", "search_code", "list_files"]
    for tool in tools.values():
        assert tool.description
        assert "sha" not in tool.args and "commit" not in tool.args


def test_tools_for_an_unknown_commit_cannot_be_made(tiny_repo):
    with pytest.raises(UnknownCommit):
        make_tools(tiny_repo.repo, "0123abc")


def test_read_file_returns_numbered_lines(tools):
    output = tools["read_file"].invoke({"path": "app/studio/models.py"})

    assert output.splitlines()[0] == "    1   from django.db import models"
    assert output.splitlines()[-1] == "    6       capacity = models.PositiveIntegerField(default=10)"


def test_read_file_returns_only_the_requested_lines(tools):
    output = tools["read_file"].invoke({"path": "app/studio/models.py", "start_line": 4, "end_line": 5})

    assert output.splitlines() == [
        "    4   class DanceClass(models.Model):",
        "    5       title = models.CharField(max_length=120)",
        "[lines 4-5 of 6; call again with start_line=6 for more]",
    ]


def test_search_code_returns_path_line_and_text(tools):
    output = tools["search_code"].invoke({"pattern": "def class_list"})

    assert output == "app/studio/views.py:6: def class_list(request):"


def test_list_files_lists_the_directory(tools):
    output = tools["list_files"].invoke({"directory": "app/studio"})

    assert output.splitlines() == ["app/studio/logo.png", "app/studio/models.py", "app/studio/views.py"]
