import pytest
from conftest import git, write

from reviewgraph.gitrepo import GitRepo, UnknownCommit
from reviewgraph.tools import MAX_HITS, MAX_LINES, is_readable, make_tools


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


# --- Hardening: the model decides what the tools receive --------------------


@pytest.fixture
def early_tools(tiny_repo) -> dict:
    """The tools at the first commit, where views.py does not exist yet."""
    return {tool.name: tool for tool in make_tools(tiny_repo.repo, tiny_repo.first_sha)}


@pytest.fixture
def messy_tools(tmp_path) -> dict:
    """Tools on a repository with secrets, a long file and a file with very long lines."""
    path = tmp_path / "messy"
    path.mkdir()
    git(path, "init", "--quiet", "-b", "main")
    write(path / ".env", "API_TOKEN=real-secret-value\n")
    write(path / ".env.example", "API_TOKEN=\n")
    write(path / "deploy/server.key", "API_TOKEN in a private key file\n")
    write(path / "app/settings.py", "API_TOKEN = env('API_TOKEN')\n")
    write(path / "app/long.py", "".join(f"value_{number} = {number}\n" for number in range(1, 251)))
    write(path / "app/static/vendor.min.js", "var a=1;" * 500 + "\n")
    git(path, "add", "-A", "--force")
    git(path, "commit", "--quiet", "-m", "everything")
    sha = git(path, "rev-parse", "HEAD")
    return {tool.name: tool for tool in make_tools(GitRepo(path), sha)}


OUTSIDE_PATHS = [
    "../../secrets.txt",
    "..",
    "app/../../secrets.txt",
    "app\\..\\..\\secrets.txt",
    "/etc/passwd",
    "C:/Windows/win.ini",
    "C:\\Windows\\win.ini",
    "",
]


@pytest.mark.parametrize("path", OUTSIDE_PATHS)
def test_read_file_refuses_paths_outside_the_repository(tools, path):
    assert tools["read_file"].invoke({"path": path}).startswith("Refused:")


@pytest.mark.parametrize("path", OUTSIDE_PATHS)
def test_search_and_list_refuse_directories_outside_the_repository(tools, path):
    assert tools["search_code"].invoke({"pattern": "models", "directory": path}).startswith("Refused:")
    assert tools["list_files"].invoke({"directory": path}).startswith("Refused:")


def test_file_added_by_a_later_commit_is_missing_at_the_earlier_one(early_tools, tools):
    assert "def class_list" in tools["read_file"].invoke({"path": "app/studio/views.py"})

    output = early_tools["read_file"].invoke({"path": "app/studio/views.py"})

    assert output.startswith("No such file at this commit: app/studio/views.py.")


def test_search_and_list_do_not_see_a_later_commit(early_tools):
    assert early_tools["search_code"].invoke({"pattern": "class_list"}) == "No matches for 'class_list' in app."
    assert "views.py" not in early_tools["list_files"].invoke({"directory": "app/studio"})
    assert "capacity" not in early_tools["read_file"].invoke({"path": "app/studio/models.py"})


def test_tools_do_not_read_the_working_tree(tiny_repo, tools):
    (tiny_repo.path / "app/studio/models.py").write_text("changed_on_disk = True\n", encoding="utf-8")
    (tiny_repo.path / "app/studio/untracked.py").write_text("changed_on_disk = True\n", encoding="utf-8")

    assert "changed_on_disk" not in tools["read_file"].invoke({"path": "app/studio/models.py"})
    assert tools["search_code"].invoke({"pattern": "changed_on_disk"}).startswith("No matches")
    assert "untracked.py" not in tools["list_files"].invoke({"directory": "app/studio"})


def test_misspelled_file_gets_a_suggestion(tools):
    output = tools["read_file"].invoke({"path": "app/studio/model.py"})

    assert output == "No such file at this commit: app/studio/model.py. Did you mean app/studio/models.py?"


def test_misspelled_directory_gets_a_suggestion(tools):
    output = tools["list_files"].invoke({"directory": "app/studios"})

    assert output == "No such directory at this commit: app/studios. Did you mean app/studio?"


def test_reading_a_directory_points_to_list_files(tools):
    output = tools["read_file"].invoke({"path": "app/studio"})

    assert output == "app/studio is a directory. Use list_files to see its files."


def test_binary_file_is_not_shown(tools):
    output = tools["read_file"].invoke({"path": "app/studio/logo.png"})

    assert output == "app/studio/logo.png is a binary file and cannot be shown."


def test_invalid_regular_expression_returns_gits_message(tools):
    output = tools["search_code"].invoke({"pattern": "(unclosed"})

    assert output.startswith("Search failed:")
    assert "(unclosed" in output          # git names the pattern it could not compile


@pytest.mark.parametrize("pattern", ["", "   "])
def test_empty_pattern_is_explained(tools, pattern):
    assert tools["search_code"].invoke({"pattern": pattern}).startswith("The search pattern is empty.")


def test_search_in_a_missing_directory_is_explained(tools):
    output = tools["search_code"].invoke({"pattern": "models", "directory": "app/nothing"})

    assert output.startswith("No such file or directory at this commit: app/nothing.")


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({"start_line": 7}, "app/studio/models.py has only 6 lines; start_line 7 is after the end."),
        ({"start_line": 900}, "app/studio/models.py has only 6 lines; start_line 900 is after the end."),
        ({"start_line": -3}, "Invalid start_line -3. Lines are counted from 1."),
        ({"start_line": 0}, "Invalid start_line 0. Lines are counted from 1."),
        ({"start_line": 5, "end_line": 2}, "Invalid line range 5-2. end_line must not be smaller than start_line."),
        ({"start_line": 3, "end_line": -1}, "Invalid line range 3--1. end_line must not be smaller than start_line."),
    ],
)
def test_bad_line_ranges_get_a_short_explanation(tools, arguments, expected):
    assert tools["read_file"].invoke({"path": "app/studio/models.py", **arguments}) == expected


def test_end_line_after_the_end_of_the_file_is_not_an_error(tools):
    output = tools["read_file"].invoke({"path": "app/studio/models.py", "start_line": 5, "end_line": 999})

    assert [line[:5].strip() for line in output.splitlines()] == ["5", "6"]


@pytest.mark.parametrize("path", [".env", "deploy/server.key"])
def test_deny_listed_file_is_refused(messy_tools, path):
    output = messy_tools["read_file"].invoke({"path": path})

    assert output == f"Refused: {path} may contain secrets and cannot be read."


def test_deny_listed_files_are_not_searched_or_listed(messy_tools):
    hits = messy_tools["search_code"].invoke({"pattern": "API_TOKEN", "directory": "."})
    listing = messy_tools["list_files"].invoke({"directory": "."})

    assert "real-secret-value" not in hits and "server.key" not in hits
    assert "app/settings.py:1:" in hits and ".env.example:1:" in hits
    assert ".env.example" in listing.splitlines()
    assert ".env" not in listing.splitlines() and "deploy/server.key" not in listing.splitlines()


@pytest.mark.parametrize(
    "path, readable",
    [
        (".env", False),
        ("app/.env", False),
        (".env.production", False),
        ("certs/site.pem", False),
        ("deploy\\server.key", False),
        ("backup.p12", False),
        ("home/id_rsa", False),
        ("home/id_rsa.pub", False),
        (".env.example", True),
        ("app/studio/environment.py", True),
        ("app/studio/keys.py", True),
        ("AGENTS.md", True),
    ],
)
def test_is_readable(path, readable):
    assert is_readable(path) is readable


def test_long_file_is_read_in_pages_of_200_lines(messy_tools):
    first = messy_tools["read_file"].invoke({"path": "app/long.py"}).splitlines()
    second = messy_tools["read_file"].invoke({"path": "app/long.py", "start_line": 201}).splitlines()

    assert len(first) == MAX_LINES + 1
    assert first[-1] == "[lines 1-200 of 250; call again with start_line=201 for more]"
    assert second[0] == "  201   value_201 = 201"
    assert second[-1] == "  250   value_250 = 250"


def test_a_large_requested_range_is_still_limited_to_200_lines(messy_tools):
    output = messy_tools["read_file"].invoke({"path": "app/long.py", "start_line": 10, "end_line": 250})

    assert output.splitlines()[-1] == "[lines 10-209 of 250; call again with start_line=210 for more]"


def test_many_hits_are_cut_with_a_note(messy_tools):
    output = messy_tools["search_code"].invoke({"pattern": "value_", "directory": "app/long.py"}).splitlines()

    assert len(output) == MAX_HITS + 1
    assert output[-1] == "[truncated, 40 of 250 hits; narrow the pattern or the directory]"


def test_very_long_lines_are_cut(messy_tools):
    hit = messy_tools["search_code"].invoke({"pattern": "var a", "directory": "app/static"})
    line = messy_tools["read_file"].invoke({"path": "app/static/vendor.min.js"})

    assert hit.endswith("[line cut]") and len(hit) < 300
    assert line.endswith("[line cut]") and len(line) < 1100


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_file", {"path": "--help"}),
        ("read_file", {"path": "-rf"}),
        ("read_file", {"path": "HEAD:app/studio/models.py"}),
        ("read_file", {"path": "app/studio/models.py:1"}),
        ("read_file", {"path": "app/studio/models.py\nAGENTS.md"}),
        ("read_file", {"path": "x" * 5000}),
        ("read_file", {"path": "app/studio/models.py", "start_line": 10**12}),
        ("search_code", {"pattern": "--help"}),
        ("search_code", {"pattern": "-e"}),
        ("search_code", {"pattern": "models", "directory": "--help"}),
        ("search_code", {"pattern": "[[:alpha:]"}),
        ("search_code", {"pattern": "a" * 5000}),
        ("list_files", {"directory": "--help"}),
        ("list_files", {"directory": ":(top)"}),
    ],
)
def test_odd_arguments_return_text_and_never_raise(tools, name, arguments):
    output = tools[name].invoke(arguments)

    assert isinstance(output, str) and output
    assert "usage: git" not in output          # nothing was read as a git option
