import pytest

from reviewgraph.gitrepo import GitError, GitRepo, PathNotFound, UnknownCommit


def test_folder_without_repository_is_rejected(tmp_path):
    with pytest.raises(GitError):
        GitRepo(tmp_path)


def test_missing_folder_is_rejected(tmp_path):
    with pytest.raises(GitError):
        GitRepo(tmp_path / "nothing-here")


def test_resolve_turns_short_sha_into_full_sha(tiny_repo):
    assert tiny_repo.repo.resolve(tiny_repo.first_sha[:7]) == tiny_repo.first_sha


def test_unknown_sha_raises_unknown_commit(tiny_repo):
    with pytest.raises(UnknownCommit):
        tiny_repo.repo.resolve("0123abc")


@pytest.mark.parametrize("rev", ["", "HEAD", "main", "--help", "-n", "abc", "zzzzzzz"])
def test_only_hex_shas_are_accepted(tiny_repo, rev):
    with pytest.raises(UnknownCommit):
        tiny_repo.repo.resolve(rev)


def test_changed_files_of_second_commit(tiny_repo):
    files = {f.path: f for f in tiny_repo.repo.changed_files(tiny_repo.second_sha)}

    assert set(files) == {"app/studio/models.py", "app/studio/views.py", "app/studio/logo.png"}
    assert (files["app/studio/models.py"].added, files["app/studio/models.py"].deleted) == (2, 1)
    assert (files["app/studio/views.py"].added, files["app/studio/views.py"].deleted) == (7, 0)


def test_binary_file_counts_as_zero_lines(tiny_repo):
    files = {f.path: f for f in tiny_repo.repo.changed_files(tiny_repo.second_sha)}

    assert (files["app/studio/logo.png"].added, files["app/studio/logo.png"].deleted) == (0, 0)


def test_changed_files_of_first_commit_lists_everything(tiny_repo):
    paths = [f.path for f in tiny_repo.repo.changed_files(tiny_repo.first_sha)]

    assert sorted(paths) == ["AGENTS.md", "app/studio/models.py"]


def test_diff_contains_only_the_requested_file(tiny_repo):
    diff = tiny_repo.repo.diff(tiny_repo.second_sha, "app/studio/models.py")

    assert "+    capacity = models.PositiveIntegerField(default=10)" in diff
    assert "-    title = models.CharField(max_length=100)" in diff
    assert "views.py" not in diff


def test_diff_of_first_commit_shows_the_whole_file(tiny_repo):
    diff = tiny_repo.repo.diff(tiny_repo.first_sha, "app/studio/models.py")

    assert "+class DanceClass(models.Model):" in diff


def test_diff_of_unchanged_file_is_empty(tiny_repo):
    assert tiny_repo.repo.diff(tiny_repo.second_sha, "AGENTS.md") == ""


def test_read_returns_the_file_as_it_was_at_the_commit(tiny_repo):
    old = tiny_repo.repo.read(tiny_repo.first_sha, "app/studio/models.py")
    new = tiny_repo.repo.read(tiny_repo.second_sha, "app/studio/models.py")

    assert "max_length=100" in old and "capacity" not in old
    assert "max_length=120" in new and "capacity" in new


def test_read_does_not_see_files_added_later(tiny_repo):
    # views.py exists in the second commit and in the working tree, but not in the first commit.
    assert (tiny_repo.path / "app/studio/views.py").exists()

    with pytest.raises(PathNotFound):
        tiny_repo.repo.read(tiny_repo.first_sha, "app/studio/views.py")


def test_read_does_not_use_the_working_tree(tiny_repo):
    (tiny_repo.path / "AGENTS.md").write_text("changed after the commit", encoding="utf-8")
    (tiny_repo.path / "app/studio/views.py").unlink()

    assert "Views never touch" in tiny_repo.repo.read(tiny_repo.second_sha, "AGENTS.md")
    assert "class_list" in tiny_repo.repo.read(tiny_repo.second_sha, "app/studio/views.py")


def test_read_accepts_backslashes(tiny_repo):
    assert "DanceClass" in tiny_repo.repo.read(tiny_repo.second_sha, "app\\studio\\models.py")


@pytest.mark.parametrize(
    "path", ["", "../secret.txt", "app/../../secret.txt", "/etc/passwd", "C:/Windows/win.ini", "app/studio"]
)
def test_read_rejects_paths_that_are_not_files_in_the_repository(tiny_repo, path):
    with pytest.raises(PathNotFound):
        tiny_repo.repo.read(tiny_repo.second_sha, path)


def test_ls_lists_the_files_at_the_commit(tiny_repo):
    assert tiny_repo.repo.ls(tiny_repo.first_sha) == ["AGENTS.md", "app/studio/models.py"]
    assert tiny_repo.repo.ls(tiny_repo.second_sha, "app/studio") == [
        "app/studio/logo.png",
        "app/studio/models.py",
        "app/studio/views.py",
    ]


def test_grep_returns_path_line_and_text(tiny_repo):
    hits = tiny_repo.repo.grep(tiny_repo.second_sha, r"class \w+\(models\.Model\)")

    assert hits == ["app/studio/models.py:4:class DanceClass(models.Model):"]


def test_grep_with_no_hit_returns_empty_list(tiny_repo):
    assert tiny_repo.repo.grep(tiny_repo.second_sha, "no_such_name_anywhere") == []


def test_grep_does_not_see_code_added_later(tiny_repo):
    assert tiny_repo.repo.grep(tiny_repo.first_sha, "class_list") == []
    assert len(tiny_repo.repo.grep(tiny_repo.second_sha, "class_list")) == 1


def test_grep_with_invalid_pattern_raises_git_error(tiny_repo):
    with pytest.raises(GitError):
        tiny_repo.repo.grep(tiny_repo.second_sha, "(unclosed")
