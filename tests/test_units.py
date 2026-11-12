import pytest

from reviewgraph.schemas import build_units, is_reviewable


@pytest.mark.parametrize(
    "path",
    [
        "app/studio/forms.py",
        "app/studio/usecases/booking.py",
        "app/studio/management/commands/run_scheduler.py",
    ],
)
def test_application_python_is_reviewable(path):
    assert is_reviewable(path)


@pytest.mark.parametrize(
    "path",
    [
        "app/studio/migrations/0001_initial.py",
        "app/studio/migrations/0002_seed_data.py",
        "app/studio/seed_data.py",
        "app/studio/management/commands/seed_demo.py",
        "app/studio/templates/classes.html",
        "app/studio/static/htmx.min.js",
        "app/tests/usecases/test_uc24_login.py",
        "app/dsw/settings.py",
        "AGENTS.md",
    ],
)
def test_everything_else_is_not_reviewable(path):
    assert not is_reviewable(path)


def test_build_units_keeps_files_with_enough_added_lines(tiny_repo):
    increments = {"I1": tiny_repo.first_sha, "I4": tiny_repo.second_sha}

    units = build_units(tiny_repo.repo, increments, min_added=5)

    # models.py: 5 added lines in the first commit, only 2 in the second.
    # views.py: 7 added lines in the second commit. AGENTS.md and logo.png are never units.
    assert [unit.id for unit in units] == ["I1:app/studio/models.py", "I4:app/studio/views.py"]


def test_units_carry_commit_lines_and_split(tiny_repo):
    increments = {"I1": tiny_repo.first_sha[:7], "I4": tiny_repo.second_sha[:7]}

    dev_unit, test_unit = build_units(tiny_repo.repo, increments, min_added=5)

    assert (dev_unit.split, dev_unit.commit, dev_unit.added_lines) == ("dev", tiny_repo.first_sha, 5)
    assert (test_unit.split, test_unit.commit, test_unit.added_lines) == ("test", tiny_repo.second_sha, 7)


def test_default_threshold_is_15_added_lines(tiny_repo):
    increments = {"I1": tiny_repo.first_sha, "I4": tiny_repo.second_sha}

    assert build_units(tiny_repo.repo, increments) == []
