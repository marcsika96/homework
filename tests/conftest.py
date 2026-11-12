import subprocess
from pathlib import Path

import pytest

from reviewgraph.gitrepo import GitRepo

MODELS_V1 = """from django.db import models


class DanceClass(models.Model):
    title = models.CharField(max_length=100)
"""

MODELS_V2 = """from django.db import models


class DanceClass(models.Model):
    title = models.CharField(max_length=120)
    capacity = models.PositiveIntegerField(default=10)
"""

VIEWS = """from django.shortcuts import render

from .models import DanceClass


def class_list(request):
    return render(request, "classes.html", {"classes": DanceClass.objects.all()})
"""

AGENTS = """# Project rules

- Views never touch the database directly.
"""


class TinyRepo:
    """A two-commit repository that stands in for the private review target."""

    def __init__(self, path: Path, first_sha: str, second_sha: str):
        self.path = path
        self.first_sha = first_sha
        self.second_sha = second_sha
        self.repo = GitRepo(path)


def git(path: Path, *args: str) -> str:
    # Name, e-mail and signing are set here, so the fixture works on a clean machine.
    command = [
        "git",
        "-c", "user.name=Test",
        "-c", "user.email=test@example.com",
        "-c", "commit.gpgsign=false",
        "-c", "core.autocrlf=false",
        "-C", str(path),
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


@pytest.fixture
def tiny_repo(tmp_path) -> TinyRepo:
    path = tmp_path / "target"
    path.mkdir()
    git(path, "init", "--quiet", "-b", "main")

    write(path / "app/studio/models.py", MODELS_V1)
    write(path / "AGENTS.md", AGENTS)
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "first")
    first_sha = git(path, "rev-parse", "HEAD")

    write(path / "app/studio/models.py", MODELS_V2)
    write(path / "app/studio/views.py", VIEWS)
    (path / "app/studio/logo.png").write_bytes(b"\x89PNG\x00\x01\x02\x00")
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "second")
    second_sha = git(path, "rev-parse", "HEAD")

    return TinyRepo(path, first_sha, second_sha)
