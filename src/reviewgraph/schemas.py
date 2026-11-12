from typing import Literal

from pydantic import BaseModel

from reviewgraph.gitrepo import GitRepo

DEV_INCREMENTS = ("I1", "I2", "I3")


class Unit(BaseModel):
    """One changed Python file in one increment commit."""

    id: str                      # "I5:app/studio/usecases/booking.py"
    increment: str               # "I1" .. "I8"
    commit: str                  # full SHA
    path: str
    added_lines: int
    split: Literal["dev", "test"]


def is_reviewable(path: str) -> bool:
    """Python under app/studio/, not a migration, not seed data."""
    if not path.startswith("app/studio/") or not path.endswith(".py"):
        return False
    if "/migrations/" in path:
        return False
    # Seed data: app/studio/seed_data.py and management/commands/seed_demo.py
    file_name = path.rsplit("/", 1)[-1]
    return not file_name.startswith("seed_")


def build_units(repo: GitRepo, increments: dict[str, str], min_added: int = 15) -> list[Unit]:
    """Review units of the increment commits: reviewable files with at least `min_added` added lines."""
    units = []
    for increment, sha in increments.items():
        commit = repo.resolve(sha)
        for changed in repo.changed_files(commit):
            if is_reviewable(changed.path) and changed.added >= min_added:
                units.append(
                    Unit(
                        id=f"{increment}:{changed.path}",
                        increment=increment,
                        commit=commit,
                        path=changed.path,
                        added_lines=changed.added,
                        split="dev" if increment in DEV_INCREMENTS else "test",
                    )
                )
    units.sort(key=lambda unit: (unit.increment, unit.path))
    return units
