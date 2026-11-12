from typing import Literal

from pydantic import BaseModel, Field

from reviewgraph.gitrepo import GitRepo

DEV_INCREMENTS = ("I1", "I2", "I3")

Category = Literal["authorization", "transaction", "model-rule", "validation",
                   "error-handling", "logic", "project-rule", "other"]
Severity = Literal["low", "medium", "high"]


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


# --- What the model returns -------------------------------------------------
# These schemas are sent to the model as the structured-output format. They stay
# flat and every field has a description, because the description is what the
# model reads.


class Finding(BaseModel):
    file: str = Field(description="Path of the reviewed file")
    line_start: int = Field(description="First affected line, numbered as in the new version of the file")
    line_end: int = Field(description="Last affected line; equal to line_start for a single line")
    category: Category = Field(description="The kind of defect")
    severity: Severity = Field(description="high: wrong behaviour or data; medium: wrong in some cases; low: cosmetic")
    explanation: str = Field(description="What is wrong and what happens because of it")
    fix: str = Field(description="The concrete change that removes the defect")


class ReviewOutput(BaseModel):
    """What the model returns. A wrapper, because structured output needs an object."""

    findings: list[Finding] = Field(description="All defects found; an empty list if nothing is wrong")


class Verification(BaseModel):
    supported: bool = Field(description="True only if the quoted code really has the described defect")
    reason: str = Field(description="One sentence that says why the finding is or is not supported")


# --- What a review run produces ---------------------------------------------


class RunStats(BaseModel):
    latency_s: float = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ReviewResult(BaseModel):
    unit_id: str
    system: Literal["agent", "baseline"]
    model: str
    findings: list[Finding] = []
    dropped: list[Finding] = []      # removed by the self-check (agent only)
    trace: list[dict] = []           # tool calls and their arguments, for failure analysis
    stats: RunStats = RunStats()
    error: str | None = None


def clean_findings(findings: list[Finding], file_text: str, path: str) -> list[Finding]:
    """Repair what models often get wrong: line numbers and the file name.

    Reversed line ranges are swapped, lines are moved into the file
    (1 .. number of lines) and `file` is set to the reviewed path.
    """
    last_line = max(1, len(file_text.splitlines()))
    cleaned = []
    for finding in findings:
        start, end = sorted((finding.line_start, finding.line_end))
        start = min(max(start, 1), last_line)
        end = min(max(end, 1), last_line)
        cleaned.append(finding.model_copy(update={"file": path, "line_start": start, "line_end": end}))
    return cleaned
