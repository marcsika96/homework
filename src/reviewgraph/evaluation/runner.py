"""Runs a system over a list of units and stores one file per unit, so a run can be continued.

The files under data/runs/<system>/ are the cached outputs of the evaluation:
everything after this stage (judge, metrics, `reviewgraph eval --cached`) reads
them and needs no key.
"""

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from reviewgraph.baseline import review_baseline
from reviewgraph.config import get_settings
from reviewgraph.gitrepo import GitRepo
from reviewgraph.graph import review_agent
from reviewgraph.llm import DAILY_QUOTA_MARKER, model_name
from reviewgraph.prompts import PROMPT_VERSION
from reviewgraph.schemas import ReviewResult, Unit

# src/reviewgraph/evaluation/runner.py -> the root of the homework repository
ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "data" / "runs"


class RunRecord(BaseModel):
    """What is stored for one unit and system: the result and where it came from."""

    result: ReviewResult             # holds the unit id, the system and the model name
    prompt_version: str
    max_tool_calls: int | None       # the tool budget; None for the baseline
    code_version: str                # Git commit of this repository when the review ran
    timestamp: str                   # when the review finished, in UTC


def load_units(path: str | Path) -> list[Unit]:
    """Read a units file: one Unit as JSON per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Unit.model_validate_json(line) for line in lines if line.strip()]


def code_version() -> str:
    """Commit of the homework repository, with "-dirty" if src/ has uncommitted changes.

    This is the proof that a test run was made with the frozen code.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        # Exit code 1 of `git diff --quiet` means "there are differences".
        # capture_output keeps git's warnings (for example about line endings) off the screen.
        changed = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src"], capture_output=True
        ).returncode
        return commit + ("-dirty" if changed else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_path(runs_dir: Path, system: str, unit_id: str) -> Path:
    """data/runs/<system>/<unit id as a file name>.json"""
    # ":" and "/" cannot be in a file name, so everything unusual becomes "_".
    file_name = re.sub(r"[^A-Za-z0-9._-]", "_", unit_id) + ".json"
    return Path(runs_dir) / system / file_name


def load_record(path: Path) -> RunRecord | None:
    """The stored record, or None if this unit has not been run."""
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name first and renamed at the end. If the run is
    # stopped while writing, there is no half file that looks like a finished unit.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def is_done(record: RunRecord | None) -> bool:
    """True if the unit was already run with the current model and prompts.

    A stored failure counts as done: failed reviews are not repeated. A record
    of another model or prompt version is run again and replaced, so one
    evaluation never mixes two configurations.
    """
    if record is None:
        return False
    current_model = model_name(verifier=record.result.system == "agent")
    return record.result.model == current_model and record.prompt_version == PROMPT_VERSION


def hit_daily_quota(result: ReviewResult) -> bool:
    """True if the review failed because the daily quota of the Gemini API is used up."""
    return result.error is not None and result.error.startswith("LLM call failed") and DAILY_QUOTA_MARKER in result.error


# What the Gemini API answers when the model is overloaded: 503 UNAVAILABLE ("high
# demand") or 504 DEADLINE_EXCEEDED. Both were seen on 2026-11-18.
OVERLOAD_MARKERS = ("UNAVAILABLE", "DEADLINE_EXCEEDED")


def hit_overload(result: ReviewResult) -> bool:
    """True if the review failed because the model was overloaded, even after the retries.

    Such a unit is not stored and the run stops: the review did not fail because
    of the unit, and every further unit would fail the same way. On the free
    tier the failed attempts count against the daily quota too (16 requests for
    four units on 2026-11-18), so going on would use up the day for nothing.
    """
    return (
        result.error is not None
        and result.error.startswith("LLM call failed")
        and any(marker in result.error for marker in OVERLOAD_MARKERS)
    )


def run_units(
    units: list[Unit],
    system: str,
    repo: GitRepo,
    model: BaseChatModel,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Review every unit that is not stored yet. Returns how many were reviewed and skipped.

    The run stops early when the daily quota is used up or the model is
    overloaded (hit_overload). That unit is not stored, because it was not
    reviewed; the next run starts with it.
    """
    summary = {"reviewed": 0, "skipped": 0, "quota_used_up": False, "overloaded": False}
    max_tool_calls = get_settings().max_tool_calls if system == "agent" else None

    for number, unit in enumerate(units, start=1):
        path = record_path(runs_dir, system, unit.id)
        label = f"[{system} {number}/{len(units)}] {unit.id}"
        if is_done(load_record(path)):
            print(f"{label}: skipped, already stored")
            summary["skipped"] += 1
            continue

        if system == "agent":
            result = review_agent(repo, unit.commit, unit.path, unit_id=unit.id, model=model, max_tool_calls=max_tool_calls)
        else:
            result = review_baseline(repo, unit.commit, unit.path, unit_id=unit.id, model=model)

        if hit_daily_quota(result):
            print(f"{label}: the daily quota is used up, stopping here")
            summary["quota_used_up"] = True
            break
        if hit_overload(result):
            print(f"{label}: the model is overloaded, stopping here")
            summary["overloaded"] = True
            break

        record = RunRecord(
            result=result,
            prompt_version=PROMPT_VERSION,
            max_tool_calls=max_tool_calls,
            code_version=code_version(),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        save_record(path, record)
        outcome = "FAILED" if result.error else f"{len(result.findings)} findings"
        print(f"{label}: {outcome} in {result.stats.latency_s:.0f} s")
        summary["reviewed"] += 1

    return summary


def load_records(units: list[Unit], system: str, runs_dir: Path = RUNS_DIR) -> list[RunRecord]:
    """The stored records of these units, in their order. Units that were not run are left out."""
    records = []
    for unit in units:
        record = load_record(record_path(runs_dir, system, unit.id))
        if record is not None:
            records.append(record)
    return records
