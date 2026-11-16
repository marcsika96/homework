"""The LLM judge: rates the stored findings as valid, minor or invalid.

The verdicts go to data/verdicts.jsonl, one line per finding. A finding that
has a verdict is skipped, so judging can be interrupted and continued like the
runs of runner.py.
"""

import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from reviewgraph.evaluation.runner import ROOT, RUNS_DIR, load_records
from reviewgraph.gitrepo import GitRepo
from reviewgraph.llm import (
    DAILY_QUOTA_MARKER,
    RETRY_MAX_ATTEMPTS,
    call_structured,
    is_transient_error,
    model_name,
    retry_wait,
)
from reviewgraph.prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_judge_request, load_rules, render_diff
from reviewgraph.schemas import Finding, ReviewResult, Unit

VERDICTS_PATH = ROOT / "data" / "verdicts.jsonl"
# The findings are judged in a shuffled order. A fixed seed gives the same order every time.
SHUFFLE_SEED = 2026


class JudgeVerdict(BaseModel):
    """What the judge model returns."""

    label: Literal["valid", "minor", "invalid"] = Field(
        description="valid: a real defect; minor: true but cosmetic; invalid: wrong or not confirmable"
    )
    reason: str = Field(description="One sentence that names the line or the rule that decided it")


class VerdictRecord(BaseModel):
    """One line of verdicts.jsonl: the finding, its verdict and where the verdict came from."""

    finding_id: str
    unit_id: str
    system: str
    finding: Finding
    verdict: JudgeVerdict
    judge_model: str
    judge_prompt_version: str
    input_tokens: int
    output_tokens: int
    timestamp: str


def select_for_judging(result: ReviewResult, limit: int = 3) -> list[Finding]:
    """Medium and high findings, high first, then reporting order, at most `limit`."""
    chosen = [finding for finding in result.findings if finding.severity in ("high", "medium")]
    # sort() keeps the order of equal items, so within "high" and within
    # "medium" the reporting order stays. False sorts before True.
    chosen.sort(key=lambda finding: finding.severity != "high")
    return chosen[:limit]


def finding_id(unit_id: str, system: str, finding: Finding) -> str:
    """Stable id of a finding: the same finding always gets the same 12 characters.

    Used as the key in verdicts.jsonl and in the hand-check sheet. It is a hash,
    so the sheet can show it without showing which system the finding is from.
    """
    parts = [unit_id, system, str(finding.line_start), str(finding.line_end), finding.category, finding.explanation]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def judge_finding(model: BaseChatModel, repo: GitRepo, unit: Unit, finding: Finding) -> tuple[JudgeVerdict, list]:
    """Ask the judge about one finding. Returns the verdict and the raw model messages (for the tokens).

    The judge sees the finding, the diff, the whole file at the commit and the rules.
    It never sees which system produced the finding: `system` is not an argument.
    """
    request = build_judge_request(
        finding,
        rules=load_rules(repo, unit.commit),
        diff=render_diff(repo, unit.commit, unit.path),
        file_text=repo.read(unit.commit, unit.path),
    )
    messages = [SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(request)]

    # The same retry rule as the two reviewers: only errors that waiting can fix.
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return call_structured(model, JudgeVerdict, messages)
        except Exception as e:
            if attempt == RETRY_MAX_ATTEMPTS or not is_transient_error(e):
                raise
            time.sleep(retry_wait(attempt))


def load_verdicts(path: Path = VERDICTS_PATH) -> dict[str, VerdictRecord]:
    """All stored verdicts by finding id. If an id was judged twice, the later line counts."""
    verdicts = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = VerdictRecord.model_validate_json(line)
                verdicts[record.finding_id] = record
    return verdicts


def append_verdict(path: Path, record: VerdictRecord) -> None:
    """Add one line to the end of the file. Earlier lines are never rewritten."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(record.model_dump_json() + "\n")


def is_current(record: VerdictRecord | None) -> bool:
    """True if the finding was already judged with the current judge model and judge prompt."""
    if record is None:
        return False
    return record.judge_model == model_name() and record.judge_prompt_version == JUDGE_PROMPT_VERSION


def findings_to_judge(units: list[Unit], systems: list[str], runs_dir: Path = RUNS_DIR) -> list[tuple[Unit, str, Finding]]:
    """The selected findings of the stored runs, as (unit, system, finding), in a shuffled order."""
    units_by_id = {unit.id: unit for unit in units}
    items = []
    for system in systems:
        for record in load_records(units, system, runs_dir):
            for finding in select_for_judging(record.result):
                items.append((units_by_id[record.result.unit_id], system, finding))
    # Shuffled, so the findings of one system are not all judged before the other's.
    random.Random(SHUFFLE_SEED).shuffle(items)
    return items


def judge_runs(
    units: list[Unit],
    systems: list[str],
    repo: GitRepo,
    model: BaseChatModel,
    runs_dir: Path = RUNS_DIR,
    verdicts_path: Path = VERDICTS_PATH,
) -> dict:
    """Judge every selected finding that has no verdict yet. Returns the counts.

    A failed judgement is not stored and is tried again on the next run. The
    run stops early when the daily quota is used up.
    """
    summary = {"judged": 0, "skipped": 0, "failed": 0, "quota_used_up": False}
    verdicts = load_verdicts(verdicts_path)
    items = findings_to_judge(units, systems, runs_dir)

    for number, (unit, system, finding) in enumerate(items, start=1):
        identifier = finding_id(unit.id, system, finding)
        label = f"[judge {number}/{len(items)}] {identifier}"
        if is_current(verdicts.get(identifier)):
            summary["skipped"] += 1
            continue

        try:
            verdict, raw_messages = judge_finding(model, repo, unit, finding)
        except Exception as e:
            if DAILY_QUOTA_MARKER in str(e):
                print(f"{label}: the daily quota is used up, stopping here")
                summary["quota_used_up"] = True
                break
            print(f"{label}: FAILED, will be tried again on the next run: {type(e).__name__}")
            summary["failed"] += 1
            continue

        usage = [message.usage_metadata or {} for message in raw_messages]
        record = VerdictRecord(
            finding_id=identifier,
            unit_id=unit.id,
            system=system,
            finding=finding,
            verdict=verdict,
            judge_model=model_name(),
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            input_tokens=sum(u.get("input_tokens", 0) for u in usage),
            output_tokens=sum(u.get("output_tokens", 0) for u in usage),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        append_verdict(verdicts_path, record)
        # The progress line shows the id only: not the system and not the label.
        print(f"{label}: judged")
        summary["judged"] += 1

    return summary
