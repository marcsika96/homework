"""Try a local Ollama model on one dev unit: the baseline, then the agent.

Usage: python scripts/bench_local.py qwen3.5:4b            (dev unit 20, budget of 2 tool calls)
       python scripts/bench_local.py qwen3.5:4b 20 8       (unit number and tool budget)

Uses the same code path as a real review (review_baseline and review_agent),
only with the given Ollama model. Prints a summary and appends the full
results to out/bench_local.jsonl. Dev unit 20 has one known defect at lines
52-54 (a DurationField that should be a DateTimeField), so the output also
says whether the model found it.
"""

import json
import sys
from pathlib import Path

from reviewgraph.baseline import review_baseline
from reviewgraph.config import get_settings
from reviewgraph.gitrepo import GitRepo
from reviewgraph.graph import review_agent
from reviewgraph.llm import get_chat_model
from reviewgraph.schemas import ReviewResult, Unit

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "out" / "bench_local.jsonl"


def summary(name: str, result: ReviewResult) -> str:
    stats = result.stats
    lines = [
        f"--- {name}: {stats.latency_s:.0f} s, {stats.llm_calls} model calls, {stats.tool_calls} tool calls, "
        f"{stats.input_tokens} input and {stats.output_tokens} output tokens"
    ]
    if result.error:
        lines.append(f"    error: {result.error[:300]}")
    for label, findings in (("finding", result.findings), ("dropped", result.dropped)):
        for finding in findings:
            lines.append(
                f"    {label}: lines {finding.line_start}-{finding.line_end} [{finding.severity}] "
                f"[{finding.category}] {' '.join(finding.explanation.split())[:160]}"
            )
    if not result.error and not result.findings and not result.dropped:
        lines.append("    no findings")
    for entry in result.trace:
        lines.append(f"    tool: {entry['status']} {entry['tool_name']}({json.dumps(entry['tool_args'])})")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    model_name = sys.argv[1]
    unit_number = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    lines = (ROOT / "data" / "units_dev.jsonl").read_text(encoding="utf-8").splitlines()
    unit = Unit.model_validate_json(lines[unit_number])

    settings = get_settings().model_copy(update={"llm_provider": "ollama", "ollama_model": model_name})
    model = get_chat_model(settings)
    repo = GitRepo(settings.target_repo_path)
    print(f"=== {model_name} on {unit.id}, budget {budget}", flush=True)

    baseline = review_baseline(repo, unit.commit, unit.path, unit_id=unit.id, model=model)
    print(summary("baseline", baseline), flush=True)
    agent = review_agent(repo, unit.commit, unit.path, unit_id=unit.id, model=model, max_tool_calls=budget)
    print(summary("agent", agent), flush=True)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8", newline="\n") as file:
        for result in (baseline, agent):
            record = result.model_dump()
            record["model"] = model_name
            record["budget"] = budget
            file.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
