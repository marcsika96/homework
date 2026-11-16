"""Run the agent on one dev unit and print its result, for comparison with try_baseline.py.

Usage: python scripts/try_agent.py            (lists the dev units)
       python scripts/try_agent.py 20         (reviews dev unit number 20)
       python scripts/try_agent.py 20 2       (the same with a budget of 2 tool calls)

One review makes up to about 12 model requests (the Gemini free tier allows 20 per day).
Only dev units can be chosen: the test units stay unopened until the evaluation.
"""

import sys
from pathlib import Path

from reviewgraph.config import ConfigError, get_settings
from reviewgraph.gitrepo import GitRepo
from reviewgraph.graph import review_agent
from reviewgraph.schemas import Unit

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    lines = (ROOT / "data" / "units_dev.jsonl").read_text(encoding="utf-8").splitlines()
    units = [Unit.model_validate_json(line) for line in lines]

    if len(sys.argv) < 2:
        for number, unit in enumerate(units):
            print(f"{number:2d}  {unit.id}  (+{unit.added_lines} lines)")
        return 0

    try:
        unit = units[int(sys.argv[1])]
        max_tool_calls = int(sys.argv[2]) if len(sys.argv) > 2 else None
    except (ValueError, IndexError):
        print(f"Error: give a unit number between 0 and {len(units) - 1}, and a whole number as budget.", file=sys.stderr)
        return 1

    try:
        repo = GitRepo(get_settings().target_repo_path)
        print(f"--- {unit.id}: reviewing with the agent, this can take a few minutes ...")
        result = review_agent(repo, unit.commit, unit.path, unit_id=unit.id, max_tool_calls=max_tool_calls)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
