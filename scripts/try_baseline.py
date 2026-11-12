"""Run the baseline on one dev unit and print what the model saw and what it found.

Usage: python scripts/try_baseline.py            (lists the dev units)
       python scripts/try_baseline.py 20         (reviews dev unit number 20)

Only dev units can be chosen: the test units stay unopened until the evaluation.
"""

import json
import sys
from pathlib import Path

from reviewgraph.baseline import review_baseline
from reviewgraph.config import get_settings
from reviewgraph.gitrepo import GitRepo
from reviewgraph.prompts import render_diff
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
    except (ValueError, IndexError):
        print(f"Error: give a unit number between 0 and {len(units) - 1}.", file=sys.stderr)
        return 1

    repo = GitRepo(get_settings().target_repo_path)
    print(f"--- {unit.id}: the change the model sees")
    print(render_diff(repo, unit.commit, unit.path))

    print("--- reviewing, this can take a minute or two ...")
    result = review_baseline(repo, unit.commit, unit.path, unit_id=unit.id)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
