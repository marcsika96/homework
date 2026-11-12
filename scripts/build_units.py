"""Build the review units from the increment commits of the target repository.

Usage: python scripts/build_units.py

Reads data/increments.json (written by scripts/get_target.py) and the repository
at TARGET_REPO_PATH. Writes data/units_dev.jsonl and data/units_test.jsonl,
one JSON object per line.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from reviewgraph.config import ConfigError, get_settings
from reviewgraph.gitrepo import GitError, GitRepo
from reviewgraph.schemas import DEV_INCREMENTS, Unit, build_units

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INCREMENTS_FILE = DATA / "increments.json"


def write_jsonl(path: Path, units: list[Unit]) -> None:
    # newline="\n" keeps the file byte-identical on Windows and Linux.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for unit in units:
            f.write(unit.model_dump_json() + "\n")


def main() -> int:
    if not INCREMENTS_FILE.exists():
        print("Error: data/increments.json is missing. Run scripts/get_target.py first.", file=sys.stderr)
        return 1
    increments = json.loads(INCREMENTS_FILE.read_text(encoding="utf-8"))

    try:
        target = Path(get_settings().target_repo_path)
        if not target.is_absolute():
            target = ROOT / target
        units = build_units(GitRepo(target), increments)
    except (ConfigError, GitError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    dev_units = [unit for unit in units if unit.split == "dev"]
    test_units = [unit for unit in units if unit.split == "test"]
    write_jsonl(DATA / "units_dev.jsonl", dev_units)
    write_jsonl(DATA / "units_test.jsonl", test_units)

    per_increment = Counter(unit.increment for unit in units)
    for increment in increments:
        split = "dev" if increment in DEV_INCREMENTS else "test"
        print(f"{increment}: {per_increment[increment]:2d} units ({split})")
    print(f"{len(dev_units)} dev units  -> data/units_dev.jsonl")
    print(f"{len(test_units)} test units -> data/units_test.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
