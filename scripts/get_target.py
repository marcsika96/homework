"""Clone (or update) the review target and record the increment commits.

Usage: python scripts/get_target.py

Reads TARGET_REPO_URL and TARGET_REPO_PATH from the environment or .env.
Writes data/increments.json, which maps I1..I8 to full commit SHAs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from reviewgraph.config import ConfigError, get_settings

ROOT = Path(__file__).resolve().parent.parent
INCREMENTS_FILE = ROOT / "data" / "increments.json"

# The commit that delivered each increment of the dance-studio app.
# I5 and I6 were followed by small fix-up commits (2be6655, e1e3f63, 807a50a);
# the increment is the first commit, which contains the generated code.
INCREMENTS = {
    "I1": "29949a3",
    "I2": "94725b9",
    "I3": "6afea22",
    "I4": "c89ea07",
    "I5": "19c7057",
    "I6": "5cad1b2",
    "I7": "de7dc35",
    "I8": "e594bc8",
}


class TargetError(Exception):
    """The target repository could not be fetched or does not look as expected."""


def run_git(args: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8"
        )
    except FileNotFoundError:
        raise TargetError("git is not installed or not on the PATH.") from None
    if result.returncode != 0:
        raise TargetError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def clone_or_fetch(url: str, target: Path) -> None:
    if (target / ".git").exists():
        # The remote branches before and after the fetch tell whether anything arrived.
        before = run_git(["for-each-ref", "refs/remotes"], cwd=target)
        run_git(["fetch", "--quiet"], cwd=target)
        after = run_git(["for-each-ref", "refs/remotes"], cwd=target)
        if before == after:
            print(f"Target is up to date: {target}")
        else:
            print(f"Fetched new commits into {target}")
        return

    if target.exists() and any(target.iterdir()):
        raise TargetError(f"{target} exists but is not a git repository. Remove it and run again.")
    if not url:
        raise TargetError("TARGET_REPO_URL is empty. Set it in .env (see .env.example).")

    print(f"Cloning {url} into {target} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_git(["clone", "--quiet", url, str(target)])
    print("Cloned.")


def resolve_increments(target: Path) -> dict[str, str]:
    resolved = {}
    missing = []
    for name, short_sha in INCREMENTS.items():
        try:
            resolved[name] = run_git(
                ["rev-parse", "--verify", "--quiet", short_sha + "^{commit}"], cwd=target
            )
        except TargetError:
            missing.append(f"{name} ({short_sha})")
    if missing:
        raise TargetError(
            "These increment commits are not in the target repository: "
            + ", ".join(missing)
            + ". Check that TARGET_REPO_URL points to dance-studio-webapp."
        )
    return resolved


def main() -> int:
    try:
        settings = get_settings()
        url = os.environ.get("TARGET_REPO_URL", "").strip()
        target = Path(settings.target_repo_path)
        if not target.is_absolute():
            target = ROOT / target

        clone_or_fetch(url, target)
        increments = resolve_increments(target)
    except (ConfigError, TargetError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    INCREMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INCREMENTS_FILE.write_text(json.dumps(increments, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(increments)} increment commits to {INCREMENTS_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
