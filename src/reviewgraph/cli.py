"""The command line of ReviewGraph.

Usage: reviewgraph review --repo <path> --commit <sha> [--system agent|baseline] [--verbose]
       python -m reviewgraph review ...        (the same)
"""

import argparse
import json
import sys

from reviewgraph.baseline import review_baseline
from reviewgraph.config import ConfigError
from reviewgraph.gitrepo import GitError, GitRepo
from reviewgraph.graph import review_agent
from reviewgraph.llm import OllamaError, check_ollama, get_chat_model
from reviewgraph.report import one_line, write_reports
from reviewgraph.schemas import ReviewResult, is_reviewable

# Exit codes: what the shell (or a CI job) sees after the command has finished.
EXIT_OK = 0
EXIT_BAD_INPUT = 2       # wrong repository, commit or configuration; argparse also uses 2
EXIT_REVIEW_FAILED = 3   # at least one file could not be reviewed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reviewgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="Review the Python files a commit changes")
    review.add_argument("--repo", required=True, help="Path of the local Git repository")
    review.add_argument("--commit", required=True, help="SHA of the commit to review")
    review.add_argument("--system", choices=["agent", "baseline"], default="agent")
    review.add_argument("--out", default="out", help="Folder for the reports; they go to <out>/<sha7>/")
    review.add_argument("--verbose", action="store_true", help="Print each graph step as it runs")

    sub.add_parser("eval", help="Run or recompute the evaluation (Part 6)")
    return parser


def print_step(node: str, update: dict) -> None:
    """--verbose: the name of the graph node that just ran, and its tool calls."""
    print(f"    [{node}]")
    if node == "lookup":
        for call in update["messages"][-1].tool_calls:
            print(f"        asks for {call['name']}({json.dumps(call['args'])})")
    elif node == "run_tools":
        for entry in update["trace"]:
            print(f"        {entry['status']}: {entry['tool_name']}")


def print_result(result: ReviewResult) -> None:
    """One progress line per file. The details are in the report at the end."""
    if result.error:
        print(f"    FAILED: {one_line(result.error)}")
    else:
        print(f"    {len(result.findings)} findings, {len(result.dropped)} dropped by the self-check")


def run_review(args: argparse.Namespace) -> int:
    # Step 1: Git only. A wrong repository or commit raises GitError here,
    # before a model exists, so these mistakes never cost an LLM call.
    repo = GitRepo(args.repo)
    sha = repo.resolve(args.commit)

    # Step 2: which files to review. A file without added lines (for example a
    # deleted file) has no new code, so it is skipped.
    paths = []
    for changed in repo.changed_files(sha):
        if is_reviewable(changed.path) and changed.added > 0:
            paths.append(changed.path)
    if not paths:
        print(f"Nothing to review: commit {sha[:7]} adds no Python code under app/studio/ "
              "(migrations and seed data are not reviewed).")
        return EXIT_OK

    # Step 3: only now create the model. A missing key raises ConfigError here,
    # and a local model that cannot be used raises OllamaError. Only the agent
    # calls tools, so only the agent needs a model with tool support.
    model = get_chat_model()
    check_ollama(needs_tools=args.system == "agent")
    on_step = print_step if args.verbose else None

    # Step 4: one file after the other. review_agent and review_baseline do not
    # raise for a failed review, they report it in result.error, so the loop goes on.
    print(f"Reviewing {len(paths)} files of commit {sha[:7]} with the {args.system}.")
    results = []
    for number, path in enumerate(paths, start=1):
        print(f"[{number}/{len(paths)}] {path}")
        if args.system == "agent":
            result = review_agent(repo, sha, path, model=model, on_step=on_step)
        else:
            result = review_baseline(repo, sha, path, model=model)
        print_result(result)
        results.append(result)

    # Step 5: the reports. The files are written first, so they exist even if
    # printing fails.
    markdown_path, json_path, markdown = write_reports(results, sha, args.system, args.out)
    print()
    print(markdown)
    print(f"Reports: {markdown_path} and {json_path}")

    # Step 6: the exit code says whether every file was reviewed.
    failed = [result for result in results if result.error]
    if failed:
        print(f"{len(failed)} of {len(results)} files could not be reviewed.", file=sys.stderr)
        return EXIT_REVIEW_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the command line. `argv` is for the tests; None means the real arguments."""
    args = build_parser().parse_args(argv)
    # The report contains text written by the model. If the terminal cannot show
    # a character, print "?" instead of stopping with a UnicodeEncodeError.
    sys.stdout.reconfigure(errors="replace")
    try:
        if args.command == "review":
            return run_review(args)
        print("Error: the eval command is not written yet (Part 6).", file=sys.stderr)
        return EXIT_BAD_INPUT
    except (GitError, ConfigError) as e:
        # Known mistakes get one readable line instead of a traceback.
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except OllamaError as e:
        # The input was fine, but the model cannot be reached: nothing was reviewed.
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_REVIEW_FAILED


if __name__ == "__main__":
    sys.exit(main())
