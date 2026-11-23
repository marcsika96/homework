"""The command line of ReviewGraph.

Usage: reviewgraph review --repo <path> --commit <sha> [--system agent|baseline] [--verbose]
       python -m reviewgraph review ...        (the same)
"""

import argparse
import json
import sys

from reviewgraph.baseline import review_baseline
from reviewgraph.config import ConfigError, get_settings
from reviewgraph.evaluation.judge import VERDICTS_PATH, finding_id, judge_runs, load_verdicts, select_for_judging
from reviewgraph.evaluation.runner import RUNS_DIR, load_records, load_units, run_units
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

    evaluate = sub.add_parser("eval", help="Run both systems over a units file, or show the stored runs")
    evaluate.add_argument("--units", default="data/units_dev.jsonl", help="Units file, one unit per line")
    evaluate.add_argument("--system", choices=["agent", "baseline", "both"], default="both")
    evaluate.add_argument("--limit", type=int, help="Only the first N units of the file")
    evaluate.add_argument("--cached", action="store_true", help="Run nothing, only read the stored runs (needs no key)")
    evaluate.add_argument("--test-set", action="store_true", help="Allow units of the test set (only after the freeze)")
    evaluate.add_argument("--runs", default=str(RUNS_DIR), help="Folder of the stored runs")
    evaluate.add_argument("--judge", action="store_true",
                          help="Review nothing; let the judge rate the stored findings that have no verdict yet")
    evaluate.add_argument("--verdicts", default=str(VERDICTS_PATH), help="File of the judge's verdicts")
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


def run_eval(args: argparse.Namespace) -> int:
    # Step 1: which units. The test set is used once, so it needs its own flag.
    try:
        units = load_units(args.units)
    except OSError as e:
        print(f"Error: cannot read the units file {args.units}: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if args.limit:
        units = units[:args.limit]
    if any(unit.split == "test" for unit in units) and not args.test_set:
        print("Error: this file has units of the test set. They are run once, after the freeze; "
              "add --test-set if this is that run.", file=sys.stderr)
        return EXIT_BAD_INPUT
    systems = ["baseline", "agent"] if args.system == "both" else [args.system]

    # Step 2: the work that needs a model. One of three:
    #   --cached   nothing, so no repository, key or model is needed
    #   --judge    the judge rates the stored findings
    #   (default)  the systems review the units that are not stored yet
    quota_used_up = False
    overloaded = False
    if not args.cached:
        repo = GitRepo(get_settings().target_repo_path)
        model = get_chat_model()
        if args.judge:
            check_ollama()
            summary = judge_runs(units, systems, repo, model, runs_dir=args.runs, verdicts_path=args.verdicts)
            quota_used_up = summary["quota_used_up"]
        else:
            check_ollama(needs_tools="agent" in systems)
            for system in systems:
                summary = run_units(units, system, repo, model, runs_dir=args.runs)
                if summary["quota_used_up"]:
                    quota_used_up = True
                    break
                if summary["overloaded"]:
                    overloaded = True
                    break

    # Step 3: a summary of what is stored for these units.
    print()
    verdicts = load_verdicts(args.verdicts)
    for system in systems:
        records = load_records(units, system, runs_dir=args.runs)
        results = [record.result for record in records]
        print(f"{system}: {len(results)} of {len(units)} units stored, "
              f"{sum(len(result.findings) for result in results)} findings, "
              f"{sum(1 for result in results if result.error)} failed, "
              f"{sum(result.stats.llm_calls for result in results)} LLM calls, "
              f"{sum(result.stats.latency_s for result in results):.0f} s")

        # How many of the findings chosen for the judge have a verdict. The
        # labels are not counted here: that is the job of the metrics (6.5).
        chosen = [finding_id(result.unit_id, system, finding)
                  for result in results for finding in select_for_judging(result)]
        judged = [identifier for identifier in chosen if identifier in verdicts]
        print(f"    judge: {len(judged)} of {len(chosen)} selected findings have a verdict")

    if quota_used_up:
        print("Stopped: the daily quota is used up. Run the same command again tomorrow to continue.", file=sys.stderr)
        return EXIT_REVIEW_FAILED
    if overloaded:
        print("Stopped: the model is overloaded right now. Run the same command again later to continue.", file=sys.stderr)
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
        return run_eval(args)
    except KeyboardInterrupt:
        # Ctrl+C. For `eval` every finished unit is already stored.
        print("Interrupted.", file=sys.stderr)
        return EXIT_REVIEW_FAILED
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
