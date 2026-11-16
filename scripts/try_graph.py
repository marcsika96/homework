"""Stream the look-up loop of the graph on one dev unit and print each step.

Usage: python scripts/try_graph.py            (lists the dev units)
       python scripts/try_graph.py 20         (runs dev unit number 20)
       python scripts/try_graph.py 20 2       (the same with a budget of 2 tool calls)

Only dev units can be chosen: the test units stay unopened until the evaluation.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from reviewgraph.config import ConfigError, get_settings
from reviewgraph.gitrepo import GitRepo
from reviewgraph.graph import build_graph
from reviewgraph.llm import get_chat_model
from reviewgraph.schemas import Unit

ROOT = Path(__file__).resolve().parent.parent
PREVIEW_CHARS = 200


def preview(text: str) -> str:
    """The text on one line, cut after PREVIEW_CHARS characters."""
    line = " ".join(str(text).split())
    return line if len(line) <= PREVIEW_CHARS else line[:PREVIEW_CHARS] + " ..."


def describe(finding) -> str:
    """One finding on one line: lines, severity, category and the start of the explanation."""
    lines = f"{finding.line_start}-{finding.line_end}"
    return f"lines {lines} [{finding.severity}] [{finding.category}] {preview(finding.explanation)}"


def print_update(node: str, update: dict) -> None:
    """Short summary of what one node wrote to the state."""
    print(f"--- {node}")
    if node == "load_context":
        print(f"    diff: {len(update['diff'].splitlines())} lines, rules: {len(update['rules'])} characters")
    elif node == "lookup":
        message = update["messages"][-1]
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"    asks for {call['name']}({json.dumps(call['args'])})")
        else:
            print(f"    no tool calls: {preview(message.text)}")
    elif node == "run_tools":
        for entry, message in zip(update["trace"], update["messages"]):
            print(f"    {entry['status']}: {entry['tool_name']}({json.dumps(entry['tool_args'])})")
            print(f"        -> {preview(message.content)}")
        print(f"    tool_calls_used: {update['tool_calls_used']}")
    elif node == "write_findings":
        print(f"    draft: {len(update['draft'])} findings")
        for finding in update["draft"]:
            print(f"    {describe(finding)}")
    elif node == "verify_finding":
        for entry in update["checked"]:
            verdict = "supported" if entry["supported"] else "not supported"
            print(f"    {verdict}: {describe(entry['finding'])}")
            print(f"        -> {preview(entry['reason'])}")
    elif node == "finalize":
        for key in ("findings", "dropped"):
            print(f"    {key}: {len(update[key])}")
            for finding in update[key]:
                print(f"        {describe(finding)}")
    else:
        print(f"    keys: {', '.join(update)}")


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

    try:
        settings = get_settings()
        max_tool_calls = int(sys.argv[2]) if len(sys.argv) > 2 else settings.max_tool_calls
        model = get_chat_model(settings)
    except ValueError:
        print("Error: the tool budget has to be a whole number.", file=sys.stderr)
        return 1
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    repo = GitRepo(settings.target_repo_path)
    graph = build_graph(model, repo, unit.commit, max_tool_calls=max_tool_calls)

    print(f"{unit.id}, budget: {max_tool_calls} tool calls")
    statuses = Counter()
    inputs = {"commit": unit.commit, "path": unit.path}
    for step in graph.stream(inputs, stream_mode="updates"):
        # With stream_mode="updates" every step is {node name: what the node returned}.
        for node, update in step.items():
            print_update(node, update)
            statuses.update(entry["status"] for entry in update.get("trace", []))

    print("--- end")
    print(f"    trace: {dict(statuses) or 'no tool calls'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
