"""Write the diagram of the compiled graph to docs/architecture.md.

Usage: python scripts/draw_graph.py

The diagram is generated from the graph that build_graph() returns, so it
cannot drift from the implementation. tests/test_agent.py fails when the file
is out of date. Needs no key, no model and no target repository.
"""

import sys
from pathlib import Path

from reviewgraph.graph import build_graph

ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_FILE = ROOT / "docs" / "architecture.md"

PAGE = """\
# Architecture

The review of one changed file is a LangGraph state graph. This diagram is generated from the compiled graph by `scripts/draw_graph.py`; do not edit it by hand.

```mermaid
{diagram}
```

| Node | LLM calls | What it does |
|------|-----------|--------------|
| `load_context` | 0 | Reads the diff of the file and `AGENTS.md` at the reviewed commit and starts the conversation. |
| `lookup` | 1 per round | The model asks for read-only tools, or says it has enough context. |
| `run_tools` | 0 | Runs the requested tool calls while the budget lasts and refuses the rest. |
| `write_findings` | 1 | Turns the conversation into a structured list of draft findings. |
| `verify_finding` | 1 per finding | Checks one finding against the code again, in a fresh context. The findings are checked in parallel. |
| `finalize` | 0 | Splits the checked findings into reported and dropped ones and sorts them. |

A solid arrow is a fixed edge. A dotted arrow is a conditional edge: `lookup` goes to `run_tools` when the model asked for tools, `run_tools` goes back to `lookup` while tool budget is left, and `write_findings` starts one `verify_finding` per draft finding, or goes straight to `finalize` when there is none.
"""


class NoModel:
    """build_graph() only binds the tools at build time; no model is called here."""

    def bind_tools(self, tools):
        return self


class NoRepo:
    """build_graph() only resolves the commit at build time; nothing is read here."""

    def resolve(self, rev: str) -> str:
        return rev


def main() -> int:
    graph = build_graph(NoModel(), NoRepo(), "HEAD")
    diagram = graph.get_graph().draw_mermaid().strip()
    ARCHITECTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARCHITECTURE_FILE.write_text(PAGE.format(diagram=diagram), encoding="utf-8", newline="\n")
    print(f"Wrote {ARCHITECTURE_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
