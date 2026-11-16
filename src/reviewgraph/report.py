"""Turns the results of a review into two reports: Markdown for people, JSON for programs."""

import json
from pathlib import Path

from reviewgraph.schemas import Finding, ReviewResult

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def file_path(result: ReviewResult) -> str:
    """The reviewed path. A unit id is "<commit or increment>:<path>"."""
    return result.unit_id.split(":", 1)[1]


def location(finding: Finding) -> str:
    """"path:line", or "path:first-last" when the finding covers several lines."""
    if finding.line_start == finding.line_end:
        return f"{finding.file}:{finding.line_start}"
    return f"{finding.file}:{finding.line_start}-{finding.line_end}"


def cell(text: str) -> str:
    """Text that is safe inside a Markdown table: one line, and no | that would end the cell."""
    return " ".join(text.split()).replace("|", "\\|")


def one_line(text: str, limit: int = 300) -> str:
    """An error message on one line, cut after `limit` characters.

    API errors can be many lines of JSON. The full text stays in review.json.
    """
    line = " ".join(text.split())
    return line if len(line) <= limit else line[:limit] + " ..."


def render_file(result: ReviewResult) -> list[str]:
    """The lines of the section of one file."""
    lines = [f"## {file_path(result)}", ""]

    # Three cases a reader has to tell apart: not reviewed, nothing found, findings.
    if result.error:
        lines += [f"**Not reviewed:** {one_line(result.error)}", ""]
        return lines
    if not result.findings:
        lines += ["Reviewed, no findings.", ""]
    else:
        lines += [
            "| Location | Category | Severity | Explanation | Fix |",
            "|----------|----------|----------|-------------|-----|",
        ]
        findings = sorted(result.findings, key=lambda f: (SEVERITY_RANK[f.severity], f.line_start))
        for f in findings:
            lines.append(f"| {location(f)} | {f.category} | {f.severity} | {cell(f.explanation)} | {cell(f.fix)} |")
        lines.append("")

    if result.dropped:
        lines += [f"Dropped by the self-check: {len(result.dropped)}", ""]
    return lines


def render_markdown(results: list[ReviewResult], commit: str, system: str) -> str:
    """The whole report: a header, one section per file, a footer with the costs."""
    model = results[0].model if results else "-"
    lines = [
        f"# Review of commit {commit[:7]}",
        "",
        f"- System: {system}",
        f"- Model: {model}",
        f"- Files: {len(results)}",
        "",
    ]

    for result in results:
        lines += render_file(result)

    stats = [result.stats for result in results]
    lines += [
        "---",
        "",
        f"LLM calls: {sum(s.llm_calls for s in stats)}, "
        f"tool calls: {sum(s.tool_calls for s in stats)}, "
        f"tokens: {sum(s.input_tokens for s in stats)} in and {sum(s.output_tokens for s in stats)} out, "
        f"time: {sum(s.latency_s for s in stats):.1f} s",
    ]
    return "\n".join(lines) + "\n"


def write_reports(results: list[ReviewResult], commit: str, system: str, out_dir: str | Path) -> tuple[Path, Path, str]:
    """Write review.md and review.json to <out_dir>/<sha7>/. Returns both paths and the Markdown text."""
    folder = Path(out_dir) / commit[:7]
    folder.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(results, commit, system)
    markdown_path = folder / "review.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    # model_dump() turns a Pydantic object into plain dicts and lists, which json can write.
    json_path = folder / "review.json"
    data = [result.model_dump() for result in results]
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return markdown_path, json_path, markdown
