"""All prompts of ReviewGraph, and the functions that prepare their input.

The baseline and the agent share REVIEW_SYSTEM_PROMPT, the rules text and the
rendered diff, so the only differences between them are the look-up loop and
the self-check.
"""

import re

from reviewgraph.gitrepo import GitRepo, PathNotFound

# Bump on every change of a prompt in this module. It is stored with every run.
PROMPT_VERSION = "1"

RULES_FILE = "AGENTS.md"
NO_RULES = "(This commit has no AGENTS.md. Review against general Django and Python practice.)"

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

REVIEW_SYSTEM_PROMPT = """\
You are a senior Django developer who reviews one changed Python file of a \
dance-studio web application. The code was written by an AI coding agent, and \
the developer wants a second opinion before accepting the change.

## What you get

- The project rules (AGENTS.md) as they were at the reviewed commit.
- The change to one file. Every line starts with its line number in the new \
version of the file. Lines marked with "+" were added by the commit, lines \
marked with "-" were removed and have no number, all other lines are unchanged \
context. "..." stands for unchanged lines that are not shown.

## What counts as a finding

Report a defect only if it is in the added lines, or is caused by them, and it is one of these:

- The code behaves wrongly: wrong result, crash, lost or corrupted data, a \
user seeing or changing something they must not.
- The code breaks a project rule from AGENTS.md.

Do not report style, naming, missing comments, missing tests, or things you \
would merely do differently. Do not guess about code you cannot see: if a \
finding depends on what another file does, report it only when the visible \
code makes the defect certain.

## Categories

- authorization: a missing or wrong check of who may do or see something
- transaction: database work that must be atomic or locked and is not; race conditions
- model-rule: a rule of the data model is violated or not enforced (uniqueness, required relations, allowed states)
- validation: user input that is not checked or is checked wrongly
- error-handling: an error that is swallowed, not handled, or reported wrongly
- logic: a wrong condition, calculation, or order of steps
- project-rule: a rule from AGENTS.md is broken and no category above fits
- other: a real defect that fits none of the above

## Severity

- high: wrong behaviour or wrong data in normal use, or a security problem
- medium: wrong behaviour only in some cases (an edge case, a race, an unusual input)
- low: correct behaviour, but fragile or misleading code

## Output

For every finding give the line range in the new version of the file, the \
category, the severity, what is wrong and what happens because of it, and the \
concrete change that fixes it. Use the line numbers printed in front of the \
lines.

If the change has no defect, return an empty list. An empty list is a good \
answer: most changes are fine, and a wrong finding costs the developer more \
time than a missed one.
"""

REVIEW_REQUEST = """\
## Project rules (AGENTS.md)

{rules}

## Change to review: {path}

{diff}
"""


def render_diff(repo: GitRepo, sha: str, path: str, context: int = 10) -> str:
    """The commit's change to `path`, with the new-file line number in front of every line.

    Added lines are marked with "+", removed lines with "-" and no number.
    Returns "" if the commit did not change the file.

        12   def book(request):
        13 +     dancer = request.user
           -     dancer = None
    """
    lines = []
    new_line = None                       # None until the first hunk header
    for line in repo.diff(sha, path, context).splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            if lines:
                lines.append("   ...")
            new_line = int(header.group(1))
        elif new_line is None or line.startswith("\\"):
            continue                      # file header, or "\ No newline at end of file"
        elif line.startswith("+"):
            lines.append(f"{new_line:>5} + {line[1:]}")
            new_line += 1
        elif line.startswith("-"):
            lines.append(f"{'':>5} - {line[1:]}")
        else:
            lines.append(f"{new_line:>5}   {line[1:]}")
            new_line += 1
    return "\n".join(lines)


def load_rules(repo: GitRepo, sha: str) -> str:
    """AGENTS.md as it was at the commit, or a placeholder if it did not exist yet."""
    try:
        return repo.read(sha, RULES_FILE).strip()
    except PathNotFound:
        return NO_RULES


def build_review_request(path: str, rules: str, diff: str) -> str:
    """The user message of a review: the rules and the rendered change."""
    return REVIEW_REQUEST.format(path=path, rules=rules, diff=diff)
