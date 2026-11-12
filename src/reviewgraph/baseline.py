"""System B of the evaluation: one prompt with the rules and the diff, no tools, no self-check."""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from reviewgraph.gitrepo import GitError, GitRepo
from reviewgraph.llm import StructuredOutputError, add_usage, call_structured, get_chat_model, model_name
from reviewgraph.prompts import REVIEW_SYSTEM_PROMPT, build_review_request, load_rules, render_diff
from reviewgraph.schemas import ReviewOutput, ReviewResult, clean_findings


def review_baseline(
    repo: GitRepo,
    sha: str,
    path: str,
    unit_id: str | None = None,
    model: BaseChatModel | None = None,
) -> ReviewResult:
    """Review the change of one file in one commit with a single LLM call.

    Problems are reported in `result.error` instead of raising: an unknown
    commit or an unchanged file (no LLM call is made), an API failure, or a
    reply that is still invalid after one retry.
    """
    result = ReviewResult(unit_id=unit_id or f"{sha[:7]}:{path}", system="baseline", model=model_name())
    start = time.perf_counter()
    try:
        _review(result, repo, sha, path, model)
    finally:
        result.stats.latency_s = round(time.perf_counter() - start, 3)
    return result


def _review(result: ReviewResult, repo: GitRepo, sha: str, path: str, model: BaseChatModel | None) -> None:
    """Fill `result` with the findings or with the error."""
    try:
        diff = render_diff(repo, sha, path)
        if not diff:
            result.error = f"Commit {sha[:7]} does not change {path}."
            return
        file_text = repo.read(sha, path)
        rules = load_rules(repo, sha)
    except GitError as e:
        result.error = str(e)
        return

    model = model or get_chat_model()
    messages = [
        SystemMessage(REVIEW_SYSTEM_PROMPT),
        HumanMessage(build_review_request(path, rules, diff)),
    ]
    try:
        output, raw_messages = call_structured(model, ReviewOutput, messages)
        add_usage(result.stats, raw_messages)
        result.findings = clean_findings(output.findings, file_text, path)
    except StructuredOutputError as e:
        add_usage(result.stats, e.raw_messages)
        result.error = str(e)
    except Exception as e:
        # API failure after the model's own retries: rate limit, network, server error.
        result.stats.llm_calls += 1
        result.error = f"LLM call failed: {type(e).__name__}: {e}"
