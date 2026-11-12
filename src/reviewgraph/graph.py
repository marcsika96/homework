import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, Send

from reviewgraph.gitrepo import GitRepo
from reviewgraph.schemas import Finding
from reviewgraph.tools import make_tools
from reviewgraph.prompts import REVIEW_SYSTEM_PROMPT, build_review_request, load_rules, render_diff


class ReviewState(TypedDict, total=False):
    # input
    commit: str
    path: str
    # written by load_context
    diff: str
    rules: str
    # look-up loop
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_used: int
    trace: Annotated[list[dict], operator.add]
    # findings
    draft: list[Finding]
    checked: Annotated[list[dict], operator.add]
    findings: list[Finding]
    dropped: list[Finding]


def build_graph(model, repo: GitRepo, sha: str, max_tool_calls: int = 8):
    tools = make_tools(repo, sha)
    tools_by_name = {t.name: t for t in tools}
    lookup_model = model.bind_tools(tools)

    def load_context(state: ReviewState) -> dict:
        """Step 1. No LLM call: read the diff and the rules, start the conversation."""
        
        # return diff, rules, tool_calls_used=0 and the first two messages
        diff = render_diff(repo, sha, state["path"])
        rules = load_rules(repo, sha)
        tool_calls_used = 0
        messages = [
            SystemMessage(REVIEW_SYSTEM_PROMPT),
            HumanMessage(build_review_request(state["path"], rules, diff)),
        ]
        return {
            "diff": diff,
            "rules": rules,
            "tool_calls_used": 0,
            "messages": messages
        }

    builder = StateGraph(ReviewState)
    builder.add_node("load_context", load_context)
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", END)
    return builder.compile()
