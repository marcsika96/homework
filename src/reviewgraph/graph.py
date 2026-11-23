import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, Send

from reviewgraph.config import get_settings
from reviewgraph.gitrepo import GitError, GitRepo
from reviewgraph.llm import RETRY_BACKOFF_FACTOR, RETRY_INITIAL_INTERVAL, RETRY_MAX_ATTEMPTS, StructuredOutputError, UsageCallback, call_structured, get_chat_model, is_transient_error, model_name, verifier_model
from reviewgraph.schemas import Finding, ReviewResult, Verification, clean_findings, ReviewOutput
from reviewgraph.tools import make_tools
from reviewgraph.prompts import FINAL_INSTRUCTION, LOOKUP_REQUEST, VERIFICATION_SYSTEM_PROMPT, build_agent_system_prompt, build_review_request, build_verification_request, load_rules, render_diff, render_lookups



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
    verify_model = verifier_model(model) #because thinking did not help the agent, but without thinking the verifier dropped correct findings.

    def load_context(state: ReviewState) -> dict:
        """Step 1. No LLM call: read the diff and the rules, start the conversation."""
        
        # return diff, rules, tool_calls_used=0 and the first two messages
        diff = render_diff(repo, sha, state["path"])
        rules = load_rules(repo, sha)
        messages = [
            SystemMessage(build_agent_system_prompt(max_tool_calls)),
            HumanMessage(build_review_request(state["path"], rules, diff)),
            HumanMessage(LOOKUP_REQUEST),
        ]
        return {
            "diff": diff,
            "rules": rules,
            "tool_calls_used": 0,
            "messages": messages
        }

    def lookup(state: ReviewState) -> dict:
        """Step 2a. The model either asks for tools or says it has enough context."""
        # TODO: lookup_model.invoke(state["messages"]) -> {"messages": [ai_message]}
        ai_message=lookup_model.invoke(state["messages"])
        return {
            "messages": [ai_message],
        }


    def run_tools(state: ReviewState) -> dict:
        """Step 2b. Execute the requested tool calls while budget remains."""
        #for each call in state["messages"][-1].tool_calls:
        #         budget left -> tools_by_name[call["name"]].invoke(call) gives a ToolMessage
        #         no budget   -> ToolMessage(content="Tool budget used up.", tool_call_id=call["id"])
        #       return the messages, the new tool_calls_used and one trace entry per call
        used_budget = state["tool_calls_used"]
        messages = []
        trace_entries = []
        for call in state["messages"][-1].tool_calls:
            if used_budget < max_tool_calls and call["name"] in tools_by_name:
                messages.append(tools_by_name[call["name"]].invoke(call))
                status = "executed"
                used_budget += 1
            elif used_budget < max_tool_calls and call["name"] not in tools_by_name:
                messages.append(ToolMessage(content=f"Tool {call['name']} not found. The available tools are: {', '.join(tools_by_name.keys())}.", tool_call_id=call["id"]))
                status = "unknown_tool"  
                used_budget += 1
            else:
                messages.append(ToolMessage(content="Tool budget used up.", tool_call_id=call["id"]))
                status = "budget_exceeded"
            trace_entry = {
                "tool_call_id": call["id"],
                "tool_name": call["name"],
                "tool_args": call["args"],
                "status": status,  
            }
            trace_entries.append(trace_entry)
      
        return {
            "messages": messages,
            "tool_calls_used": used_budget,
            "trace": trace_entries            
        }
    
    def after_lookup(state: ReviewState) -> Literal["run_tools", "write_findings"]:
        if state["messages"][-1].tool_calls:
            return "run_tools"
        else:
            return "write_findings"

    def after_tools(state: ReviewState) -> Literal["lookup", "write_findings"]:
        if state["tool_calls_used"] < max_tool_calls:
            return "lookup"
        else:
            return "write_findings"

    def write_findings(state: ReviewState) -> dict: 
       """Step 3. Structured ReviewOutput from everything gathered so far."""
       final_prompt = HumanMessage(FINAL_INSTRUCTION)
       output, raw_messages = call_structured(model, ReviewOutput, state["messages"] + [final_prompt])
       return {"draft": clean_findings(output.findings, repo.read(sha, state["path"]), state["path"])}

    def fan_out(state: ReviewState):
        """One verify_finding branch per draft finding."""
        if not state["draft"]:
            return "finalize"
        looked_up = render_lookups(state["messages"], state.get("trace", []))
        return [Send("verify_finding", {"finding": f, "commit": state["commit"], "looked_up": looked_up})
            for f in state["draft"]]

    def verify_finding(payload: dict) -> dict:
        """Step 4. Re-check one finding against the code it points at."""
        # `payload` is what Send passed, not the graph state.
        #  read the file around the finding's lines from the repo, ask for a verification
        finding = payload["finding"]
        file_text = repo.read(payload["commit"], finding.file)
        lines = file_text.splitlines()
        # set context to 20 lines before and after the finding, but not outside the file
        context=20
        first = max(1, finding.line_start - context)          # not before line 1
        last = min(len(lines), finding.line_end + context)    # not after the last line
        excerpt = lines[first - 1:last]               # slice end is exclusive, so `last` is included
        numbered = [f"{number:>5}   {line}" for number, line in enumerate(excerpt, start=first)]
        context_text = "\n".join(numbered)

        rules = load_rules(repo, payload["commit"])
        # Perform the verification logic here
        messages = [
            SystemMessage(VERIFICATION_SYSTEM_PROMPT),
                        HumanMessage(build_verification_request(finding, rules, context_text, payload["looked_up"])),
        ]
        try:
            output, raw_messages = call_structured(verify_model, Verification, messages)
            return {"checked": [{"finding": finding, "supported": output.supported, "reason": output.reason}]}
        except StructuredOutputError as e:
            return {"checked": [{"finding": finding, "supported": True, "reason": f"Not verified: {e}"}]}
        

    def sort_key(finding: Finding) -> tuple[int, int]:
        SEVERITY_RANK = {"high": 0 , "medium": 1  , "low": 2}
        return SEVERITY_RANK.get(finding.severity, 3), finding.line_start  # Default to 3 for unknown severities
    
    def finalize(state: ReviewState) -> dict:        
        #split state["checked"] into findings and dropped; sort by severity, then line
        checked = state.get("checked", [])
        findings = [c["finding"] for c in checked if c["supported"]]
        dropped = [c["finding"] for c in checked if not c["supported"]]
        findings.sort(key=sort_key)
        dropped.sort(key=sort_key)
        return {"findings": findings, "dropped": dropped}

    model_retry =RetryPolicy(max_attempts=RETRY_MAX_ATTEMPTS, initial_interval=RETRY_INITIAL_INTERVAL, backoff_factor=RETRY_BACKOFF_FACTOR, retry_on=is_transient_error)
    builder = StateGraph(ReviewState)
    builder.add_node("load_context", load_context)
    builder.add_node("lookup", lookup, retry_policy=model_retry)
    builder.add_node("run_tools", run_tools)
    builder.add_node("write_findings", write_findings, retry_policy=model_retry)
    builder.add_node("verify_finding", verify_finding, retry_policy=model_retry)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "lookup")
    builder.add_conditional_edges("lookup", after_lookup, {"run_tools": "run_tools", "write_findings": "write_findings"})
    builder.add_conditional_edges("run_tools", after_tools, {"lookup": "lookup", "write_findings": "write_findings"}) 
    builder.add_conditional_edges("write_findings", fan_out, {"verify_finding": "verify_finding", "finalize": "finalize"})
    builder.add_edge("verify_finding", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


# --- Exercise 4.4: one entry function ---------------------------------------
# Everything below this line was written by Claude Code (see notes.md).


def recursion_limit(max_tool_calls: int) -> int:
    """Upper bound for the steps of one run, as a safety net behind the tool budget.

    Every look-up round is two steps (lookup, run_tools) and uses at least one
    tool call while budget is left, so there are at most `max_tool_calls` rounds.
    The other nodes and one refused round fit into the 10 extra steps.
    """
    return 2 * max_tool_calls + 10


def review_agent(
    repo: GitRepo,
    sha: str,
    path: str,
    unit_id: str | None = None,
    model: BaseChatModel | None = None,
    max_tool_calls: int | None = None,
    on_step=None,
) -> ReviewResult:
    """Review the change of one file in one commit with the graph.

    The counterpart of review_baseline, with the same first arguments. Problems
    are reported in `result.error` instead of raising: an unknown commit or an
    unchanged file (no LLM call is made), an API failure after the retries, a
    reply that is still invalid after one retry, or a run that hits the
    recursion limit. Only a configuration error (missing key) is raised.

    After a failure `findings` is empty, but `stats` and `trace` hold what
    happened up to the last finished step.

    `on_step` is an optional function `on_step(node_name, update)`. It is called
    after every graph step with what the node returned; the command line uses
    it for --verbose.
    """
    result = ReviewResult(unit_id=unit_id or f"{sha[:7]}:{path}", system="agent", model=model_name(verifier=True))
    start = time.perf_counter()
    try:
        _review_with_graph(result, repo, sha, path, model, max_tool_calls, on_step)
    finally:
        result.stats.latency_s = round(time.perf_counter() - start, 3)
    return result


def _review_with_graph(
    result: ReviewResult,
    repo: GitRepo,
    sha: str,
    path: str,
    model: BaseChatModel | None,
    max_tool_calls: int | None,
    on_step=None,
) -> None:
    """Fill `result` with the findings or with the error."""
    # The same checks as the baseline, before any model is created or called.
    try:
        if not render_diff(repo, sha, path):
            result.error = f"Commit {sha[:7]} does not change {path}."
            return
    except GitError as e:
        result.error = str(e)
        return

    if max_tool_calls is None:
        max_tool_calls = get_settings().max_tool_calls
    model = model or get_chat_model()
    graph = build_graph(model, repo, sha, max_tool_calls=max_tool_calls)
    config = {
        "recursion_limit": recursion_limit(max_tool_calls),
        # The self-check branches run in parallel; two at a time is enough for the free tier.
        "max_concurrency": 2,
        "callbacks": [UsageCallback(result.stats)],
    }

    # Two stream modes at once, so every item is a pair (mode, data):
    # - "values" is the whole state after a step. Keeping the last one means the
    #   trace survives a failure in a later step.
    # - "updates" is {node name: what the node returned}, which on_step gets.
    state: dict = {}
    try:
        for mode, data in graph.stream({"commit": sha, "path": path}, config, stream_mode=["values", "updates"]):
            if mode == "values":
                state = data
            elif on_step:
                for node, update in data.items():
                    on_step(node, update)
        result.findings = state.get("findings", [])
        result.dropped = state.get("dropped", [])
    except StructuredOutputError as e:
        result.error = str(e)
    except GraphRecursionError:
        result.error = f"The review did not finish within {config['recursion_limit']} steps."
    except Exception as e:
        # API failure after the retries of the graph: rate limit, network, server error.
        result.error = f"LLM call failed: {type(e).__name__}: {e}"
    finally:
        result.trace = state.get("trace", [])
        result.stats.tool_calls = state.get("tool_calls_used", 0)
