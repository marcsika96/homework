"""Exercise 4.5: the graph without a model, a key or a network.

Layer 1: the routing functions as plain functions, called with small state dicts.
Layer 2: the whole graph with a scripted model on the fixture repository.
"""

import ollama
from conftest import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Send

from reviewgraph.graph import build_graph
from reviewgraph.schemas import Finding, ReviewOutput, Verification

VIEWS = "app/studio/views.py"
MODELS = "app/studio/models.py"
BUDGET = 2


def routing(tiny_repo, node: str, name: str, max_tool_calls: int = BUDGET):
    """A routing function of the graph, as the plain Python function that was passed in.

    The functions are defined inside build_graph(), so they cannot be imported.
    The compiled graph keeps them: builder.branches[source node][function name].
    """
    graph = build_graph(ScriptedChatModel(), tiny_repo.repo, tiny_repo.second_sha, max_tool_calls=max_tool_calls)
    return graph.builder.branches[node][name].path.func


def asks_for_a_tool() -> AIMessage:
    return AIMessage(content="", tool_calls=[tool_call("read_file", "call-1", path=MODELS)])


def make_finding(line: int, severity: str = "medium") -> Finding:
    return Finding(
        file=VIEWS,
        line_start=line,
        line_end=line,
        category="project-rule",
        severity=severity,
        explanation=f"Line {line} reads from the database directly.",
        fix="Move the query into a database function and call that.",
    )


# --- Layer 1: routing functions ---------------------------------------------


def test_after_lookup_runs_the_tools_when_the_model_asked_for_some(tiny_repo):
    after_lookup = routing(tiny_repo, "lookup", "after_lookup")

    assert after_lookup({"messages": [asks_for_a_tool()], "tool_calls_used": 0}) == "run_tools"


def test_after_lookup_runs_the_tools_also_when_the_budget_is_used(tiny_repo):
    # Decision of 2026-11-16: every tool call needs a ToolMessage, so the calls
    # go to run_tools, which refuses them, and not straight to write_findings.
    after_lookup = routing(tiny_repo, "lookup", "after_lookup")

    assert after_lookup({"messages": [asks_for_a_tool()], "tool_calls_used": BUDGET}) == "run_tools"


def test_after_lookup_writes_the_findings_when_there_are_no_tool_calls(tiny_repo):
    after_lookup = routing(tiny_repo, "lookup", "after_lookup")

    state = {"messages": [AIMessage(content="I have what I need.")], "tool_calls_used": 1}
    assert after_lookup(state) == "write_findings"


def test_after_tools_goes_back_to_lookup_while_budget_is_left(tiny_repo):
    after_tools = routing(tiny_repo, "run_tools", "after_tools")

    assert after_tools({"tool_calls_used": BUDGET - 1}) == "lookup"


def test_after_tools_writes_the_findings_when_the_budget_is_used(tiny_repo):
    after_tools = routing(tiny_repo, "run_tools", "after_tools")

    assert after_tools({"tool_calls_used": BUDGET}) == "write_findings"


def test_after_tools_with_a_budget_of_zero_never_goes_back(tiny_repo):
    after_tools = routing(tiny_repo, "run_tools", "after_tools", max_tool_calls=0)

    assert after_tools({"tool_calls_used": 0}) == "write_findings"


def test_fan_out_sends_one_branch_per_draft_finding(tiny_repo):
    fan_out = routing(tiny_repo, "write_findings", "fan_out")
    draft = [make_finding(3), make_finding(7)]
    messages = [asks_for_a_tool(), ToolMessage(content="class DanceClass(models.Model):", tool_call_id="call-1")]
    trace = [{"tool_call_id": "call-1", "tool_name": "read_file", "tool_args": {"path": MODELS}, "status": "executed"}]

    sends = fan_out({"draft": draft, "commit": tiny_repo.second_sha, "messages": messages, "trace": trace})

    assert all(isinstance(send, Send) and send.node == "verify_finding" for send in sends)
    assert [send.arg["finding"] for send in sends] == draft
    # Every branch gets the same rendered look-ups (render_lookups).
    for send in sends:
        assert send.arg["looked_up"] == f"### read_file(path='{MODELS}')\n\nclass DanceClass(models.Model):"


def test_fan_out_goes_straight_to_finalize_without_findings(tiny_repo):
    fan_out = routing(tiny_repo, "write_findings", "fan_out")

    assert fan_out({"draft": [], "commit": tiny_repo.second_sha}) == "finalize"


# --- Layer 2: the whole graph with a scripted model -------------------------


def scripted_run(tiny_repo, max_tool_calls: int) -> tuple[dict, ScriptedChatModel]:
    """The run the plan describes: three tool calls, then the findings and two verdicts."""
    three_calls = AIMessage(
        content="",
        tool_calls=[
            tool_call("read_file", "call-1", path=MODELS),
            tool_call("search_code", "call-2", pattern="DanceClass"),
            tool_call("list_files", "call-3", directory="app/studio"),
        ],
    )

    def verdict(messages) -> Verification:
        # The two self-checks run in parallel, so the reply depends on the finding, not on the order.
        return Verification(supported="- lines: 7-7\n" in messages[-1].content, reason="Scripted verdict.")

    replies = [three_calls]
    if max_tool_calls > 3:
        # With budget left the graph asks the model again, and it says it is done.
        replies.append(AIMessage(content="I have what I need."))
    replies += [ReviewOutput(findings=[make_finding(3), make_finding(7)]), verdict, verdict]

    model = ScriptedChatModel(replies=replies)
    graph = build_graph(model, tiny_repo.repo, tiny_repo.second_sha, max_tool_calls=max_tool_calls)
    state = graph.invoke({"commit": tiny_repo.second_sha, "path": VIEWS})
    assert model.replies == []          # every prepared reply was used
    return state, model


def test_scripted_run_ends_with_one_finding_and_one_dropped(tiny_repo):
    state, model = scripted_run(tiny_repo, max_tool_calls=8)

    assert state["findings"] == [make_finding(7)]
    assert state["dropped"] == [make_finding(3)]
    assert len(state["draft"]) == len(state["findings"]) + len(state["dropped"])
    assert {entry["supported"] for entry in state["checked"]} == {True, False}
    assert state["tool_calls_used"] == 3
    assert [entry["status"] for entry in state["trace"]] == ["executed"] * 3
    assert len(model.received) == 5     # two look-ups, the findings, two self-checks


def test_with_a_budget_of_two_the_third_tool_call_gets_the_budget_message(tiny_repo):
    state, model = scripted_run(tiny_repo, max_tool_calls=2)

    answers = [message for message in state["messages"] if isinstance(message, ToolMessage)]
    assert [answer.tool_call_id for answer in answers] == ["call-1", "call-2", "call-3"]
    assert answers[2].content == "Tool budget used up."
    assert [entry["status"] for entry in state["trace"]] == ["executed", "executed", "budget_exceeded"]
    assert state["tool_calls_used"] == 2
    # The budget is used up, so the model is not asked for more tools.
    assert len(model.received) == 4
    assert state["findings"] == [make_finding(7)] and state["dropped"] == [make_finding(3)]


def test_self_check_uses_the_verifier_model_and_the_review_does_not(tiny_repo, monkeypatch):
    # The verifier variant (OLLAMA_VERIFY_REASONING) as its own scripted model, so
    # the test sees which model each node called.
    verdict = Verification(supported=True, reason="Scripted verdict.")
    verifier = ScriptedChatModel(replies=[verdict, verdict])
    monkeypatch.setattr("reviewgraph.graph.verifier_model", lambda model: verifier)
    model = ScriptedChatModel(replies=[
        AIMessage(content="Nothing to look up."),
        ReviewOutput(findings=[make_finding(3), make_finding(7)]),
    ])

    graph = build_graph(model, tiny_repo.repo, tiny_repo.second_sha, max_tool_calls=BUDGET)
    state = graph.invoke({"commit": tiny_repo.second_sha, "path": VIEWS})

    assert len(model.received) == 2         # the look-up and the findings
    assert len(verifier.received) == 2      # one self-check per draft finding
    assert state["findings"] == [make_finding(3), make_finding(7)]


def test_self_check_gets_a_fresh_context_with_the_code_around_the_finding(tiny_repo):
    state, model = scripted_run(tiny_repo, max_tool_calls=8)

    look_up_conversation = model.received[0]
    for self_check in model.received[-2:]:
        assert len(self_check) == 2                                  # system prompt and one request
        assert self_check[0].content != look_up_conversation[0].content
        request = self_check[1].content
        assert "Views never touch the database directly." in request      # the rules
        assert "    7       return render(request" in request             # numbered code of views.py
        # The tool results the reviewer saw (decision of 2026-11-23) ...
        assert f"### read_file(path='{MODELS}')" in request
        assert "PositiveIntegerField" in request                          # from models.py
        # ... but not the reviewer's own words or the look-up instructions.
        assert "I have what I need." not in request
        assert "Step 1 of 2: look-up." not in request


def test_an_unreadable_look_up_answer_ends_the_look_up_and_the_review_goes_on(tiny_repo):
    # qwen3.5:9b on dev unit 25 (2026-11-23): Ollama could not parse the model's tool call.
    broken = ollama.ResponseError("XML syntax error on line 3: element <function> closed by </parameter>")
    model = ScriptedChatModel(replies=[
        asks_for_a_tool(),                                   # round 1: one tool call, executed
        broken,                                              # round 2: the answer cannot be read
        ReviewOutput(findings=[make_finding(7)]),
        Verification(supported=True, reason="Scripted verdict."),
    ])

    graph = build_graph(model, tiny_repo.repo, tiny_repo.second_sha, max_tool_calls=8)
    state = graph.invoke({"commit": tiny_repo.second_sha, "path": VIEWS})

    assert len(model.received) == 4                         # not retried, no third look-up round
    assert [entry["status"] for entry in state["trace"]] == ["executed", "unreadable_reply"]
    assert "XML syntax error" in state["trace"][-1]["error"]
    assert state["tool_calls_used"] == 1
    # The findings are written with what was looked up before the broken answer.
    assert "PositiveIntegerField" in "\n".join(str(m.content) for m in model.received[2])
    assert state["findings"] == [make_finding(7)]
