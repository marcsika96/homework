"""Ask the configured model for one structured review and print the tokens.

Usage: python scripts/check_llm.py        (uses LLM_PROVIDER from .env)
"""

from langchain_core.messages import HumanMessage, SystemMessage

from reviewgraph.llm import add_usage, call_structured, get_chat_model, model_name
from reviewgraph.schemas import ReviewOutput, RunStats

CODE = """1 def average(numbers):
2     total = sum(numbers)
3     return total / (len(numbers) - 1)
"""

messages = [
    SystemMessage("You are a code reviewer. Report the defects of the file discount.py."),
    HumanMessage(CODE),
]
output, raw_messages = call_structured(get_chat_model(), ReviewOutput, messages)

stats = RunStats()
add_usage(stats, raw_messages)
print("model:", model_name())
for finding in output.findings:
    print(finding.model_dump_json(indent=2))
print("usage_metadata:", raw_messages[-1].usage_metadata)
print("stats:", stats)
