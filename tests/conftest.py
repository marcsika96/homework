import subprocess
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from reviewgraph.gitrepo import GitRepo

MODELS_V1 = """from django.db import models


class DanceClass(models.Model):
    title = models.CharField(max_length=100)
"""

MODELS_V2 = """from django.db import models


class DanceClass(models.Model):
    title = models.CharField(max_length=120)
    capacity = models.PositiveIntegerField(default=10)
"""

VIEWS = """from django.shortcuts import render

from .models import DanceClass


def class_list(request):
    return render(request, "classes.html", {"classes": DanceClass.objects.all()})
"""

AGENTS = """# Project rules

- Views never touch the database directly.
"""


class TinyRepo:
    """A two-commit repository that stands in for the private review target."""

    def __init__(self, path: Path, first_sha: str, second_sha: str):
        self.path = path
        self.first_sha = first_sha
        self.second_sha = second_sha
        self.repo = GitRepo(path)


def git(path: Path, *args: str) -> str:
    # Name, e-mail and signing are set here, so the fixture works on a clean machine.
    command = [
        "git",
        "-c", "user.name=Test",
        "-c", "user.email=test@example.com",
        "-c", "commit.gpgsign=false",
        "-c", "core.autocrlf=false",
        "-C", str(path),
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


@pytest.fixture
def tiny_repo(tmp_path) -> TinyRepo:
    path = tmp_path / "target"
    path.mkdir()
    git(path, "init", "--quiet", "-b", "main")

    write(path / "app/studio/models.py", MODELS_V1)
    write(path / "AGENTS.md", AGENTS)
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "first")
    first_sha = git(path, "rev-parse", "HEAD")

    write(path / "app/studio/models.py", MODELS_V2)
    write(path / "app/studio/views.py", VIEWS)
    (path / "app/studio/logo.png").write_bytes(b"\x89PNG\x00\x01\x02\x00")
    git(path, "add", "-A")
    git(path, "commit", "--quiet", "-m", "second")
    second_sha = git(path, "rev-parse", "HEAD")

    return TinyRepo(path, first_sha, second_sha)


# --- A chat model that needs no network ------------------------------------


def raw_message(input_tokens: int, output_tokens: int) -> AIMessage:
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return AIMessage(content="", usage_metadata=usage)


def reply(parsed, error=None, tokens=(100, 20)) -> dict:
    """One result of with_structured_output(..., include_raw=True)."""
    return {"raw": raw_message(*tokens), "parsed": parsed, "parsing_error": error}


class FakeModel:
    """Stands in for a chat model: returns the prepared replies one after the other."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.received = []          # the message list of every call

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw
        return self

    def invoke(self, messages):
        self.received.append(messages)
        next_reply = self.replies.pop(0)
        if isinstance(next_reply, Exception):
            raise next_reply            # an API failure
        return next_reply


# --- A scripted chat model for the graph ------------------------------------

REPLY_TOKENS = (100, 20)    # input and output tokens reported for every scripted reply


class ScriptedChatModel(BaseChatModel):
    """A real LangChain chat model that replays prepared replies, without a network.

    Unlike FakeModel it goes through LangChain's own invoke(), so callbacks and
    the retry policy of the graph see its calls. A reply is one of:

    - an AIMessage: returned as it is (a look-up answer, with or without tool calls),
    - a Pydantic object: returned as JSON text, which with_structured_output parses,
    - an Exception: raised, as an API failure,
    - a function of the received messages that returns one of the above. Use it
      for the self-check: its branches run in parallel, so the order in which
      they take their replies is not fixed.
    """

    replies: list = Field(default_factory=list)
    received: list = Field(default_factory=list)      # the message list of every call

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(messages)
        next_reply = self.replies.pop(0)
        if callable(next_reply) and not isinstance(next_reply, BaseModel):
            next_reply = next_reply(messages)
        if isinstance(next_reply, Exception):
            raise next_reply
        if isinstance(next_reply, BaseModel) and not isinstance(next_reply, AIMessage):
            next_reply = AIMessage(content=next_reply.model_dump_json())
        if next_reply.usage_metadata is None:
            input_tokens, output_tokens = REPLY_TOKENS
            next_reply.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        return ChatResult(generations=[ChatGeneration(message=next_reply)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        assert include_raw

        def call(messages) -> dict:
            raw = self.invoke(messages)
            try:
                return {"raw": raw, "parsed": schema.model_validate_json(raw.content), "parsing_error": None}
            except ValueError as error:
                return {"raw": raw, "parsed": None, "parsing_error": error}

        return RunnableLambda(call)


def tool_call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
