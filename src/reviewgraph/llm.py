import threading
from functools import lru_cache

import httpx
import ollama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from reviewgraph.config import ConfigError, Settings, get_settings
from reviewgraph.schemas import RunStats

# The chat model's own retries are switched off (max_retries=0); with two layers
# the attempts would multiply. API errors are retried in one place per system:
# by the RetryPolicy on the agent's model nodes in graph.py, and by the loop in
# baseline.py. Both use these numbers and is_transient_error() below, so the
# two systems of the evaluation are treated the same.
RETRY_MAX_ATTEMPTS = 4          # the first try included
RETRY_INITIAL_INTERVAL = 2.0    # seconds before the second attempt
RETRY_BACKOFF_FACTOR = 2.0      # every further wait is this much longer: 2 s, 4 s, 8 s
# Text in Gemini's 429 error when the daily quota is used up. The quota id is
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier"; seen on 2026-11-16.
DAILY_QUOTA_MARKER = "PerDay"
# Without a timeout a request that never gets an answer blocks the run for ever.
API_TIMEOUT_SECONDS = 300

# The context window and the limit of one answer's length of a local model are
# the settings OLLAMA_NUM_CTX and OLLAMA_NUM_PREDICT (config.py).

# OLLAMA_REASONING as ChatOllama's `reasoning`: "true" and "false" become
# booleans, a level ("low", "medium", "high") is passed on as it is.
REASONING_VALUES = {"true": True, "false": False}


class StructuredOutputError(Exception):
    """The model did not return a valid object, even after one retry."""

    def __init__(self, message: str, raw_messages: list[AIMessage]):
        super().__init__(message)
        # Kept so that the tokens of the failed attempts can still be counted.
        self.raw_messages = raw_messages


def is_transient_error(error: Exception) -> bool:
    """True if the same model call can succeed when it is tried again a few seconds later.

    Made for `RetryPolicy(retry_on=is_transient_error)`. LangGraph's default
    retries almost every exception, also the ones waiting cannot fix.

    Retried: a server error of the Gemini API (5xx), a rate limit that is not
    the daily quota, and a timeout or a lost connection on the way to the API.

    Not retried: the daily quota (it comes back the next day), a wrong key, an
    invalid request, an unknown model, a prompt that is too long, a reply that
    is still invalid after call_structured's own retry, a missing Ollama server
    (the Ollama client raises the built-in ConnectionError), and any other error.
    """
    if isinstance(error, GoogleRateLimitError):
        return DAILY_QUOTA_MARKER not in str(error)
    return isinstance(error, (GoogleAPIError, httpx.TransportError))


def retry_wait(failed_attempts: int) -> float:
    """Seconds to wait before the next attempt, after this many failed ones."""
    return RETRY_INITIAL_INTERVAL * RETRY_BACKOFF_FACTOR ** (failed_attempts - 1)


@lru_cache
def get_rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    # Cached, so every model created in this process shares one limiter.
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Chat model for LLM_PROVIDER, temperature 0."""
    settings = settings or get_settings()

    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama to use a local model."
            )
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0,
            max_retries=0,
            timeout=API_TIMEOUT_SECONDS,
            rate_limiter=get_rate_limiter(settings.requests_per_minute),
        )

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        reasoning=REASONING_VALUES.get(settings.ollama_reasoning, settings.ollama_reasoning),
    )


class OllamaError(Exception):
    """Ollama cannot be used right now. The message says what to do about it."""


def check_ollama(needs_tools: bool = False, settings: Settings | None = None) -> None:
    """Raise OllamaError if LLM_PROVIDER is ollama and the local model cannot be used.

    Checked: the server runs, the model is installed and, with `needs_tools`
    (the agent), the model supports tool calling. One small request to the
    local server, no model call. Does nothing for Gemini.
    """
    settings = settings or get_settings()
    if settings.llm_provider != "ollama":
        return

    client = ollama.Client(host=settings.ollama_base_url, timeout=10)
    try:
        # The same information as `ollama show <model>` in the terminal.
        details = client.show(settings.ollama_model)
        # `capabilities` is a list such as ["completion", "tools"]; an old server sends none.
        if needs_tools and details.capabilities and "tools" not in details.capabilities:
            raise OllamaError(
                f"The Ollama model {settings.ollama_model} does not support tool calling, which the agent needs. "
                "Set OLLAMA_MODEL to a model with tool support, or use --system baseline."
            )
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise OllamaError(
                f"The Ollama model {settings.ollama_model} is not installed. "
                f"Run `ollama pull {settings.ollama_model}` and try again."
            ) from None
        raise OllamaError(f"Ollama answered with an error: {e}") from None
    except (ConnectionError, httpx.TransportError):
        raise OllamaError(
            f"Ollama is not running at {settings.ollama_base_url}. "
            "Start the Ollama app or run `ollama serve`, and try again."
        ) from None


def model_name(settings: Settings | None = None) -> str:
    """Name of the model that get_chat_model() returns, for reports and cached runs."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    # The thinking mode is part of the name, so that stored runs of one mode are
    # not taken for runs of another (is_done in evaluation/runner.py).
    if settings.ollama_reasoning is None:
        return settings.ollama_model
    return f"{settings.ollama_model}, reasoning {settings.ollama_reasoning}"


def call_structured(
    model: BaseChatModel, schema: type[BaseModel], messages: list
) -> tuple[BaseModel, list[AIMessage]]:
    """Return (parsed, raw_messages). Retry once if the reply does not validate.

    The retry tells the model what was wrong. Exception: a local model that
    thinks and was cut off at OLLAMA_NUM_PREDICT is asked the same again with
    thinking switched off (without_thinking), because with thinking it would
    most likely run into the limit again (dev unit 20 on 2026-11-23: three
    answers of 16,384 tokens, 11 minutes each, none of them with a reply).

    `raw_messages` has one message per attempt (one or two), so the caller can
    count every LLM call and its tokens with add_usage().
    """
    raw_messages = []
    attempt_messages = list(messages)
    attempt_model = model

    for attempt in range(2):
        runnable = attempt_model.with_structured_output(schema, include_raw=True)
        # The result is a dict with "raw", "parsed" and "parsing_error".
        result = runnable.invoke(attempt_messages)
        raw = result["raw"]
        raw_messages.append(raw)
        if result["parsed"] is not None and result["parsing_error"] is None:
            return result["parsed"], raw_messages

        error = result["parsing_error"] or "the reply was empty"
        fallback = without_thinking(attempt_model) if was_cut_off(raw) else None
        if fallback is not None:
            attempt_model = fallback
            attempt_messages = list(messages)
        else:
            attempt_messages = list(messages) + [
                HumanMessage(
                    f"Your previous reply was not valid for the required format: {error}\n"
                    "Answer again and follow the format exactly."
                )
            ]

    raise StructuredOutputError(
        f"The model did not return a valid {schema.__name__} in two attempts: {error} "
        f"(last reply: {describe_reply(raw_messages[-1])})",
        raw_messages,
    )


def was_cut_off(raw: AIMessage) -> bool:
    """True if a local model stopped the answer at OLLAMA_NUM_PREDICT (Ollama's done_reason "length")."""
    return (raw.response_metadata or {}).get("done_reason") == "length"


def without_thinking(model: BaseChatModel) -> BaseChatModel | None:
    """The same model with thinking switched off, or None if there is nothing to switch off.

    Only when thinking was switched on by OLLAMA_REASONING (true or a level).
    Unset leaves the model's own default, and a model that cannot think might
    refuse the `think` setting, so it keeps the normal retry.
    """
    if getattr(model, "reasoning", None) in (None, False):
        return None
    return model.model_copy(update={"reasoning": False})


def describe_reply(raw: AIMessage) -> str:
    """Why an answer ended and how long it was, for error messages.

    Ollama calls the reason done_reason ("stop", or "length" at OLLAMA_NUM_PREDICT),
    Gemini finish_reason.
    """
    metadata = raw.response_metadata or {}
    usage = raw.usage_metadata or {}
    reason = metadata.get("done_reason") or metadata.get("finish_reason") or "unknown"
    return (
        f"stop reason {reason}, {usage.get('input_tokens', '?')} prompt tokens, "
        f"{usage.get('output_tokens', '?')} answer tokens"
    )


def add_usage(stats: RunStats, raw_messages: list[AIMessage]) -> None:
    """Add the LLM calls and their token counts to `stats`."""
    for raw in raw_messages:
        usage = raw.usage_metadata or {}
        stats.llm_calls += 1
        stats.input_tokens += usage.get("input_tokens", 0)
        stats.output_tokens += usage.get("output_tokens", 0)


class UsageCallback(BaseCallbackHandler):
    """Counts every model call of a graph run into `stats`, also the failed ones.

    Passed as `config={"callbacks": [UsageCallback(stats)]}` to the graph, it sees
    the model calls of all nodes. LangChain's UsageMetadataCallbackHandler adds
    up the tokens only; the evaluation also needs the number of calls.
    """

    def __init__(self, stats: RunStats):
        self.stats = stats
        # The verify_finding branches run in parallel threads.
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(generation, "message", None)
        with self._lock:
            if isinstance(message, AIMessage):
                add_usage(self.stats, [message])
            else:
                self.stats.llm_calls += 1

    def on_llm_error(self, error, **kwargs) -> None:
        # A failed call is still a request that was sent, as in the baseline.
        with self._lock:
            self.stats.llm_calls += 1
