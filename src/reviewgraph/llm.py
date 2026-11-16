import threading
from functools import lru_cache

import httpx
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

# Ollama's default context window is small and it cuts longer prompts without a
# warning. A diff plus AGENTS.md plus tool results needs more.
OLLAMA_NUM_CTX = 16384


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
        num_ctx=OLLAMA_NUM_CTX,
    )


def model_name(settings: Settings | None = None) -> str:
    """Name of the model that get_chat_model() returns, for reports and cached runs."""
    settings = settings or get_settings()
    return settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model


def call_structured(
    model: BaseChatModel, schema: type[BaseModel], messages: list
) -> tuple[BaseModel, list[AIMessage]]:
    """Return (parsed, raw_messages). Retry once if the reply does not validate.

    `raw_messages` has one message per attempt (one or two), so the caller can
    count every LLM call and its tokens with add_usage().
    """
    runnable = model.with_structured_output(schema, include_raw=True)
    raw_messages = []
    attempt_messages = list(messages)

    for attempt in range(2):
        # The result is a dict with "raw", "parsed" and "parsing_error".
        result = runnable.invoke(attempt_messages)
        raw_messages.append(result["raw"])
        if result["parsed"] is not None and result["parsing_error"] is None:
            return result["parsed"], raw_messages

        error = result["parsing_error"] or "the reply was empty"
        attempt_messages = list(messages) + [
            HumanMessage(
                f"Your previous reply was not valid for the required format: {error}\n"
                "Answer again and follow the format exactly."
            )
        ]

    raise StructuredOutputError(
        f"The model did not return a valid {schema.__name__} in two attempts: {error}", raw_messages
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
