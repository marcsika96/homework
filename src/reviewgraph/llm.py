from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from reviewgraph.config import ConfigError, Settings, get_settings
from reviewgraph.schemas import RunStats

# The only retry layer for API errors (rate limit, timeout, server error) is the
# chat model's own. Nothing else in the project retries a failed API call.
API_MAX_RETRIES = 3
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
            max_retries=API_MAX_RETRIES,
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
