import httpx
import pytest
from conftest import FakeModel, raw_message, reply
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)
from langchain_ollama import ChatOllama

from reviewgraph.config import ConfigError, Settings
from reviewgraph.llm import (
    StructuredOutputError,
    add_usage,
    call_structured,
    get_chat_model,
    is_transient_error,
    model_name,
)
from reviewgraph.schemas import ReviewOutput, RunStats

VALID = ReviewOutput(findings=[])


def test_valid_reply_needs_one_call():
    model = FakeModel([reply(VALID)])

    parsed, raw_messages = call_structured(model, ReviewOutput, [HumanMessage("review this")])

    assert parsed == VALID
    assert len(raw_messages) == 1
    assert len(model.received) == 1


def test_invalid_reply_is_retried_once_with_the_error():
    model = FakeModel([reply(None, ValueError("severity must be low, medium or high")), reply(VALID)])
    messages = [HumanMessage("review this")]

    parsed, raw_messages = call_structured(model, ReviewOutput, messages)

    assert parsed == VALID
    assert len(raw_messages) == 2
    retry_messages = model.received[1]
    assert retry_messages[0] == messages[0]
    assert "severity must be low, medium or high" in retry_messages[-1].content
    # The caller's list is not changed.
    assert len(messages) == 1


def test_two_invalid_replies_raise_structured_output_error():
    model = FakeModel([reply(None, ValueError("bad json")), reply(None, ValueError("bad json again"))])

    with pytest.raises(StructuredOutputError) as error:
        call_structured(model, ReviewOutput, [HumanMessage("review this")])

    assert "ReviewOutput" in str(error.value)
    assert len(error.value.raw_messages) == 2
    assert len(model.received) == 2


def test_empty_reply_counts_as_invalid():
    model = FakeModel([reply(None), reply(VALID)])

    parsed, raw_messages = call_structured(model, ReviewOutput, [HumanMessage("review this")])

    assert parsed == VALID
    assert len(raw_messages) == 2


def test_add_usage_counts_every_attempt():
    stats = RunStats()

    add_usage(stats, [raw_message(100, 20), raw_message(130, 25)])
    add_usage(stats, [AIMessage(content="no usage reported")])

    assert (stats.llm_calls, stats.input_tokens, stats.output_tokens) == (3, 230, 45)


def test_gemini_without_key_names_the_variable_and_the_free_path():
    settings = Settings(llm_provider="gemini", gemini_api_key=None)

    with pytest.raises(ConfigError) as error:
        get_chat_model(settings)

    assert "GEMINI_API_KEY" in str(error.value)
    assert "LLM_PROVIDER=ollama" in str(error.value)


def test_gemini_model_is_built_from_settings():
    settings = Settings(llm_provider="gemini", gemini_api_key="test-key", gemini_model="gemini-3.5-flash")

    model = get_chat_model(settings)

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.temperature == 0
    assert model.rate_limiter is not None
    assert model_name(settings) == "gemini-3.5-flash"


def test_gemini_models_share_one_rate_limiter():
    settings = Settings(llm_provider="gemini", gemini_api_key="test-key")

    assert get_chat_model(settings).rate_limiter is get_chat_model(settings).rate_limiter


def test_ollama_model_is_built_from_settings():
    settings = Settings(llm_provider="ollama", ollama_model="qwen2.5-coder:7b")

    model = get_chat_model(settings)

    assert isinstance(model, ChatOllama)
    assert model.temperature == 0
    assert model.num_ctx == 16384         # Ollama's own default would cut the prompt
    assert model.num_predict == 4096      # a local model cannot write for ever
    assert model_name(settings) == "qwen2.5-coder:7b"


def test_ollama_answer_limit_and_context_come_from_settings():
    settings = Settings(llm_provider="ollama", ollama_num_predict=16384, ollama_num_ctx=32768)

    model = get_chat_model(settings)

    assert model.num_predict == 16384
    assert model.num_ctx == 32768


# The 429 of 2026-11-16, shortened. The quota id is what marks the daily limit.
DAILY_QUOTA_MESSAGE = (
    "Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'. Please retry in 18h23m44s."
)
MINUTE_QUOTA_MESSAGE = (
    "Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'. Please retry in 31s."
)


@pytest.mark.parametrize(
    "error",
    [
        GoogleAPIError(code=503, response_json={"error": {"code": 503, "status": "UNAVAILABLE"}}),
        GoogleRateLimitError(MINUTE_QUOTA_MESSAGE),
        httpx.ReadTimeout("the request timed out"),
        httpx.ConnectError("no route to host"),
    ],
    ids=["server error", "rate limit per minute", "timeout", "lost connection"],
)
def test_errors_that_waiting_can_fix_are_retried(error):
    assert is_transient_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        GoogleRateLimitError(DAILY_QUOTA_MESSAGE),
        GoogleAuthenticationError("Error calling model 'gemini-3.5-flash' (UNAUTHENTICATED): API key not valid."),
        GoogleInvalidRequestError("Error calling model 'gemini-3.5-flash' (INVALID_ARGUMENT): bad request."),
        GoogleModelNotFoundError("Error calling model 'gemini-0-flash' (NOT_FOUND): no such model."),
        StructuredOutputError("The model did not return a valid ReviewOutput in two attempts: ...", []),
        ConnectionError("Failed to connect to Ollama."),
        ValueError("a bug in our own code"),
    ],
    ids=["daily quota", "wrong key", "invalid request", "unknown model", "invalid reply", "no Ollama server", "bug"],
)
def test_errors_that_waiting_cannot_fix_are_not_retried(error):
    assert is_transient_error(error) is False


def test_gemini_model_does_not_retry_by_itself():
    # The RetryPolicy of the graph is the only retry layer; two layers would multiply the attempts.
    model = get_chat_model(Settings(llm_provider="gemini", gemini_api_key="test-key"))

    assert model.max_retries == 0
