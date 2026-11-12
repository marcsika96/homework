import pytest
from conftest import FakeModel, raw_message, reply
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from reviewgraph.config import ConfigError, Settings
from reviewgraph.llm import (
    OLLAMA_NUM_CTX,
    StructuredOutputError,
    add_usage,
    call_structured,
    get_chat_model,
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
    assert model.num_ctx == OLLAMA_NUM_CTX
    assert model_name(settings) == "qwen2.5-coder:7b"
