import httpx
import ollama
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
    is_unreadable_reply,
    model_name,
    verifier_model,
    without_thinking,
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


def cut_off_reply() -> dict:
    """An answer that Ollama stopped at OLLAMA_NUM_PREDICT while the model was thinking."""
    raw = AIMessage(
        content="",
        response_metadata={"done_reason": "length"},
        usage_metadata={"input_tokens": 11600, "output_tokens": 16384, "total_tokens": 27984},
    )
    return {"raw": raw, "parsed": None, "parsing_error": ValueError("Invalid json output: ")}


class ThinkingFakeModel(FakeModel):
    """A FakeModel with ChatOllama's `reasoning` setting and model_copy()."""

    def __init__(self, replies: list[dict], reasoning):
        super().__init__(replies)
        self.reasoning = reasoning
        self.used_reasoning = []        # the reasoning setting of every call

    def invoke(self, messages):
        self.used_reasoning.append(self.reasoning)
        return super().invoke(messages)

    def model_copy(self, update):
        # The copy shares the replies and the records, so the test sees both models' calls.
        copy = ThinkingFakeModel([], update["reasoning"])
        copy.replies, copy.received, copy.used_reasoning = self.replies, self.received, self.used_reasoning
        return copy


def test_cut_off_answer_is_asked_again_without_thinking():
    model = ThinkingFakeModel([cut_off_reply(), reply(VALID)], reasoning="low")
    messages = [HumanMessage("review this")]

    parsed, raw_messages = call_structured(model, ReviewOutput, messages)

    assert parsed == VALID
    assert len(raw_messages) == 2
    assert model.used_reasoning == ["low", False]
    # The same request again: no correction message, which would not help.
    assert model.received[1] == messages
    # The caller's model keeps its setting.
    assert model.reasoning == "low"


def test_cut_off_answer_without_reasoning_setting_gets_the_normal_retry():
    # OLLAMA_REASONING unset: the model's own default, which call_structured does not touch.
    model = ThinkingFakeModel([cut_off_reply(), reply(VALID)], reasoning=None)

    call_structured(model, ReviewOutput, [HumanMessage("review this")])

    assert model.used_reasoning == [None, None]
    assert "not valid for the required format" in model.received[1][-1].content


def test_failure_message_says_why_the_last_answer_ended():
    model = ThinkingFakeModel([cut_off_reply(), cut_off_reply()], reasoning="low")

    with pytest.raises(StructuredOutputError) as error:
        call_structured(model, ReviewOutput, [HumanMessage("review this")])

    message = str(error.value)
    assert "stop reason length" in message
    assert "11600 prompt tokens" in message
    assert "16384 answer tokens" in message


def test_without_thinking_copies_a_thinking_ollama_model():
    model = ChatOllama(model="qwen3.5:9b", reasoning="low")

    copy = without_thinking(model)

    assert copy.reasoning is False
    assert copy.model == "qwen3.5:9b"
    assert model.reasoning == "low"
    assert without_thinking(ChatOllama(model="qwen3.5:9b")) is None
    assert without_thinking(ChatOllama(model="qwen3.5:9b", reasoning=False)) is None


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


@pytest.mark.parametrize("setting, sent", [(None, None), ("true", True), ("false", False), ("low", "low")])
def test_ollama_thinking_comes_from_settings(setting, sent):
    settings = Settings(llm_provider="ollama", ollama_model="qwen3.5:9b", ollama_reasoning=setting)

    assert get_chat_model(settings).reasoning == sent


def test_thinking_mode_is_part_of_the_model_name():
    # So that stored runs of one mode are not skipped as done for another.
    assert model_name(Settings(llm_provider="ollama", ollama_model="qwen3.5:9b")) == "qwen3.5:9b"
    low = Settings(llm_provider="ollama", ollama_model="qwen3.5:9b", ollama_reasoning="low")
    assert model_name(low) == "qwen3.5:9b, reasoning low"


def test_verifier_thinking_is_in_the_name_of_the_agent_only():
    settings = Settings(llm_provider="ollama", ollama_model="qwen3.5:9b", ollama_reasoning="false",
                        ollama_verify_reasoning="low")

    assert model_name(settings, verifier=True) == "qwen3.5:9b, reasoning false, verifier low"
    assert model_name(settings) == "qwen3.5:9b, reasoning false"      # baseline and judge


def test_verifier_model_is_a_copy_with_its_own_thinking():
    settings = Settings(llm_provider="ollama", ollama_model="qwen3.5:9b", ollama_reasoning="false",
                        ollama_verify_reasoning="low")
    model = get_chat_model(settings)

    verifier = verifier_model(model, settings)

    assert verifier is not model
    assert verifier.reasoning == "low"
    assert verifier.model == "qwen3.5:9b" and verifier.num_predict == model.num_predict
    assert model.reasoning is False             # the review keeps its own setting


def test_verifier_model_true_and_false_become_booleans():
    settings = Settings(llm_provider="ollama", ollama_verify_reasoning="true")

    assert verifier_model(get_chat_model(settings), settings).reasoning is True


def test_verifier_model_without_its_setting_is_the_model_itself():
    settings = Settings(llm_provider="ollama", ollama_reasoning="low")
    model = get_chat_model(settings)

    assert verifier_model(model, settings) is model


def test_verifier_model_leaves_other_models_alone():
    # Gemini, and the scripted model of the tests, have no Ollama thinking setting.
    settings = Settings(llm_provider="ollama", ollama_verify_reasoning="low")
    model = FakeModel([])

    assert verifier_model(model, settings) is model


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


# --- An answer Ollama could not read --------------------------------------------


def test_a_broken_tool_call_in_the_answer_stream_is_an_unreadable_reply():
    error = ollama.ResponseError("XML syntax error on line 3: element <function> closed by </parameter>")

    assert is_unreadable_reply(error) is True
    assert is_transient_error(error) is False      # the same request would most likely break again


@pytest.mark.parametrize("error", [
    ollama.ResponseError('model "qwen3.5:9b" not found', status_code=404),
    ollama.ResponseError("internal error", status_code=500),
    ConnectionError("Failed to connect to Ollama"),
    ValueError("XML syntax error"),
])
def test_a_refused_request_or_another_error_is_not_an_unreadable_reply(error):
    assert is_unreadable_reply(error) is False
