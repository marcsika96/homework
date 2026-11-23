import pytest

from reviewgraph.config import ConfigError, get_settings


@pytest.fixture(autouse=True)
def fresh_settings():
    # get_settings() is cached, so every test has to start with an empty cache.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MAX_TOOL_CALLS", "5")

    settings = get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.max_tool_calls == 5


def test_empty_variable_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    assert get_settings().gemini_api_key is None


def test_typo_in_provider_raises_config_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemni")

    with pytest.raises(ConfigError) as error:
        get_settings()

    message = str(error.value)
    assert "LLM_PROVIDER" in message
    assert "\n" not in message


def test_answer_limit_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "16384")

    assert get_settings().ollama_num_predict == 16384


def test_answer_limit_must_be_positive(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "0")

    with pytest.raises(ConfigError, match="OLLAMA_NUM_PREDICT"):
        get_settings()


def test_thinking_level_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_REASONING", "low")

    assert get_settings().ollama_reasoning == "low"


def test_unknown_thinking_level_names_the_variable(monkeypatch):
    monkeypatch.setenv("OLLAMA_REASONING", "very low")

    with pytest.raises(ConfigError, match="OLLAMA_REASONING"):
        get_settings()


def test_verifier_thinking_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_VERIFY_REASONING", "medium")

    assert get_settings().ollama_verify_reasoning == "medium"


def test_unknown_verifier_thinking_names_the_variable(monkeypatch):
    monkeypatch.setenv("OLLAMA_VERIFY_REASONING", "yes")

    with pytest.raises(ConfigError, match="OLLAMA_VERIFY_REASONING"):
        get_settings()


def test_context_window_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")

    assert get_settings().ollama_num_ctx == 32768


def test_non_numeric_limit_raises_config_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MAX_TOOL_CALLS", "eight")

    with pytest.raises(ConfigError, match="MAX_TOOL_CALLS"):
        get_settings()
