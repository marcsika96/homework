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


def test_non_numeric_limit_raises_config_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MAX_TOOL_CALLS", "eight")

    with pytest.raises(ConfigError, match="MAX_TOOL_CALLS"):
        get_settings()
