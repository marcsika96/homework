import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class ConfigError(Exception):
    """A setting is missing or invalid. The message names the environment variable."""


class Settings(BaseModel):
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    target_repo_path: str = "data/target"
    max_tool_calls: int = 8
    requests_per_minute: float = 10


@lru_cache
def get_settings() -> Settings:
    load_dotenv()

    # Every field is read from the environment variable with the same name in
    # upper case. Unset and empty variables fall back to the default.
    values = {}
    for field in Settings.model_fields:
        value = os.environ.get(field.upper(), "").strip()
        if value:
            values[field] = value

    try:
        return Settings(**values)
    except ValidationError as e:
        problems = []
        for error in e.errors():
            field = error["loc"][0]
            problems.append(f"{field.upper()}={values[field]!r}: {error['msg']}")
        raise ConfigError("; ".join(problems)) from None
