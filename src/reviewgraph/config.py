import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ConfigError(Exception):
    """A setting is missing or invalid. The message names the environment variable."""


class Settings(BaseModel):
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    # The most tokens a local model may write in one answer, thinking included.
    # Ollama has no limit by default, and on 2026-11-16 phi4-mini wrote more than
    # 7,600 tokens in one look-up answer (45 minutes on the CPU) and was still
    # going. 4,096 stops such runaways on the laptop; a thinking model on a GPU
    # (qwen3.5:9b on Colab) needs more, so the Colab notebook sets a higher value.
    ollama_num_predict: int = Field(4096, gt=0)
    # Ollama's context window: the prompt AND the answer, thinking included, must
    # fit. Ollama's own default is small and it cuts longer prompts without a
    # warning. A diff, AGENTS.md and a few tool results take 10-13k tokens, so on
    # the GPU (Colab) the notebook sets 32,768 to leave room for the thinking.
    ollama_num_ctx: int = Field(16384, gt=0)
    # Thinking of a local model: true, false, or a level (low, medium, high), sent
    # as Ollama's `think`. Unset: the model's own default, as before this setting.
    # On 2026-11-23 qwen3.5:9b thought for 2,363 tokens by default and 944 with
    # "low" on a one-line question; on dev unit 20 its default thinking ran into
    # OLLAMA_NUM_PREDICT (16,384) three times in a row and gave no answer.
    ollama_reasoning: Literal["true", "false", "low", "medium", "high"] | None = None
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
