from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import OpenAIConfig, require_non_empty


def build_llm(cfg: OpenAIConfig) -> ChatOpenAI:
    require_non_empty(cfg.api_key, "OPENAI_API_KEY")

    # langchain_openai will read OPENAI_API_KEY from env,
    # but we validate early for clearer errors.
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=cfg.api_key,
    )
