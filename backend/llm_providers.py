"""
LangChain chat LLM factory — Ollama (default, local / free) or Anthropic API.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import (
    ANTHROPIC_API_KEY,
    LLM_PROVIDER,
    MAX_TOKENS,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def build_langchain_chat_llm() -> BaseChatModel:
    """Return the shared chat model used by entity, smoking_gun, and story chains."""
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "\n\nLLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing.\n"
                "Set it in .env or use LLM_PROVIDER=ollama (default) for local models.\n"
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=MODEL_NAME,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=MAX_TOKENS,
            temperature=0.1,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=MAX_TOKENS,
    )
