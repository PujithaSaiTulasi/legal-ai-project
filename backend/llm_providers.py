"""
LangChain chat LLM factory — local Ollama.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from backend.config import (
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def build_langchain_chat_llm() -> BaseChatModel:
    """Return the shared chat model used by entity, smoking_gun, and story chains."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=MAX_TOKENS,
    )
