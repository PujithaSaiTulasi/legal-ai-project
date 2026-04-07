"""
backend/config.py — Central Configuration
==========================================

WHY HAVE A CONFIG FILE?
------------------------
Instead of scattering magic numbers and settings throughout your code,
you put them all in one place. This means:

  - To change the AI model, you edit ONE line here instead of hunting
    through 5 different files.
  - When you deploy to production vs. development, you can swap configs
    easily.
  - New teammates can understand the system's settings at a glance.

This is a standard software engineering pattern called "single source of truth."
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider ─────────────────────────────────────────────────────────────
# "ollama" = local models via Ollama (default, no API billing).
# "anthropic" = Claude via ANTHROPIC_API_KEY.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in ("ollama", "anthropic"):
    raise ValueError(
        f"LLM_PROVIDER must be 'ollama' or 'anthropic', got: {LLM_PROVIDER!r}"
    )

# Ollama: install from https://ollama.com — then e.g. `ollama pull llama3` and
# `ollama pull nomic-embed-text` for chat + embeddings.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Anthropic (only when LLM_PROVIDER=anthropic) ────────────────────────────

MODEL_NAME = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Optional OpenAI key: used for LlamaIndex embeddings when LLM_PROVIDER=anthropic
# and you want OpenAI embeddings instead of mock vectors.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── LlamaIndex / Chunking Settings ────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_TOP_K = 3
