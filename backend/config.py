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

# ── Ollama (local LLM + embeddings) ──────────────────────────────────────────
# Install Ollama from https://ollama.com — then pull the models:
#   ollama pull llama3
#   ollama pull nomic-embed-text
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

MAX_TOKENS = 1024

# ── LlamaIndex / Chunking Settings ────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_TOP_K = 3
