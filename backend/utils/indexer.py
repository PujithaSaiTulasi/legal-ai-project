"""
backend/utils/indexer.py — LlamaIndex Document Ingestion & RAG
===============================================================

WHAT IS LLAMAINDEX?
-------------------
LlamaIndex is a framework specifically built for connecting LLMs to your data.
Its superpower is RAG — Retrieval-Augmented Generation.

WHAT IS RAG?
------------
RAG solves a fundamental problem with LLMs: they only know what they were
trained on. They can't answer questions about YOUR specific documents
(a deposition, an internal email, a proprietary contract).

RAG works in two phases:

  PHASE 1 — INDEXING (happens once):
  ┌────────────────────────────────────────────────┐
  │  Raw text → split into chunks → convert each   │
  │  chunk into a "vector" (a list of numbers that │
  │  captures the chunk's meaning) → store in an   │
  │  index (like a semantic search engine)         │
  └────────────────────────────────────────────────┘

  PHASE 2 — RETRIEVAL + GENERATION (happens per question):
  ┌────────────────────────────────────────────────┐
  │  Question → convert to a vector → find the most │
  │  similar chunks in the index → send those       │
  │  chunks + the question to the LLM → answer      │
  │  using only the relevant passages               │
  └────────────────────────────────────────────────┘

WHY IS THIS POWERFUL FOR EDISCOVERY?
-------------------------------------
A real case might have 500,000 documents. You can't send them all to the LLM
at once (context window limits + cost). RAG lets you index everything once,
then retrieve only the 3 most relevant paragraphs for each specific question.
"""

import warnings

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter

from backend.config import (
    LLM_PROVIDER,
    MODEL_NAME,
    ANTHROPIC_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_TOKENS,
    OPENAI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EMBED_MODEL,
    SIMILARITY_TOP_K,
)

# VectorStoreIndex always calls resolve_embed_model(); that function imports LangChain first,
# which breaks on Python 3.14. We pre-build BaseEmbedding instances and short-circuit.
_EMBED_RESOLVE_PATCHED = False


def _patch_resolve_embed_model_for_prebuilt_instances() -> None:
    global _EMBED_RESOLVE_PATCHED
    if _EMBED_RESOLVE_PATCHED:
        return

    import llama_index.core.embeddings.utils as embed_utils
    import llama_index.core.indices.vector_store.base as vs_index

    _original = embed_utils.resolve_embed_model

    def resolve_embed_model(embed_model=None, callback_manager=None):
        if isinstance(embed_model, BaseEmbedding):
            embed_model.callback_manager = (
                callback_manager or Settings.callback_manager
            )
            return embed_model
        return _original(embed_model, callback_manager=callback_manager)

    embed_utils.resolve_embed_model = resolve_embed_model
    vs_index.resolve_embed_model = resolve_embed_model
    _EMBED_RESOLVE_PATCHED = True


def _ensure_llama_index_embed_model_anthropic_path() -> None:
    """
    When using Claude for generation, avoid Settings.embed_model's default
    resolution on Python 3.14 (LangChain bridge import crash).
    """
    if Settings._embed_model is not None:
        return

    if OPENAI_API_KEY:
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings._embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
        return

    warnings.warn(
        "OPENAI_API_KEY is not set. Using MockEmbedding — vector retrieval is not "
        "semantic. Set OPENAI_API_KEY for OpenAI embeddings, or use "
        "LLM_PROVIDER=ollama with a local embedding model (e.g. nomic-embed-text).",
        stacklevel=1,
    )
    Settings._embed_model = MockEmbedding(embed_dim=8)


def _configure_llama_index_llm_and_embed() -> None:
    """Point LlamaIndex at the same backend as LangChain (Ollama or Anthropic)."""
    if LLM_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        Settings._llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            request_timeout=120.0,
            additional_kwargs={"num_predict": MAX_TOKENS},
        )
        Settings._embed_model = OllamaEmbedding(
            model_name=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        return

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY in your environment."
        )

    from llama_index.llms.anthropic import Anthropic as LlamaAnthropic

    Settings.llm = LlamaAnthropic(
        model=MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS,
    )
    _ensure_llama_index_embed_model_anthropic_path()


def build_index(text: str) -> VectorStoreIndex:
    """
    Takes raw document text and builds a queryable LlamaIndex vector index.

    Parameters:
        text (str): The full text of the legal document.

    Returns:
        VectorStoreIndex: An in-memory index you can query with natural language.
    """

    _configure_llama_index_llm_and_embed()

    doc = Document(
        text=text,
        metadata={
            "source": "eDiscovery_upload",
            "case_id": "DEMO-001",
            "doc_type": "email_chain",
        }
    )

    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = parser.get_nodes_from_documents([doc])

    _patch_resolve_embed_model_for_prebuilt_instances()
    index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)

    return index


def query_index(index: VectorStoreIndex, question: str) -> str:
    """
    Ask a specific natural language question and get an answer
    grounded in the document.
    """

    query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)

    response = query_engine.query(question)

    return str(response)
