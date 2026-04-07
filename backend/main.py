"""
backend/main.py — Pipeline Orchestrator & Entry Point
======================================================

This is the "conductor" of the orchestra. It imports every module we've built
and calls them in the right order.

THE FULL PIPELINE:
------------------

                    ┌──────────────────────┐
                    │   Raw Document Text   │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │        LlamaIndex Indexer        │
              │  (indexer.py → build_index())    │
              │  Chunks the doc, builds a        │
              │  queryable vector index          │
              └────────────────┬────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
         ▼                     ▼                      │
┌─────────────────┐  ┌──────────────────────┐        │
│  Entity Chain   │  │  Smoking Gun Chain   │        │
│ (LangChain)     │  │  (LangChain)         │        │
│                 │  │                      │        │
│ Outputs:        │  │  Outputs:            │        │
│ • Persons       │  │  • Top 3 evidence    │        │
│ • Orgs          │  │  • Legal categories  │        │
│ • Dates         │  │  • Why it matters    │        │
│ • Money         │  │                      │        │
└────────┬────────┘  └─────────┬────────────┘        │
         │                     │                      │
         └──────────┬──────────┘                      │
                    │                                 │
                    ▼                                 │
         ┌──────────────────────┐                    │
         │     Story Chain      │◄───────────────────┘
         │     (LangChain)      │   (also receives original doc)
         │                      │
         │  Takes ALL above +   │
         │  original doc and    │
         │  synthesizes the     │
         │  legal narrative     │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │    LlamaIndex RAG    │
         │  Targeted follow-up  │
         │  questions answered  │
         │  from the index      │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │   Display Results    │
         │   (display.py)       │
         └──────────────────────┘

HOW TO RUN:
    cd legalstoryos
    source venv/bin/activate
    python backend/main.py
"""

import sys
from pathlib import Path

# Running `python backend/main.py` puts `backend/` on sys.path, not the repo root;
# `backend.*` imports need the parent directory.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import warnings

# Third-party noise on Python 3.14 (Anthropic SDK pydantic.v1 bridge; Pydantic Field()).
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
    module="anthropic._compat",
)
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
    module="langchain_core._api.deprecation",
)
warnings.filterwarnings(
    "ignore",
    message=".*validate_default.*",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)

# Import our modules — each does one focused job
from backend.config import LLM_PROVIDER
from backend.llm_providers import build_langchain_chat_llm
from backend.utils.indexer import build_index, query_index
from backend.utils.display import console, print_step, print_success, print_results
from backend.chains.entity_chain import build_entity_chain
from backend.chains.smoking_gun_chain import build_smoking_gun_chain
from backend.chains.story_chain import build_story_chain


# ── Sample Document ───────────────────────────────────────────────────────────
# In a real system, this would come from:
#   - A file upload (Flask/FastAPI endpoint)
#   - An S3/cloud storage bucket
#   - A Relativity or Concordance eDiscovery export
#   - A database of ingested documents
SAMPLE_DOCUMENT = """
CONFIDENTIAL — INTERNAL EMAIL CHAIN
From: Marcus Holt <m.holt@nexagen.com>
To: Sandra Voss <s.voss@nexagen.com>
Date: March 14, 2021
Subject: Re: Q1 Safety Results

Sandra,

I've reviewed the lab reports from Dr. Karim's team. The failure rate on batch NX-447
came in at 18.3% — significantly above our acceptable threshold of 2%. I told the board
it was below 3%. You and I both know we cannot disclose this to the FDA before the
April launch. Let's just bury the December data in the appendix. Nobody reads those.

— Marcus

---
From: Sandra Voss <s.voss@nexagen.com>
To: Marcus Holt <m.holt@nexagen.com>
Date: March 15, 2021

Marcus, agreed. But if Karim talks we have a problem. He's been asking too many questions.
I'll handle the personnel review.

Also — the $4.2M payment to Vertex Consulting last quarter. Make sure that's coded as
"R&D" not "consulting." Auditors are reviewing Q4.

S.

---
From: Dr. Priya Karim <p.karim@nexagen.com>
To: FDA Safety Desk <safety@fda.hhs.gov>
Date: April 2, 2021

I am a senior researcher at NexaGen Pharmaceuticals. I am writing to report potential
safety data suppression related to product batch NX-447 prior to its April 10 commercial
launch. The actual failure rate is 18.3%, not the 2.8% reported to the board and the FDA.
"""


def run_pipeline(document_text: str) -> dict:
    """
    The main pipeline function. Orchestrates all modules in sequence.

    Parameters:
        document_text (str): The raw text of the legal document to analyze.

    Returns:
        dict: Contains entities, smoking_guns, story, and rag_answers.

    This function is intentionally kept lean — it imports and calls,
    but doesn't contain implementation logic. That lives in the modules.
    This is the "Orchestrator pattern."
    """

    # ── Initialize the LLM ────────────────────────────────────────────────────
    # Default: Ollama (local). Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY for Claude.
    llm = build_langchain_chat_llm()

    # ── STEP 1: LlamaIndex Indexing ───────────────────────────────────────────
    print_step(1, "LlamaIndex — Ingesting & indexing document")
    index = build_index(document_text)
    print_success(f"Document indexed and ready for retrieval")

    # ── STEP 2: Build LangChain Chains ────────────────────────────────────────
    print_step(2, "LangChain — Building analysis chains")
    entity_chain = build_entity_chain(llm)
    smoking_gun_chain = build_smoking_gun_chain(llm)
    story_chain = build_story_chain(llm)
    print_success("All 3 chains initialized (entity, smoking_gun, story)")

    # ── STEP 3: Run Analysis Chains ───────────────────────────────────────────
    print_step(3, "Running LangChain analysis chains")

    # .invoke() is how you run a chain in LangChain.
    # It's synchronous — it blocks until the model responds.
    # In production, use .ainvoke() for async execution,
    # which lets entity_chain and smoking_gun_chain run in PARALLEL.
    console.print("  Extracting entities...")
    entities = entity_chain.invoke(document_text)
    print_success("Entity extraction complete")

    console.print("  Detecting smoking guns...")
    smoking_guns = smoking_gun_chain.invoke(document_text)
    print_success("Smoking gun detection complete")

    # The story chain needs all three inputs as a dict.
    # The prompt template maps each key to its {placeholder}.
    console.print("  Synthesizing legal narrative...")
    story = story_chain.invoke({
        "document": document_text,
        "entities": entities,
        "smoking_guns": smoking_guns,
    })
    print_success("Legal story synthesized")

    # ── STEP 4: LlamaIndex RAG Queries ───────────────────────────────────────
    # Now we use the index we built in Step 1 to answer targeted questions.
    # This demonstrates that LlamaIndex isn't just for indexing —
    # it's a full RAG system you can query at any time.
    print_step(4, "LlamaIndex — Running RAG targeted queries")

    rag_questions = [
        "Who was responsible for suppressing the safety data?",
        "What was the actual batch failure rate vs. what was reported?",
        "Did anyone report this misconduct to authorities?",
    ]

    rag_answers = {}
    for question in rag_questions:
        answer = query_index(index, question)
        rag_answers[question] = answer
        print_success(f"Answered: '{question[:50]}...'")

    # ── Return all results ────────────────────────────────────────────────────
    return {
        "entities": entities,
        "smoking_guns": smoking_guns,
        "story": story,
        "rag_answers": rag_answers,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
# This is standard Python: the code inside `if __name__ == "__main__"`
# only runs when you execute this file directly (python main.py).
# It does NOT run when this module is imported by another module.
# This lets main.py also be imported as a library without side effects.
if __name__ == "__main__":
    from rich.panel import Panel

    _backend = "Ollama (local)" if LLM_PROVIDER == "ollama" else "Anthropic Claude"
    console.print(Panel.fit(
        "[bold]⚖ LegalStoryOS[/bold] — Story-Driven eDiscovery Intelligence\n"
        f"[dim]LangChain · LlamaIndex · {_backend} · Python[/dim]",
        border_style="gold1",
    ))

    results = run_pipeline(SAMPLE_DOCUMENT)
    print_results(results)
