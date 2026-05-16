# ⚖️ LegalStoryOS — Complete Setup & Concept Guide

> A story-driven eDiscovery intelligence tool powered by LangChain, LlamaIndex, and a local Ollama LLM.
> Built to demonstrate production-grade AI pipeline architecture for Altumatim.

---

## 📁 Project Structure

```
legalstoryos/
│
├── README.md                  ← You are here. Full guide + concept explanations.
│
├── .env                       ← Optional overrides (OLLAMA_MODEL, OLLAMA_BASE_URL, etc.)
├── requirements.txt           ← All Python packages the project needs
│
├── backend/                   ← All Python / AI code lives here
│   ├── main.py                ← Entry point — runs the full pipeline in your terminal
│   ├── config.py              ← Central config (model names, chunk sizes, etc.)
│   ├── llm_providers.py       ← LangChain ChatOllama factory
│   │
│   ├── chains/                ← LangChain "reasoning" modules
│   │   ├── __init__.py
│   │   ├── entity_chain.py    ← Extracts people, orgs, dates, money
│   │   ├── smoking_gun_chain.py ← Detects the most legally damaging evidence
│   │   └── story_chain.py     ← Synthesizes the full legal narrative
│   │
│   └── utils/                 ← Helper utilities
│       ├── __init__.py
│       ├── indexer.py         ← LlamaIndex document ingestion & RAG queries
│       └── display.py         ← Pretty terminal output using Rich
│
├── frontend/                  ← The browser demo UI
│   └── index.html             ← Single HTML file — open directly in any browser
│
└── docs/                      ← Learning resources
    └── concepts.md            ← Deep-dive: LLMs, LangChain, LlamaIndex explained
```

---

## 🧠 The Big Picture — What Are We Actually Building?

Before we touch any code, let's understand **what problem we're solving** and **why we need these tools**.

### The Problem

Imagine a legal case with 500,000 emails, PDFs, and documents. A lawyer needs to find the 3 emails that prove the CEO knew about the fraud. Manually reading everything takes months. That's eDiscovery.

**LegalStoryOS solves this with a 4-step AI pipeline:**

```
Raw Document Text
      │
      ▼
┌─────────────────────────────────┐
│  STEP 1: LlamaIndex             │  ← "The Librarian"
│  Reads, chunks & indexes the    │    Turns a wall of text into something
│  document for smart retrieval   │    you can ask questions about
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  STEP 2: LangChain Chains       │  ← "The Analysts"
│  3 specialized AI prompts run   │    Each expert does ONE thing well:
│  in sequence:                   │    extract → detect → synthesize
│   • Entity extraction           │
│   • Smoking gun detection       │
│   • Story synthesis             │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  STEP 3: Local LLM (Ollama)     │  ← "The Brain"
│  Powers every AI decision.      │    A model running on your own machine
│  LangChain and LlamaIndex both  │    reads text, understands meaning, and
│  call Ollama under the hood     │    generates intelligent responses.
└─────────────────────────────────┘
      │
      ▼
  Structured Output:
  - Legal narrative story
  - Smoking gun evidence
  - Key entities
  - Case timeline
```

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Install Python (if you don't have it)

Check if Python is installed:
```bash
python3 --version
```

You need Python 3.9 or higher. If you don't have it, download from https://python.org.

---

### Step 2: Clone / Download This Project

If you have it as a zip, unzip it. Then open your terminal and navigate into the folder:
```bash
cd legalstoryos
```

---

### Step 3: Create a Virtual Environment

**What is this?** A virtual environment is an isolated bubble for your project's Python packages. Without it, packages from different projects can clash with each other. Think of it as a clean room for your project.

```bash
# Create the virtual environment (creates a folder called "venv")
python3 -m venv venv

# Activate it — you must do this every time you open a new terminal
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

After activation, your terminal prompt will show `(venv)` at the start. That means you're inside the clean room.

---

### Step 4: Install All Dependencies

**What is requirements.txt?** It's a shopping list of all the Python packages this project needs. One command installs everything:

```bash
pip install -r requirements.txt
```

This will download and install:
- `langchain` — the AI chain orchestration framework
- `langchain-ollama` — LangChain's plugin for Ollama
- `llama-index` — the document indexing & retrieval framework
- `llama-index-llms-ollama` — LlamaIndex's plugin for Ollama as the LLM
- `llama-index-embeddings-ollama` — local embedding model via Ollama
- `ollama` — the official Ollama Python client
- `python-dotenv` — reads your `.env` file
- `rich` — makes your terminal output beautiful
- `flask` — optional, for running a local web server

---

### Step 5: Install Ollama & Pull the Models

1. Download Ollama from https://ollama.com and install it.
2. Start the local server (it usually auto-starts; otherwise run `ollama serve`):
   ```bash
   ollama serve
   ```
3. Pull the chat model and the embedding model:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

**Why two models?** `llama3` is the chat/reasoning model used by every LangChain chain. `nomic-embed-text` turns text into the vectors that LlamaIndex uses for retrieval (RAG).

The models run entirely on your machine — no API keys, no rate limits, no per-token cost.

---

### Step 6: (Optional) Customize via .env

You only need a `.env` file if you want to change the defaults. Create one in the project root:

```bash
touch .env
```

Add any of these lines as overrides:
```
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

---

### Step 7: Run the Backend Pipeline

```bash
python run.py
```

You'll see the analysis run step-by-step in your terminal with colored output.

---

### Step 8: Open the Frontend Demo

The browser demo talks to your local Ollama server directly. Because the page is served via `file://`, you may need to start Ollama with permissive CORS so the browser can reach it:

```bash
# Stop any running ollama server, then:
OLLAMA_ORIGINS='*' ollama serve
```

Then open the HTML file:
```bash
# On Mac:
open frontend/index.html

# On Windows:
start frontend/index.html

# Or just double-click the file in your file explorer
```

Choose the model name (defaults to `llama3`), paste a document, and click **Analyze**.

---

## 🧪 Quick Test

To verify everything works:
```bash
# From the project root, with venv activated and Ollama running:
python run.py
```

Expected output: colored panels showing entities, smoking guns, and the legal narrative.

---

## 💡 Talking Points for the Interview

When they ask "walk me through your project":

1. **"I separated concerns clearly"** — LlamaIndex handles the data/retrieval layer, LangChain handles the reasoning layer. They have different jobs.

2. **"Each chain does one thing"** — Entity extraction, smoking gun detection, and story synthesis are separate chains. This makes them testable, swappable, and debuggable independently.

3. **"It runs fully local"** — The whole stack uses Ollama on the developer's machine. There's no external API call, no key management, and no per-token cost during development.

4. **"It scales"** — The LlamaIndex VectorStoreIndex can be swapped for Elasticsearch (which Altumatim uses) to handle millions of documents with zero code changes.

5. **"The frontend is decoupled"** — The HTML demo can call the Ollama API directly. In production, you'd route through a Flask/FastAPI backend for auth and rate limiting.
