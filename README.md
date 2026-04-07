# ⚖️ LegalStoryOS — Complete Setup & Concept Guide

> A story-driven eDiscovery intelligence tool powered by LangChain, LlamaIndex, and Claude.
> Built to demonstrate production-grade AI pipeline architecture for Altumatim.

---

## 📁 Project Structure

```
legalstoryos/
│
├── README.md                  ← You are here. Full guide + concept explanations.
│
├── .env                       ← Your secret API key (never commit this to git!)
├── requirements.txt           ← All Python packages the project needs
│
├── backend/                   ← All Python / AI code lives here
│   ├── main.py                ← Entry point — runs the full pipeline in your terminal
│   ├── config.py              ← Central config (model names, chunk sizes, etc.)
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
│  STEP 3: Claude API (the LLM)   │  ← "The Brain"
│  Powers every AI decision.      │    Reads text, understands meaning,
│  LangChain and LlamaIndex both  │    generates intelligent responses
│  call Claude under the hood     │
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
- `langchain-anthropic` — LangChain's plugin for Claude
- `llama-index` — the document indexing & retrieval framework
- `llama-index-llms-anthropic` — LlamaIndex's plugin for Claude
- `anthropic` — the official Anthropic SDK
- `python-dotenv` — reads your .env file
- `rich` — makes your terminal output beautiful
- `flask` — optional, for running a local web server

---

### Step 5: Get Your Anthropic API Key

1. Go to https://console.anthropic.com
2. Sign in (or create an account)
3. Click "API Keys" in the left sidebar
4. Click "Create Key" — copy the key (it starts with `sk-ant-...`)

**What is an API Key?** It's a password that proves to Anthropic's servers that you're allowed to use Claude. Every time your code calls Claude, it sends this key. Keep it secret — never put it in code you share.

---

### Step 6: Create Your .env File

Create a file called `.env` in the root of the project:
```bash
# Create and edit the .env file
touch .env
```

Open it and add one line:
```
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

**What is a .env file?** A simple text file that stores secret values (like passwords and API keys) separate from your code. The `python-dotenv` library reads this file at runtime and makes the values available as environment variables. This way, your secrets never get accidentally committed to GitHub.

---

### Step 7: Run the Backend Pipeline

```bash
cd backend
python main.py
```

You'll see the analysis run step-by-step in your terminal with colored output.

---

### Step 8: Open the Frontend Demo

No server needed — just open the file directly:
```bash
# On Mac:
open frontend/index.html

# On Windows:
start frontend/index.html

# Or just double-click the file in your file explorer
```

Enter your API key in the browser UI, paste a document, and click Analyze.

---

## 🧪 Quick Test

To verify everything works before the interview:
```bash
# From the legalstoryos/ folder, with venv activated:
python backend/main.py
```

Expected output: colored panels showing entities, smoking guns, and the legal narrative.

---

## 💡 Talking Points for the Interview

When they ask "walk me through your project":

1. **"I separated concerns clearly"** — LlamaIndex handles the data/retrieval layer, LangChain handles the reasoning layer. They have different jobs.

2. **"Each chain does one thing"** — Entity extraction, smoking gun detection, and story synthesis are separate chains. This makes them testable, swappable, and debuggable independently.

3. **"It scales"** — The LlamaIndex VectorStoreIndex can be swapped for Elasticsearch (which Altumatim uses) to handle millions of documents with zero code changes.

4. **"The frontend is decoupled"** — The HTML demo can call the API directly. In production, you'd route through a Flask/FastAPI backend for auth and rate limiting.
