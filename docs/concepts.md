# 🧠 Concepts Deep-Dive — How All This Actually Works

This document explains every concept used in LegalStoryOS from first principles.
Read it before the interview so you can explain any part of the system fluently.

---

## 1. What Is an LLM?

An **LLM (Large Language Model)** is a neural network trained to predict the next
word in a sequence, over and over, on hundreds of billions of words of text.

After training, an LLM has "learned" statistical patterns about language, facts,
reasoning, and code. Claude is Anthropic's LLM.

### How we use it
We send Claude a "prompt" (a text message) and Claude predicts what a helpful,
intelligent response would look like. Under the hood it's generating one token
at a time, each token probabilistically chosen based on all the preceding text.

**Key concept: context window**
Claude can only "see" a certain amount of text at once (its context window).
Claude Sonnet can handle ~200,000 tokens. This is why LlamaIndex chunks documents —
for massive eDiscovery datasets, you can't fit everything in at once.

---

## 2. What Is the Anthropic API?

Instead of running Claude on your computer (it requires specialized hardware),
you send HTTP requests to Anthropic's servers. They run Claude and send back
the response.

### HTTP Request structure (simplified)
```
POST https://api.anthropic.com/v1/messages
Headers:
  x-api-key: sk-ant-...
  Content-Type: application/json

Body:
{
  "model": "claude-sonnet-4-20250514",
  "messages": [{"role": "user", "content": "Analyze this document..."}]
}
```

This is exactly what both `fetch()` in JavaScript and `ChatAnthropic` in LangChain
do — they just wrap this HTTP call in a more convenient interface.

**Why do you need an API key?**
The key identifies you. Anthropic counts how many tokens you use and charges you.
Without the key, the server rejects your request (HTTP 401 Unauthorized).

---

## 3. What Is LangChain?

LangChain is a framework for building applications powered by LLMs.

### The core problem it solves
Without LangChain, building an AI pipeline means:
  - Manually formatting prompt strings
  - Manually parsing LLM responses
  - Manually chaining outputs into the next prompt
  - No standardized way to swap models or components

LangChain gives you:

```
Prompt Template + Variables
         ↓
    [chain them with |]
         ↓
         LLM
         ↓
    [chain them with |]
         ↓
    Output Parser
```

### The | (pipe) operator — LCEL
LangChain Expression Language (LCEL) uses the `|` operator to connect steps:

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"document": "some text"})
```

This is similar to Unix pipes: `cat file.txt | grep "error" | wc -l`
Each step's output becomes the next step's input.

### Why separate chains for each task?
You could put all three tasks (entities, smoking guns, story) in one giant prompt.
But separate chains are better because:
  - **Focused prompts** = better results (models do one thing at a time better)
  - **Testable** = you can test entity extraction independently
  - **Composable** = in production, run entity + smoking gun in parallel
  - **Debuggable** = if story synthesis fails, you know exactly where

---

## 4. What Is LlamaIndex?

LlamaIndex specializes in **connecting LLMs to your data**. Its killer feature is
making RAG easy.

### The problem without LlamaIndex
You have a 500-page legal document. Claude's context window is ~200k tokens,
which is roughly ~150,000 words. The document is 125,000 words — it fits!
But what if you have 1,000 such documents? Or 500,000 emails?

You can't send everything to Claude every time. You need to **find the relevant parts**.

### LlamaIndex's approach

**Indexing (one time):**
```
Document text
    → split into chunks (SentenceSplitter)
    → embed each chunk (convert to a vector of numbers)
    → store in VectorStoreIndex (a searchable database of vectors)
```

**Querying (per question):**
```
Your question
    → embed the question (same vector conversion)
    → find the chunks with the most similar vectors
    → send those chunks + your question to Claude
    → Claude answers using only those chunks
```

### What is a vector / embedding?
An **embedding** is a list of numbers (typically 1536 numbers) that represents
the *meaning* of a piece of text in multi-dimensional space.

The key insight: **similar meanings → similar vectors → close together in space**

"The defendant concealed evidence" and "documents were hidden from prosecutors"
have similar embeddings even though they share no words.

This is fundamentally different from keyword search (which looks for exact words).
Embedding-based search understands *semantic meaning*.

### VectorStoreIndex vs Elasticsearch
In this demo: `VectorStoreIndex` stores vectors in memory. Fast for a demo.

In production (like Altumatim): Elasticsearch with vector search plugin.
Same concept, but:
  - Persists to disk (data survives restarts)
  - Scales to billions of documents
  - Supports hybrid search (keywords + vectors)
  - Multiple users can query it simultaneously

---

## 5. What Is RAG?

**RAG = Retrieval-Augmented Generation**

The standard way to let LLMs answer questions about your private documents.

**Without RAG:** Claude only knows what it was trained on (public internet data up to a cutoff date). It knows nothing about your specific documents.

**With RAG:**
```
1. [Retrieval] Find the relevant passages from your documents
2. [Augmentation] Add those passages to the prompt as context
3. [Generation] Claude generates an answer based on that context
```

The LLM is "augmented" with retrieved information. It only answers based on
what you gave it, which means:
  - No hallucination about documents it hasn't seen
  - Answers are grounded in actual document text
  - You can cite exactly which passage led to each answer

---

## 6. What Is Prompt Engineering?

The practice of designing prompts that get LLMs to produce the output you want.

### Techniques used in LegalStoryOS

**1. Expert Persona**
> "You are a seasoned trial litigator with 20 years of experience..."

Gives the LLM a role that activates relevant knowledge and tone.

**2. Structured Output**
> "Return ONLY valid JSON with NO markdown. Structure: {...}"

Forces the LLM to produce predictable, parseable output. Critical when your
code needs to parse the response programmatically.

**3. Output Format Specification**
> "For each finding, output in this exact format: 🔴 [CATEGORY]: [finding]"

Constrains the format so your display code can rely on it.

**4. Constraint Enumeration**
> "Categories: DATA_SUPPRESSION | FRAUD | OBSTRUCTION | ..."

Limits the space of possible outputs, making responses more consistent.

**5. Chain-of-Thought Forcing**
The story chain's output structure forces Claude to write the plaintiff narrative,
then the defense narrative, then legal risks. This sequential structure improves
reasoning quality — writing the defense view helps identify weaknesses in the
plaintiff's story.

---

## 7. How Python Imports Work

When you write `from backend.config import ANTHROPIC_API_KEY`, Python:
1. Looks for a folder called `backend`
2. Checks it has `__init__.py` (making it a "package")
3. Looks for a file called `config.py` inside it
4. Runs `config.py` and makes `ANTHROPIC_API_KEY` available

The folder structure maps directly to the import path:
```
backend/config.py         → from backend.config import X
backend/chains/entity.py  → from backend.chains.entity import X
```

---

## 8. What Is a Virtual Environment?

Python packages (like LangChain) are installed globally by default.
If Project A needs LangChain 0.2 and Project B needs LangChain 0.3, they conflict.

A virtual environment creates an isolated copy of Python + packages for each project:
```
legalstoryos/venv/
  ├── bin/python     ← this project's own Python interpreter
  ├── bin/pip        ← this project's own pip
  └── lib/           ← packages installed only for this project
```

When you `source venv/bin/activate`, your terminal uses this isolated Python.
Other projects' packages can't interfere.

---

## 9. How the Frontend Calls the API

The browser demo uses the `fetch()` API to call Anthropic directly:

```javascript
const response = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: {
    'x-api-key': apiKey,
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01',
  },
  body: JSON.stringify({ model: '...', messages: [...] })
});
const data = await response.json();
```

This is the **same HTTP request** that the Python `ChatAnthropic` object makes —
just written in JavaScript instead.

**Why async/await?**
Network requests take time (100-3000ms). JavaScript is single-threaded — if it
blocked waiting for the response, the browser UI would freeze. `async/await` lets
JavaScript say "I'm waiting for this, go do other things, come back when it's ready."

---

## 10. Key Interview Talking Points

| Question | Answer |
|---|---|
| "Why LlamaIndex and LangChain — aren't they the same?" | LlamaIndex owns the data layer (indexing, retrieval, RAG). LangChain owns the reasoning layer (chains, agents, orchestration). They complement each other. |
| "How would this scale to millions of documents?" | Swap VectorStoreIndex for Elasticsearch with vector search. LlamaIndex supports this with one line of config change. |
| "Why separate chains instead of one big prompt?" | Focused prompts produce better results. Separate chains are testable, composable, and debuggable independently. In production, entity + smoking gun chains can run in parallel. |
| "What's the difference between RAG and fine-tuning?" | Fine-tuning bakes knowledge into model weights (expensive, static). RAG retrieves knowledge at query time (cheap, dynamic, updatable). For eDiscovery, RAG is correct — you need to search new documents constantly. |
| "How would you add authentication?" | Flask/FastAPI backend holds the API key. Frontend calls your backend. Backend validates user auth, then calls Claude. The key never leaves your server. |
