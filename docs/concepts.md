# Concepts Deep-Dive — How LegalStoryOS Works

This document explains the project from first principles. The current stack is
Ollama-only: local chat generation, local embeddings, LangChain orchestration,
and LlamaIndex retrieval.

---

## 1. What Is an LLM?

An **LLM (Large Language Model)** is a neural network trained to predict the next
token in a sequence. After training, it can follow instructions, summarize text,
extract facts, reason over context, and produce structured output.

### How we use it

LegalStoryOS sends legal document text to a local Ollama model with task-specific
instructions. The model generates analysis one token at a time based on the
prompt and the document text.

**Key concept: context window**

Every model can only see a limited amount of text at once. This is why
LegalStoryOS uses LlamaIndex to chunk documents and retrieve only the most
relevant passages for targeted questions.

---

## 2. What Is Ollama?

Ollama runs LLMs locally on your machine and exposes a small HTTP API at:

```text
http://127.0.0.1:11434
```

This project uses Ollama for both:

- Chat generation with `llama3`
- Local embeddings with `nomic-embed-text`

Install and prepare the models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama serve
```

The frontend calls the local Ollama API directly. The Python backend uses
`langchain-ollama` and `llama-index` Ollama integrations.

---

## 3. What Is LangChain?

LangChain is a framework for building applications powered by LLMs.

Without LangChain, building an AI pipeline means:

- Manually formatting prompt strings
- Manually parsing model responses
- Manually chaining outputs into the next prompt
- Repeating orchestration code across tasks

LangChain gives you:

```text
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

### The `|` Pipe Operator

LangChain Expression Language uses the `|` operator to connect steps:

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"document": "some text"})
```

Each step's output becomes the next step's input.

### Why Separate Chains?

LegalStoryOS uses separate chains for entity extraction, smoking-gun detection,
and story synthesis because focused prompts are easier to debug and usually
produce more reliable outputs.

---

## 4. What Is LlamaIndex?

LlamaIndex connects LLMs to your data. Its main job here is RAG:
Retrieval-Augmented Generation.

### Indexing

```text
Document text
    → split into chunks
    → embed each chunk as a vector
    → store chunks in VectorStoreIndex
```

### Querying

```text
Question
    → embed the question
    → find the most similar chunks
    → send those chunks + the question to the local model
    → answer using the retrieved context
```

### What Is an Embedding?

An embedding is a list of numbers that represents the meaning of text.
Similar meanings produce similar vectors, which lets the system search by
meaning instead of exact keywords.

---

## 5. What Is RAG?

**RAG = Retrieval-Augmented Generation**

RAG lets an LLM answer questions about private documents without retraining the
model. It works in three steps:

```text
1. Retrieval: find relevant passages from the indexed document
2. Augmentation: add those passages to the prompt as context
3. Generation: ask the local model to answer using that context
```

For eDiscovery, RAG is useful because the document set changes constantly and
the system needs to answer based on case-specific evidence.

---

## 6. What Is Prompt Engineering?

Prompt engineering is designing instructions that get the model to produce the
output you need.

Techniques used in LegalStoryOS:

- **Expert persona:** asks the model to act like a legal analyst or litigator
- **Structured output:** requires predictable sections or JSON
- **Output format specification:** makes downstream rendering easier
- **Constraint enumeration:** limits categories to a known legal taxonomy
- **Sequential synthesis:** feeds extracted facts into the narrative chain

---

## 7. How Python Imports Work

When you write `from backend.config import OLLAMA_MODEL`, Python:

1. Looks for a folder called `backend`
2. Checks it has `__init__.py`
3. Looks for `config.py` inside that folder
4. Runs `config.py` and makes `OLLAMA_MODEL` available

The folder structure maps directly to the import path:

```text
backend/config.py              → from backend.config import X
backend/chains/entity_chain.py → from backend.chains.entity_chain import X
```

---

## 8. What Is a Virtual Environment?

A virtual environment creates an isolated Python installation for this project:

```text
legalstoryos/venv/
  ├── bin/python
  ├── bin/pip
  └── lib/
```

When you activate it, this project uses its own dependencies instead of whatever
is installed globally on your machine.

---

## 9. How the Frontend Calls Ollama

The browser demo uses `fetch()` to call the local Ollama chat API:

```javascript
const response = await fetch('http://127.0.0.1:11434/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama3',
    stream: false,
    format: 'json',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: documentText }
    ]
  })
});

const data = await response.json();
const result = JSON.parse(data.message.content);
```

Because the page is opened from `file://`, Ollama should be started with browser
origins allowed:

```bash
OLLAMA_ORIGINS='*' ollama serve
```

### Why async/await?

Model requests take time. `async/await` lets the browser keep the UI responsive
while it waits for Ollama to return a response.

---

## 10. Key Interview Talking Points

| Question | Answer |
|---|---|
| "Why LlamaIndex and LangChain?" | LlamaIndex owns the data/retrieval layer. LangChain owns the reasoning/orchestration layer. |
| "Why Ollama?" | It keeps the demo local, avoids cloud API keys, and makes the project runnable without paid endpoints. |
| "How would this scale to millions of documents?" | Keep the chain design, but replace the in-memory vector index with a production vector database or Elasticsearch vector search. |
| "Why separate chains instead of one big prompt?" | Focused prompts are more reliable, easier to test, and easier to debug. |
| "What's the difference between RAG and fine-tuning?" | Fine-tuning changes model behavior. RAG retrieves current case documents at query time. For eDiscovery, RAG is the better fit. |
| "How would you add authentication?" | Put a backend between the browser and any shared service, validate users there, and keep access controls server-side. |
