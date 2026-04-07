"""
backend/chains/entity_chain.py — Entity Extraction Chain
=========================================================

WHAT IS A LANGCHAIN CHAIN?
----------------------------
A "chain" in LangChain is a sequence of steps where the output of one step
becomes the input of the next. The simplest chain is:

    Prompt Template → LLM → Output Parser

Think of it like an assembly line:
  1. The Prompt Template takes your variables and builds the full prompt text
  2. The LLM (Claude) reads the prompt and generates a response
  3. The Output Parser cleans up the response into a usable format

WHY NOT JUST CALL CLAUDE DIRECTLY?
------------------------------------
You could just write:
    response = anthropic.messages.create(prompt="Extract entities from: " + text)

But LangChain chains give you:
  - Composability: chain this with other chains
  - Testability: swap the LLM, change the parser, test each piece independently
  - Observability: LangChain can log every input/output for debugging
  - Reusability: import this chain anywhere in the codebase

WHAT IS THIS CHAIN'S JOB?
--------------------------
Extract structured information from raw document text:
  - Person names
  - Organization names
  - Dates
  - Monetary amounts
  - Legal concepts (fraud, suppression, etc.)

This is called "Named Entity Recognition" (NER) — a classic NLP task,
but here we use Claude instead of a traditional ML model, which means
we get much better results on complex legal language.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough


def build_entity_chain(llm: BaseChatModel):
    """
    Builds and returns the entity extraction chain.

    Parameters:
        llm: A configured chat model (Ollama or Anthropic), shared across chains.

    Returns:
        A LangChain runnable chain you can call with .invoke(document_text)

    Why accept llm as a parameter instead of creating it here?
    This is called "dependency injection" — it makes testing easier
    (you can pass a mock LLM) and avoids creating multiple LLM instances.
    """

    # ── The Prompt Template ───────────────────────────────────────────────────
    #
    # ChatPromptTemplate builds a structured chat conversation.
    # It takes a list of (role, message) tuples:
    #   "system" = the instructions that define how Claude should behave
    #   "human"  = the actual user message (the document text)
    #
    # Notice {document} — this is a placeholder variable.
    # When you call .invoke({"document": some_text}), LangChain replaces
    # {document} with the actual text before sending to Claude.
    #
    # Writing good system prompts is a skill called "prompt engineering."
    # The key principles used here:
    #   1. Clear role definition ("You are a legal entity extractor")
    #   2. Exact output format specified (so parsing is reliable)
    #   3. "Output nothing else" — prevents Claude from adding preamble text
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal entity extractor specialized in eDiscovery.
Extract all named entities from the document and return them in EXACTLY this format:

PERSONS: [comma-separated full names]
ORGANIZATIONS: [comma-separated organization names]
DATES: [comma-separated dates in any format found]
MONETARY: [comma-separated dollar amounts or financial figures]
LEGAL_CONCEPTS: [comma-separated legal concepts, e.g. fraud, suppression, whistleblower, FDA violation]

Be exhaustive — extract every entity. Output nothing else. No explanations, no preamble."""),

        ("human", "Analyze this legal document and extract all entities:\n\n{document}")
    ])

    # ── The Output Parser ─────────────────────────────────────────────────────
    #
    # StrOutputParser is the simplest parser — it just takes Claude's response
    # and returns it as a plain Python string.
    #
    # Other parsers exist for more structured output:
    #   - JsonOutputParser: parses Claude's response as JSON
    #   - PydanticOutputParser: validates against a Pydantic model schema
    #   - CommaSeparatedListOutputParser: splits a comma-separated list
    #
    # We use StrOutputParser here for simplicity, then display the raw text.
    parser = StrOutputParser()

    # ── The Chain: Pipe Operator ──────────────────────────────────────────────
    #
    # The | (pipe) operator is LangChain's way of connecting steps.
    # It works exactly like Unix pipes: output of left → input of right.
    #
    # This chain reads as:
    #   "Take the input dict, put it through the prompt template,
    #    send the result to the LLM, then parse the output as a string."
    #
    # RunnablePassthrough() is needed here because the prompt template
    # expects a dict with a "document" key, and we're passing a plain string.
    # RunnablePassthrough wraps the input string into {"document": input}.
    chain = (
        {"document": RunnablePassthrough()}
        | entity_prompt
        | llm
        | parser
    )

    return chain
