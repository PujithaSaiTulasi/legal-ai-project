"""
backend/chains/smoking_gun_chain.py — Smoking Gun Detection Chain
=================================================================

WHAT IS THIS CHAIN'S JOB?
--------------------------
Identify the 3 most legally significant pieces of evidence in a document —
the moments that would "win or lose" the case. This is the core of altumatimOS's
value proposition: surfacing the needle in the haystack.

HOW IS THIS DIFFERENT FROM THE ENTITY CHAIN?
---------------------------------------------
  - Entity chain: Extracts WHAT exists (people, dates, amounts) — factual extraction
  - Smoking gun chain: Identifies WHAT MATTERS legally — requires reasoning and judgment

This distinction is why we use separate chains. Each has a completely different
system prompt that gives Claude a different "expert persona" and different instructions.
Mixing them into one prompt would make the output worse — models perform better
with focused, single-task prompts.

PROMPT ENGINEERING TECHNIQUE USED HERE: "Expert Persona"
---------------------------------------------------------
We tell Claude "You are a seasoned litigator" instead of just "find important things."
This activates Claude's knowledge about legal strategy, evidence standards, and
courtroom dynamics. The persona shapes the lens through which it analyzes the text.

Other techniques visible in this prompt:
  - Output format specification (numbered list with specific fields)
  - Category enumeration (DATA_SUPPRESSION | FRAUD | etc.) constrains and
    structures the response
  - "Rank by legal severity" — instructs ordering, not just listing
  - "Be specific — cite names, dates, figures" — prevents vague generalities
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough


def build_smoking_gun_chain(llm: BaseChatModel):
    """
    Builds the smoking gun detection chain.

    Parameters:
        llm: Shared chat model instance.

    Returns:
        A LangChain runnable chain.
    """

    smoking_gun_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a seasoned trial litigator with 20 years of experience in
corporate fraud and regulatory violations. Your job is to identify the 3 most
damaging pieces of evidence in this document — the "smoking guns" that would
be decisive in court.

For each, output in this exact format:

🔴 [CATEGORY]: [Direct quote or precise paraphrase with names/figures]
   WHY IT MATTERS: [One sentence on the specific legal significance]

Categories (use the most specific one):
DATA_SUPPRESSION | FRAUD | OBSTRUCTION | WHISTLEBLOWER | FINANCIAL_MISCONDUCT | CONSPIRACY | PERJURY

Rules:
- Rank by legal severity (most damaging first)
- Always cite specific names, dates, and dollar figures when present
- Quote directly from the document where possible
- Do not speculate — only what the document explicitly shows"""),

        ("human", "Identify the smoking gun evidence in this document:\n\n{document}")
    ])

    parser = StrOutputParser()

    # Same pipe pattern as entity_chain — this is LangChain's LCEL syntax
    # (LangChain Expression Language). Every chain follows this pattern.
    chain = (
        {"document": RunnablePassthrough()}
        | smoking_gun_prompt
        | llm
        | parser
    )

    return chain
