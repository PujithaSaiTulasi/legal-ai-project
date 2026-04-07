"""
backend/chains/story_chain.py — Legal Story Synthesis Chain
============================================================

WHAT IS THIS CHAIN'S JOB?
--------------------------
Take ALL the outputs from the previous chains (entities + smoking guns)
and synthesize them into a structured legal narrative — the "story" of
what happened, told from multiple perspectives.

THIS CHAIN IS DIFFERENT: IT TAKES MULTIPLE INPUTS
--------------------------------------------------
The entity and smoking gun chains each took one input: {document}.

This chain takes THREE inputs:
  - {document}     — the original text
  - {entities}     — output from the entity chain
  - {smoking_guns} — output from the smoking gun chain

This is the "chain of chains" pattern — earlier outputs feed into later ones.
In LangChain terms, this is called a "sequential chain" or "pipeline."

WHY FEED PREVIOUS OUTPUTS INTO THIS CHAIN?
-------------------------------------------
Claude works better with structured context. Instead of re-discovering
entities and key evidence from scratch, we hand it the already-extracted
information and say "now synthesize this into a story."

This mirrors how human lawyers work: analysts extract facts, paralegals
organize the timeline, then the senior attorney weaves it into a narrative.
Each specialist does their focused job, then passes to the next.

PROMPT ENGINEERING TECHNIQUE: "Chain-of-Thought + Structure"
-------------------------------------------------------------
The prompt forces a specific output structure with clear section headers.
This does two things:
  1. Makes the output predictable and parseable
  2. Forces Claude to reason through the problem step-by-step before
     synthesizing — structured output = structured thinking

The "Defense Counter-Narrative" section is especially valuable —
it shows you've thought about both sides, which is what good lawyers do
and what impresses interviewers.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel


def build_story_chain(llm: BaseChatModel):
    """
    Builds the story synthesis chain.

    Parameters:
        llm: Shared chat model instance.

    Returns:
        A LangChain chain that accepts a dict with keys:
        document, entities, smoking_guns
    """

    story_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are altumatimOS's story engine — the AI that turns raw legal
evidence into the narrative that wins cases. Your job is to synthesize facts
into a compelling, airtight legal story that a senior attorney could use
directly in trial preparation.

Output in EXACTLY this structure:

─────────────────────────────────────
THE STORY (plaintiff perspective):
─────────────────────────────────────
[2-3 paragraph narrative grounded in specific facts. Name names. Cite figures.
Show the sequence of events that constitutes the wrongdoing.]

─────────────────────────────────────
DEFENSE COUNTER-NARRATIVE:
─────────────────────────────────────
[1 paragraph — how would the defense tell this story? What would they argue?
What facts would they emphasize or reframe?]

─────────────────────────────────────
KEY LEGAL RISKS & EXPOSURE:
─────────────────────────────────────
• [Specific legal risk with citation to the evidence]
• [Second legal risk]
• [Third legal risk]

─────────────────────────────────────
RECOMMENDED NEXT DISCOVERY STEPS:
─────────────────────────────────────
• [What documents/depositions should be subpoenaed next?]
• [Second recommendation]

Ground everything in the provided facts. No speculation."""),

        ("human", """Synthesize the legal story from this information:

=== ORIGINAL DOCUMENT ===
{document}

=== EXTRACTED ENTITIES ===
{entities}

=== SMOKING GUN EVIDENCE ===
{smoking_guns}

Now synthesize the complete legal narrative.""")
    ])

    # Note: this chain does NOT use RunnablePassthrough because it already
    # receives a dict with multiple keys. The prompt template directly maps
    # {document}, {entities}, and {smoking_guns} from the input dict.
    chain = story_prompt | llm | StrOutputParser()

    return chain
