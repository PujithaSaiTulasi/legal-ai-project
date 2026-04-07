"""
backend/utils/display.py — Terminal Output Formatting
======================================================

WHY HAVE A SEPARATE DISPLAY MODULE?
-------------------------------------
Mixing display logic with business logic is an anti-pattern.
If you want to change how results look (maybe export to JSON, or
send to Slack instead), you'd have to dig through your pipeline code.

By isolating display here, you can swap it out without touching anything else.
This is the "Single Responsibility Principle" — each module does ONE thing.

Rich is a Python library for beautiful terminal output.
It supports: colored text, panels, tables, progress bars, syntax highlighting.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

# Console is Rich's main object. It's like print() but with superpowers.
# You call console.print() instead of print() and get colors, markup, etc.
console = Console()


def print_step(step_number: int, description: str):
    """
    Prints a styled step header so the user can follow pipeline progress.
    Example output: ► Step 1: LlamaIndex — Indexing document...
    """
    console.print(
        f"\n[bold cyan]► Step {step_number}: {description}[/bold cyan]"
    )
    # [bold cyan]...[/bold cyan] is Rich's markup for bold + cyan colored text.
    # Rich markup looks like HTML but it's just for terminal styling.


def print_success(message: str):
    """Prints a green checkmark line for successful sub-steps."""
    console.print(f"  [green]✓[/green] {message}")


def print_results(results: dict):
    """
    Takes the final analysis dictionary and renders it as beautiful
    terminal panels and tables.

    Parameters:
        results (dict): Must contain keys: story, smoking_guns, entities, rag_answers
    """

    # Rule prints a horizontal divider line with text in the center
    console.print("\n")
    console.rule("[bold gold1]⚖  LegalStoryOS — Analysis Complete[/bold gold1]")
    console.print("\n")

    # ── Panel 1: Legal Narrative ──────────────────────────────────────────────
    # Panel draws a box around text with a title.
    # padding=(1, 2) means 1 line of vertical padding, 2 chars of horizontal.
    console.print(Panel(
        results["story"],
        title="[bold]📖 Legal Narrative[/bold]",
        border_style="gold1",    # gold colored border
        padding=(1, 2),
    ))

    # ── Panel 2: Smoking Guns ─────────────────────────────────────────────────
    console.print(Panel(
        results["smoking_guns"],
        title="[bold red]🔴 Smoking Gun Evidence[/bold red]",
        border_style="red",
        padding=(1, 2),
    ))

    # ── Panel 3: Entities ─────────────────────────────────────────────────────
    console.print(Panel(
        results["entities"],
        title="[bold cyan]🔍 Key Entities Extracted[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Table: RAG Answers ────────────────────────────────────────────────────
    # Table creates a structured grid. Great for showing Q&A pairs.
    table = Table(
        title="💬 LlamaIndex RAG — Targeted Queries",
        box=box.ROUNDED,         # rounded corner style
        border_style="green",
        show_header=True,
        header_style="bold green",
    )

    # Add columns — first arg is the column name
    table.add_column("Question", style="bold", width=35)
    table.add_column("Answer (from document)", width=60)

    # Add a row for each RAG question/answer pair
    for question, answer in results["rag_answers"].items():
        # Truncate long answers so the table stays readable
        truncated = answer[:250] + "..." if len(answer) > 250 else answer
        table.add_row(question, truncated)

    console.print(table)

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print(
        "\n[bold green]✓ Pipeline complete.[/bold green] "
        "[dim]LlamaIndex handled retrieval. "
        "LangChain handled reasoning. "
        "Claude powered the LLM layer.[/dim]\n"
    )
