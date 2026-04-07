"""
run.py — Project Entry Point
=============================
Run this file from the legalstoryos/ folder instead of backend/main.py directly.

WHY THIS FILE EXISTS:
When you run `python backend/main.py`, Python sets the "module root" to the
backend/ folder. So `from backend.config import ...` fails because Python
is already INSIDE backend/ and doesn't see it as a package.

When you run `python run.py` from legalstoryos/, Python sets the module root
to legalstoryos/ — so it can find `backend/` as a sub-package correctly.

USAGE:
    cd legalstoryos
    python run.py
"""

# This line adds the current directory to Python's module search path,
# ensuring all `from backend.x import y` statements work correctly.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can safely import and run the main pipeline
from backend.main import run_pipeline, SAMPLE_DOCUMENT
from backend.utils.display import console, print_results
from rich.panel import Panel

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]⚖ LegalStoryOS[/bold] — Story-Driven eDiscovery Intelligence\n"
        "[dim]LangChain · LlamaIndex · Claude API · Python[/dim]",
        border_style="gold1",
    ))

    results = run_pipeline(SAMPLE_DOCUMENT)
    print_results(results)