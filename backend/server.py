"""
backend/server.py — Flask wrapper around run_pipeline()
========================================================

Exposes the same pipeline the terminal runs (python run.py) over HTTP so the
browser demo can render exactly the same panels.

Run from the project root:
    ./venv/bin/python backend/server.py

Then open frontend/index.html.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.main import run_pipeline

app = Flask(__name__)
CORS(app)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("document") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'document' in request body"}), 400

    results = run_pipeline(text)
    return jsonify({
        "entities":     results["entities"],
        "smoking_guns": results["smoking_guns"],
        "story":        results["story"],
        "rag_answers": [
            {"question": q, "answer": a}
            for q, a in results["rag_answers"].items()
        ],
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
