"""Lesson 5.3: rank local text candidates with all-MiniLM-L6-v2.

Run only when ready to load (and possibly download) the model:
    uv run --extra embeddings python -m scripts.practice_embeddings --run
"""

import argparse
import sys

from app.semantic_search import rank_candidates


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUERY = "How can I prepare for a software engineering interview?"
DEFAULT_CANDIDATES = (
    "Practice coding questions and explain your solutions aloud.",
    "Review your resume and prepare stories about past projects.",
    "Buy groceries for dinner tonight.",
)


def load_model():
    """Import the optional package only after the user opts in with --run."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL)


def run_experiment(query: str, candidates: list[str]) -> int:
    try:
        model = load_model()
    except ImportError:
        print("Install the optional embeddings extra before using --run.", file=sys.stderr)
        return 2

    results = rank_candidates(query, candidates, model)
    print(f"Model: {MODEL}")
    print(f"Query: {query}")
    print(f"Dimensions: {results[0].dimensions}")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.4f}  {result.text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Load the local model and rank texts")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text to search for")
    parser.add_argument("--candidate", action="append", help="Candidate text; repeat as needed")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    candidates = args.candidate if args.candidate is not None else list(DEFAULT_CANDIDATES)
    if not args.query.strip() or any(not text.strip() for text in candidates):
        parser.error("query and candidates must be nonempty")
    return run_experiment(args.query, candidates)


if __name__ == "__main__":
    raise SystemExit(main())
