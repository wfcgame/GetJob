"""Rank text candidates using embeddings supplied by an encoder."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.vector_math import cosine_similarity


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> object: ...


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    dimensions: int


def rank_candidates(
    query: str, candidates: Sequence[str], embedder: TextEmbedder
) -> list[SearchResult]:
    """Embed once, then rank candidates by descending cosine similarity.

    The encoder is injected so ranking can be tested without loading a model.
    Equal scores keep the candidates' original order.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be nonempty text.")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or any(
        not isinstance(text, str) or not text.strip() for text in candidates
    ):
        raise ValueError("Candidates must be a sequence of nonempty texts.")
    if not callable(getattr(embedder, "encode", None)):
        raise ValueError("Embedder must provide a callable encode method.")
    if not candidates:
        return []

    embeddings = embedder.encode([query, *candidates])
    if callable(getattr(embeddings, "tolist", None)):
        embeddings = embeddings.tolist()
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
        raise ValueError("Encoder must return a sequence of embeddings.")
    if len(embeddings) != len(candidates) + 1:
        raise ValueError("Encoder returned the wrong number of embeddings.")

    # SentenceTransformer returns NumPy arrays; tolist() converts their scalar
    # values to ordinary Python numbers accepted by cosine_similarity().
    vectors = [
        vector.tolist() if callable(getattr(vector, "tolist", None)) else vector
        for vector in embeddings
    ]
    if any(
        not isinstance(vector, Sequence) or isinstance(vector, (str, bytes))
        for vector in vectors
    ):
        raise ValueError("Each embedding must be a vector of real numbers.")
    query_vector = vectors[0]
    results = [
        SearchResult(
            text=text,
            score=cosine_similarity(query_vector, vector),
            dimensions=len(vector),
        )
        for text, vector in zip(candidates, vectors[1:])
    ]
    return sorted(results, key=lambda result: result.score, reverse=True)
