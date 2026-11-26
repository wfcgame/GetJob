"""Offline ranking tests with fake embeddings; no model is loaded."""

import math

import pytest

from app.semantic_search import rank_candidates
from scripts import practice_embeddings


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return self.vectors


class ArrayLikeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class ArrayLikeBatch(ArrayLikeVector):
    pass


def test_ranks_by_cosine_similarity_and_preserves_text():
    embedder = FakeEmbedder([[1, 0], [0, 1], [1, 1], [1, 0]])

    results = rank_candidates("query", ["unrelated", "related", "exact"], embedder)

    assert embedder.calls == [["query", "unrelated", "related", "exact"]]
    assert [result.text for result in results] == ["exact", "related", "unrelated"]
    assert [result.score for result in results] == pytest.approx([1, 2**-0.5, 0])
    assert [result.dimensions for result in results] == [2, 2, 2]


def test_accepts_array_like_vectors_from_encoder():
    embedder = FakeEmbedder([ArrayLikeVector([1.0, 0.0]), ArrayLikeVector([1.0, 0.0])])

    assert rank_candidates("query", ["match"], embedder)[0].score == pytest.approx(1.0)


def test_accepts_numpy_style_batch_output():
    embedder = FakeEmbedder(ArrayLikeBatch([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))

    results = rank_candidates("query", ["low", "high"], embedder)

    assert [result.text for result in results] == ["high", "low"]
    assert all(result.dimensions == 2 for result in results)


def test_equal_scores_keep_original_order():
    embedder = FakeEmbedder([[1, 0], [1, 1], [1, -1]])

    results = rank_candidates("query", ["first", "second"], embedder)

    assert [result.text for result in results] == ["first", "second"]


def test_empty_candidates_do_not_call_encoder():
    embedder = FakeEmbedder([])

    assert rank_candidates("query", [], embedder) == []
    assert embedder.calls == []


@pytest.mark.parametrize("query,candidates", [
    ("", ["candidate"]),
    ("   ", ["candidate"]),
    (None, ["candidate"]),
    (123, ["candidate"]),
    ("query", [""]),
    ("query", ["  "]),
    ("query", [None]),
    ("query", [123]),
])
def test_rejects_empty_text(query, candidates):
    with pytest.raises(ValueError, match="nonempty"):
        rank_candidates(query, candidates, FakeEmbedder([]))


@pytest.mark.parametrize("candidates", [None, 123, "single text", b"bytes", {"text"}, (text for text in ["one"])])
def test_rejects_invalid_candidate_container(candidates):
    with pytest.raises(ValueError, match="sequence"):
        rank_candidates("query", candidates, FakeEmbedder([]))


@pytest.mark.parametrize("embedder", [None, object(), {"encode": lambda texts: []}])
def test_rejects_missing_encode_method(embedder):
    with pytest.raises(ValueError, match="encode method"):
        rank_candidates("query", ["candidate"], embedder)


@pytest.mark.parametrize("output", [None, 123, "vectors", b"vectors", {"one": [1, 0]}, iter([[1, 0], [1, 0]])])
def test_rejects_malformed_embedding_batch(output):
    with pytest.raises(ValueError, match="sequence of embeddings"):
        rank_candidates("query", ["candidate"], FakeEmbedder(output))


def test_rejects_wrong_embedding_count():
    with pytest.raises(ValueError, match="number of embeddings"):
        rank_candidates("query", ["candidate"], FakeEmbedder([[1, 0]]))


@pytest.mark.parametrize("vectors", [
    [[1, 0], None],
    [[1, 0], 5],
    [[1, 0], "1, 0"],
    [ArrayLikeVector(1.0), [1, 0]],
])
def test_rejects_malformed_embedding_vectors(vectors):
    with pytest.raises(ValueError, match="vector of real numbers"):
        rank_candidates("query", ["candidate"], FakeEmbedder(vectors))


@pytest.mark.parametrize("vectors", [
    [[], []],
    [[0, 0], [1, 0]],
    [[1, 0], [0, 0]],
    [[1, 0], [math.nan, 1]],
    [[1, 0], [True, 1]],
])
def test_reuses_cosine_validation_for_invalid_values(vectors):
    with pytest.raises(ValueError):
        rank_candidates("query", ["candidate"], FakeEmbedder(vectors))


def test_reuses_vector_validation_for_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions"):
        rank_candidates("query", ["candidate"], FakeEmbedder([[1, 0], [1]]))


def test_cli_without_run_never_loads_model(monkeypatch, capsys):
    monkeypatch.setattr(practice_embeddings, "load_model", lambda: pytest.fail("model loaded"))
    monkeypatch.setattr("sys.argv", ["practice_embeddings.py"])

    assert practice_embeddings.main() == 0
    assert "--run" in capsys.readouterr().out


def test_cli_run_prints_dimensions_and_ranked_scores(monkeypatch, capsys):
    embedder = FakeEmbedder([[1, 0], [0, 1], [1, 0]])
    monkeypatch.setattr(practice_embeddings, "load_model", lambda: embedder)
    monkeypatch.setattr(
        "sys.argv",
        ["practice_embeddings.py", "--run", "--query", "query", "--candidate", "low", "--candidate", "high"],
    )

    assert practice_embeddings.main() == 0
    output = capsys.readouterr().out
    assert "Dimensions: 2" in output
    assert "1. score=1.0000  high" in output
    assert "2. score=0.0000  low" in output
    assert embedder.calls == [["query", "low", "high"]]
