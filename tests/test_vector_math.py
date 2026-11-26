"""Offline mathematical and validation tests for cosine similarity."""

import math

import pytest

from app.vector_math import cosine_similarity


@pytest.mark.parametrize(("a", "b", "expected"), [
    ([1, 2, 3], [1, 2, 3], 1.0),
    ([1, 2], [3, 6], 1.0),
    ([1, 2], [-1, -2], -1.0),
    ([1, 0], [0, 1], 0.0),
    ([1, 1], [1, -1], 0.0),
    ([1, 0], [1, 1], 1 / math.sqrt(2)),
    ([1, 2, 3], [4, 5, 6], 32 / math.sqrt(14 * 77)),
    ([-1, 0], [1, 1], -1 / math.sqrt(2)),
    ([2], [7], 1.0),
    ([2], [-7], -1.0),
    ((0.5, 0.0), (0.5, 0.5), 1 / math.sqrt(2)),
    ([1e308, 1e308], [1e308, 0], 1 / math.sqrt(2)),
    ([1e-300, 1e-300], [1e-300, 0], 1 / math.sqrt(2)),
    ([1e308, 0], [1e-300, 1e-300], 1 / math.sqrt(2)),
])
def test_mathematical_cases(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)
    assert cosine_similarity(b, a) == pytest.approx(expected)


@pytest.mark.parametrize(("a", "b", "message"), [
    ([], [], "empty"),
    ([], [1], "empty"),
    ([1], [], "empty"),
    ([0, 0], [1, 2], "Zero"),
    ([1, 2], [0, 0], "Zero"),
    ([0], [0], "Zero"),
    ([-0.0], [1], "Zero"),
    ([1], [1, 2], "dimensions"),
    ([1, 2], [1], "dimensions"),
])
def test_invalid_vectors(a, b, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(a, b)


@pytest.mark.parametrize("value", [
    math.nan, math.inf, -math.inf, "2", "text", None, True, False,
    complex(1, 2), [], {}, object(),
])
@pytest.mark.parametrize("side", ["left", "right"])
def test_invalid_components(value, side):
    a, b = [1, 2], [3, 4]
    (a if side == "left" else b)[1] = value
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


@pytest.mark.parametrize("vector", [None, 3, "12", b"12", {1, 2}, {1: 2}])
def test_invalid_containers(vector):
    with pytest.raises(ValueError, match="sequences"):
        cosine_similarity(vector, [1, 2])


def test_inputs_are_not_modified():
    a, b = [1, 2, 3], [4, 5, 6]
    cosine_similarity(a, b)
    assert a == [1, 2, 3]
    assert b == [4, 5, 6]
