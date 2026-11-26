"""Pure Python vector math with no external service dependencies."""

import math
from collections.abc import Sequence
from numbers import Real


def _validated_vector(vector: Sequence[Real]) -> list[float]:
    """Convert a nonempty sequence of finite real numbers to floats."""
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError("Vectors must be sequences of real numbers.")
    if len(vector) == 0:
        raise ValueError("Vectors must not be empty.")

    values = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Vector values must be real numbers, excluding booleans.")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Vector values must be finite floats.") from exc
        if not math.isfinite(number):
            raise ValueError("Vector values must be finite floats.")
        values.append(number)
    return values


def cosine_similarity(a: Sequence[Real], b: Sequence[Real]) -> float:
    """Return dot(a, b) / (magnitude(a) * magnitude(b)), in [-1, 1].

    Raise ValueError for empty, zero, or differently sized vectors, or values
    that are not finite real numbers representable as floats. Booleans and
    numeric strings are rejected. Inputs are never modified.
    """
    left = _validated_vector(a)
    right = _validated_vector(b)
    if len(left) != len(right):
        raise ValueError("Vectors must have matching dimensions.")

    left_scale = max(abs(value) for value in left)
    right_scale = max(abs(value) for value in right)
    if left_scale == 0 or right_scale == 0:
        raise ValueError("Zero vectors have undefined cosine similarity.")

    # Positive scaling preserves the angle and prevents overflow/underflow
    # when multiplying very large or very small input components.
    left = [value / left_scale for value in left]
    right = [value / right_scale for value in right]
    dot_product = math.fsum(x * y for x, y in zip(left, right))
    left_magnitude = math.sqrt(math.fsum(x * x for x in left))
    right_magnitude = math.sqrt(math.fsum(y * y for y in right))
    similarity = dot_product / (left_magnitude * right_magnitude)
    return max(-1.0, min(1.0, similarity))
