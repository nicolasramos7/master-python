"""Shared type aliases.

`Floats` is the return type of every pricing function: a float64 array.
`Numeric` is what the pricing functions accept: a plain float, or an array of
them. Deliberately narrow -- NumPy ships a much broader `npt.ArrayLike` that
also covers lists, bytes and strings, and arithmetic on a union that wide
makes mypy enumerate every combination it could possibly be. Narrow types at
the boundary are a kindness to your type checker and to the next reader.
"""

import numpy.typing as npt
from numpy import float64

Floats = npt.NDArray[float64]
"""A float64 array of any shape."""

Numeric = float | Floats
"""A float, or a float64 array. What every pricing function takes."""
