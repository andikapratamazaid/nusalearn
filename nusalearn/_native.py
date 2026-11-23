"""Accessors for the optional C and C++ extension modules.

Both modules are conveniences, not requirements: every function here
falls back to NumPy when the compiled module is unavailable.
"""

import numpy as np

try:
    from . import _dot

    DOT_AVAILABLE = True
except Exception:
    _dot = None
    DOT_AVAILABLE = False

try:
    from . import _matops

    MATOPS_AVAILABLE = True
except Exception:
    _matops = None
    MATOPS_AVAILABLE = False


def dot(a, b):
    """Dot product of two 1D arrays, via the C extension when present."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if DOT_AVAILABLE:
        return _dot.dot(a, b)
    return float(np.dot(a, b))


def matmul(A, B):
    """Matrix product, via the C++ extension when present."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    if MATOPS_AVAILABLE:
        return _matops.matmul(A, B)
    return A @ B
