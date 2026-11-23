"""Pairwise distances with an optional compiled accelerator.

The Cython module ``_fast`` provides the fast path. When it has not
been compiled (for example in a pure Python checkout), a NumPy
implementation is used instead. Setting the environment variable
NUSALEARN_NO_KERNEL=1 forces the pure path, which the test suite uses
to verify both implementations agree.
"""

import os

import numpy as np

_FORCE_PURE = os.environ.get("NUSALEARN_NO_KERNEL", "") == "1"

try:
    if _FORCE_PURE:
        raise ImportError("pure path forced by NUSALEARN_NO_KERNEL")
    from . import _fast

    KERNEL_AVAILABLE = True
except Exception:  # ImportError or a broken build
    _fast = None
    KERNEL_AVAILABLE = False


def pairwise_distances_numpy(X, Y):
    """Euclidean distances between rows of X and rows of Y, in NumPy."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_sq = np.sum(X * X, axis=1, keepdims=True)
    y_sq = np.sum(Y * Y, axis=1)
    sq = x_sq + y_sq - 2.0 * (X @ Y.T)
    # Guard against tiny negative values from floating point rounding.
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq)


def pairwise_distances(X, Y=None):
    """Euclidean pairwise distances, using the compiled kernel if present."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    self_pairs = Y is None
    if self_pairs:
        Y = X
    else:
        Y = np.ascontiguousarray(Y, dtype=np.float64)
    if KERNEL_AVAILABLE:
        distances = _fast.pairwise_euclidean(X, Y)
    else:
        distances = pairwise_distances_numpy(X, Y)
    if self_pairs:
        # The distance from a point to itself is exactly zero; the
        # Gram matrix formulation can leave tiny rounding residue on
        # the diagonal, so set it explicitly.
        np.fill_diagonal(distances, 0.0)
    return distances
