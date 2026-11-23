"""Compiled-kernel correctness against the NumPy reference.

These tests run in both modes: with the kernels available and with
NUSALEARN_NO_KERNEL=1 forcing the pure path. Either way the numbers
must agree with plain NumPy.
"""

import numpy as np
import pytest

from nusalearn import _native
from nusalearn._pairwise import pairwise_distances, pairwise_distances_numpy


def test_pairwise_distances_345_triangle():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0]])
    assert pairwise_distances(X, Y)[0, 0] == pytest.approx(5.0)


def test_pairwise_distances_match_numpy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(17, 4))
    Y = rng.normal(size=(11, 4))
    fast = pairwise_distances(X, Y)
    ref = pairwise_distances_numpy(X, Y)
    assert fast.shape == (17, 11)
    assert np.allclose(fast, ref, atol=1e-10)
    # Distance from a point to itself is zero.
    self_d = pairwise_distances(X)
    assert np.allclose(np.diag(self_d), 0.0, atol=1e-10)


def test_native_dot_matches_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert _native.dot(a, b) == pytest.approx(32.0)


def test_native_matmul_matches_numpy():
    if not _native.MATOPS_AVAILABLE:
        pytest.skip("C++ matmul module not compiled in this environment")
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 4))
    B = rng.normal(size=(4, 5))
    assert np.allclose(_native.matmul(A, B), A @ B, atol=1e-10)
