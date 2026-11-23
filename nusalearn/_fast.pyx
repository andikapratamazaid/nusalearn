# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Compiled pairwise Euclidean distances.

This is the hot loop behind k-nearest neighbours and KMeans. It is a
straightforward typed implementation written for NusaLearn; the
algorithm itself (sum of squared differences, then a square root) is
the textbook definition of Euclidean distance.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


def pairwise_euclidean(double[:, ::1] X, double[:, ::1] Y):
    """Return the (n_samples_X, n_samples_Y) matrix of Euclidean distances."""
    cdef Py_ssize_t n_x = X.shape[0]
    cdef Py_ssize_t n_y = Y.shape[0]
    cdef Py_ssize_t n_f = X.shape[1]
    cdef Py_ssize_t i, j, k
    cdef double diff, acc
    out = np.empty((n_x, n_y), dtype=np.float64)
    cdef double[:, ::1] out_view = out
    for i in range(n_x):
        for j in range(n_y):
            acc = 0.0
            for k in range(n_f):
                diff = X[i, k] - Y[j, k]
                acc += diff * diff
            out_view[i, j] = sqrt(acc)
    return out
