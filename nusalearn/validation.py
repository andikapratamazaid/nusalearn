"""Input validation helpers used across the library."""

import numpy as np


def check_array(X, name="X"):
    """Convert X to a 2D float ndarray and validate it."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("%s must be 2-dimensional, got %d dimensions" % (name, arr.ndim))
    if arr.shape[0] == 0:
        raise ValueError("%s must contain at least one sample" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains NaN or infinite values" % name)
    return arr


def check_X_y(X, y):
    """Validate a feature matrix and a target vector together."""
    X = check_array(X)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "X and y have inconsistent lengths: %d and %d" % (X.shape[0], y.shape[0])
        )
    if y.shape[0] == 0:
        raise ValueError("y must contain at least one sample")
    return X, y


def check_consistent_n_features(estimator, X):
    """Ensure X has the same number of features seen during fit."""
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            "X has %d features, but this estimator was fitted with %d features"
            % (X.shape[1], expected)
        )
