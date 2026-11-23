"""Train/test splitting and cross-validation helpers."""

import copy

import numpy as np

from .metrics import accuracy_score, r2_score
from .validation import check_X_y


def train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None,
                     stratify=False):
    """Split arrays into random train and test subsets.

    Returns X_train, X_test, y_train, y_test. With stratify=True the
    class proportions of y are preserved in both subsets.
    """
    X_arr, y_arr = check_X_y(X, y)
    n = X_arr.shape[0]
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("float test_size must be between 0 and 1")
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)
    if n_test < 1 or n_test >= n:
        raise ValueError("test_size leaves an empty train or test subset")
    rng = np.random.default_rng(random_state)
    if stratify:
        test_idx = []
        train_idx = []
        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            if shuffle:
                rng.shuffle(cls_idx)
            n_cls_test = max(1, int(round(len(cls_idx) * n_test / n)))
            n_cls_test = min(n_cls_test, len(cls_idx) - 1) if len(cls_idx) > 1 else 0
            test_idx.extend(cls_idx[:n_cls_test])
            train_idx.extend(cls_idx[n_cls_test:])
        test_idx = np.array(sorted(test_idx))
        train_idx = np.array(sorted(train_idx))
    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        test_idx = np.sort(indices[:n_test])
        train_idx = np.sort(indices[n_test:])
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def kfold_indices(n_samples, n_splits=5, shuffle=True, random_state=None):
    """Yield (train_indices, test_indices) pairs for k-fold splitting."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed n_samples")
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, test_idx


def cross_val_score(estimator, X, y, cv=5, scoring=None, random_state=None):
    """Evaluate an estimator with k-fold cross-validation.

    The estimator is deep copied for every fold, so the object passed
    in is left unfitted. scoring may be None (the estimator's own score
    method), "accuracy" or "r2".
    """
    X_arr, y_arr = check_X_y(X, y)
    scores = []
    for train_idx, test_idx in kfold_indices(
        X_arr.shape[0], n_splits=cv, shuffle=True, random_state=random_state
    ):
        fold_estimator = copy.deepcopy(estimator)
        fold_estimator.fit(X_arr[train_idx], y_arr[train_idx])
        if scoring is None:
            value = fold_estimator.score(X_arr[test_idx], y_arr[test_idx])
        elif scoring == "accuracy":
            value = accuracy_score(y_arr[test_idx], fold_estimator.predict(X_arr[test_idx]))
        elif scoring == "r2":
            value = r2_score(y_arr[test_idx], fold_estimator.predict(X_arr[test_idx]))
        else:
            raise ValueError("scoring must be None, 'accuracy' or 'r2'")
        scores.append(float(value))
    return np.array(scores)
