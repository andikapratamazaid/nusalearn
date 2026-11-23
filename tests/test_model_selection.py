"""Data splitting and cross-validation behaviour."""

import numpy as np
import pytest

from nusalearn.model_selection import cross_val_score, kfold_indices, train_test_split
from nusalearn.neighbors import KNeighborsClassifier


def test_train_test_split_sizes():
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.arange(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    assert len(X_tr) == 75 and len(X_te) == 25
    # No overlap and full coverage.
    assert sorted(np.concatenate([y_tr, y_te]).tolist()) == list(range(100))


def test_train_test_split_stratified():
    X = np.zeros((90, 2))
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    _, _, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=True)
    for cls in (0, 1, 2):
        assert int(np.sum(y_te == cls)) == 9
        assert int(np.sum(y_tr == cls)) == 21


def test_train_test_split_reproducible():
    X = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50)
    a = train_test_split(X, y, test_size=0.2, random_state=7)
    b = train_test_split(X, y, test_size=0.2, random_state=7)
    assert np.array_equal(a[3], b[3])


def test_kfold_indices_cover_all_samples():
    pairs = list(kfold_indices(23, n_splits=5, shuffle=True, random_state=42))
    assert len(pairs) == 5
    test_parts = []
    for train_idx, test_idx in pairs:
        # Train and test are disjoint and together cover everything.
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(train_idx) + len(test_idx) == 23
        test_parts.append(test_idx)
    all_test = np.concatenate(test_parts)
    assert sorted(all_test.tolist()) == list(range(23))
    sizes = sorted(len(t) for t in test_parts)
    assert sizes == [4, 4, 5, 5, 5]


def test_cross_val_score_knn_iris(iris):
    X, y = iris
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=5, random_state=42)
    assert len(scores) == 5
    assert float(np.mean(scores)) >= 0.93


def test_get_params_and_set_params():
    clf = KNeighborsClassifier()
    params = clf.get_params()
    assert params["n_neighbors"] == 5
    clf.set_params(n_neighbors=3)
    assert clf.n_neighbors == 3
    with pytest.raises(ValueError):
        clf.set_params(not_a_param=1)
