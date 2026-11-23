"""A CART decision tree classifier with Gini or entropy splits."""

import numpy as np

from .base import BaseEstimator, ClassifierMixin
from .validation import check_X_y, check_array, check_consistent_n_features


def _gini(counts, total):
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - float(np.sum(p * p))


def _entropy(counts, total):
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba", "prediction",
                 "n_samples", "impurity")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.proba = None
        self.prediction = None
        self.n_samples = 0
        self.impurity = 0.0

    @property
    def is_leaf(self):
        return self.left is None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Decision tree classifier grown greedily by impurity decrease.

    Candidate thresholds are the midpoints between consecutive sorted
    unique values of each feature. Splitting stops at max_depth, when a
    node holds fewer than min_samples_split samples, when a child would
    hold fewer than min_samples_leaf samples, or when the node is pure.
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._class_index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_index[v] for v in y])
        self._impurity_fn = _gini if self.criterion == "gini" else _entropy
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(X, y_idx, depth=0)
        self.depth_ = self._measure_depth(self.tree_)
        self.n_leaves_ = self._count_leaves(self.tree_)
        return self

    def _measure_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._measure_depth(node.left), self._measure_depth(node.right))

    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _candidate_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError("max_features must be None, an int, 'sqrt' or 'log2'")
        return self._rng.choice(n_features, size=k, replace=False)

    def _build(self, X, y_idx, depth):
        node = _Node()
        n = X.shape[0]
        node.n_samples = n
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        node.impurity = self._impurity_fn(counts, n)
        node.proba = counts / n
        node.prediction = self.classes_[int(np.argmax(counts))]
        if (
            node.impurity == 0.0
            or n < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        split = self._best_split(X, y_idx, counts, node.impurity)
        if split is None:
            return node
        feature, threshold, left_mask = split
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y_idx[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y_idx[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y_idx, parent_counts, parent_impurity):
        n, d = X.shape
        best = None
        best_gain = 1e-12
        for feature in self._candidate_features(d):
            column = X[:, feature]
            order = np.argsort(column, kind="stable")
            sorted_values = column[order]
            sorted_y = y_idx[order]
            left_counts = np.zeros(len(self.classes_))
            right_counts = parent_counts.copy()
            for i in range(n - 1):
                cls = sorted_y[i]
                left_counts[cls] += 1
                right_counts[cls] -= 1
                if sorted_values[i] == sorted_values[i + 1]:
                    continue
                n_left = i + 1
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                weighted = (
                    n_left * self._impurity_fn(left_counts, n_left)
                    + n_right * self._impurity_fn(right_counts, n_right)
                ) / n
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    threshold = 0.5 * (sorted_values[i] + sorted_values[i + 1])
                    best = (feature, threshold, column <= threshold)
        return best

    def _apply_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        self._check_is_fitted("tree_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return np.array([self._apply_one(row, self.tree_).prediction for row in X])

    def predict_proba(self, X):
        self._check_is_fitted("tree_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return np.array([self._apply_one(row, self.tree_).proba for row in X])
