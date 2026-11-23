"""K-nearest neighbours classifier and regressor."""

import numpy as np

from ._pairwise import pairwise_distances
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .validation import check_array, check_X_y, check_consistent_n_features


class _KNeighborsBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                "n_neighbors=%d is larger than the %d training samples"
                % (self.n_neighbors, X.shape[0])
            )
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.n_features_in_ = X.shape[1]
        self._X_train = X
        self._y_train = y
        return self

    def _neighbors(self, X):
        X = check_array(X)
        check_consistent_n_features(self, X)
        distances = pairwise_distances(X, self._X_train)
        order = np.argsort(distances, axis=1, kind="stable")
        neighbor_idx = order[:, : self.n_neighbors]
        neighbor_dist = np.take_along_axis(distances, neighbor_idx, axis=1)
        return neighbor_idx, neighbor_dist

    def _weights(self, neighbor_dist):
        if self.weights == "uniform":
            return np.ones_like(neighbor_dist)
        # Exact matches get infinite weight so they dominate the vote,
        # matching the usual convention for distance weighting.
        with np.errstate(divide="ignore"):
            weights = 1.0 / neighbor_dist
        exact = neighbor_dist == 0.0
        if np.any(exact):
            weights = np.where(exact, 1.0, 0.0)
        return weights


class KNeighborsClassifier(_KNeighborsBase, ClassifierMixin):
    """Classify by majority vote of the k nearest training samples."""

    def fit(self, X, y):
        self._fit(X, y)
        self.classes_ = np.unique(self._y_train)
        return self

    def predict(self, X):
        self._check_is_fitted("_X_train")
        X = check_array(X)
        neighbor_idx, neighbor_dist = self._neighbors(X)
        weights = self._weights(neighbor_dist)
        neighbor_labels = self._y_train[neighbor_idx]
        out = np.empty(X.shape[0], dtype=self._y_train.dtype)
        for i in range(X.shape[0]):
            scores = {}
            for label, weight in zip(neighbor_labels[i], weights[i]):
                scores[label] = scores.get(label, 0.0) + weight
            # Ties resolve to the smallest class label, deterministically.
            out[i] = min(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return out

    def predict_proba(self, X):
        self._check_is_fitted("_X_train")
        X = check_array(X)
        neighbor_idx, neighbor_dist = self._neighbors(X)
        weights = self._weights(neighbor_dist)
        neighbor_labels = self._y_train[neighbor_idx]
        proba = np.zeros((X.shape[0], len(self.classes_)))
        class_pos = {c: j for j, c in enumerate(self.classes_)}
        for i in range(X.shape[0]):
            for label, weight in zip(neighbor_labels[i], weights[i]):
                proba[i, class_pos[label]] += weight
            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
        return proba


class KNeighborsRegressor(_KNeighborsBase, RegressorMixin):
    """Predict by the (optionally distance weighted) mean of neighbours."""

    def fit(self, X, y):
        self._fit(X, y)
        self._y_train = self._y_train.astype(float)
        return self

    def predict(self, X):
        self._check_is_fitted("_X_train")
        neighbor_idx, neighbor_dist = self._neighbors(X)
        weights = self._weights(neighbor_dist)
        neighbor_values = self._y_train[neighbor_idx]
        weighted = np.sum(neighbor_values * weights, axis=1)
        return weighted / np.sum(weights, axis=1)
