"""KMeans clustering with k-means++ initialisation."""

import numpy as np

from ._pairwise import pairwise_distances
from .base import BaseEstimator
from .validation import check_array, check_consistent_n_features


class KMeans(BaseEstimator):
    """Partition samples into k clusters by Lloyd's algorithm.

    Centroids are seeded with k-means++ unless init is "random" or an
    array of starting centroids is passed. The objective reported in
    inertia_ is the sum of squared distances to the nearest centroid.
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, X, rng):
        n = X.shape[0]
        if isinstance(self.init, np.ndarray):
            centroids = np.array(self.init, dtype=float)
            if centroids.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("init array has the wrong shape")
            return centroids
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()
        if self.init != "k-means++":
            raise ValueError("init must be 'k-means++', 'random' or an array")
        centroids = np.empty((self.n_clusters, X.shape[1]))
        centroids[0] = X[rng.integers(n)]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
        for j in range(1, self.n_clusters):
            total = np.sum(closest_sq)
            if total == 0.0:
                centroids[j] = X[rng.integers(n)]
            else:
                probs = closest_sq / total
                centroids[j] = X[rng.choice(n, p=probs)]
            dist_sq = np.sum((X - centroids[j]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, dist_sq)
        return centroids

    def _single_run(self, X, rng):
        centroids = self._init_centroids(X, rng)
        previous_inertia = np.inf
        labels = np.zeros(X.shape[0], dtype=int)
        for iteration in range(1, self.max_iter + 1):
            distances = pairwise_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(X.shape[0]), labels] ** 2))
            new_centroids = centroids.copy()
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members) > 0:
                    new_centroids[j] = np.mean(members, axis=0)
                else:
                    # Reinitalise an empty cluster at the worst served point.
                    worst = int(np.argmax(distances[np.arange(X.shape[0]), labels]))
                    new_centroids[j] = X[worst]
            shift = float(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            if previous_inertia - inertia < self.tol and shift < self.tol:
                previous_inertia = inertia
                break
            previous_inertia = inertia
        distances = pairwise_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(X.shape[0]), labels] ** 2))
        return centroids, labels, inertia, iteration

    def fit(self, X, y=None):
        X = check_array(X)
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        n_runs = 1 if isinstance(self.init, np.ndarray) else self.n_init
        best = None
        for _ in range(n_runs):
            result = self._single_run(X, rng)
            if best is None or result[2] < best[2]:
                best = result
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        self._check_is_fitted("cluster_centers_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        distances = pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_
