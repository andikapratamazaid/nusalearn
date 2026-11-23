"""KMeans on the synthetic blobs, where ground truth is known."""

import numpy as np

from nusalearn.cluster import KMeans


def _best_match_accuracy(y_true, labels):
    """Accuracy after the best permutation of cluster ids."""
    from itertools import permutations

    best = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[label] for label in labels])
        best = max(best, float(np.mean(mapped == y_true)))
    return best


def test_kmeans_recovers_blobs(synthetic_blobs):
    X, y = synthetic_blobs
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    assert km.cluster_centers_.shape == (3, 2)
    assert len(km.labels_) == len(y)
    assert np.isfinite(km.inertia_) and km.inertia_ > 0
    assert _best_match_accuracy(y, km.labels_) >= 0.90


def test_kmeans_fit_predict_consistent(synthetic_blobs):
    X, _ = synthetic_blobs
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)
    assert labels.tolist() == km.labels_.tolist()
    assert km.predict(X).tolist() == labels.tolist()


def test_kmeans_two_points_two_clusters():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(X)
    assert km.labels_[0] != km.labels_[1]
    assert km.inertia_ == 0.0
