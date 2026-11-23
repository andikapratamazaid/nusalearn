"""Cluster the synthetic blobs with KMeans.

Run from the repository root:

    python examples/kmeans_clustering.py

Data: data/synthetic_blobs.csv, SYNTHETIC data generated locally by
data/make_synthetic.py with seed 42 (three Gaussian blobs, known
labels). Because the generating process is known, the label match
below measures implementation correctness only. See data/SOURCES.md.
"""

from itertools import permutations
from pathlib import Path

import numpy as np

from nusalearn.cluster import KMeans

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic_blobs.csv"


def load_blobs():
    raw = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding=None)
    X = np.column_stack([raw["x0"], raw["x1"]]).astype(float)
    y = np.asarray(raw["label"]).astype(int)
    return X, y


def best_match_accuracy(y_true, labels, n_clusters):
    best = 0.0
    for perm in permutations(range(n_clusters)):
        mapped = np.array([perm[label] for label in labels])
        best = max(best, float(np.mean(mapped == y_true)))
    return best


def main():
    X, y = load_blobs()
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    print(f"Synthetic blobs: {X.shape[0]} samples, {X.shape[1]} features, "
          f"3 true clusters")
    print(f"KMeans(n_clusters=3) converged in {km.n_iter_} iterations")
    print(f"  inertia (sum of squared distances to centroids): {km.inertia_:.2f}")
    print(f"  cluster sizes: {np.bincount(km.labels_).tolist()}")
    print(f"  centroids:\n{np.round(km.cluster_centers_, 2)}")
    acc = best_match_accuracy(y, km.labels_, 3)
    print(f"  label match after best permutation: {acc:.4f}")


if __name__ == "__main__":
    main()
