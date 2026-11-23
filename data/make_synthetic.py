"""Generate the SYNTHETIC datasets bundled with NusaLearn.

Both files are artificial and are produced with a fixed seed (42) so
that every run creates byte-identical output:

- synthetic_regression.csv: y = 3*x0 - 2*x1 + 0.5*x2 + 5 + noise
- synthetic_blobs.csv: three Gaussian blobs in 2D for clustering

Run from anywhere:  python3 data/make_synthetic.py
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # Regression: known ground truth coefficients.
    n = 300
    X = rng.normal(size=(n, 3))
    true_coef = np.array([3.0, -2.0, 0.5])
    y = X @ true_coef + 5.0 + rng.normal(scale=0.5, size=n)
    with open(os.path.join(HERE, "synthetic_regression.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "x1", "x2", "target"])
        for row, target in zip(X, y):
            writer.writerow([f"{v:.6f}" for v in row] + [f"{target:.6f}"])

    # Clustering: three Gaussian blobs.
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
    parts = []
    labels = []
    for label, center in enumerate(centers):
        parts.append(center + rng.normal(scale=0.8, size=(100, 2)))
        labels.extend([label] * 100)
    blobs = np.vstack(parts)
    labels = np.array(labels)
    order = rng.permutation(len(blobs))
    blobs, labels = blobs[order], labels[order]
    with open(os.path.join(HERE, "synthetic_blobs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "x1", "label"])
        for row, label in zip(blobs, labels):
            writer.writerow([f"{v:.6f}" for v in row] + [int(label)])

    print("wrote synthetic_regression.csv (300 rows, SYNTHETIC, seed 42)")
    print("wrote synthetic_blobs.csv (300 rows, SYNTHETIC, seed 42)")


if __name__ == "__main__":
    main()
