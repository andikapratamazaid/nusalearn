"""Classify the Iris dataset with k-nearest neighbors and a decision tree.

Run from the repository root:

    python examples/iris_classification.py

Data: data/iris.csv, from the UCI Machine Learning Repository
(Fisher, 1936). See data/SOURCES.md.
"""

from pathlib import Path

import numpy as np

from nusalearn.metrics import accuracy_score, confusion_matrix
from nusalearn.model_selection import train_test_split
from nusalearn.neighbors import KNeighborsClassifier
from nusalearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parent.parent / "data" / "iris.csv"


def load_iris():
    raw = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding=None)
    X = np.column_stack([raw[name] for name in raw.dtype.names[:-1]]).astype(float)
    y = np.asarray(raw[raw.dtype.names[-1]])
    return X, y


def main():
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=True
    )
    print(f"Iris: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    print(f"Split: {len(y_train)} train, {len(y_test)} test (stratified, seed 42)")

    models = [
        ("KNeighborsClassifier(k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    ]
    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"\n{name}")
        print(f"  test accuracy: {acc:.4f}")
        print(f"  confusion matrix (rows = true, cols = predicted):")
        for row in confusion_matrix(y_test, pred):
            print(f"    {row.tolist()}")


if __name__ == "__main__":
    main()
