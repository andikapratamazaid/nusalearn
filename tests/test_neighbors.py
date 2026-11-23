"""k-nearest neighbors: hand cases and dataset checks."""

import numpy as np
import pytest

from nusalearn.metrics import accuracy_score, r2_score
from nusalearn.model_selection import train_test_split
from nusalearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from nusalearn.preprocessing import StandardScaler


def test_knn_classifier_k1_memorises():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    y = np.array(["a", "a", "b", "b"])
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert clf.predict([[0.1, 0.2]])[0] == "a"
    assert clf.predict([[4.9, 5.2]])[0] == "b"
    assert accuracy_score(y, clf.predict(X)) == 1.0


def test_knn_classifier_majority_vote():
    X = np.array([[0.0], [0.2], [0.4], [10.0]])
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    # Neighbours of 0.3 are 0.2 (class 0), 0.4 (class 1), 0.0 (class 0).
    assert clf.predict([[0.3]])[0] == 0


def test_knn_regressor_k1_and_k2():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])
    reg1 = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    assert reg1.predict([[1.0]])[0] == pytest.approx(10.0)
    reg2 = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    # Neighbours of 0.9 are x=1 (10) and x=0 (0): mean 5.
    assert reg2.predict([[0.9]])[0] == pytest.approx(5.0)


def test_knn_on_iris(iris):
    X, y = iris
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=True)
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) >= 0.93


def test_knn_on_wine_scaled(wine):
    X, y = wine
    Xs = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=True)
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) >= 0.90


def test_knn_regressor_on_synthetic(synthetic_regression):
    X, y = synthetic_regression
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    reg = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
    assert r2_score(y_te, reg.predict(X_te)) > 0.85
