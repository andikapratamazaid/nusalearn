"""Linear and logistic regression: exact recovery and dataset checks."""

import numpy as np
import pytest

from nusalearn.linear_model import LinearRegression, LogisticRegression
from nusalearn.metrics import accuracy_score, r2_score
from nusalearn.model_selection import train_test_split
from nusalearn.preprocessing import StandardScaler


def test_linear_regression_recovers_exact_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * X[:, 0] + 2.0
    reg = LinearRegression().fit(X, y)
    assert reg.coef_[0] == pytest.approx(3.0, abs=1e-8)
    assert reg.intercept_ == pytest.approx(2.0, abs=1e-8)
    assert r2_score(y, reg.predict(X)) == pytest.approx(1.0)


def test_linear_regression_gradient_descent_solver():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([1.5, -0.5]) + 0.3
    reg = LinearRegression(solver="gd", learning_rate=0.1, max_iter=3000).fit(X, y)
    assert np.allclose(reg.coef_, [1.5, -0.5], atol=0.05)
    assert reg.intercept_ == pytest.approx(0.3, abs=0.05)


def test_linear_regression_on_synthetic_data(synthetic_regression):
    X, y = synthetic_regression
    reg = LinearRegression().fit(X, y)
    # Ground truth used by the generator: 3, -2, 0.5 and intercept 5.
    assert np.allclose(reg.coef_, [3.0, -2.0, 0.5], atol=0.15)
    assert reg.intercept_ == pytest.approx(5.0, abs=0.15)
    assert r2_score(y, reg.predict(X)) > 0.95


def test_linear_regression_no_intercept():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    assert reg.intercept_ == 0.0
    assert reg.coef_[0] == pytest.approx(2.0, abs=1e-8)


def test_logistic_regression_separable():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(learning_rate=0.5, max_iter=2000).fit(X, y)
    assert accuracy_score(y, clf.predict(X)) == 1.0
    proba = clf.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # The far positive point should be confidently class 1.
    assert proba[3, 1] > 0.9


def test_logistic_regression_on_iris(iris):
    X, y = iris
    Xs = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=True)
    clf = LogisticRegression(learning_rate=0.1, max_iter=2000).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) >= 0.90


def test_logistic_regression_on_wine_scaled(wine):
    X, y = wine
    Xs = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=True)
    clf = LogisticRegression(learning_rate=0.05, max_iter=3000).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) >= 0.88
