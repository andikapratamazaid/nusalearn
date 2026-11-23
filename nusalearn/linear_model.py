"""Linear models: ordinary least squares and logistic regression."""

import numpy as np

from . import _native
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .validation import check_array, check_X_y, check_consistent_n_features


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class LinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary least squares regression.

    With solver="normal", the coefficients solve the normal equations
    (X^T X) w = X^T y via a least squares solve. The X^T X product uses
    the compiled C++ matmul when it is available. With solver="gd",
    batch gradient descent is used instead.
    """

    def __init__(self, fit_intercept=True, solver="normal", learning_rate=0.01,
                 max_iter=5000, tol=1e-10):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = y.astype(float)
        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            self._x_mean = np.mean(X, axis=0)
            self._y_mean = float(np.mean(y))
            X_work = X - self._x_mean
            y_work = y - self._y_mean
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._y_mean = 0.0
            X_work = X
            y_work = y
        if self.solver == "normal":
            xtx = _native.matmul(X_work.T, X_work)
            xty = X_work.T @ y_work
            coef, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
            self.coef_ = coef
            self.n_iter_ = 1
        elif self.solver == "gd":
            coef = np.zeros(X.shape[1])
            previous = np.inf
            n = X.shape[0]
            for iteration in range(1, self.max_iter + 1):
                residual = X_work @ coef - y_work
                loss = float(residual @ residual) / n
                gradient = (2.0 / n) * (X_work.T @ residual)
                coef = coef - self.learning_rate * gradient
                if abs(previous - loss) < self.tol:
                    previous = loss
                    break
                previous = loss
            self.coef_ = coef
            self.n_iter_ = iteration
        else:
            raise ValueError("solver must be 'normal' or 'gd'")
        if self.fit_intercept:
            self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        self._check_is_fitted("coef_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        if _native.DOT_AVAILABLE:
            out = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                out[i] = _native.dot(X[i], self.coef_) + self.intercept_
            return out
        return X @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression trained by gradient descent.

    Features are used as given; scaling them first (for example with
    StandardScaler) usually helps convergence a lot. Probabilities come
    from a softmax over one weight vector per class, which reduces to
    the binary sigmoid case when there are two classes.
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-8, l2=0.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        class_index = np.searchsorted(self.classes_, y)
        Y = np.zeros((n, k))
        Y[np.arange(n), class_index] = 1.0
        W = np.zeros((d, k))
        b = np.zeros(k)
        previous = np.inf
        for iteration in range(1, self.max_iter + 1):
            logits = X @ W + b
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            loss = -float(np.mean(np.log(np.clip(probs[np.arange(n), class_index], 1e-15, 1.0))))
            loss += 0.5 * self.l2 * float(np.sum(W * W))
            error = (probs - Y) / n
            grad_W = X.T @ error + self.l2 * W
            grad_b = np.sum(error, axis=0)
            W = W - self.learning_rate * grad_W
            b = b - self.learning_rate * grad_b
            if abs(previous - loss) < self.tol:
                previous = loss
                break
            previous = loss
        self.coef_ = W.T
        self.intercept_ = b
        self.n_iter_ = iteration
        return self

    def predict_proba(self, X):
        self._check_is_fitted("coef_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        logits = X @ self.coef_.T + self.intercept_
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
