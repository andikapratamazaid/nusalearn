"""Feature scaling transformers."""

import numpy as np

from .base import BaseEstimator
from .validation import check_array, check_consistent_n_features


class StandardScaler(BaseEstimator):
    """Standardise features to zero mean and unit variance.

    Constant features get a scale of 1.0 so they map to zero instead of
    causing a division by zero.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = np.std(X, axis=0)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X):
        self._check_is_fitted("mean_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted("mean_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """Scale features into a fixed range, by default zero to one."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        span = self.data_max_ - self.data_min_
        span[span == 0.0] = 1.0
        self.data_range_ = span
        low, high = self.feature_range
        self.scale_ = (high - low) / self.data_range_
        self.min_ = low - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        self._check_is_fitted("scale_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return X * self.scale_ + self.min_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted("scale_")
        X = check_array(X)
        check_consistent_n_features(self, X)
        return (X - self.min_) / self.scale_
