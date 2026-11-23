"""Scalers on tiny arrays with known statistics."""

import numpy as np
import pytest

from nusalearn.preprocessing import MinMaxScaler, StandardScaler


def test_standard_scaler_known_values():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    # Column means are 2 and 3, population standard deviations are 1.
    assert scaler.mean_.tolist() == [2.0, 3.0]
    assert Xt.tolist() == [[-1.0, -1.0], [1.0, 1.0]]


def test_standard_scaler_inverse_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)) * 5 + 10
    scaler = StandardScaler()
    back = scaler.inverse_transform(scaler.fit_transform(X))
    assert np.allclose(back, X)


def test_standard_scaler_constant_column():
    X = np.array([[1.0, 5.0], [1.0, 7.0]])
    Xt = StandardScaler().fit_transform(X)
    # A constant column has zero variance and must not produce NaNs.
    assert np.isfinite(Xt).all()
    assert Xt[:, 0].tolist() == [0.0, 0.0]


def test_minmax_scaler_known_values():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = MinMaxScaler()
    Xt = scaler.fit_transform(X)
    assert Xt.min() == 0.0 and Xt.max() == 1.0
    assert Xt[1].tolist() == [0.5, 0.5]
    back = scaler.inverse_transform(Xt)
    assert np.allclose(back, X)


def test_scaler_unfitted_raises():
    with pytest.raises(Exception):
        StandardScaler().transform([[1.0, 2.0]])
