"""Shared fixtures: locate the repository data folder robustly.

The suite runs either from the repository root or from a copied
folder (see test.sh), so the data directory is searched in a few
places: next to the repository checkout, relative to the installed
package (editable installs point back at the source tree), and via
the NUSALEARN_DATA environment variable.
"""

import os
from pathlib import Path

import numpy as np
import pytest


def _find_data_dir():
    candidates = []
    env = os.environ.get("NUSALEARN_DATA")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "data")
    try:
        import nusalearn

        candidates.append(Path(nusalearn.__file__).resolve().parent.parent / "data")
    except Exception:
        pass
    candidates.append(Path.cwd() / "data")
    for cand in candidates:
        if (cand / "iris.csv").exists():
            return cand
    raise RuntimeError("could not locate the data folder with iris.csv")


DATA_DIR = _find_data_dir()


def _load_csv(name):
    return np.genfromtxt(DATA_DIR / name, delimiter=",", names=True, dtype=None, encoding=None)


@pytest.fixture(scope="session")
def iris():
    raw = _load_csv("iris.csv")
    X = np.column_stack([raw[name] for name in raw.dtype.names[:-1]]).astype(float)
    y = np.asarray(raw[raw.dtype.names[-1]])
    return X, y


@pytest.fixture(scope="session")
def wine():
    raw = _load_csv("wine.csv")
    names = raw.dtype.names
    y = np.asarray(raw[names[0]]).astype(int)
    X = np.column_stack([raw[name] for name in names[1:]]).astype(float)
    return X, y


@pytest.fixture(scope="session")
def synthetic_regression():
    raw = _load_csv("synthetic_regression.csv")
    X = np.column_stack([raw[name] for name in ("x0", "x1", "x2")]).astype(float)
    y = np.asarray(raw["target"]).astype(float)
    return X, y


@pytest.fixture(scope="session")
def synthetic_blobs():
    raw = _load_csv("synthetic_blobs.csv")
    X = np.column_stack([raw[name] for name in ("x0", "x1")]).astype(float)
    y = np.asarray(raw["label"]).astype(int)
    return X, y
