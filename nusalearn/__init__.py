"""NusaLearn: a small machine learning library written from scratch.

Every estimator here is an original implementation built on NumPy.
The API follows the fit / predict / score conventions popularised by
scikit-learn, but no scikit-learn code is used at runtime and the
project is not affiliated with scikit-learn.
"""

from . import cluster, linear_model, metrics, model_selection, neighbors
from . import preprocessing, tree
from ._pairwise import KERNEL_AVAILABLE

__version__ = "0.1.0"

__all__ = [
    "cluster",
    "linear_model",
    "metrics",
    "model_selection",
    "neighbors",
    "preprocessing",
    "tree",
    "KERNEL_AVAILABLE",
    "__version__",
]
