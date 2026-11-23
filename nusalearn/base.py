"""Base estimator class shared by all NusaLearn estimators.

The interface deliberately mirrors the conventions popularised by
scikit-learn (fit returns self, fitted attributes end with an
underscore), but the implementation here is original.
"""

import inspect

import numpy as np


class BaseEstimator:
    """Minimal base class providing parameter introspection."""

    def get_params(self):
        """Return the constructor parameters as a dictionary."""
        signature = inspect.signature(self.__init__)
        params = {}
        for name in signature.parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        """Set constructor parameters and return self."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    "Invalid parameter %r for estimator %s"
                    % (key, type(self).__name__)
                )
            setattr(self, key, value)
        return self

    def _check_is_fitted(self, attributes):
        if isinstance(attributes, str):
            attributes = [attributes]
        if not any(hasattr(self, attr) for attr in attributes):
            raise ValueError(
                "This %s instance is not fitted yet. Call fit first."
                % type(self).__name__
            )


class ClassifierMixin:
    """Adds a default score method based on accuracy."""

    def score(self, X, y):
        from .metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


class RegressorMixin:
    """Adds a default score method based on R squared."""

    def score(self, X, y):
        from .metrics import r2_score

        return r2_score(y, self.predict(X))
