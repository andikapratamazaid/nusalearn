"""Hand-checkable cases for every metric."""

import numpy as np
import pytest

from nusalearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def test_accuracy_perfect_and_half():
    y = [0, 1, 1, 0]
    assert accuracy_score(y, y) == 1.0
    assert accuracy_score(y, [0, 0, 1, 1]) == 0.5


def test_confusion_matrix_counts():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [1, 2]]


def test_precision_recall_f1_binary():
    # TP=2, FP=1, FN=1 for the positive class, so all three are 2/3.
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    assert precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_precision_recall_macro():
    # Per class precision: 1/2, 2/3, 1 -> macro 13/18.
    # Per class recall: 1/2, 1, 1/2 -> macro 2/3.
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert precision_score(y_true, y_pred, average="macro") == pytest.approx(13 / 18)
    assert recall_score(y_true, y_pred, average="macro") == pytest.approx(2 / 3)
    expected_f1 = 2 * (13 / 18) * (2 / 3) / (13 / 18 + 2 / 3)
    assert f1_score(y_true, y_pred, average="macro") == pytest.approx(expected_f1)


def test_precision_perfect_prediction_macro():
    y = [0, 1, 2, 0, 1, 2]
    assert precision_score(y, y, average="macro") == 1.0
    assert recall_score(y, y, average="macro") == 1.0
    assert f1_score(y, y, average="macro") == 1.0


def test_roc_auc_known_values():
    y = [0, 0, 1, 1]
    assert roc_auc_score(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc_score(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    # Classic textbook example: 3 of 4 positive/negative pairs ordered.
    assert roc_auc_score(y, [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    # All scores tied: AUC is one half.
    assert roc_auc_score(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_mean_squared_error():
    assert mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mean_squared_error([1.0, 2.0], [2.0, 4.0]) == pytest.approx(2.5)


def test_r2_score():
    y = [1.0, 2.0, 3.0, 4.0]
    assert r2_score(y, y) == pytest.approx(1.0)
    # Predicting the mean gives R2 of zero.
    assert r2_score(y, [2.5, 2.5, 2.5, 2.5]) == pytest.approx(0.0)


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError):
        accuracy_score([0, 1], [0])
