"""Evaluation metrics for classification and regression."""

import numpy as np


def _check_targets(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred have inconsistent lengths")
    if y_true.shape[0] == 0:
        raise ValueError("metrics are undefined for empty inputs")
    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Fraction of predictions that exactly match the true labels."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix with rows as true labels and columns as predicted."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix


def _tp_fp_fn(y_true, y_pred, positive):
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    return tp, fp, fn


def precision_score(y_true, y_pred, average="binary", zero_division=0.0):
    """Precision, either for the positive class or macro averaged."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    labels = np.unique(y_true)
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary precision requires exactly two classes")
        positive = labels[-1]
        tp, fp, _ = _tp_fp_fn(y_true, y_pred, positive)
        return tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
    if average == "macro":
        values = []
        for label in labels:
            tp, fp, _ = _tp_fp_fn(y_true, y_pred, label)
            values.append(tp / (tp + fp) if (tp + fp) > 0 else float(zero_division))
        return float(np.mean(values))
    raise ValueError("average must be 'binary' or 'macro'")


def recall_score(y_true, y_pred, average="binary", zero_division=0.0):
    """Recall, either for the positive class or macro averaged."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    labels = np.unique(y_true)
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary recall requires exactly two classes")
        positive = labels[-1]
        tp, _, fn = _tp_fp_fn(y_true, y_pred, positive)
        return tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
    if average == "macro":
        values = []
        for label in labels:
            tp, _, fn = _tp_fp_fn(y_true, y_pred, label)
            values.append(tp / (tp + fn) if (tp + fn) > 0 else float(zero_division))
        return float(np.mean(values))
    raise ValueError("average must be 'binary' or 'macro'")


def f1_score(y_true, y_pred, average="binary", zero_division=0.0):
    """F1 score, either for the positive class or macro averaged."""
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    if precision + recall == 0:
        return float(zero_division)
    return 2.0 * precision * recall / (precision + recall)


def roc_auc_score(y_true, y_score):
    """Area under the ROC curve for binary labels, via the rank statistic.

    Tied scores contribute one half, matching the Mann-Whitney U
    definition of AUC.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score have inconsistent lengths")
    labels = np.unique(y_true)
    if len(labels) != 2:
        raise ValueError("roc_auc_score requires exactly two classes")
    positive = labels[-1]
    is_pos = y_true == positive
    n_pos = int(np.sum(is_pos))
    n_neg = int(np.sum(~is_pos))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires samples of both classes")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[is_pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def mean_squared_error(y_true, y_pred):
    """Mean of squared differences between true and predicted values."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def r2_score(y_true, y_pred):
    """Coefficient of determination R squared."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot
