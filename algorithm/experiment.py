"""Evaluation helpers for clustering experiments."""

import itertools

import numpy as np


def _comb2(values):
    """Compute n choose 2 for each value and return the sum."""
    values = np.asarray(values, dtype=float)
    return np.sum(values * (values - 1) / 2)


def purity(y_true, y_pred):
    """Return clustering purity."""
    clusters = np.unique(y_pred)
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    majority_counts = []

    for cluster in clusters:
        idx = np.where(y_pred == cluster)[0]
        majority_counts.append(np.bincount(y_true[idx, :].reshape(-1)).max())

    return np.sum(majority_counts) / y_true.shape[0]


def ARI(y_true, y_pred):
    """Return the adjusted Rand index."""
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    contingency = np.zeros((len(true_labels), len(pred_labels)), dtype=int)

    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            contingency[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    sum_comb = _comb2(contingency)
    true_comb = _comb2(np.sum(contingency, axis=1))
    pred_comb = _comb2(np.sum(contingency, axis=0))
    total_comb = _comb2([len(y_true)])

    expected = true_comb * pred_comb / total_comb if total_comb != 0 else 0
    maximum = (true_comb + pred_comb) / 2
    denominator = maximum - expected
    if denominator == 0:
        return 1.0
    return (sum_comb - expected) / denominator


def validation(clients):
    """Evaluate predictions stored in client dictionaries with Purity and ARI."""
    y_true = list(itertools.chain.from_iterable(client["label"] for client in clients))
    y_pred = list(itertools.chain.from_iterable(client["predict"] for client in clients))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "Purity": purity(y_true, y_pred),
        "ARI": ARI(y_true, y_pred),
    }
