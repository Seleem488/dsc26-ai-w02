"""Decision metrics for a mule-risk score."""
from __future__ import annotations

import math
import numpy as np


def _validate_inputs(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
    if len(y) == 0 or len(s) == 0:
        raise ValueError("Inputs must not be empty")
    if len(y) != len(s):
        raise ValueError("Input lengths must match")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be 0 or 1")
    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")

    return y.astype(int), s


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return true/false positive and negative counts."""
    y, s = _validate_inputs(y_true, scores)
    flagged = s >= threshold

    return {
        "tp": int(np.sum(flagged & (y == 1))),
        "fp": int(np.sum(flagged & (y == 0))),
        "tn": int(np.sum(~flagged & (y == 0))),
        "fn": int(np.sum(~flagged & (y == 1))),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at the threshold."""
    c = confusion_at(y_true, scores, threshold)
    predicted = c["tp"] + c["fp"]
    positives = c["tp"] + c["fn"]

    precision = c["tp"] / predicted if predicted else 0.0
    recall = c["tp"] / positives if positives else 0.0

    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores; ties keep original order."""
    y, s = _validate_inputs(y_true, scores)

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(y))
    order = np.argsort(-s, kind="stable")
    return float(np.mean(y[order[:k]] == 1))


def expected_cost(
    y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float
) -> float:
    """Return fp * cost_fp + fn * cost_fn."""
    c = confusion_at(y_true, scores, threshold)
    return float(c["fp"] * cost_fp + c["fn"] * cost_fn)


def best_threshold(
    y_true, scores, *, cost_fp: float, cost_fn: float
) -> float:
    """Minimize expected cost; ties favor the highest threshold."""
    y, s = _validate_inputs(y_true, scores)
    candidates = list(np.unique(s)) + [math.inf]

    best = math.inf
    best_cost = expected_cost(
        y, s, best, cost_fp=cost_fp, cost_fn=cost_fn
    )

    for threshold in candidates:
        cost = expected_cost(
            y, s, threshold, cost_fp=cost_fp, cost_fn=cost_fn
        )
        if cost < best_cost or (cost == best_cost and threshold > best):
            best = float(threshold)
            best_cost = cost

    return best


def threshold_for_precision(
    y_true, scores, min_precision: float
) -> float | None:
    """Return the lowest distinct score meeting min_precision, or None."""
    y, s = _validate_inputs(y_true, scores)

    for threshold in sorted(np.unique(s)):
        precision, _ = precision_recall_at(y, s, threshold)
        if precision >= min_precision:
            return float(threshold)

    return None
```
