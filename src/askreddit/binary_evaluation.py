"""Exact binary metrics from collected (label, probability) arrays. Positive class = high popularity."""

import numpy as np


def _arrays(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Labels and scores must be nonempty 1-D arrays of equal length")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise ValueError("Labels must be 0/1 and scores finite")
    return y, p


def roc_auc(y, p) -> float:
    """Mann–Whitney AUC with average ranks for tied scores (exact, no binning)."""
    y, p = _arrays(y, p)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(p, kind="mergesort")
    _, first, counts = np.unique(p[order], return_index=True, return_counts=True)
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def threshold_curve(y, p):
    """Cumulative counts when predicting positive for score >= each distinct threshold (descending)."""
    y, p = _arrays(y, p)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    last = np.r_[np.nonzero(np.diff(ps))[0], ys.size - 1]
    tp = np.cumsum(ys)[last]
    fp = last + 1 - tp
    return ps[last], tp, fp, int(y.sum()), int(y.size - y.sum())


def average_precision(y, p) -> float:
    _, tp, fp, positives, _ = threshold_curve(y, p)
    if positives == 0:
        raise ValueError("Average precision needs positives")
    recall = tp / positives
    precision = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_f1_threshold(y, p) -> float:
    thresholds, tp, fp, positives, _ = threshold_curve(y, p)
    f1 = 2 * tp / (tp + fp + positives)
    return float(thresholds[int(np.argmax(f1))])


def metrics_at_threshold(y, p, threshold: float) -> dict:
    y, p = _arrays(y, p)
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / y.size, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "predicted_positive_rate": (tp + fp) / y.size,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_layout": "rows = true [low, high], columns = predicted [low, high]"}


def roc_points(y, p, max_points: int = 201) -> dict:
    _, tp, fp, positives, negatives = threshold_curve(y, p)
    fpr = np.r_[0.0, fp / negatives]
    tpr = np.r_[0.0, tp / positives]
    if fpr.size > max_points:
        keep = np.unique(np.r_[np.linspace(0, fpr.size - 1, max_points).round().astype(int), fpr.size - 1])
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": [round(float(v), 6) for v in fpr], "tpr": [round(float(v), 6) for v in tpr]}


def calibration_summary(y, p, offset: float) -> dict:
    """Mean probability and Brier score before/after adding ``offset`` to the logit (prior correction)."""
    y, p = _arrays(y, p)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    corrected = 1.0 / (1.0 + np.exp(-(np.log(clipped) - np.log1p(-clipped) + offset)))
    return {"prior_logit_offset": float(offset), "positive_rate": float(y.mean()),
            "mean_raw_probability": float(p.mean()), "mean_calibrated_probability": float(corrected.mean()),
            "brier_raw": float(np.mean((p - y) ** 2)), "brier_calibrated": float(np.mean((corrected - y) ** 2))}


def evaluate_scores(y, p, threshold: float, *, seen=None) -> dict:
    y, p = _arrays(y, p)
    positives = int(y.sum())
    rate = positives / y.size
    result = {
        "records": int(y.size), "positives": positives, "positive_rate": rate,
        "auc_roc": roc_auc(y, p), "pr_auc": average_precision(y, p),
        "pr_auc_random_baseline": rate,
        "all_high_f1_reference": 2 * rate / (1 + rate),
        **metrics_at_threshold(y, p, threshold),
        "at_probability_0_5": metrics_at_threshold(y, p, 0.5),
        "roc_curve": roc_points(y, p),
        "zero_division": "Undefined precision/recall/specificity are reported as 0.",
    }
    if seen is not None:
        unseen = ~np.asarray(seen, dtype=bool)
        if unseen.any() and 0 < y[unseen].sum() < unseen.sum():
            slice_metrics = metrics_at_threshold(y[unseen], p[unseen], threshold)
            result["unseen_train_titles"] = {"records": int(unseen.sum()), "auc_roc": roc_auc(y[unseen], p[unseen]),
                                             **{k: slice_metrics[k] for k in ("precision", "recall", "f1")}}
    return result
