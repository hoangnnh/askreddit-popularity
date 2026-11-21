"""Unweighted eight-class and cumulative-threshold metrics from 8x8 counts."""

import csv
from pathlib import Path

from pyspark.sql import functions as F

LABELS = tuple(range(8))


def metrics_from_confusion(matrix: list[list[int]]) -> dict:
    if len(matrix) != 8 or any(len(row) != 8 for row in matrix) or any(
        not isinstance(v, int) or v < 0 for row in matrix for v in row
    ):
        raise ValueError("Confusion matrix must contain 8x8 nonnegative integer counts")
    support = [sum(row) for row in matrix]
    predicted = [sum(matrix[i][j] for i in LABELS) for j in LABELS]
    total = sum(support)
    per_class = []
    for k in LABELS:
        tp = matrix[k][k]
        precision = tp / predicted[k] if predicted[k] else 0.0
        recall = tp / support[k] if support[k] else 0.0
        f1 = 2 * tp / (support[k] + predicted[k]) if support[k] + predicted[k] else 0.0
        per_class.append({"label": k, "precision": precision, "recall": recall, "f1": f1, "support": support[k]})
    threshold_metrics = []
    for j in range(1, 8):
        tp = sum(matrix[a][b] for a in range(j, 8) for b in range(j, 8))
        fp = sum(matrix[a][b] for a in range(j) for b in range(j, 8))
        fn = sum(matrix[a][b] for a in range(j, 8) for b in range(j))
        threshold_metrics.append({"threshold": j, "tp": tp, "fp": fp, "fn": fn,
                                  "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
    return {"records": total, "label_universe": list(LABELS), "confusion_matrix": matrix,
            "row_normalized_confusion_matrix": [[v / support[i] if support[i] else 0.0 for v in row] for i, row in enumerate(matrix)],
            "per_class": per_class, "macro_f1": sum(v["f1"] for v in per_class) / 8,
            "balanced_accuracy": sum(v["recall"] for v in per_class) / 8,
            "accuracy": sum(matrix[k][k] for k in LABELS) / total if total else 0.0,
            "weighted_f1": sum(v["f1"] * v["support"] for v in per_class) / total if total else 0.0,
            "threshold_metrics": threshold_metrics,
            "mean_threshold_f1": sum(v["f1"] for v in threshold_metrics) / 7,
            "zero_division": "Undefined precision/recall/F1 and absent-class contributions are zero; always average over 8 classes or 7 thresholds.",
            "weighting": "Natural evaluation frequencies; no training class weights applied."}


def evaluate_predictions(predictions) -> dict:
    counts = predictions.groupBy("label", "prediction").count().collect()
    matrix = [[0] * 8 for _ in LABELS]
    for row in counts:
        if row.label is None or row.prediction is None or row.label not in LABELS or row.prediction not in LABELS:
            raise ValueError("Labels and predictions must be integers in 0–7")
        matrix[int(row.label)][int(row.prediction)] += row["count"]
    return metrics_from_confusion(matrix)


def confusion_figure(metrics: dict, destination: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, key, subtitle in zip(axes, ("confusion_matrix", "row_normalized_confusion_matrix"), ("Counts", "Row normalized")):
        matrix = metrics[key]
        im = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if subtitle == "Row normalized" else None)
        for i in LABELS:
            for j in LABELS:
                value = matrix[i][j]
                axis.text(j, i, f"{value:.2f}" if subtitle == "Row normalized" else str(value),
                          ha="center", va="center", fontsize=7,
                          color="white" if value > im.norm.vmax * 0.55 else "black")
        axis.set(xticks=LABELS, yticks=LABELS, xlabel="Predicted label", ylabel="True label", title=subtitle)
        fig.colorbar(im, ax=axis, shrink=0.8)
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def comparison_csv(rows: list[dict], path: Path) -> None:
    fields = ["candidate", "split", "selected", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "mean_threshold_f1", "records", "training_seconds", "prediction_seconds", "feature_dimension"]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
