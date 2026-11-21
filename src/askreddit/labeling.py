"""Chronological splitting and exact upper-remainder median labels (Fang/Chua)."""

from fractions import Fraction

from pyspark.sql import DataFrame, Window, functions as F

from askreddit.clean import utc_seconds

VERSION = "fang-chua-upper-median-v1"
SPLITS = ("train", "validation", "test")
REFERENCES = ["https://aclanthology.org/W16-6209.pdf", "https://aclanthology.org/2021.acl-srw.10.pdf"]


def split_intervals(dates: dict) -> dict:
    intervals = {name: tuple(utc_seconds(d) for d in dates[name]) for name in SPLITS}
    if any(start >= end for start, end in intervals.values()):
        raise ValueError("Split intervals must have positive duration")
    if intervals["train"][1] != intervals["validation"][0] or intervals["validation"][1] != intervals["test"][0]:
        raise ValueError("Chronological split intervals must be adjacent and disjoint")
    return intervals


def assign_splits(frame: DataFrame, dates: dict) -> DataFrame:
    expression = F.lit(None).cast("string")
    for name, (start, end) in split_intervals(dates).items():
        expression = F.when((F.col("created_utc") >= start) & (F.col("created_utc") < end), name).otherwise(expression)
    return frame.withColumn("split", expression)


def exact_median(histogram: DataFrame) -> tuple[Fraction, int]:
    """Find both central records from exact distributed score-frequency aggregates.

    Only two central scores and a count reach Python. The ordered executor window
    operates on distinct scores, never individual post ranks or tie-breaking IDs.
    """
    total = histogram.agg(F.sum("count")).first()[0]
    if not total:
        raise ValueError("Cannot fit a median to an empty upper remainder")
    lower_rank, upper_rank = (total + 1) // 2, total // 2 + 1
    cumulative = histogram.withColumn("cumulative", F.sum("count").over(
        Window.orderBy("score").rowsBetween(Window.unboundedPreceding, Window.currentRow)))
    central = cumulative.agg(
        F.min(F.when(F.col("cumulative") >= lower_rank, F.col("score"))).alias("low"),
        F.min(F.when(F.col("cumulative") >= upper_rank, F.col("score"))).alias("high"),
    ).first()
    return Fraction(int(central.low) + int(central.high), 2), int(total)


def fit_thresholds(train: DataFrame) -> dict:
    if not {"score", "split"}.issubset(train.columns):
        raise ValueError("Threshold fitting requires score and an explicit train split")
    if train.filter(F.col("split").isNull() | (F.col("split") != "train") | F.col("score").isNull()).limit(1).count():
        raise ValueError("Thresholds must be fitted on valid TRAIN scores only")
    if dict(train.dtypes)["score"] not in {"bigint", "int"}:
        raise ValueError("Exact labeling requires integer scores")
    histogram = train.filter(F.col("score") > 1).groupBy("score").count().cache()
    boundaries, steps = [2], []
    try:
        remainder = histogram
        for label in range(1, 7):
            median, n = exact_median(remainder)
            threshold = -(-median.numerator // median.denominator)
            lower_count = remainder.filter(F.col("score") < threshold).agg(F.sum("count")).first()[0] or 0
            if threshold <= boundaries[-1] or lower_count == 0 or lower_count == n:
                raise ValueError(f"Degenerate median split at label {label}: median={median}, lower={lower_count}, total={n}")
            steps.append({"label": label, "remainder_records": n,
                          "median_numerator": median.numerator, "median_denominator": median.denominator,
                          "median_exact": str(median), "effective_upper_exclusive": threshold,
                          "assigned_records": int(lower_count), "remaining_records": n - int(lower_count)})
            boundaries.append(threshold)
            remainder = remainder.filter(F.col("score") >= threshold)
        return {
            "algorithm_version": VERSION, "fitted_on": "train", "references": REFERENCES,
            "effective_boundaries": boundaries, "steps": steps,
            "median_convention": "Exact record-frequency weighted ordinary median; average the two central integer values for even N, represented as a rational number.",
            "interval_convention": "Scores < median leave the upper remainder; integer cutoffs are ceil(median). Equal scores never split.",
            "mapping": [{"label": k, "lower_inclusive": None if k == 0 else boundaries[k - 1],
                         "upper_exclusive": boundaries[k] if k < 7 else None} for k in range(8)],
        }
    finally:
        histogram.unpersist()


def apply_labels(frame: DataFrame, model: dict) -> DataFrame:
    bounds = model["effective_boundaries"]
    if len(bounds) != 7 or bounds[0] != 2 or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Expected base cutoff 2 and six strictly increasing frozen cutoffs")
    expression = F.lit(7)
    for label in reversed(range(7)):
        expression = F.when(F.col("score") < F.lit(bounds[label]), label).otherwise(expression)
    return frame.withColumn("label", F.when(F.col("score").isNotNull(), expression).cast("int"))


def class_distributions(frame: DataFrame) -> dict:
    counts = {(r.split, r.label): r["count"] for r in frame.groupBy("split", "label").count().collect()}
    if any(label not in range(8) for _, label in counts):
        raise ValueError("Unexpected/null popularity label")
    result = {}
    for split in SPLITS:
        total = sum(counts.get((split, k), 0) for k in range(8))
        result[split] = {"total": total, "labels": [
            {"label": k, "count": counts.get((split, k), 0),
             "percentage": 100 * counts.get((split, k), 0) / total if total else None} for k in range(8)]}
    return result


def class_weights(counts: list[int], alpha: float) -> list[float]:
    if len(counts) != 8 or min(counts) <= 0 or alpha not in (0, 0.5, 1):
        raise ValueError("Weights require eight nonempty training classes and alpha 0, 0.5 or 1")
    n = sum(counts)
    raw = [(n / (8 * count)) ** alpha for count in counts]
    mean = sum(count * weight for count, weight in zip(counts, raw)) / n
    return [weight / mean for weight in raw]
