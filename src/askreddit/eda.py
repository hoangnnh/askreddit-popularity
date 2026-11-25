"""Reproducible full-year exploratory analysis of the cleaned AskReddit submissions.

Spark computes every aggregate from the cleaned Parquet dataset; only small
summary tables reach Python. Coverage, volume and score-distribution tables
describe the whole year. Popularity *associations* (title length, question
patterns, posting time, correlations) use the TRAIN interval only, so that the
validation and test months never inform modeling decisions. Label thresholds
are fitted on TRAIN with the same exact procedure as the experiment.

Outputs (JSON, PNG figures and a Vietnamese summary) are written once into a
fresh directory. ``--render-only`` rebuilds figures and the summary from the
saved JSON without starting Spark.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
from fractions import Fraction
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import DataFrame, functions as F

from askreddit.clean import CLEAN_SCHEMA, create_spark, file_fingerprint, guard_paths, write_json
from askreddit.features import STRUCTURED_COLUMNS, base_features
from askreddit.labeling import SPLITS, apply_labels, assign_splits, class_distributions, fit_thresholds

LOG = logging.getLogger(__name__)

QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)
SCORE_TAIL_CUTOFFS = (2, 10, 46, 100, 271, 1000, 10000, 100000)
CHAR_EDGES = (0, 20, 40, 60, 80, 100, 150, 200)
WORD_EDGES = (0, 4, 7, 10, 13, 16, 21, 31)
UPPERCASE_EDGES = (0.0, 0.05, 0.1, 0.2, 0.5)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# Matched against the NFKC-normalized, lowercased title used by the model features.
# "starts_*" patterns skip one leading bracketed tag such as "[serious] " or "(nsfw) ".
LEAD = r"^(?:[\[(][^\])]{0,30}[\])]\s*)?"
PATTERNS = {
    "starts_what": LEAD + r"what\b",
    "starts_why": LEAD + r"why\b",
    "starts_how": LEAD + r"how\b",
    "starts_who": LEAD + r"who\b",
    "starts_which": LEAD + r"which\b",
    "starts_when_where": LEAD + r"(when|where)\b",
    "starts_yes_no_verb": LEAD + r"(do|does|did|is|are|was|were|have|has|can|could|would|should|will)\b",
    "starts_if": LEAD + r"if\b",
    "contains_if_you_could": r"\bif you could\b",
    "contains_would_you_rather": r"\bwould you rather\b",
    "addresses_reader": r"\b(you|your|yours|yourself)\b",
    "first_person": r"\b(i|i'm|i’m|my|me)\b",
    "serious_tag": r"[\[(]\s*serious\s*[\])]",
    "nsfw_mention": r"\bnsfw\b",
    "mentions_reddit": r"\b(reddit|redditors?)\b",
}
OUTPUT_FILES = ("profile", "monthly_volume", "score_distribution", "score_frequencies", "title_characteristics",
                "posting_time", "associations", "label_feasibility")


def quantiles_from_frequencies(pairs, probabilities=QUANTILES) -> dict:
    """Exact inverse empirical CDF: the smallest value whose cumulative count reaches ceil(p*N)."""
    pairs = sorted((value, int(count)) for value, count in pairs)
    if any(count < 0 for _, count in pairs):
        raise ValueError("Frequencies must be nonnegative")
    total = sum(count for _, count in pairs)
    if total == 0:
        raise ValueError("Cannot compute quantiles of an empty distribution")
    result = {}
    for p in probabilities:
        exact = Fraction(str(p))
        if not 0 < exact <= 1:
            raise ValueError(f"Quantile probability must be in (0, 1]: {p}")
        rank, cumulative = max(1, math.ceil(exact * total)), 0
        for value, count in pairs:
            cumulative += count
            if cumulative >= rank:
                result[str(p)] = value
                break
    return result


def bucket_labels(edges, integer=True) -> list[str]:
    labels = []
    for i, low in enumerate(edges):
        if i == len(edges) - 1:
            labels.append(f"≥{low}" if integer else f"≥{low:.2f}")
        elif integer:
            high = edges[i + 1] - 1
            labels.append(str(low) if high == low else f"{low}–{high}")
        else:
            labels.append(f"{low:.2f}–{edges[i + 1]:.2f}")
    return labels


def bucket_column(column: str, edges):
    """Index of the last edge <= value; values below the first edge stay null."""
    if list(edges) != sorted(set(edges)):
        raise ValueError("Bucket edges must be strictly increasing")
    expression = F.lit(None).cast("int")
    for index, edge in enumerate(edges):
        expression = F.when(F.col(column) >= F.lit(edge), F.lit(index)).otherwise(expression)
    return expression


def analysis_frame(raw: DataFrame, config: dict) -> DataFrame:
    """Analysis columns that reuse the exact model feature definitions for title statistics."""
    features = base_features(raw, config["features"]).select("id", "normalized_title", *STRUCTURED_COLUMNS[:5])
    timestamp = F.timestamp_seconds("created_utc")
    frame = raw.select("id", "created_utc", "score", "selftext", "year", "month").join(features, "id")
    frame = assign_splits(frame, config["split_dates"]).withColumns({
        "hour": F.hour(timestamp),
        "weekday": F.pmod(F.dayofweek(timestamp) + 5, F.lit(7)),  # Monday=0, independent of locale
        "date": F.date_format(timestamp, "yyyy-MM-dd"),
        "title_only": F.trim(F.coalesce(F.col("selftext"), F.lit(""))) == "",
        "char_bucket": bucket_column("title_characters", CHAR_EDGES),
        "word_bucket": bucket_column("title_words", WORD_EDGES),
        "uppercase_bucket": bucket_column("uppercase_ratio", UPPERCASE_EDGES),
        "question_bucket": F.least(F.col("question_count"), F.lit(2.0)).cast("int"),
        **{name: F.col("normalized_title").rlike(pattern) for name, pattern in PATTERNS.items()},
    })
    return frame.drop("selftext")


def popularity_aggregates():
    return [F.count("*").alias("records"),
            F.avg(F.col("popular").cast("double")).alias("share_score_ge_2"),
            F.avg(F.col("high").cast("double")).alias("share_label_ge_6"),
            F.avg(F.col("label").cast("double")).alias("mean_label"),
            F.median("score").alias("median_score"),
            F.avg(F.col("score").cast("double")).alias("mean_score")]


def grouped(frame: DataFrame, *columns) -> list[dict]:
    return [row.asDict() for row in frame.groupBy(*columns).agg(*popularity_aggregates()).orderBy(*columns).collect()]


def frequencies(frame: DataFrame, column: str) -> list[list]:
    return [[row[column], row["count"]] for row in frame.groupBy(column).count().orderBy(column).collect()]


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def profile_section(raw: DataFrame, frame: DataFrame) -> dict:
    columns = [name for name, _ in CLEAN_SCHEMA]
    stats = raw.agg(
        F.count("*").alias("records"), F.countDistinct("id").alias("distinct_ids"),
        F.min("created_utc").alias("min_created_utc"), F.max("created_utc").alias("max_created_utc"),
        F.min("score").alias("min_score"), F.max("score").alias("max_score"),
        F.count(F.when(F.trim("title") == "", 1)).alias("blank_titles"),
        F.count(F.when(F.trim(F.coalesce("selftext", F.lit(""))) == "", 1)).alias("title_only_posts"),
        *[F.count(F.when(F.col(c).isNull(), 1)).alias("null_" + c) for c in columns],
    ).first().asDict()
    titles = frame.groupBy("normalized_title").count()
    repeated = titles.agg(
        F.count("*").alias("distinct_normalized_titles"),
        F.count(F.when(F.col("count") > 1, 1)).alias("repeated_distinct_titles"),
        F.coalesce(F.sum(F.when(F.col("count") > 1, F.col("count"))), F.lit(0)).alias("records_with_repeated_title"),
        F.max("count").alias("max_repeats_of_one_title"),
    ).first().asDict()
    top = [{"normalized_title": r.normalized_title, "records": r["count"]}
           for r in titles.orderBy(F.desc("count"), "normalized_title").limit(10).collect()]
    records = stats["records"]
    return {
        "schema": [{"column": name, "type": kind} for name, kind in raw.dtypes],
        "records": records, "distinct_ids": stats["distinct_ids"], "ids_unique": records == stats["distinct_ids"],
        "coverage_utc": [iso(stats["min_created_utc"]), iso(stats["max_created_utc"])],
        "score_range": [stats["min_score"], stats["max_score"]],
        "null_counts": {c: stats["null_" + c] for c in columns},
        "blank_titles": stats["blank_titles"],
        "title_only_posts": stats["title_only_posts"], "title_only_share": stats["title_only_posts"] / records,
        "normalized_titles": {**repeated, "share_records_with_repeated_title": repeated["records_with_repeated_title"] / records,
                              "top_repeated": top},
    }


def monthly_section(frame: DataFrame) -> dict:
    months = []
    for row in frame.groupBy("year", "month").agg(*popularity_aggregates(),
            F.avg(F.col("title_only").cast("double")).alias("share_title_only"),
            F.avg((F.col("score") <= 1).cast("double")).alias("share_score_le_1"),
            F.countDistinct("date").alias("days")).orderBy("year", "month").collect():
        months.append({**row.asDict(), "posts_per_day": row.records / row.days})
    daily = [[r.date, r["count"]] for r in frame.groupBy("date").count().orderBy("date").collect()]
    counts = sorted(c for _, c in daily)
    return {"months": months, "daily": daily,
            "daily_summary": {"days": len(counts), "min": counts[0], "max": counts[-1],
                              "median": quantiles_from_frequencies([(c, 1) for c in counts], (0.5,))["0.5"],
                              "mean": sum(counts) / len(counts)},
            "note": "Popularity columns use labels from TRAIN-fitted thresholds; validation/test months are descriptive only."}


def score_section(frame: DataFrame) -> tuple[dict, dict]:
    overall = frequencies(frame, "score")
    by_split = {s: [] for s in SPLITS}
    for row in frame.groupBy("split", "score").count().orderBy("split", "score").collect():
        by_split[row.split].append([row.score, row["count"]])

    def describe(pairs):
        total = sum(c for _, c in pairs)
        count = dict(pairs)
        return {"records": total, "distinct_values": len(pairs), "min": pairs[0][0], "max": pairs[-1][0],
                "mean": sum(v * c for v, c in pairs) / total,
                "zero": count.get(0, 0), "zero_share": count.get(0, 0) / total,
                "one": count.get(1, 0), "one_share": count.get(1, 0) / total,
                "share_le_1": (count.get(0, 0) + count.get(1, 0)) / total,
                "quantiles": quantiles_from_frequencies(pairs),
                "tail": [{"score_at_least": k, "records": sum(c for v, c in pairs if v >= k),
                          "share": sum(c for v, c in pairs if v >= k) / total} for k in SCORE_TAIL_CUTOFFS]}

    small = dict(overall)
    summary = {"all": describe(overall), "by_split": {s: describe(p) for s, p in by_split.items()},
               "small_scores": [{"score": v, "records": small.get(v, 0)} for v in range(21)],
               "quantile_definition": "Exact inverse empirical CDF: smallest score whose cumulative count reaches ceil(p*N)."}
    return summary, {"all": overall, "by_split": by_split}


def title_section(frame: DataFrame) -> dict:
    # Feature columns are doubles holding integral counts; report them as integers.
    characters, words, questions = ([[int(v), c] for v, c in frequencies(frame, name)]
                                    for name in ("title_characters", "title_words", "question_count"))
    uppercase = frame.select(F.percentile("uppercase_ratio", list(QUANTILES)).alias("q")).first().q
    shares = frame.agg(F.avg(F.col("question_present")).alias("question_present_share"),
                       F.avg(F.col("title_only").cast("double")).alias("title_only_share"),
                       *[F.avg(F.col(p).cast("double")).alias(p) for p in PATTERNS]).first().asDict()
    return {
        "definitions": "Same definitions as model features: NFKC + whitespace-collapsed title length in characters; "
                       "word count from the feature tokenizer; question marks; uppercase-to-letter ratio.",
        "title_characters": {"quantiles": quantiles_from_frequencies(characters),
                             "mean": sum(v * c for v, c in characters) / sum(c for _, c in characters),
                             "frequencies": [[int(v), c] for v, c in characters]},
        "title_words": {"quantiles": quantiles_from_frequencies(words),
                        "mean": sum(v * c for v, c in words) / sum(c for _, c in words),
                        "frequencies": [[int(v), c] for v, c in words]},
        "question_count": {"frequencies": [[int(v), c] for v, c in questions]},
        "uppercase_ratio": {"quantiles": dict(zip(map(str, QUANTILES), uppercase))},
        "question_present_share": shares.pop("question_present_share"),
        "title_only_share": shares.pop("title_only_share"),
        "pattern_shares": shares, "patterns": PATTERNS,
    }


def pattern_section(train: DataFrame) -> dict:
    aggregates = popularity_aggregates()
    for name in PATTERNS:
        flag = F.col(name)
        aggregates += [F.count(F.when(flag, 1)).alias(f"{name}__records"),
                       F.avg(F.when(flag, F.col("popular").cast("double"))).alias(f"{name}__share_score_ge_2"),
                       F.avg(F.when(flag, F.col("high").cast("double"))).alias(f"{name}__share_label_ge_6"),
                       F.avg(F.when(flag, F.col("label").cast("double"))).alias(f"{name}__mean_label"),
                       F.median(F.when(flag, F.col("score"))).alias(f"{name}__median_score")]
    row = train.agg(*aggregates).first().asDict()
    overall = {k: row[k] for k in ("records", "share_score_ge_2", "share_label_ge_6", "mean_label", "median_score", "mean_score")}
    patterns = []
    for name, regex in PATTERNS.items():
        item = {"pattern": name, "regex": regex, **{k: row[f"{name}__{k}"] for k in (
            "records", "share_score_ge_2", "share_label_ge_6", "mean_label", "median_score")}}
        item["share_of_train"] = item["records"] / overall["records"]
        for key in ("share_score_ge_2", "share_label_ge_6"):
            item[key.replace("share", "lift")] = item[key] / overall[key] if item[key] is not None and overall[key] else None
        patterns.append(item)
    return {"overall": overall, "patterns": patterns}


def correlation_section(train: DataFrame) -> dict:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    columns = ["score", "title_characters", "title_words", "question_count", "uppercase_ratio"]
    vectors = VectorAssembler(inputCols=columns, outputCol="values").transform(
        train.select(*[F.col(c).cast("double").alias(c) for c in columns]))
    spearman = Correlation.corr(vectors, "values", "spearman").first()[0].toArray()
    pearson = train.agg(*[F.corr(F.log1p(F.col("score").cast("double")), F.col(c).cast("double")).alias(c)
                          for c in columns[1:]]).first().asDict()
    return {"spearman_with_score": {c: float(spearman[0][i]) for i, c in enumerate(columns) if i},
            "pearson_with_log1p_score": pearson,
            "note": "TRAIN split only. Spearman uses average ranks for ties; scores 0/1 dominate, so ties are extensive."}


def posting_time_section(frame: DataFrame, train: DataFrame) -> dict:
    def merge(column, labels):
        volume = dict(frame.groupBy(column).count().collect())
        popularity = {r[column]: r for r in grouped(train, column)}
        return [{column: k, "label": labels[k], "records_all_year": volume.get(k, 0),
                 "train": {key: value for key, value in popularity[k].items() if key != column} if k in popularity else None}
                for k in range(len(labels))]
    cells = [{"weekday": r["weekday"], "hour": r["hour"], **{k: v for k, v in r.items() if k not in ("weekday", "hour")}}
             for r in grouped(train, "weekday", "hour")]
    return {"timezone": "UTC", "weekday_convention": "Monday=0 … Sunday=6",
            "by_hour": merge("hour", [f"{h:02d}" for h in range(24)]),
            "by_weekday": merge("weekday", list(WEEKDAYS)),
            "train_weekday_hour": cells}


def association_section(train: DataFrame) -> dict:
    def labelled(column, labels):
        return [{"bucket": labels[r[column]], **{k: v for k, v in r.items() if k != column}}
                for r in grouped(train.filter(F.col(column).isNotNull()), column)]
    return {
        "scope": "TRAIN split only (2025-01-01 to 2025-10-01 UTC). Descriptive associations, not causal effects.",
        "popularity_measures": {"share_score_ge_2": "Share of posts with score >= 2 (label >= 1)",
                                "share_label_ge_6": "Share of posts in labels 6–7 (score >= 46 with the fitted thresholds)",
                                "mean_label": "Mean ordinal label 0–7", "median_score": "Exact median archived score"},
        "title_characters": labelled("char_bucket", bucket_labels(CHAR_EDGES)),
        "title_words": labelled("word_bucket", bucket_labels(WORD_EDGES)),
        "uppercase_ratio": labelled("uppercase_bucket", bucket_labels(UPPERCASE_EDGES, integer=False)),
        "question_marks": labelled("question_bucket", ["0", "1", "≥2"]),
        "title_only": [{"bucket": "title only" if r["title_only"] else "has body", **{k: v for k, v in r.items() if k != "title_only"}}
                       for r in grouped(train, "title_only")],
        "patterns": pattern_section(train),
        "correlations": correlation_section(train),
    }


def label_section(frame: DataFrame, thresholds: dict) -> dict:
    distributions = class_distributions(frame)
    nonempty = {s: all(r["count"] > 0 for r in d["labels"]) for s, d in distributions.items()}
    baseline = {}
    for split, values in distributions.items():
        p = values["labels"][0]["percentage"] / 100
        baseline[split] = {"accuracy": p, "macro_f1": (2 * p / (1 + p)) / 8, "balanced_accuracy": 1 / 8}
    boundaries = thresholds["effective_boundaries"]
    train = frame.filter(F.col("split") == "train")
    at_boundary = dict(train.filter(F.col("score").isin(boundaries)).groupBy("score").count().collect())
    ties = [{"boundary": b, "train_records_at_boundary_score": at_boundary.get(b, 0),
             "assigned_label": k + 1} for k, b in enumerate(boundaries)]
    monthly_train = train.select("month", "score", "split").persist(StorageLevel.MEMORY_AND_DISK)
    stability = []
    try:
        for month in sorted(r.month for r in monthly_train.select("month").distinct().collect()):
            try:
                fitted = fit_thresholds(monthly_train.filter(F.col("month") == month).drop("month"))
                stability.append({"month": month, "boundaries": fitted["effective_boundaries"]})
            except ValueError as error:
                stability.append({"month": month, "boundaries": None, "error": str(error)})
        # The first upper-remainder median is degenerate (median = 2, empty label 1)
        # whenever at least half of the score >= 2 posts have score exactly 2.
        upper = monthly_train.groupBy("month").agg(F.count(F.when(F.col("score") >= 2, 1)).alias("ge2"),
                                                   F.count(F.when(F.col("score") == 2, 1)).alias("eq2")).orderBy("month").collect()
    finally:
        monthly_train.unpersist()
    pooled_ge2, pooled_eq2 = sum(r.ge2 for r in upper), sum(r.eq2 for r in upper)
    score_two = {"pooled_train": {"score_ge_2": pooled_ge2, "score_eq_2": pooled_eq2, "share_eq_2": pooled_eq2 / pooled_ge2},
                 "by_train_month": [{"month": r.month, "score_ge_2": r.ge2, "score_eq_2": r.eq2,
                                     "share_eq_2": r.eq2 / r.ge2 if r.ge2 else None} for r in upper],
                 "note": "Label 1 exists only while fewer than half of the score >= 2 posts have score exactly 2."}
    return {"thresholds": thresholds, "class_distributions": distributions, "nonempty_classes": nonempty,
            "all_classes_nonempty": all(nonempty.values()), "boundary_ties": ties,
            "score_2_share_of_upper_remainder": score_two,
            "tie_rule": "Scores equal to a boundary go to the higher label; equal scores never split.",
            "monthly_train_threshold_stability": stability,
            "analytical_majority_baseline": baseline,
            "note": "Thresholds use TRAIN only. Monthly refits are a stability diagnostic, never the fitted mapping."}


def run_eda(spark, input_path: Path, config: dict, output: Path, config_path: str = "") -> dict:
    started = perf_counter()
    input_files = sorted(input_path.rglob("*.parquet"))
    if not input_files or not (input_path / "_SUCCESS").is_file():
        raise ValueError("A completed local cleaned Parquet dataset is required")
    guard_paths(input_files, [output])
    output.mkdir(parents=True)
    fingerprints = [file_fingerprint(p) for p in input_files]
    raw = spark.read.parquet(str(input_path))
    if dict(raw.dtypes) != dict(CLEAN_SCHEMA):
        raise ValueError(f"Unexpected cleaned schema: {raw.dtypes}")
    base = analysis_frame(raw, config).persist(StorageLevel.DISK_ONLY)
    timings = {}
    try:
        def timed(name, function, *args):
            began = perf_counter()
            value = function(*args)
            timings[name] = perf_counter() - began
            LOG.info("EDA section %s finished in %.1f s", name, timings[name])
            return value
        records = timed("materialize", base.count)
        if base.filter(F.col("split").isNull()).limit(1).count():
            raise ValueError("Every cleaned record must fall inside a chronological split interval")
        thresholds = timed("thresholds", fit_thresholds, base.filter(F.col("split") == "train").select("score", "split"))
        frame = apply_labels(base, thresholds).withColumns({"popular": F.col("label") >= 1, "high": F.col("label") >= 6})
        train = frame.filter(F.col("split") == "train")
        profile = timed("profile", profile_section, raw, frame)
        if profile["records"] != records:
            raise ValueError("Profile and analysis row counts disagree")
        score_summary, score_frequencies = timed("score_distribution", score_section, frame)
        sections = {
            "profile": profile,
            "monthly_volume": timed("monthly_volume", monthly_section, frame),
            "score_distribution": score_summary,
            "score_frequencies": score_frequencies,
            "title_characteristics": timed("title_characteristics", title_section, frame),
            "posting_time": timed("posting_time", posting_time_section, frame, train),
            "associations": timed("associations", association_section, train),
            "label_feasibility": timed("label_feasibility", label_section, frame, thresholds),
        }
    finally:
        base.unpersist()
    if [file_fingerprint(p) for p in input_files] != fingerprints:
        raise ValueError("Cleaned inputs changed during EDA")
    for name in OUTPUT_FILES:
        write_json(output / f"{name}.json", sections[name])
    run = {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
           "elapsed_seconds": perf_counter() - started, "section_seconds": timings,
           "records": records, "input": str(input_path), "input_files": fingerprints,
           "config_path": config_path, "split_dates": config["split_dates"], "feature_config": config["features"],
           "spark_version": spark.version, "python_version": platform.python_version(),
           "association_scope": "train", "outputs": [f"{n}.json" for n in OUTPUT_FILES]}
    write_json(output / "run.json", run)
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/fang_chua_v2.json",
                        help="Experiment config supplying input path, split dates and feature definitions")
    parser.add_argument("--output", default="reports/eda", help="Fresh output directory")
    parser.add_argument("--render-only", action="store_true", help="Rebuild figures and summary from saved JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from askreddit.eda_report import render_all
    output = Path(args.output).resolve()
    if args.render_only:
        print("\n".join(str(p) for p in render_all(output)))
        return 0
    spark = None
    try:
        config = json.loads(Path(args.config).read_text())
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        scratch = Path("artifacts/eda_spark_tmp").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch)
        spark = create_spark(config["spark"]["driver_memory"], config["spark"]["shuffle_partitions"])
        run = run_eda(spark, Path(config["input"]).resolve(), config, output, args.config)
        LOG.info("EDA completed for %s records in %.1f s", f"{run['records']:,}", run["elapsed_seconds"])
    except BaseException:
        LOG.exception("EDA failed")
        return 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    print("\n".join(str(p) for p in render_all(output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
