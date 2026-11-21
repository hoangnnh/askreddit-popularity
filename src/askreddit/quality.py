"""Distributed quality aggregates; collect only scalar summaries and month/file counts."""

from datetime import datetime, timezone

from pyspark.sql import DataFrame, functions as F

from askreddit.clean import COLUMNS, CLEAN_SCHEMA, Policy, trim_whitespace


def count_when(condition, name):
    return F.count(F.when(condition, 1)).alias(name)


def iso_utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat() if seconds is not None else None


def duplicate_summary(groups: DataFrame) -> dict:
    values = groups.agg(
        F.count("*").alias("duplicate_ids"),
        F.coalesce(F.sum(F.col("rows") - 1), F.lit(0)).alias("excess_duplicate_records"),
        F.coalesce(F.sum(F.col("rows") - F.col("original_variants")), F.lit(0)).alias("exact_duplicate_excess"),
        count_when(F.col("original_variants") == 1, "exact_only_duplicate_ids"),
        count_when(F.col("original_variants") > 1, "conflicting_duplicate_ids"),
        *[count_when(F.col("variants_" + c) > 1, "conflicting_" + c + "_ids") for c in COLUMNS[1:]],
    ).first().asDict()
    values["definition"] = (
        "Null IDs are excluded; blank/invalid non-null IDs are included in the raw profile. "
        "Raw groups use the original ID; eligible groups use the trimmed ID. Exact "
        "duplicates compare all six original logical fields, including nulls. "
        "exact_duplicate_excess can include repeated rows inside conflicting groups."
    )
    return values


def profile_records(frame: DataFrame) -> dict:
    body = trim_whitespace(F.col("raw_selftext"))
    values = frame.agg(
        F.count("*").alias("total_records"),
        *[count_when(F.col("raw_" + c).isNull(), "null_" + c) for c in COLUMNS],
        count_when(F.col("raw_title").isNotNull() & (F.col("title") == ""), "empty_or_whitespace_titles"),
        count_when(F.array_contains("rejection_reasons", "outside_askreddit"), "outside_askreddit"),
        count_when(F.array_contains("rejection_reasons", "invalid_id"), "invalid_ids"),
        count_when(F.col("_time_problem").isNotNull(), "invalid_or_suspicious_timestamps"),
        count_when(F.col("_time_problem") == "missing_timestamp", "missing_timestamps"),
        count_when(F.col("_time_problem") == "invalid_timestamp", "invalid_timestamps"),
        count_when(F.col("_time_problem") == "suspicious_timestamp", "suspicious_timestamps"),
        count_when(F.col("_time_problem") == "subsecond_timestamp", "subsecond_timestamps"),
        count_when(F.col("_score_problem") == "missing_score", "missing_scores"),
        count_when(F.col("_score_problem") == "invalid_score", "invalid_scores"),
        count_when(F.col("raw_score").isNotNull() & ~F.col("_score_text").rlike(r"^[+-]?[0-9]+$"),
                   "non_integer_score_representations"),
        count_when(F.col("score") < 0, "negative_scores"),
        count_when(F.col("score") == 0, "zero_scores"),
        count_when(F.col("score") > 0, "positive_scores"),
        count_when(body == "", "empty_or_whitespace_selftext"),
        count_when(F.lower(body) == "[removed]", "removed_selftext"),
        count_when(F.lower(body) == "[deleted]", "deleted_selftext"),
        count_when(F.col("raw_id") != F.col("id"), "trimmed_ids"),
        count_when(F.col("raw_title") != F.col("title"), "normalized_titles"),
        count_when(~F.col("raw_selftext").eqNullSafe(F.col("selftext")), "normalized_selftext"),
        count_when(F.col("raw_subreddit") != F.col("subreddit"), "normalized_subreddits"),
        F.min("_time_integer").alias("min_created_utc_raw_numeric"),
        F.max("_time_integer").alias("max_created_utc_raw_numeric"),
        F.min("created_utc").alias("min_created_utc"), F.max("created_utc").alias("max_created_utc"),
        F.min("score").alias("min_score"), F.max("score").alias("max_score"),
    ).first().asDict()
    total = values["total_records"]
    values["nulls"] = {c: {"count": values.pop("null_" + c)} for c in COLUMNS}
    for entry in values["nulls"].values():
        entry["percentage"] = 100 * entry["count"] / total if total else None
    values["min_created_utc_iso"] = iso_utc(values["min_created_utc"])
    values["max_created_utc_iso"] = iso_utc(values["max_created_utc"])
    values["monthly_records"] = [r.asDict() for r in frame.groupBy("year", "month").count().orderBy("year", "month").collect()]
    values["source_files"] = [r.asDict() for r in frame.groupBy("_source_file").agg(
        F.count("*").alias("records"), F.min("created_utc").alias("min_created_utc"),
        F.max("created_utc").alias("max_created_utc")).orderBy("_source_file").collect()]
    values["definitions"] = {
        "timestamp_extrema": "Raw numeric extrema retain the selected source unit. Normalized extrema/months include only timestamps within the configured UTC sanity bounds; invalid timestamps form a null-month group.",
        "scores": "Extrema and sign counts include all parseable signed 64-bit integer scores, independently of other rejection reasons.",
        "null_percentage": "Percentage of all input rows; null for an empty dataset.",
        "non_numeric_scores": "invalid_scores includes non-integer syntax, blanks, overflow, NaN and infinity. No rounding or coercion of fractional values.",
        "normalization_counts": "Count changes across all input rows, including rows later quarantined.",
    }
    return values


def verify_cleaned(cleaned: DataFrame, quarantine: DataFrame, input_count: int, policy: Policy) -> dict:
    if cleaned.dtypes != CLEAN_SCHEMA:
        raise ValueError(f"Read-back schema mismatch: {cleaned.dtypes}")
    stats = cleaned.agg(
        F.count("*").alias("total_cleaned_records"),
        *[count_when(F.col(c).isNull(), "null_" + c) for c, _ in CLEAN_SCHEMA],
        count_when(F.col("id").isNull() | ~F.col("id").rlike(r"^[a-z0-9]+$"), "invalid_ids"),
        count_when(F.col("subreddit").isNull() | (F.col("subreddit") != "AskReddit"), "outside_askreddit"),
        count_when(F.col("title").isNull() | F.col("title").rlike(r"(?U)^\s*$"), "invalid_titles"),
        count_when(F.col("created_utc").isNull() | ~F.col("created_utc").between(policy.min_created_utc, policy.max_created_utc), "invalid_timestamps"),
        count_when(~F.col("year").eqNullSafe(F.year(F.timestamp_seconds("created_utc"))) |
                   ~F.col("month").eqNullSafe(F.month(F.timestamp_seconds("created_utc"))), "invalid_partitions"),
        F.min("created_utc").alias("min_created_utc"), F.max("created_utc").alias("max_created_utc"),
        F.min("score").alias("min_score"), F.max("score").alias("max_score"),
        count_when(F.col("score") < 0, "negative_scores"),
        count_when(F.col("score") == 0, "zero_scores"),
        count_when(F.col("score") > 0, "positive_scores"),
    ).first().asDict()
    stats["remaining_duplicate_ids"] = cleaned.groupBy("id").count().filter(F.col("count") > 1).count()
    stats["nulls"] = {c: stats.pop("null_" + c) for c, _ in CLEAN_SCHEMA}
    stats["monthly_records"] = [r.asDict() for r in cleaned.groupBy("year", "month").count().orderBy("year", "month").collect()]
    stats["primary_rejection_counts"] = {r.primary_reason: r["count"] for r in quarantine.groupBy("primary_reason").count().collect()}
    stats["rejection_counts_by_reason"] = {r.reason: r["count"] for r in quarantine.select(
        F.explode("rejection_reasons").alias("reason")).groupBy("reason").count().collect()}
    stats["total_quarantined_records"] = quarantine.count()
    stats["invalid_quarantine_records"] = quarantine.filter(
        F.col("rejection_reasons").isNull() | (F.size("rejection_reasons") == 0) |
        F.col("primary_reason").isNull() | F.col("_raw_json").isNull() | F.col("_source_file").isNull()).count()
    stats["total_duplicate_records_removed"] = sum(
        stats["primary_rejection_counts"].get(reason, 0) for reason in ("duplicate_exact", "duplicate_conflict"))
    stats["total_rejected_records"] = stats["total_quarantined_records"] - stats["total_duplicate_records_removed"]
    stats["total_input_records"] = input_count
    stats["retained_percentage"] = 100 * stats["total_cleaned_records"] / input_count if input_count else None
    stats["min_created_utc_iso"] = iso_utc(stats["min_created_utc"])
    stats["max_created_utc_iso"] = iso_utc(stats["max_created_utc"])
    stats["invariants"] = {
        "unique_ids": stats["remaining_duplicate_ids"] == 0,
        "valid_ids": stats["invalid_ids"] == 0,
        "askreddit_only": stats["outside_askreddit"] == 0,
        "nonempty_titles": stats["invalid_titles"] == 0,
        "valid_timestamps": stats["invalid_timestamps"] == 0,
        "valid_integer_scores": stats["nulls"]["score"] == 0,
        "no_required_nulls": sum(stats["nulls"].values()) == 0,
        "utc_partition_values": stats["invalid_partitions"] == 0,
        "parquet_read_back": True,
        "quarantine_has_originals_and_reasons": stats["invalid_quarantine_records"] == 0,
        "counts_reconcile": input_count == stats["total_cleaned_records"] + stats["total_quarantined_records"],
        "primary_reasons_reconcile": sum(stats["primary_rejection_counts"].values()) == stats["total_quarantined_records"],
    }
    stats["reconciliation_definition"] = (
        "input = cleaned + quarantined; quarantined = rejected_invalid + duplicate_losers. "
        "Invalid rows never compete in deduplication. All reasons are retained and their "
        "counts can overlap; primary reasons are exclusive in order: community, ID, "
        "title, timestamp, score. Duplicate losers have exactly one duplicate reason."
    )
    failed = [name for name, valid in stats["invariants"].items() if not valid]
    if failed:
        raise ValueError(f"Post-cleaning invariants failed: {failed}; statistics: {stats}")
    return stats
