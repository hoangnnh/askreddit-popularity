"""Clean explicit local AskReddit Parquet inputs with PySpark; do not label posts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import glob
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F


COLUMNS = ("id", "subreddit", "title", "selftext", "created_utc", "score")
SCHEMA = list(zip(COLUMNS, ("string",) * 4 + ("bigint", "bigint")))
DEDUPLICATION_RULE = (
    "After title/body filtering, retain one complete source row per id across all inputs: "
    "earliest created_utc (nulls last), then lexicographically smallest SHA-256 of "
    "the JSON struct of the six source columns in their original order, including nulls. "
    "Identical rows are interchangeable; hash ties use source-column values ascending, "
    "nulls last. No score is averaged, filled, clipped, or otherwise rewritten."
)


def resolve_inputs(arguments: list[str]) -> list[Path]:
    """Resolve explicit files or local globs, failing on any missing argument."""
    paths = set()
    for argument in arguments:
        if "://" in argument:
            raise ValueError(f"Only local Parquet files are accepted: {argument}")
        expanded = str(Path(argument).expanduser())
        matches = glob.glob(expanded) if glob.has_magic(expanded) else [expanded]
        if not matches:
            raise ValueError(f"Input pattern matched no files: {argument}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix != ".parquet":
                raise ValueError(f"Input must be an existing Parquet file: {match}")
            paths.add(path)
    if not paths:
        raise ValueError("At least one input file is required")
    return sorted(paths)


def resolve_output(argument: str) -> Path:
    if "://" in argument or glob.has_magic(argument):
        raise ValueError("Output must be an explicit local directory path")
    path = Path(argument).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    return path


def validate_schema(frame: DataFrame) -> None:
    if frame.dtypes != SCHEMA:
        raise ValueError(f"Expected the six ingestion columns and types {SCHEMA}; got {frame.dtypes}")


def score_aggregates() -> list:
    return [
        F.count("*").alias("row_count"),
        F.count("score").alias("known_score_count"),
        F.count(F.when(F.col("score") == 1, 1)).alias("score_eq_1_count"),
        F.expr("percentile(score, array(0.5, 0.75))").alias("score_quantiles"),
    ]


def score_report(values: dict) -> dict:
    known = values["known_score_count"]
    quantiles = values["score_quantiles"] or [None, None]
    return {
        "row_count": values["row_count"],
        "missing_score_count": values["row_count"] - known,
        "known_score_count": known,
        "score_eq_1_count": values["score_eq_1_count"],
        "score_eq_1_share": values["score_eq_1_count"] / known if known else None,
        "median_score": quantiles[0],
        "score_p75": quantiles[1],
    }


@contextmanager
def cleaned_records(frame: DataFrame):
    """Yield cached cleaned records and sequential removal/statistics counts."""
    validate_schema(frame)
    raw = frame.persist(StorageLevel.MEMORY_AND_DISK)
    cleaned = None
    try:
        # Unicode whitespace, including tabs/newlines; predicates never rewrite values.
        invalid_title = F.col("title").isNull() | F.col("title").rlike(r"(?U)^\s*$")
        trimmed_body = F.regexp_replace("selftext", r"(?U)^\s+|\s+$", "")
        removed_body = F.coalesce(trimmed_body.isin("[removed]", "[deleted]"), F.lit(False))
        invalid_subreddit = F.col("subreddit").isNull() | (F.lower("subreddit") != "askreddit")
        invalid_id = F.col("id").isNull() | F.col("id").rlike(r"(?U)^\s*$")
        values = raw.agg(
            *score_aggregates(),
            F.count(F.when(invalid_subreddit, 1)).alias("invalid_subreddit_count"),
            F.count(F.when(invalid_id, 1)).alias("invalid_id_count"),
            F.count(F.when(invalid_title, 1)).alias("removed_titles"),
            F.count(F.when(~invalid_title & removed_body, 1)).alias("removed_bodies"),
        ).first().asDict()
        if values["invalid_subreddit_count"]:
            raise ValueError(f"Input contains {values['invalid_subreddit_count']} rows outside r/AskReddit")
        if values["invalid_id_count"]:
            raise ValueError(f"Input contains {values['invalid_id_count']} null/blank IDs; cannot deduplicate safely")

        eligible = raw.filter(~invalid_title & ~removed_body)
        content_hash = F.sha2(
            F.to_json(F.struct(*COLUMNS), options={"ignoreNullFields": "false"}), 256
        )
        ordering = Window.partitionBy("id").orderBy(
            F.col("created_utc").asc_nulls_last(),
            content_hash.asc(),
            *[F.col(column).asc_nulls_last() for column in COLUMNS],
        )
        cleaned = (
            eligible.withColumn("_dedup_rank", F.row_number().over(ordering))
            .filter(F.col("_dedup_rank") == 1)
            .select(*COLUMNS)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        output_values = cleaned.agg(*score_aggregates()).first().asDict()
        after_titles = values["row_count"] - values["removed_titles"]
        after_bodies = after_titles - values["removed_bodies"]
        output_count = output_values["row_count"]
        report = {
            "input_row_count": values["row_count"],
            "output_row_count": output_count,
            "removed_row_count": values["row_count"] - output_count,
            "steps": [
                {"step": "verify_subreddit_and_id", "removed": 0, "remaining": values["row_count"]},
                {"step": "drop_null_or_blank_title", "removed": values["removed_titles"], "remaining": after_titles},
                {"step": "drop_removed_or_deleted_selftext", "removed": values["removed_bodies"], "remaining": after_bodies},
                {"step": "deduplicate_id_globally", "removed": after_bodies - output_count, "remaining": output_count},
            ],
            "input_scores": score_report(values),
            "output_scores": score_report(output_values),
            "score_statistics_definition": (
                "Missing scores are retained. score_eq_1_share uses non-null scores as its denominator. "
                "Median and p75 are exact linearly interpolated percentiles of non-null scores. "
                "Share and percentiles are null if no scores are present."
            ),
            "deduplication_rule": DEDUPLICATION_RULE,
            "statistics_use": (
                "Exploratory descriptions only, including the January pilot. "
                "Labels and thresholds belong to the later chronological split stage."
            ),
        }
        yield cleaned, report
    finally:
        if cleaned is not None:
            cleaned.unpersist()
        raw.unpersist()


def run_cleaning(spark: SparkSession, inputs: list[Path], output: Path) -> dict:
    started = perf_counter()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    # Inspect every schema; Spark's default schema inference can overlook drift in other files.
    for path in inputs:
        validate_schema(spark.read.parquet(str(path)))
    raw = spark.read.parquet(*[str(path) for path in inputs])
    with cleaned_records(raw) as (cleaned, report):
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
            dataset = Path(staging) / "parquet"
            cleaned.write.mode("errorifexists").option("compression", "zstd").parquet(str(dataset))
            written = spark.read.parquet(str(dataset))
            validate_schema(written)
            if written.count() != report["output_row_count"]:
                raise ValueError("Written Parquet row count does not match the cleaned frame")
            report.update({
                "input_files": [str(path) for path in inputs],
                "output_path": str(output),
                "source_columns": list(COLUMNS),
                "spark_version": spark.version,
                "elapsed_seconds": round(perf_counter() - started, 3),
            })
            (dataset / "_cleaning_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            if output.exists():
                raise ValueError(f"Output appeared during cleaning; refusing to replace it: {output}")
            dataset.rename(output)
    return report


def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("askreddit-clean")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="explicit local Parquet file(s) or quoted file glob(s)")
    parser.add_argument("--output", required=True, help="new local output directory for cleaned Parquet and report")
    args = parser.parse_args(argv)
    spark = None
    try:
        inputs = resolve_inputs(args.input)
        output = resolve_output(args.output)
        spark = create_spark()
        report = run_cleaning(spark, inputs, output)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        print("Cleaning interrupted; rerun with the same inputs and a new/absent output directory.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Cleaning failed: {error}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
