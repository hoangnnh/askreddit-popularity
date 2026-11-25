"""Title-level deduplication: keep one post per normalized title, the one with the highest score.

Many AskReddit questions are posted again and again; the same title can score 1 on
one posting and thousands on another. This step keeps, for every normalized title,
the single posting with the highest archived score (ties: earliest ``created_utc``,
then smallest ``id``), and drops moderator/Reddit placeholder titles such as
``[ Removed by moderator ]``. The label then reads "the best score this question
reached in 2025".

Deduplication is global over the whole year, so no title appears in more than one
chronological split. The choice of which posting to keep uses every posting's
score, including postings in later (validation/test) months; reports disclose this.

Input and output share the cleaned schema (``CLEAN_SCHEMA``), partitioned by year
and month, so every downstream experiment can read either dataset.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import Window, functions as F, types as T

from askreddit.clean import CLEAN_SCHEMA, create_spark, file_fingerprint, guard_paths, write_json
from askreddit.features import normalize_unicode

LOG = logging.getLogger(__name__)
PLACEHOLDER_PATTERN = r"^\[\s*(removed|deleted)\b[^\]]*\]$"
RULE = ("Key = NFKC + whitespace collapse + lowercase title, curly quotes mapped to straight quotes. "
        "Placeholder titles matching " + PLACEHOLDER_PATTERN + " are dropped. Within each key keep the posting "
        "with the highest score; ties by earliest created_utc, then smallest id. Global over all of 2025.")


def title_key(column):
    normalize = F.udf(normalize_unicode, T.StringType(), useArrow=False)
    return F.translate(F.lower(normalize(column)), "’‘“”", "''\"\"")


def deduplicate(frame):
    """Return (kept rows with CLEAN_SCHEMA, keyed frame) for reporting."""
    keyed = frame.withColumn("title_key", title_key("title")).withColumn(
        "is_placeholder", F.col("title_key").rlike(PLACEHOLDER_PATTERN))
    ordering = Window.partitionBy("title_key").orderBy(F.col("score").desc(), F.col("created_utc"), F.col("id"))
    ranked = keyed.filter(~F.col("is_placeholder")).withColumns({
        "title_rank": F.row_number().over(ordering),
        "title_postings": F.count("*").over(Window.partitionBy("title_key")),
    })
    return ranked, keyed


def score_profile(frame):
    row = frame.agg(F.count("*").alias("records"),
                    F.avg((F.col("score") <= 1).cast("double")).alias("share_score_le_1"),
                    F.avg((F.col("score") >= 2).cast("double")).alias("share_score_ge_2"),
                    F.sum((F.col("score") >= 271).cast("long")).alias("records_score_ge_271"),
                    F.expr("percentile(score, 0.5)").alias("median_score"),
                    F.expr("percentile(score, 0.9)").alias("p90_score")).first().asDict()
    return {k: (float(v) if isinstance(v, float) else v) for k, v in row.items()}


def run(spark, source: Path, output: Path, reports: Path, write_partitions: int) -> dict:
    started = perf_counter()
    inputs = sorted(source.rglob("*.parquet"))
    if not inputs or not (source / "_SUCCESS").is_file():
        raise ValueError("A completed cleaned Parquet dataset is required")
    guard_paths(inputs, [output, reports])
    fingerprints = [file_fingerprint(p) for p in inputs]
    raw = spark.read.parquet(str(source))
    if dict(raw.dtypes) != dict(CLEAN_SCHEMA):
        raise ValueError(f"Unexpected cleaned schema: {raw.dtypes}")
    ranked, keyed = deduplicate(raw)
    ranked = ranked.persist(StorageLevel.DISK_ONLY)
    try:
        placeholders = keyed.filter("is_placeholder").count()
        groups = ranked.filter("title_rank = 1").agg(
            F.count("*").alias("distinct_titles"),
            F.sum((F.col("title_postings") > 1).cast("long")).alias("repeated_titles"),
            F.sum(F.when(F.col("title_postings") > 1, F.col("title_postings"))).alias("postings_in_repeated_titles"),
        ).first().asDict()
        largest = [r.asDict() for r in ranked.filter("title_rank = 1").orderBy(F.col("title_postings").desc(), "title_key")
                   .select("title_key", "title_postings", F.col("score").alias("kept_score")).limit(15).collect()]
        kept = ranked.filter("title_rank = 1").select(*[c for c, _ in CLEAN_SCHEMA])
        writer = kept.repartition(write_partitions, "year", "month").sortWithinPartitions("year", "month", "id")
        writer.write.mode("errorifexists").option("compression", "zstd").option("maxRecordsPerFile", 1000000) \
            .partitionBy("year", "month").parquet(str(output))
        written = spark.read.parquet(str(output))
        if dict(written.dtypes) != dict(CLEAN_SCHEMA):
            raise ValueError("Deduplicated output schema differs from the cleaned schema")
        written_count = written.count()
        distinct_keys = written.select(title_key("title").alias("k")).distinct().count()
        if written_count != groups["distinct_titles"] or distinct_keys != written_count:
            raise ValueError("Read-back must contain exactly one row per normalized title")
        before, after = score_profile(raw), score_profile(written)
        if [file_fingerprint(p) for p in inputs] != fingerprints:
            raise ValueError("Input files changed during deduplication")
        report = {
            "status": "completed", "rule": RULE, "input": str(source), "output": str(output),
            "input_records": before["records"], "placeholder_titles_dropped": placeholders,
            "distinct_titles": groups["distinct_titles"], "repeated_titles": groups["repeated_titles"],
            "postings_in_repeated_titles": groups["postings_in_repeated_titles"],
            "non_best_postings_dropped": before["records"] - placeholders - written_count,
            "output_records": written_count, "score_before": before, "score_after": after,
            "largest_title_groups": largest,
            "disclosures": [
                "The kept posting is the luckiest of its repeats: scores of repeated questions are biased upward.",
                "Selection uses all 2025 scores, including validation/test months; titles never span splits.",
                "Repeated titles include karma-farming reposts; deduplication removes their volume, not their existence.",
            ],
            "input_files": fingerprints, "spark_version": spark.version, "python_version": sys.version.split()[0],
            "elapsed_seconds": round(perf_counter() - started, 3), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(reports / "run.json", report)
        return report
    finally:
        ranked.unpersist()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/askreddit_clean")
    parser.add_argument("--output", default="data/processed/askreddit_dedup")
    parser.add_argument("--reports-dir", default="reports/data_quality_dedup")
    parser.add_argument("--write-partitions", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spark = None
    try:
        spark = create_spark("2g", 16)
        report = run(spark, Path(args.input).resolve(), Path(args.output).resolve(), Path(args.reports_dir).resolve(),
                     args.write_partitions)
        print(json.dumps({k: v for k, v in report.items() if k != "input_files"}, indent=2, ensure_ascii=False))
    except BaseException:
        LOG.exception("Deduplication failed")
        return 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
