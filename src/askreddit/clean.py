"""Validate and clean existing local Parquet with PySpark; never download or label."""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window, functions as F

COLUMNS = ("id", "subreddit", "title", "selftext", "created_utc", "score")
SCHEMA = list(zip(COLUMNS, ("string",) * 4 + ("bigint", "bigint")))
CLEAN_SCHEMA = SCHEMA + [("year", "int"), ("month", "int")]
LOG = logging.getLogger(__name__)
DEDUPLICATION_RULE = (
    "Only otherwise valid rows compete. Within each trimmed ID, retain the earliest "
    "UTC timestamp, then smallest SHA-256 of the normalized six-field JSON, then "
    "normalized JSON, original JSON, and source file lexicographically. Include nulls "
    "in JSON. Identical rows from the same file are interchangeable. Never rank by "
    "highest score. This reproducible choice does not establish authority or recency. "
    "All losing rows go to quarantine with the retained original-row fingerprint. "
    "duplicate_exact means all original six-field values in the group agree; "
    "duplicate_conflict means at least one differs. Mixed groups use duplicate_conflict."
)


@dataclass(frozen=True)
class Policy:
    # Sanity bounds, not assumptions about the dataset's observed coverage.
    min_created_utc: int = 1104537600  # 2005-01-01T00:00:00Z
    max_created_utc: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    timestamp_unit: str = "seconds"

    def __post_init__(self):
        if not 0 <= self.min_created_utc <= self.max_created_utc <= 253402300799:
            raise ValueError("Timestamp bounds must be ordered Unix seconds within years 1970–9999")
        if self.timestamp_unit not in {"seconds", "milliseconds"}:
            raise ValueError("timestamp_unit must be seconds or milliseconds")


def resolve_inputs(arguments: list[str]) -> list[Path]:
    """Explicit files/globs only; never silently skip corrupt or missing files."""
    paths = set()
    for argument in arguments:
        if "://" in argument:
            raise ValueError(f"Only local Parquet files are accepted: {argument}")
        expanded = str(Path(argument).expanduser())
        matches = glob.glob(expanded, include_hidden=True) if glob.has_magic(expanded) else [expanded]
        if not matches:
            raise ValueError(f"Input pattern matched no files: {argument}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix.lower() != ".parquet":
                raise ValueError(f"Input must be an existing Parquet file: {match}")
            paths.add(path)
    if not paths:
        raise ValueError("At least one input Parquet file is required")
    return sorted(paths)


def resolve_output(argument: str) -> Path:
    if "://" in argument or glob.has_magic(argument):
        raise ValueError("Output must be an explicit local directory path")
    path = Path(argument).expanduser().resolve()
    if path.exists():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    return path


def guard_paths(inputs: list[Path], outputs: list[Path]) -> None:
    """Resolve symlinks, refuse existing targets and any ancestor/descendant overlap."""
    resolved = [resolve_output(str(path)) for path in outputs]
    for i, path in enumerate(resolved):
        for other in resolved[i + 1:] + [p.resolve() for p in inputs]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(f"Input/output paths overlap: {path} and {other}")


def validate_schema(frame: DataFrame) -> None:
    actual = dict(frame.dtypes)
    if len(frame.columns) != len(COLUMNS) or set(actual) != set(COLUMNS):
        raise ValueError(f"Expected exactly the six logical fields {COLUMNS}; got {frame.dtypes}")
    for column in COLUMNS:
        allowed = {"string"} if column in COLUMNS[:4] else {"string", "tinyint", "smallint", "int", "bigint"}
        if actual[column] not in allowed:
            raise ValueError(f"Unsupported type for {column}: {actual[column]}; allowed: {sorted(allowed)}")


def file_fingerprint(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def inspect_sources(spark: SparkSession, inputs: list[Path]) -> dict:
    files = []
    for path in inputs:
        entry = file_fingerprint(path)
        try:
            frame = spark.read.parquet(str(path))
            entry["schema"] = frame.schema.jsonValue()
            validate_schema(frame)
            entry["status"] = "valid"
        except Exception as error:
            entry.update(status="error", error=str(error))
        files.append(entry)
    schemas = {json.dumps(entry["schema"], sort_keys=True) for entry in files if "schema" in entry}
    return {
        "files": files, "file_count": len(files), "distinct_schema_count": len(schemas),
        "schemas_differ": len(schemas) > 1,
        "conversion_policy": (
            "Text fields must be strings. Integral numeric types widen safely to long. "
            "String scores/timestamps require a whole-number lexical representation and "
            "successful try_cast to long. Floats, decimals, native timestamp types, extra "
            "or missing fields fail schema validation; nothing is rounded or truncated. "
            "Different supported schemas are read separately before union, avoiding "
            "Parquet physical-type mismatches. Original values are preserved in raw JSON."
        ),
    }


def raw_records(frame: DataFrame) -> DataFrame:
    validate_schema(frame.select(*COLUMNS))
    source = F.col("_source_file") if "_source_file" in frame.columns else F.lit("<in-memory>")
    return frame.select(
        *[F.col(c).cast("string").alias(c) for c in COLUMNS],
        source.alias("_source_file"),
        F.to_json(F.struct(*COLUMNS), {"ignoreNullFields": "false"}).alias("_raw_json"),
    )


def read_sources(spark: SparkSession, inspection: dict) -> DataFrame:
    groups = {}
    for entry in inspection["files"]:
        key = json.dumps(entry["schema"], sort_keys=True)
        groups.setdefault(key, []).append(entry["path"])
    frames = [raw_records(spark.read.parquet(*paths).withColumn("_source_file", F.input_file_name()))
              for paths in groups.values()]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def trim_whitespace(column):
    return F.regexp_replace(column, r"(?U)^\s+|\s+$", "")


def prepare_records(raw: DataFrame, policy: Policy) -> DataFrame:
    """Lossless raw audit values plus normalized fields and all rejection reasons."""
    if "_raw_json" not in raw.columns:
        validate_schema(raw)
        raw = raw_records(raw)
    frame = raw.select(*[F.col(c).alias(f"raw_{c}") for c in COLUMNS], "_source_file", "_raw_json")
    frame = frame.withColumns({
        "id": trim_whitespace(F.col("raw_id")),
        "subreddit": F.regexp_replace(F.lower(trim_whitespace(F.col("raw_subreddit"))), r"^r/", ""),
        "title": F.regexp_replace(trim_whitespace(F.col("raw_title")), r"(?U)\s+", " "),
        "selftext": F.coalesce(trim_whitespace(F.col("raw_selftext")), F.lit("")),
        "_time_text": trim_whitespace(F.col("raw_created_utc")),
        "_score_text": trim_whitespace(F.col("raw_score")),
    }).withColumns({
        "selftext": F.when(F.lower("selftext").isin("[removed]", "[deleted]"), "").otherwise(F.col("selftext")),
        "_time_integer": F.when(F.col("_time_text").rlike(r"^[+-]?[0-9]+$"), F.expr("try_cast(_time_text as bigint)")),
        "score": F.when(F.col("_score_text").rlike(r"^[+-]?[0-9]+$"), F.expr("try_cast(_score_text as bigint)")),
    })
    if policy.timestamp_unit == "milliseconds":
        seconds = F.expr("_time_integer DIV 1000")
        subsecond = (F.col("_time_integer") % 1000 != 0)
    else:
        seconds = F.col("_time_integer")
        subsecond = F.lit(False)
    frame = frame.withColumn("_seconds", seconds).withColumns({
        "_time_problem": F.when(F.col("raw_created_utc").isNull(), "missing_timestamp")
            .when(F.col("_time_integer").isNull(), "invalid_timestamp")
            .when(subsecond, "subsecond_timestamp")
            .when(~F.col("_seconds").between(policy.min_created_utc, policy.max_created_utc), "suspicious_timestamp"),
        "_score_problem": F.when(F.col("raw_score").isNull(), "missing_score")
            .when(F.col("score").isNull(), "invalid_score"),
    })
    frame = frame.withColumn("created_utc", F.when(F.col("_time_problem").isNull(), F.col("_seconds")).cast("long"))
    reasons = F.array(
        F.when(F.col("subreddit").isNull() | (F.col("subreddit") != "askreddit"), "outside_askreddit"),
        F.when(F.col("id").isNull() | ~F.col("id").rlike(r"^[a-z0-9]+$"), "invalid_id"),
        F.when(F.col("title").isNull() | (F.col("title") == ""), "invalid_title"),
        F.col("_time_problem"), F.col("_score_problem"),
    )
    return frame.withColumns({
        "subreddit": F.when(F.col("subreddit") == "askreddit", "AskReddit").otherwise(F.col("subreddit")),
        "rejection_reasons": F.filter(reasons, lambda reason: reason.isNotNull()),
        "year": F.year(F.timestamp_seconds("created_utc")),
        "month": F.month(F.timestamp_seconds("created_utc")),
    })


def duplicate_groups(frame: DataFrame, *, raw: bool = False) -> DataFrame:
    prefix = "raw_" if raw else ""
    identifier = prefix + "id"
    base = frame.filter(F.col(identifier).isNotNull())
    repeated = base.groupBy(identifier).count().filter(F.col("count") > 1).select(identifier)
    # First narrow to repeated IDs. Multiple distinct aggregates expand each input
    # row internally; doing that for millions of unique IDs wastes the laptop heap.
    duplicates = base.join(repeated, identifier, "left_semi")
    # Struct counts null as a distinct value, unlike countDistinct(column).
    return duplicates.groupBy(identifier).agg(
        F.count("*").alias("rows"),
        F.countDistinct("_raw_json").alias("original_variants"),
        *[F.countDistinct(F.struct(F.col("raw_" + c))).alias("variants_" + c) for c in COLUMNS[1:]],
    )


def split_records(prepared: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return typed winners and a complete rejection/deduplication audit trail."""
    eligible = prepared.filter(F.size("rejection_reasons") == 0)
    groups = duplicate_groups(eligible).select(
        "id", F.col("rows").alias("duplicate_group_size"), "original_variants",
        F.filter(F.array(*[F.when(F.col("variants_" + c) > 1, c) for c in COLUMNS[1:]]),
                 lambda c: c.isNotNull()).alias("duplicate_conflicting_fields"),
    )
    payload = F.to_json(F.struct(*COLUMNS), {"ignoreNullFields": "false"})
    ordering = Window.partitionBy("id").orderBy(
        F.col("created_utc"), F.sha2(payload, 256), payload, F.col("_raw_json"), F.col("_source_file"),
    )
    ranked = eligible.join(groups, "id", "left").withColumns({
        "dedup_rank": F.row_number().over(ordering),
        "retained_original_sha256": F.first(F.sha2("_raw_json", 256)).over(ordering),
    })
    cleaned = ranked.filter(F.col("dedup_rank") == 1).select(*[c for c, _ in CLEAN_SCHEMA])
    duplicates = ranked.filter(F.col("dedup_rank") > 1).withColumn(
        "rejection_reasons", F.array(F.when(F.col("original_variants") == 1, "duplicate_exact").otherwise("duplicate_conflict")),
    )
    rejected = prepared.filter(F.size("rejection_reasons") > 0)
    audit = rejected.unionByName(duplicates, allowMissingColumns=True).withColumn(
        "primary_reason", F.element_at("rejection_reasons", 1),
    ).select(
        *["raw_" + c for c in COLUMNS], "_raw_json", "_source_file",
        *[c for c, _ in CLEAN_SCHEMA], "rejection_reasons", "primary_reason",
        "dedup_rank", "duplicate_group_size", "original_variants",
        "duplicate_conflicting_fields", "retained_original_sha256",
    )
    return cleaned, audit


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each report is written once. A new run must use a fresh report directory.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def create_spark(driver_memory: str = "2g", shuffle_partitions: int = 8) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (SparkSession.builder.appName("askreddit-validation-cleaning").master("local[2]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.files.ignoreCorruptFiles", "false")
        .config("spark.sql.files.ignoreMissingFiles", "false")
        .config("spark.sql.files.maxPartitionBytes", str(8 * 1024 * 1024))
        .config("spark.sql.inMemoryColumnarStorage.batchSize", "1000")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_cleaning(
    spark: SparkSession, inputs: list[Path], output: Path, quarantine: Path,
    reports: Path, policy: Policy, *, validate_only: bool = False, write_partitions: int = 4,
) -> dict:
    from askreddit.quality import duplicate_summary, profile_records, verify_cleaned

    started = perf_counter()
    inputs = resolve_inputs([str(p) for p in inputs])
    output, quarantine, reports = [p.resolve() for p in (output, quarantine, reports)]
    guard_paths(inputs, [output, quarantine, reports])
    if write_partitions < 1:
        raise ValueError("write_partitions must be positive")
    if spark.conf.get("spark.sql.session.timeZone") != "UTC":
        raise ValueError("Spark session timezone must be UTC")
    reports.mkdir(parents=True, exist_ok=False)
    cached = []
    try:
        LOG.info("Inspecting schemas and SHA-256 fingerprints for %d input files", len(inputs))
        inspection = inspect_sources(spark, inputs)
        write_json(reports / "before_cleaning/schema.json", inspection)
        bad = [entry["path"] for entry in inspection["files"] if entry["status"] != "valid"]
        if bad:
            raise ValueError(f"Problematic files; see before_cleaning/schema.json: {bad}")
        # Audit rows contain both raw and normalized text. Keep this wide cache on
        # disk rather than competing with aggregations for the local JVM heap.
        prepared = prepare_records(read_sources(spark, inspection), policy).persist(StorageLevel.DISK_ONLY)
        cached.append(prepared)
        LOG.info("Profiling all source rows using timestamp policy %s", asdict(policy))
        before = profile_records(prepared)
        before.update(policy=asdict(policy), duplicates=duplicate_summary(duplicate_groups(prepared, raw=True)))
        write_json(reports / "before_cleaning/profile.json", before)
        # Full details stay distributed; only aggregate counts reach the driver.
        duplicate_groups(prepared, raw=True).coalesce(write_partitions).write.mode("errorifexists").parquet(
            str(reports / "before_cleaning/duplicate_ids"))
        if validate_only:
            result = {"status": "validated_only", "input_records": before["total_records"], "policy": asdict(policy)}
        else:
            eligible_duplicates = duplicate_summary(duplicate_groups(prepared.filter(F.size("rejection_reasons") == 0)))
            cleaned, audit = split_records(prepared)
            cleaned = cleaned.persist(StorageLevel.DISK_ONLY)
            audit = audit.persist(StorageLevel.DISK_ONLY)
            cached.extend([cleaned, audit])
            cleaned_count, audit_count = cleaned.count(), audit.count()
            LOG.info("Writing %d cleaned and %d quarantined rows", cleaned_count, audit_count)
            writer = cleaned.repartition(write_partitions, "year", "month").sortWithinPartitions("year", "month", "id")
            writer = writer.write.mode("errorifexists").option("compression", "zstd").option("maxRecordsPerFile", 1000000)
            # An empty partitioned write has no data files from which to infer schema.
            if cleaned_count:
                writer.partitionBy("year", "month").parquet(str(output))
            else:
                cleaned.coalesce(1).write.mode("errorifexists").parquet(str(output))
            audit.coalesce(write_partitions).write.mode("errorifexists").option("compression", "zstd").option(
                "maxRecordsPerFile", 250000).parquet(str(quarantine))
            LOG.info("Reading back both datasets and checking invariants")
            written = spark.read.parquet(str(output)).select(*[c for c, _ in CLEAN_SCHEMA])
            written_audit = spark.read.parquet(str(quarantine))
            after = verify_cleaned(written, written_audit, before["total_records"], policy)
            if after["total_cleaned_records"] != cleaned_count or after["total_quarantined_records"] != audit_count:
                raise ValueError("Written row counts differ from the computed datasets")
            after["eligible_duplicate_groups"] = eligible_duplicates
            after["deduplication_policy"] = DEDUPLICATION_RULE
            write_json(reports / "after_cleaning/profile.json", after)
            result = {"status": "completed", **after, "policy": asdict(policy)}
        # Detect source changes; never modify the input files ourselves.
        expected = [{k: entry[k] for k in ("path", "bytes", "sha256")} for entry in inspection["files"]]
        unchanged = [file_fingerprint(p) for p in inputs] == expected
        if not unchanged:
            raise ValueError("An input file changed during this run; outputs are not certified")
        result.update(
            input_files=[str(p) for p in inputs], output_path=str(output), quarantine_path=str(quarantine),
            reports_path=str(reports), input_files_unchanged=True, spark_version=spark.version,
            python_version=sys.version.split()[0], elapsed_seconds=round(perf_counter() - started, 3),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            warnings=["Timestamp sanity bounds are policy; observed coverage is in the profiles.",
                      "Duplicate conflict resolution is deterministic, not evidence of which snapshot is authoritative.",
                      "Score is reserved for future target construction and must never be a predictive feature."],
        )
        write_json(reports / "run.json", result)
        return result
    except BaseException as error:
        write_json(reports / "failure.json", {"status": "failed", "error": str(error),
            "note": "Any partial outputs are uncertified. Inspect them and choose fresh paths for a rerun."})
        raise
    finally:
        for frame in reversed(cached):
            # A crashed JVM must not mask the original exception or its report.
            with suppress(Exception):
                frame.unpersist()


def utc_seconds(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.microsecond:
            raise ValueError("a timezone and whole seconds are required")
        return int(parsed.timestamp())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected an ISO timestamp with timezone: {value}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Explicit local Parquet files or quoted file globs")
    parser.add_argument("--output", default="data/processed/askreddit_clean")
    parser.add_argument("--quarantine", default="data/processed/askreddit_quarantine")
    parser.add_argument("--reports-dir", default="reports/data_quality")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timestamp-unit", choices=["seconds", "milliseconds"], default="seconds")
    parser.add_argument("--min-created-utc", type=utc_seconds, default=1104537600)
    parser.add_argument("--max-created-utc", type=utc_seconds, default=int(datetime.now(timezone.utc).timestamp()))
    parser.add_argument("--driver-memory", default="2g")
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--write-partitions", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spark = None
    try:
        inputs = resolve_inputs(args.input)
        outputs = [resolve_output(p) for p in (args.output, args.quarantine, args.reports_dir)]
        guard_paths(inputs, outputs)
        policy = Policy(args.min_created_utc, args.max_created_utc, args.timestamp_unit)
        if args.shuffle_partitions < 1 or args.write_partitions < 1:
            raise ValueError("Partition counts must be positive")
        spark = create_spark(args.driver_memory, args.shuffle_partitions)
        report = run_cleaning(spark, inputs, *outputs, policy, validate_only=args.validate_only,
                              write_partitions=args.write_partitions)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except KeyboardInterrupt:
        LOG.error("Interrupted. Any partial outputs are uncertified; use fresh output/report paths.")
        return 130
    except Exception:
        LOG.exception("Validation/cleaning failed")
        return 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
