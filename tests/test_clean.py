"""Local synthetic fixtures only; test auditability, invariants and repeatability."""

import json

import pytest
from pyspark.sql import functions as F

from askreddit import clean
from askreddit.quality import duplicate_summary, profile_records

T = 1735689600
POLICY = clean.Policy(max_created_utc=1795046400)
SCHEMA = "id string, subreddit string, title string, selftext string, created_utc long, score long"
STRING_SCHEMA = "id string, subreddit string, title string, selftext string, created_utc string, score string"


@pytest.fixture(scope="module")
def spark():
    session = clean.create_spark(driver_memory="1g", shuffle_partitions=2)
    yield session
    session.stop()


def row(identifier="a1", title="Question?", body="", timestamp=T, score=1, subreddit="AskReddit"):
    return identifier, subreddit, title, body, timestamp, score


def run_fixture(spark, tmp_path, rows, schema=SCHEMA):
    source = tmp_path / "source"
    spark.createDataFrame(rows, schema).coalesce(1).write.parquet(str(source))
    inputs = sorted(source.glob("*.parquet"))
    output, quarantine, reports = [tmp_path / name for name in ("cleaned", "quarantine", "reports")]
    report = clean.run_cleaning(spark, inputs, output, quarantine, reports, POLICY, write_partitions=1)
    return report, spark.read.parquet(str(output)), spark.read.parquet(str(quarantine)), reports


def test_end_to_end_quality_and_audit(spark, tmp_path):
    exact = row("same", title="  Keep   emojis 😀?!  ", body=None, score=0)
    rows = [
        exact, exact,
        row("conflict", title="Earlier", score=-2),
        row("conflict", title="Later", timestamp=T + 10, score=999),
        row("removed", body=" \t[removed]\n "), row("deleted", body="[deleted]"),
        row("mention", body="[deleted] explanation"),
        row("nulltitle", title=None), row("emptytitle", title=""),
        row("whitetitle", title="\t \u00a0\n"),
        row("outside", subreddit="python"), row("prefix", subreddit=" R/ASKREDDIT "),
        row(None), row("  "), row("bad-id"), row(" trimmed "),
        row("nulltime", timestamp=None), row("future", timestamp=9999999999999),
        row("nullscore", score=None),
        row("invalidboth", title=None, score=None),
    ]
    report, written, audit, reports = run_fixture(spark, tmp_path, rows)
    assert report["total_input_records"] == 20
    assert report["total_cleaned_records"] == 7
    assert report["total_quarantined_records"] == 13
    assert report["total_rejected_records"] == 11
    assert report["total_duplicate_records_removed"] == 2
    assert all(report["invariants"].values())
    actual = {r.id: r.asDict() for r in written.collect()}
    assert actual["same"]["title"] == "Keep emojis 😀?!"
    assert actual["same"]["selftext"] == ""
    assert actual["same"]["score"] == 0
    assert actual["conflict"]["title"] == "Earlier" and actual["conflict"]["score"] == -2
    assert actual["removed"]["selftext"] == actual["deleted"]["selftext"] == ""
    assert actual["mention"]["selftext"] == "[deleted] explanation"
    assert actual["prefix"]["subreddit"] == "AskReddit"
    assert "trimmed" in actual
    assert all(r["year"] == 2025 and r["month"] == 1 for r in actual.values())
    rejected = {r.raw_id: r.asDict() for r in audit.collect()}
    assert rejected["future"]["raw_created_utc"] == "9999999999999"
    assert rejected["invalidboth"]["rejection_reasons"] == ["invalid_title", "missing_score"]
    assert rejected["same"]["primary_reason"] == "duplicate_exact"
    assert rejected["conflict"]["primary_reason"] == "duplicate_conflict"
    assert set(rejected["conflict"]["duplicate_conflicting_fields"]) == {"title", "created_utc", "score"}
    assert len(rejected["conflict"]["retained_original_sha256"]) == 64
    assert json.loads(rejected["conflict"]["_raw_json"])["score"] == 999
    before = json.loads((reports / "before_cleaning/profile.json").read_text())
    assert before["duplicates"]["duplicate_ids"] == 2
    assert before["duplicates"]["exact_duplicate_excess"] == 1
    assert before["duplicates"]["conflicting_score_ids"] == 1
    assert before["nulls"]["title"] == {"count": 2, "percentage": 10.0}
    assert before["empty_or_whitespace_titles"] == 2
    assert before["removed_selftext"] == before["deleted_selftext"] == 1
    assert json.loads((reports / "run.json").read_text()) == report
    assert len(list((tmp_path / "cleaned").rglob("*.parquet"))) == 1


def test_deterministic_duplicates_and_conflicts(spark):
    rows = [row("same", score=0), row("same", score=100), row("same", score=-5),
            row("body", body=None), row("body", body="[removed]"), row("body", body="text"),
            row(" tie ", title="Alpha"), row("tie", title="Beta"), row("same", score=0)]
    results = []
    for ordered, partitions in [(rows, 1), (list(reversed(rows)), 3)]:
        prepared = clean.prepare_records(spark.createDataFrame(ordered, SCHEMA).repartition(partitions), POLICY)
        cleaned, rejected = clean.split_records(prepared)
        results.append((sorted(tuple(r) for r in cleaned.collect()),
                        sorted((r.raw_id, r.raw_score, r.primary_reason, r.retained_original_sha256) for r in rejected.collect())))
    assert results[0] == results[1]
    assert len(results[0][0]) == 3


def test_integer_parsing_and_timestamp_units(spark):
    rows = [row("negative", timestamp=str(T), score=" -2 "), row("zero", timestamp=str(T), score="0"),
            row("nullscore", timestamp=str(T), score=None), row("nonnumeric", timestamp=str(T), score="NaN"),
            row("fraction", timestamp=str(T), score="1.5"), row("overflow", timestamp=str(T), score=str(2**63)),
            row("badtime", timestamp="not-a-date", score="1"), row("milliseconds", timestamp=str(T * 1000), score="1"),
            row("fractionaltime", timestamp=str(T) + ".5", score="1"), row("ancient", timestamp="1", score="1")]
    prepared = clean.prepare_records(spark.createDataFrame(rows, STRING_SCHEMA), POLICY)
    cleaned, rejected = clean.split_records(prepared)
    assert {r.id: r.score for r in cleaned.collect()} == {"negative": -2, "zero": 0}
    bad = {r.raw_id: r for r in rejected.collect()}
    assert bad["milliseconds"].primary_reason == "suspicious_timestamp"
    assert bad["badtime"].primary_reason == "invalid_timestamp"
    assert bad["overflow"].raw_score == str(2**63)
    stats = profile_records(prepared)
    assert stats["invalid_scores"] == 3 and stats["missing_scores"] == 1
    ms = clean.prepare_records(spark.createDataFrame([
        row("whole", timestamp=str(T * 1000), score="1"),
        row("fractional", timestamp=str(T * 1000 + 1), score="1"),
    ], STRING_SCHEMA), clean.Policy(max_created_utc=POLICY.max_created_utc, timestamp_unit="milliseconds"))
    kept, rejected = clean.split_records(ms)
    assert kept.first().created_utc == T
    assert rejected.first().primary_reason == "subsecond_timestamp"


@pytest.mark.parametrize("rows", [[], [row("a1", score=None)]])
def test_empty_output_is_readable_and_reconciles(spark, tmp_path, rows):
    report, written, audit, _ = run_fixture(spark, tmp_path, rows)
    assert written.count() == 0
    assert report["total_input_records"] == audit.count() == len(rows)
    assert report["total_cleaned_records"] == 0
    assert report["min_score"] is None and report["min_created_utc"] is None
    assert all(report["invariants"].values())
    assert report["retained_percentage"] == (0.0 if rows else None)


def test_schema_drift_is_reported_and_safely_read(spark, tmp_path):
    integer_source, string_source = tmp_path / "ints", tmp_path / "strings"
    spark.createDataFrame([row("integer")], SCHEMA).write.parquet(str(integer_source))
    spark.createDataFrame([row("text", timestamp=str(T), score="-3")], STRING_SCHEMA).write.parquet(str(string_source))
    files = sorted(integer_source.glob("*.parquet")) + sorted(string_source.glob("*.parquet"))
    report = clean.run_cleaning(spark, files, tmp_path / "out", tmp_path / "bad", tmp_path / "reports", POLICY)
    assert report["total_cleaned_records"] == 2
    schema = json.loads((tmp_path / "reports/before_cleaning/schema.json").read_text())
    assert schema["schemas_differ"] and schema["distinct_schema_count"] == 2


def test_incorrect_and_corrupt_schemas_fail_with_file_report(spark, tmp_path):
    source = tmp_path / "missing"
    spark.createDataFrame([row()], SCHEMA).drop("score").write.parquet(str(source))
    corrupt = tmp_path / "zero.parquet"
    corrupt.touch()
    files = sorted(source.glob("*.parquet")) + [corrupt]
    with pytest.raises(ValueError, match="Problematic files"):
        clean.run_cleaning(spark, files, tmp_path / "out", tmp_path / "bad", tmp_path / "reports", POLICY)
    report = json.loads((tmp_path / "reports/before_cleaning/schema.json").read_text())
    assert all(f["status"] == "error" for f in report["files"])
    assert (tmp_path / "reports/failure.json").exists()
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="Unsupported type"):
        clean.validate_schema(spark.createDataFrame([row()], SCHEMA).withColumn("score", F.col("score").cast("double")))


def test_validation_only_and_output_guards(spark, tmp_path):
    source = tmp_path / "source"
    spark.createDataFrame([row()], SCHEMA).write.parquet(str(source))
    files = sorted(source.glob("*.parquet"))
    before = [clean.file_fingerprint(p) for p in files]
    output, quarantine, reports = [tmp_path / name for name in ("out", "bad", "reports")]
    result = clean.run_cleaning(spark, files, output, quarantine, reports, POLICY, validate_only=True)
    assert result["status"] == "validated_only"
    assert not output.exists() and not quarantine.exists()
    assert before == [clean.file_fingerprint(p) for p in files]
    with pytest.raises(ValueError, match="already exists"):
        clean.run_cleaning(spark, files, output, quarantine, reports, POLICY)
    with pytest.raises(ValueError, match="overlap"):
        clean.guard_paths(files, [output, output / "nested"])
    with pytest.raises(ValueError):
        clean.guard_paths(files, [source])
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        clean.guard_paths(files, [alias])


def test_input_resolution_and_missing_input_before_startup(tmp_path, monkeypatch):
    source = tmp_path / "existing.parquet"
    source.touch()
    hidden = tmp_path / ".temp.parquet"
    hidden.touch()
    assert clean.resolve_inputs([str(source), str(source)]) == [source]
    assert clean.resolve_inputs([str(tmp_path / "*.parquet")]) == sorted([source, hidden])
    for path in ["https://example.com/data.parquet", str(tmp_path / "absent.parquet")]:
        with pytest.raises(ValueError):
            clean.resolve_inputs([path])
    monkeypatch.setattr(clean, "create_spark", lambda *a: pytest.fail("must reject paths before starting Spark"))
    assert clean.main(["--input", str(tmp_path / "absent.parquet")]) == 1
    with pytest.raises(ValueError):
        clean.resolve_output("s3://bucket/output")


def test_utc_partition_boundary_and_null_conflicts(spark):
    rows = [row("jan", timestamp=T - 1), row("feb", timestamp=T),
            row("body", body=None), row("body", body="[deleted]")]
    prepared = clean.prepare_records(spark.createDataFrame(rows, SCHEMA), POLICY)
    groups = duplicate_summary(clean.duplicate_groups(prepared, raw=True))
    assert groups["conflicting_selftext_ids"] == 1
    kept, _ = clean.split_records(prepared)
    dates = {r.id: (r.year, r.month) for r in kept.collect()}
    assert dates["jan"] == (2024, 12) and dates["feb"] == (2025, 1)
