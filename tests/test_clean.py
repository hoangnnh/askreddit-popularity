"""Exercise cleaning with small local Parquet fixtures, never production extracts."""

import json

import duckdb
import pytest

from askreddit import clean


@pytest.fixture(scope="module")
def spark():
    session = clean.create_spark()
    yield session
    session.stop()


def write_fixture(path, rows):
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE records (
            id VARCHAR, subreddit VARCHAR, title VARCHAR, selftext VARCHAR,
            created_utc BIGINT, score BIGINT
        )""")
        if rows:
            con.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", rows)
        con.execute("COPY records TO ? (FORMAT PARQUET)", [str(path)])
    return path


@pytest.fixture
def inputs(tmp_path):
    t = 1735689600
    first = [
        ("empty", "AskReddit", "  Title stays  ", "", t, 1),
        ("nullbody", "askreddit", "Valid", None, t, None),
        ("whitebody", "ASKREDDIT", "Works", " \t\n ", t, -2),
        ("nulltitle", "AskReddit", None, "", t, 1),
        ("blanktitle", "AskReddit", "\t \n\u00a0", "[deleted]", t, 99),
        ("removed", "AskReddit", "Valid", " \t[removed]\n ", t, 10),
        ("dup", "AskReddit", "Later record", "", t + 100, 50),
        ("tie", "AskReddit", "Alpha", "", t + 50, 7),
    ]
    second = [
        ("deleted", "AskReddit", "Valid", "\n[deleted]\t", t, 11),
        ("dup", "AskReddit", "Earliest record", "body", t + 1, 3),
        ("mention", "AskReddit", "Valid", "[deleted] explanation", t, 9),
        ("tie", "AskReddit", "Beta", None, t + 50, 7),
        first[0],
        ("spaces", "AskReddit", "   ", None, t, 4),
        ("zero", "AskReddit", "A", "", t, 0),
    ]
    return [write_fixture(tmp_path / "one.parquet", first), write_fixture(tmp_path / "two.parquet", second)]


def test_fixture_end_to_end(spark, inputs, tmp_path):
    output = tmp_path / "cleaned"
    report = clean.run_cleaning(spark, inputs, output)
    assert report["input_row_count"] == 15
    assert report["output_row_count"] == 7
    assert report["removed_row_count"] == 8
    assert [step["removed"] for step in report["steps"]] == [0, 3, 2, 3]
    assert [step["remaining"] for step in report["steps"]] == [15, 12, 10, 7]
    assert report["input_scores"]["missing_score_count"] == 1
    assert report["input_scores"]["score_eq_1_share"] == pytest.approx(3 / 14)
    assert report["input_scores"]["median_score"] == 5.5
    assert report["input_scores"]["score_p75"] == 9.75
    assert report["output_scores"] == {
        "row_count": 7, "missing_score_count": 1, "known_score_count": 6,
        "score_eq_1_count": 1, "score_eq_1_share": 1 / 6,
        "median_score": 2.0, "score_p75": 6.0,
    }
    written = spark.read.parquet(str(output))
    assert written.dtypes == clean.SCHEMA
    rows = {row.id: row.asDict() for row in written.collect()}
    assert len(rows) == 7
    assert rows["empty"]["title"] == "  Title stays  "
    assert rows["empty"]["selftext"] == ""
    assert rows["nullbody"]["selftext"] is None and rows["nullbody"]["score"] is None
    assert rows["whitebody"]["selftext"] == " \t\n " and rows["whitebody"]["score"] == -2
    assert rows["dup"]["score"] == 3 and rows["dup"]["title"] == "Earliest record"
    assert rows["mention"]["selftext"] == "[deleted] explanation"
    assert rows["zero"]["score"] == 0
    assert rows["tie"]["title"] in {"Alpha", "Beta"}
    assert json.loads((output / "_cleaning_report.json").read_text()) == report
    assert not list(tmp_path.glob(".cleaned-*"))

    # Reordering files and shuffle partitions must not change duplicate selection.
    reversed_frame = spark.read.parquet(*map(str, reversed(inputs))).repartition(3)
    with clean.cleaned_records(reversed_frame) as (reversed_cleaned, repeated_report):
        assert {row.id: row.asDict() for row in reversed_cleaned.collect()} == rows
        assert repeated_report["output_scores"] == report["output_scores"]


def test_single_file_pilot(spark, inputs, tmp_path):
    report = clean.run_cleaning(spark, [inputs[0]], tmp_path / "single")
    assert report["input_row_count"] == 8
    assert report["output_row_count"] == 5
    assert [step["removed"] for step in report["steps"]] == [0, 2, 1, 0]


@pytest.mark.parametrize("subreddit,identifier", [("python", "a"), (None, "a"), ("AskReddit", None), ("AskReddit", " ")])
def test_rejects_foreign_subreddits_and_invalid_ids(spark, tmp_path, subreddit, identifier):
    path = write_fixture(tmp_path / "bad.parquet", [(identifier, subreddit, "Title", "", 1735689600, 1)])
    output = tmp_path / "bad-output"
    with pytest.raises(ValueError):
        clean.run_cleaning(spark, [path], output)
    assert not output.exists()


@pytest.mark.parametrize("rows", [[], [("a", "AskReddit", "Title", "", 1735689600, None)]])
def test_empty_and_all_missing_scores(spark, tmp_path, rows):
    path = write_fixture(tmp_path / "missing.parquet", rows)
    report = clean.run_cleaning(spark, [path], tmp_path / "missing-output")
    assert report["output_row_count"] == len(rows)
    stats = report["output_scores"]
    assert stats["known_score_count"] == 0
    assert stats["missing_score_count"] == len(rows)
    assert stats["score_eq_1_share"] is None
    assert stats["median_score"] is None and stats["score_p75"] is None


def test_rejects_schema_drift_in_later_file(spark, inputs, tmp_path):
    bad = tmp_path / "bad-schema.parquet"
    with duckdb.connect() as con:
        con.execute("COPY (SELECT id, subreddit, title, selftext, created_utc, score::VARCHAR AS score FROM read_parquet($input)) TO $output (FORMAT PARQUET)", {"input": str(inputs[1]), "output": str(bad)})
    with pytest.raises(ValueError, match="six ingestion columns"):
        clean.run_cleaning(spark, [inputs[0], bad], tmp_path / "bad-schema-output")


def test_explicit_paths_and_existing_output(inputs, tmp_path):
    assert clean.resolve_inputs([str(tmp_path / "*.parquet")]) == sorted(inputs)
    assert clean.resolve_inputs([str(inputs[0]), str(inputs[0])]) == [inputs[0]]
    for argument in ["hf://datasets/repo/data.parquet", str(tmp_path / "missing.parquet"), str(tmp_path / "absent-*.parquet")]:
        with pytest.raises(ValueError):
            clean.resolve_inputs([argument])
    with pytest.raises(ValueError, match="Output already exists"):
        clean.resolve_output(str(tmp_path))
    with pytest.raises(ValueError, match="local directory"):
        clean.resolve_output("s3://bucket/output")


def test_missing_month_fails_before_spark_startup(inputs, tmp_path, monkeypatch):
    def unexpected_start():
        pytest.fail("invalid paths must be rejected before starting Spark")
    monkeypatch.setattr(clean, "create_spark", unexpected_start)
    assert clean.main(["--input", str(inputs[0]), str(tmp_path / "missing-month.parquet"), "--output", str(tmp_path / "year")]) == 1
    assert not (tmp_path / "year").exists()
