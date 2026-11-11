"""Local Parquet integration tests; no remote dataset reads are needed."""

import json
import duckdb
import pytest

from askreddit import ingest


@pytest.fixture
def connection():
    with duckdb.connect() as connection:
        yield connection


@pytest.fixture
def config():
    return ingest.load_config(ingest.ROOT)


@pytest.fixture
def source(connection, tmp_path):
    start, end = ingest.month_bounds("2025-01")
    connection.execute(
        """CREATE TABLE source (
            id VARCHAR, subreddit VARCHAR, title VARCHAR, selftext VARCHAR,
            created_utc BIGINT, score BIGINT, ignored_column VARCHAR
        )"""
    )
    rows = [
        ("first", "AskReddit", "Question?", "", start, 0, "extra"),
        ("same", "askreddit", "Another?", None, start + 1, None, "extra"),
        ("same", "ASKREDDIT", "Last?", "[deleted]", end - 1, -5, "extra"),
        ("removed", "AskReddit", "Why?", "[removed]", start + 2, 42, "extra"),
        ("before", "AskReddit", "Old?", "", start - 1, 1, "extra"),
        ("after", "AskReddit", "New?", "", end, 1, "extra"),
        ("other", "python", "Other?", "", start, 1, "extra"),
        ("null_sub", None, "Unknown?", "", start, 1, "extra"),
        ("null_time", "AskReddit", "When?", "", None, 1, "extra"),
    ]
    connection.executemany("INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    path = tmp_path / "source.parquet"
    connection.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    return path


@pytest.fixture
def local_remote(monkeypatch, source):
    monkeypatch.setattr(ingest, "source_path", lambda *args: str(source))
    monkeypatch.setattr(ingest, "resolve_revision", lambda config: "a" * 40)


@pytest.mark.parametrize(
    ("month", "start", "end"),
    [("2025-01", 1735689600, 1738368000), ("2025-12", 1764547200, 1767225600)],
)
def test_utc_month_boundaries(month, start, end):
    assert ingest.month_bounds(month) == (start, end)
    leap_start, leap_end = ingest.month_bounds("2024-02")
    assert leap_end - leap_start == 29 * 86400


@pytest.mark.parametrize("month", ["2025-1", "2025-00", "2025-13", "2025/01", "0000-01"])
def test_invalid_months(month):
    with pytest.raises(ValueError):
        ingest.month_bounds(month)


def test_source_and_cli_selection(config):
    base = "hf://datasets/open-index/arctic/data/submissions/2025/01/"
    assert ingest.source_path(config, "2025-01") == base + "*.parquet"
    assert ingest.source_path(config, "2025-01", sample=True) == base + "000.parquet"
    assert ingest.parse_args(["--year", "2025"]).months == [f"2025-{m:02d}" for m in range(1, 13)]
    assert ingest.parse_args(["--month", "2025-01", "--sample"]).sample
    for args in ([], ["--year", "2025", "--sample"], ["--month", "2025-01", "--year", "2025"]):
        with pytest.raises(SystemExit):
            ingest.parse_args(args)


def test_copy_filters_and_preserves_values(connection, source, tmp_path):
    output = tmp_path / "output.parquet"
    copied = ingest.copy_month(connection, str(source), output, "2025-01")
    stats = ingest.validate_parquet(connection, output, "2025-01", expected_rows=copied)
    assert stats["row_count"] == 4
    assert stats["distinct_id_count"] == 3  # Ingestion must not silently deduplicate.
    actual = connection.execute(
        "SELECT subreddit, selftext, score FROM read_parquet(?) ORDER BY created_utc",
        [str(output)],
    ).fetchall()
    assert actual == [
        ("AskReddit", "", 0), ("askreddit", None, None),
        ("AskReddit", "[removed]", 42), ("ASKREDDIT", "[deleted]", -5),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE false",
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE id = 'before'",
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE id = 'after'",
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE id = 'other'",
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE id = 'null_sub'",
        "SELECT id, subreddit, title, selftext, created_utc, score FROM source WHERE id = 'null_time'",
        "SELECT subreddit, id, title, selftext, created_utc, score FROM source WHERE id = 'first'",
        "SELECT id, subreddit, title, selftext, created_utc, score::VARCHAR AS score FROM source WHERE id = 'first'",
    ],
)
def test_validation_rejects_bad_outputs(connection, source, tmp_path, query):
    output = tmp_path / "bad.parquet"
    connection.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output)])
    with pytest.raises(ValueError):
        ingest.validate_parquet(connection, output, "2025-01")


def test_copy_count_must_match(connection, source, tmp_path):
    output = tmp_path / "output.parquet"
    ingest.copy_month(connection, str(source), output, "2025-01")
    with pytest.raises(ValueError, match="row count"):
        ingest.validate_parquet(connection, output, "2025-01", expected_rows=5)


def test_manifest_and_validated_resume(connection, tmp_path, config, local_remote, monkeypatch):
    first = ingest.ingest_month(connection, tmp_path, config, "2025-01")
    assert not first["reused"]
    manifest_path = tmp_path / "artifacts/ingest_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    entry = json.loads(manifest_bytes)["months"]["2025-01"]
    assert entry["source_revision"] == "a" * 40
    assert entry["row_count"] == 4
    assert entry["elapsed_seconds"] >= 0
    assert entry["file_size_bytes"] == (tmp_path / entry["output_path"]).stat().st_size
    assert entry["source_path"].endswith("source.parquet")

    def unexpected_copy(*args):
        pytest.fail("a valid month with matching source must not be extracted again")

    monkeypatch.setattr(ingest, "copy_month", unexpected_copy)
    second = ingest.ingest_month(connection, tmp_path, config, "2025-01")
    assert second["reused"]
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("change", ["file", "checksum", "revision", "source", "manifest", "unverified"])
def test_changed_or_unverifiable_month_is_rebuilt(
    connection, tmp_path, config, local_remote, monkeypatch, change
):
    first = ingest.ingest_month(connection, tmp_path, config, "2025-01")
    path = tmp_path / first["output_path"]
    manifest_path = tmp_path / "artifacts/ingest_manifest.json"
    if change == "file":
        path.write_bytes(b"interrupted download")
    elif change == "revision":
        monkeypatch.setattr(ingest, "resolve_revision", lambda config: "b" * 40)
    elif change == "unverified":
        monkeypatch.setattr(ingest, "resolve_revision", lambda config: None)
    elif change == "manifest":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        key = "sha256" if change == "checksum" else "source_path"
        manifest["months"]["2025-01"][key] = "different"
        ingest.write_manifest(manifest_path, manifest)
    result = ingest.ingest_month(connection, tmp_path, config, "2025-01")
    assert not result["reused"]
    assert ingest.validate_parquet(connection, path, "2025-01")["row_count"] == 4


def test_failed_copy_preserves_previous_output_and_manifest(
    connection, tmp_path, config, local_remote, monkeypatch
):
    first = ingest.ingest_month(connection, tmp_path, config, "2025-01")
    output = tmp_path / first["output_path"]
    original = output.read_bytes()
    manifest_path = tmp_path / "artifacts/ingest_manifest.json"
    manifest = manifest_path.read_bytes()
    monkeypatch.setattr(ingest, "resolve_revision", lambda config: "b" * 40)

    def broken_copy(connection, source, destination, month):
        destination.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(ingest, "copy_month", broken_copy)
    with pytest.raises(KeyboardInterrupt):
        ingest.ingest_month(connection, tmp_path, config, "2025-01")
    assert output.read_bytes() == original
    assert manifest_path.read_bytes() == manifest
    assert not list(output.parent.glob(".*.tmp.parquet"))


def test_revision_change_during_copy_is_not_promoted(
    connection, tmp_path, config, local_remote, monkeypatch
):
    revisions = iter(["a" * 40, "b" * 40])
    monkeypatch.setattr(ingest, "resolve_revision", lambda config: next(revisions))
    with pytest.raises(ValueError, match="revision changed"):
        ingest.ingest_month(connection, tmp_path, config, "2025-01")
    assert not list((tmp_path / "data/extracted").glob("*.parquet"))
    assert not (tmp_path / "artifacts/ingest_manifest.json").exists()


def test_sample_does_not_mark_full_month_complete(connection, tmp_path, config, local_remote):
    result = ingest.ingest_month(connection, tmp_path, config, "2025-01", sample=True)
    assert result["output_path"] == "data/samples/askreddit-2025-01-sample.parquet"
    assert (tmp_path / result["output_path"]).is_file()
    assert not (tmp_path / "artifacts/ingest_manifest.json").exists()


def test_malformed_manifest_is_not_discarded(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version": 1, "months": []}')
    with pytest.raises(ValueError, match="manifest"):
        ingest.read_manifest(manifest)


def test_year_runs_in_order_and_stops_on_failure(tmp_path, config, monkeypatch, connection):
    seen = []
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    monkeypatch.setattr(ingest, "load_config", lambda root: config)
    monkeypatch.setattr(ingest, "connect", lambda root: connection)

    def run(connection, root, config, month, sample):
        seen.append(month)
        if month == "2025-03":
            raise ValueError("simulated remote failure")
        return {"month": month}

    monkeypatch.setattr(ingest, "ingest_month", run)
    assert ingest.main(["--year", "2025"]) == 1
    assert seen == ["2025-01", "2025-02", "2025-03"]
