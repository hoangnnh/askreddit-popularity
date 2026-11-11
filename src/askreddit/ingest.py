"""Extract and validate monthly AskReddit Parquet files with DuckDB/httpfs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile
from time import perf_counter
import tomllib
from urllib.parse import quote
from urllib.request import urlopen

import duckdb


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)
COLUMNS = ("id", "subreddit", "title", "selftext", "created_utc", "score")
SCHEMA = list(zip(COLUMNS, ("VARCHAR",) * 4 + ("BIGINT", "BIGINT")))
MANIFEST_VERSION = 1


def month_bounds(month: str) -> tuple[int, int]:
    """Return inclusive/exclusive Unix seconds, independent of host timezone."""
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month):
        raise ValueError("month must have the form YYYY-MM")
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=timezone.utc)
    end = datetime(year + (number == 12), number % 12 + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def load_config(root: Path) -> dict:
    with (root / "configs/project.toml").open("rb") as handle:
        config = tomllib.load(handle)
    selection = config["selection"]
    if (selection["subreddit"].lower() != "askreddit"
            or selection["timezone"] != "UTC"
            or tuple(selection["columns"]) != COLUMNS):
        raise ValueError("config must select AskReddit, UTC, and the six output columns")
    if config["output"]["format"] != "parquet":
        raise ValueError("output format must be parquet")
    return config


def source_path(config: dict, month: str, sample: bool = False) -> str:
    year, number = map(int, month.split("-"))
    source = config["source"]
    path = source["monthly_path_template"].format(year=year, month=number)
    if sample:
        if not path.endswith("/*.parquet"):
            raise ValueError("sample mode requires a monthly /*.parquet source pattern")
        path = path.removesuffix("*.parquet") + "000.parquet"
    revision = source["revision"]
    suffix = "" if revision == "main" else f"@{revision}"
    return f"hf://datasets/{source['dataset_id']}{suffix}/{path}"


def resolve_revision(config: dict) -> str | None:
    """Verify the configured Hugging Face revision; never infer a commit ID."""
    source = config["source"]
    dataset = quote(source["dataset_id"], safe="/")
    revision = quote(source["revision"], safe="")
    url = f"https://huggingface.co/api/datasets/{dataset}/revision/{revision}"
    try:
        with urlopen(url, timeout=20) as response:
            sha = json.load(response).get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            raise ValueError("dataset API did not return a commit ID")
        return sha
    except (OSError, ValueError) as error:
        LOG.warning("Source revision could not be verified: %s", error)
        return None


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    if duckdb.__version__ != "1.5.4":
        raise RuntimeError("Install the project dependencies: DuckDB 1.5.4 is required")
    extensions = root / "artifacts/duckdb_extensions"
    extensions.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(config={
        "extension_directory": str(extensions),
        "threads": 4,
        "memory_limit": "2GB",
    })
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute("INSTALL httpfs FROM 'https://extensions.duckdb.org'")
        connection.execute("LOAD httpfs")
        # Bound concurrent reads and allow slow transfers to finish before retry.
        # httplib returned HTTP 0 for this mirror; retain the curl backend.
        connection.execute("SET httpfs_client_implementation = 'curl'")
        connection.execute("SET http_timeout = 120")
        connection.execute("SET http_retries = 5")
    except BaseException:
        connection.close()
        raise
    return connection


def validate_parquet(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    month: str,
    expected_rows: int | None = None,
) -> dict:
    """Read the file back and reject incomplete or out-of-scope extracts."""
    start, end = month_bounds(month)
    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    if [(row[0], row[1]) for row in schema] != SCHEMA:
        raise ValueError(f"Unexpected column names, order, or types in {path}")
    rows, distinct_ids, minimum, maximum, bad_subreddit, bad_time = connection.execute(
        """
        SELECT count(*), count(DISTINCT id), min(created_utc), max(created_utc),
               count(*) FILTER (WHERE subreddit IS NULL
                                OR lower(subreddit) <> 'askreddit'),
               count(*) FILTER (WHERE created_utc IS NULL
                                OR created_utc < ? OR created_utc >= ?)
        FROM read_parquet(?)
        """,
        [start, end, str(path)],
    ).fetchone()
    if rows == 0 or (expected_rows is not None and rows != expected_rows):
        raise ValueError(f"Invalid row count in {path}: {rows}, expected {expected_rows}")
    if bad_subreddit or bad_time:
        raise ValueError(f"Invalid subreddit or UTC range in {path}")
    compression = connection.execute(
        "SELECT DISTINCT compression FROM parquet_metadata(?)", [str(path)]
    ).fetchall()
    if compression != [("ZSTD",)]:
        raise ValueError(f"Expected ZSTD compression in {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "row_count": rows,
        "distinct_id_count": distinct_ids,
        "file_size_bytes": path.stat().st_size,
        "min_created_utc": minimum,
        "max_created_utc": maximum,
        "sha256": digest,
    }


def copy_month(
    connection: duckdb.DuckDBPyConnection, source: str, destination: Path, month: str
) -> int:
    start, end = month_bounds(month)
    # No casts, score transformation, selftext cleaning, or deduplication here.
    return connection.execute(
        """
        COPY (
            SELECT id, subreddit, title, selftext, created_utc, score
            FROM read_parquet($source)
            WHERE lower(subreddit) = 'askreddit'
              AND created_utc >= $start AND created_utc < $end
        ) TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        {"source": source, "start": start, "end": end, "destination": str(destination)},
    ).fetchone()[0]


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "months": {}}
    with path.open() as handle:
        manifest = json.load(handle)
    if (not isinstance(manifest, dict)
            or manifest.get("version") != MANIFEST_VERSION
            or not isinstance(manifest.get("months"), dict)
            or any(not isinstance(entry, dict) for entry in manifest["months"].values())):
        raise ValueError(f"Unsupported or invalid manifest: {path}")
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_month(
    connection: duckdb.DuckDBPyConnection,
    root: Path,
    config: dict,
    month: str,
    sample: bool = False,
) -> dict:
    started = perf_counter()
    year, number = map(int, month.split("-"))
    template = "sample_path_template" if sample else "monthly_path_template"
    relative = config["output"][template].format(year=year, month=number)
    destination = root / relative
    source = source_path(config, month, sample)
    manifest_path = root / config["paths"]["ingest_manifest"]
    manifest = read_manifest(manifest_path) if not sample else None
    revision_before = resolve_revision(config)

    if not sample and destination.is_file():
        entry = manifest["months"].get(month, {})
        matches_source = (
            entry.get("month") == month
            and entry.get("source_path") == source
            and entry.get("output_path") == relative
            and revision_before is not None
            and entry.get("source_revision") == revision_before
        )
        if matches_source:
            try:
                statistics = validate_parquet(
                    connection, destination, month, expected_rows=entry.get("row_count")
                )
                if any(entry.get(key) != value for key, value in statistics.items()):
                    raise ValueError("file statistics or checksum differ from the manifest")
            except (OSError, ValueError, duckdb.Error) as error:
                LOG.warning("Re-extracting %s: %s", month, error)
            else:
                LOG.info("Reusing validated month %s (%s rows)", month, statistics["row_count"])
                return {**entry, "reused": True}
        else:
            LOG.info("Re-extracting %s: no matching, verified source record", month)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.stem}.",
        suffix=".tmp.parquet", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        LOG.info("Reading %s", source)
        copied_rows = copy_month(connection, source, temporary, month)
        statistics = validate_parquet(connection, temporary, month, expected_rows=copied_rows)
        revision_after = resolve_revision(config)
        if revision_before and revision_after and revision_before != revision_after:
            raise ValueError("Source revision changed during extraction; rerun this month")
        verified_revision = revision_before if revision_before == revision_after else None
        entry = {
            "month": month,
            "source_path": source,
            "source_revision": verified_revision,
            "output_path": relative,
            **statistics,
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
        # Both files are on the same filesystem; a failed copy leaves the old final intact.
        temporary.replace(destination)
        if not sample:
            manifest["months"][month] = entry
            write_manifest(manifest_path, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("Wrote %s (%s rows)", relative, statistics["row_count"])
    return {**entry, "reused": False}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--month", help="UTC month in YYYY-MM form")
    period.add_argument("--year", type=int, help="process all 12 UTC months sequentially")
    parser.add_argument("--sample", action="store_true", help="read only 000.parquet (requires --month)")
    args = parser.parse_args(argv)
    if args.sample and not args.month:
        parser.error("--sample requires --month")
    if args.year is not None and not 1 <= args.year <= 9998:
        parser.error("--year must be between 1 and 9998")
    try:
        args.months = [args.month] if args.month else [f"{args.year:04d}-{m:02d}" for m in range(1, 13)]
        for month in args.months:
            month_bounds(month)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(ROOT)
        selection = config["selection"]
        for month in args.months:
            start, end = month_bounds(month)
            if (start < selection["start_utc"].timestamp()
                    or end > selection["end_utc_exclusive"].timestamp()):
                raise ValueError(f"{month} is outside the configured UTC date range")
        # One process owns the manifest for the duration of a run (WSL/Linux).
        import fcntl

        lock_path = ROOT / config["paths"]["artifacts_dir"] / "ingest.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with connect(ROOT) as connection:
                for month in args.months:
                    result = ingest_month(connection, ROOT, config, month, args.sample)
                    print(json.dumps(result, sort_keys=True), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, duckdb.Error) as error:
        LOG.error("Ingestion failed: %s", error)
        return 1
    except KeyboardInterrupt:
        LOG.error("Ingestion interrupted; rerun to resume completed months")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
