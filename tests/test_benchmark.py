"""Benchmark workloads must be portable SQL that returns identical answers on both engines."""

from decimal import Decimal

import duckdb
import pytest

from askreddit.benchmark import load_queries, normalize_rows, render, rows_match, run_benchmark

TITLES = ["What is your favorite food?", "Why do cats purr?", "How do you relax?", "Best advice you got?", "what now"]


def write_fixture(path):
    path.mkdir()
    with duckdb.connect(config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}) as connection:
        connection.execute(f"""
            COPY (
              SELECT 't' || i AS id, 'AskReddit' AS subreddit,
                     list_extract(?, (i % 5) + 1) AS title, '' AS selftext,
                     1735689600 + i * 63000 AS created_utc, CAST((i * 37) % 300 AS BIGINT) AS score,
                     year(to_timestamp(1735689600 + i * 63000) AT TIME ZONE 'UTC') AS year,
                     month(to_timestamp(1735689600 + i * 63000) AT TIME ZONE 'UTC') AS month
              FROM range(503) t(i)
            ) TO '{path}' (FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE)""", [TITLES])
    (path / "_SUCCESS").touch()


def test_row_comparison_and_normalization():
    assert normalize_rows([(Decimal("0.5"), 1, "a")]) == [(0.5, 1, "a")]
    assert rows_match([(1, 0.1 + 0.2)], [(1, 0.3)])
    assert not rows_match([(1, 0.3)], [(2, 0.3)])
    assert not rows_match([(1, 0.3)], [(1, 0.31)])
    assert not rows_match([(1,)], [(1,), (2,)])
    assert len(load_queries()) == 8


def test_both_engines_agree_on_every_workload(tmp_path):
    source = tmp_path / "clean"
    write_fixture(source)
    results = run_benchmark(source, tmp_path / "report", [1], repeats=1, warmup=0, spark_memory="1g", duckdb_memory="512MB")
    assert results["all_results_match"], [r["query"] for r in results["summary"] if not r["results_match"]]
    assert results["result_rows"]["q1_scan_summary"][0][0] == 503
    assert {r["query"] for r in results["summary"]} == set(load_queries())
    assert all(r["spark_median_seconds"] > 0 and r["duckdb_median_seconds"] > 0 for r in results["summary"])
    assert sum(row[2] for row in results["result_rows"]["q4_label_distribution"]) == 503
    paths = render(tmp_path / "report")
    assert all(p.exists() for p in paths)
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark(source, tmp_path / "report", [1], 1, 0, "1g", "512MB")
