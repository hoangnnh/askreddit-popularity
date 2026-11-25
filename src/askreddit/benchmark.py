"""Spark SQL versus DuckDB: identical SQL text over the same cleaned Parquet dataset.

Each workload in ``benchmarks/queries/*.sql`` runs unchanged on both engines
against a view named ``posts``. Neither engine caches data or results; both
re-read Parquet for every run (the OS page cache is warm after the warm-up run).
Thread counts are matched per configuration. Timings are wall-clock seconds
around query planning, execution and fetching the (small) result into Python.
Results from both engines must agree before timings are reported.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

from askreddit.clean import write_json

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
QUERY_DIR = ROOT / "benchmarks" / "queries"
ENGINES = ("spark", "duckdb")


def load_queries(directory: Path = QUERY_DIR) -> dict[str, str]:
    queries = {path.stem: path.read_text() for path in sorted(directory.glob("*.sql"))}
    if not queries:
        raise ValueError(f"No SQL workloads found in {directory}")
    return queries


def normalize_value(value):
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalars
        return value.item()
    return value


def normalize_rows(rows) -> list[tuple]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def rows_match(left, right, rel_tol=1e-9, abs_tol=1e-9) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, float) or isinstance(y, float):
                if x is None or y is None or not math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=abs_tol):
                    return False
            elif x != y:
                return False
    return True


def time_runs(function, repeats: int, warmup: int):
    result = None
    for _ in range(warmup):
        result = function()
    times = []
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        times.append(perf_counter() - started)
    return result, times


def spark_session(threads: int, memory: str):
    from pyspark.sql import SparkSession
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (SparkSession.builder.appName("askreddit-benchmark").master(f"local[{threads}]")
            .config("spark.driver.memory", memory)
            .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "UTC").config("spark.sql.ansi.enabled", "true")
            .config("spark.sql.shuffle.partitions", str(4 * threads))
            .config("spark.sql.adaptive.enabled", "true").getOrCreate())


def run_spark(queries, input_path: Path, threads: int, memory: str, repeats: int, warmup: int) -> dict:
    started = perf_counter()
    spark = spark_session(threads, memory)
    spark.sparkContext.setLogLevel("WARN")
    try:
        spark.read.parquet(str(input_path)).createOrReplaceTempView("posts")
        startup = perf_counter() - started
        version = spark.version
        results = {}
        for name, sql in queries.items():
            rows, times = time_runs(lambda: normalize_rows(spark.sql(sql).collect()), repeats, warmup)
            results[name] = {"rows": rows, "seconds": times}
            LOG.info("spark threads=%d %s median %.3f s", threads, name, statistics.median(times))
        return {"engine": "spark", "version": version, "startup_seconds": startup, "queries": results,
                "settings": {"master": f"local[{threads}]", "driver_memory": memory, "shuffle_partitions": 4 * threads,
                             "adaptive_query_execution": True, "caching": "none"}}
    finally:
        spark.stop()


def run_duckdb(queries, input_path: Path, threads: int, memory: str, repeats: int, warmup: int) -> dict:
    import duckdb
    started = perf_counter()
    connection = duckdb.connect(config={"threads": threads, "memory_limit": memory,
                                        "autoinstall_known_extensions": False, "autoload_known_extensions": False})
    try:
        pattern = str(input_path / "**" / "*.parquet").replace("'", "''")
        connection.execute(f"CREATE VIEW posts AS SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)")
        startup = perf_counter() - started
        results = {}
        for name, sql in queries.items():
            rows, times = time_runs(lambda: normalize_rows(connection.execute(sql).fetchall()), repeats, warmup)
            results[name] = {"rows": rows, "seconds": times}
            LOG.info("duckdb threads=%d %s median %.3f s", threads, name, statistics.median(times))
        return {"engine": "duckdb", "version": duckdb.__version__, "startup_seconds": startup, "queries": results,
                "settings": {"threads": threads, "memory_limit": memory, "caching": "none (view over read_parquet)"}}
    finally:
        connection.close()


def environment(input_path: Path) -> dict:
    def first(path, key):
        with suppress(OSError):
            for line in Path(path).read_text().splitlines():
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
        return None
    java = None
    with suppress(Exception):
        home = os.environ.get("JAVA_HOME")
        binary = str(Path(home) / "bin" / "java") if home else "java"
        java = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=30).stderr.splitlines()[0]
    files = sorted(input_path.rglob("*.parquet"))
    return {"cpu": first("/proc/cpuinfo", "model name"), "logical_cpus": os.cpu_count(),
            "memory_total": first("/proc/meminfo", "MemTotal"), "platform": platform.platform(),
            "python": platform.python_version(), "java": java, "input": str(input_path),
            "input_files": len(files), "input_bytes": sum(p.stat().st_size for p in files)}


def anchor_checks(results: dict) -> list[dict]:
    """Cross-check engine answers against independently produced project reports when they exist."""
    checks = []
    distribution = ROOT / "reports/experiments/fang_chua_20261121_v1/class_distributions.json"
    rows = results.get("q4_label_distribution")
    if distribution.exists() and rows is not None:
        expected = json.loads(distribution.read_text())
        actual = {(split, label): count for split, label, count in rows}
        ok = all(actual.get((split, r["label"]), 0) == r["count"] for split, d in expected.items() for r in d["labels"])
        checks.append({"query": "q4_label_distribution", "reference": str(distribution.relative_to(ROOT)), "matches": ok})
    return checks


def write_csv(path: Path, runs: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threads", "engine", "query", "run", "seconds"])
        for run in runs:
            for engine in ENGINES:
                for query, result in run[engine]["queries"].items():
                    for index, seconds in enumerate(result["seconds"], 1):
                        writer.writerow([run["threads"], engine, query, index, f"{seconds:.6f}"])


def summarize(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        for query in run["spark"]["queries"]:
            spark, duck = run["spark"]["queries"][query]["seconds"], run["duckdb"]["queries"][query]["seconds"]
            rows.append({"threads": run["threads"], "query": query,
                         "spark_median_seconds": statistics.median(spark), "spark_min_seconds": min(spark), "spark_max_seconds": max(spark),
                         "duckdb_median_seconds": statistics.median(duck), "duckdb_min_seconds": min(duck), "duckdb_max_seconds": max(duck),
                         "spark_to_duckdb_ratio": statistics.median(spark) / statistics.median(duck),
                         "results_match": run["agreement"][query]})
    return rows


def run_benchmark(input_path: Path, output: Path, threads: list[int], repeats: int, warmup: int,
                  spark_memory: str, duckdb_memory: str) -> dict:
    if not (input_path / "_SUCCESS").is_file():
        raise ValueError("A completed local cleaned Parquet dataset is required")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new run ID: {output}")
    queries = load_queries()
    output.mkdir(parents=True)
    runs = []
    for count in threads:
        spark = run_spark(queries, input_path, count, spark_memory, repeats, warmup)
        duck = run_duckdb(queries, input_path, count, duckdb_memory, repeats, warmup)
        agreement = {q: rows_match(spark["queries"][q]["rows"], duck["queries"][q]["rows"]) for q in queries}
        runs.append({"threads": count, "spark": spark, "duckdb": duck, "agreement": agreement})
    reference = {q: r["rows"] for q, r in runs[0]["duckdb"]["queries"].items()}
    summary = summarize(runs)
    results = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(input_path),
        "method": {"repeats": repeats, "warmup_runs": warmup, "timer": "time.perf_counter wall clock",
                   "scope": "Query planning + execution + fetching the result rows into Python; excludes engine startup.",
                   "caching": "No engine data/result caching; Parquet re-read every run; OS page cache warm after warm-up.",
                   "equivalence": "Identical SQL text from benchmarks/queries on both engines; float results compared with rel_tol 1e-9."},
        "queries": {q: queries[q] for q in queries},
        "all_results_match": all(all(r["agreement"].values()) for r in runs),
        "anchor_checks": anchor_checks(reference),
        "startup_seconds": [{"threads": r["threads"], "spark": r["spark"]["startup_seconds"], "duckdb": r["duckdb"]["startup_seconds"]} for r in runs],
        "engines": {"spark": runs[0]["spark"]["version"], "duckdb": runs[0]["duckdb"]["version"]},
        "settings": [{"threads": r["threads"], "spark": r["spark"]["settings"], "duckdb": r["duckdb"]["settings"]} for r in runs],
        "summary": summary,
        "result_rows": reference,
    }
    write_json(output / "results.json", results)
    write_csv(output / "timings.csv", runs)
    return results


def render(output: Path) -> list[Path]:
    results = json.loads((output / "results.json").read_text())
    from askreddit.eda_report import BLUE, INK, MUTED, ORANGE, pyplot, save, table
    plt = pyplot()
    configs = sorted({r["threads"] for r in results["summary"]})
    fig, axes = plt.subplots(1, len(configs), figsize=(6.2 * len(configs), 5.2), sharey=True, constrained_layout=True, squeeze=False)
    for axis, threads in zip(axes[0], configs):
        rows = [r for r in results["summary"] if r["threads"] == threads][::-1]
        positions = range(len(rows))
        axis.barh([p + 0.2 for p in positions], [r["spark_median_seconds"] for r in rows], height=0.38, color=BLUE, label="Spark SQL")
        axis.barh([p - 0.2 for p in positions], [r["duckdb_median_seconds"] for r in rows], height=0.38, color=ORANGE, label="DuckDB")
        axis.set_yticks(list(positions), [r["query"] for r in rows])
        axis.set_xscale("log")
        axis.grid(axis="x", visible=True)
        axis.grid(axis="y", visible=False)
        axis.set(title=f"Median query time, {threads} threads", xlabel="Seconds (log scale)")
        for p, r in zip(positions, rows):
            axis.text(r["spark_median_seconds"], p + 0.2, f" {r['spark_median_seconds']:.2f}", va="center", fontsize=7.5, color=INK)
            axis.text(r["duckdb_median_seconds"], p - 0.2, f" {r['duckdb_median_seconds']:.3f}", va="center", fontsize=7.5, color=MUTED)
        from matplotlib.ticker import FuncFormatter
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.legend(loc="upper right")
    (output / "figures").mkdir(exist_ok=True)
    figure = save(fig, output / "figures" / "median_times.png")
    env = results["environment"]
    lines = [
        "# Benchmark Spark SQL và DuckDB", "",
        "Hai engine chạy **cùng một đoạn SQL** (thư mục `benchmarks/queries/`) trên **cùng bộ Parquet đã làm sạch**, với số luồng "
        "bằng nhau trong từng cấu hình. Không engine nào cache dữ liệu hay kết quả; mỗi lần chạy đều đọc lại Parquet "
        f"(page cache của hệ điều hành đã ấm sau {results['method']['warmup_runs']} lần chạy khởi động). Mỗi truy vấn được đo "
        f"{results['method']['repeats']} lần; bảng báo cáo median. Thời gian gồm lập kế hoạch, thực thi và lấy kết quả về Python; "
        "không gồm thời gian khởi động engine.", "",
        f"**Kết quả hai engine khớp nhau ở mọi truy vấn: {'có' if results['all_results_match'] else 'KHÔNG'}.** "
        + " ".join(f"Đối chiếu `{c['query']}` với `{c['reference']}`: {'khớp' if c['matches'] else 'KHÔNG khớp'}."
                   for c in results["anchor_checks"]), "",
        "## Môi trường", "",
        *table(["Thành phần", "Giá trị"], [["CPU", env["cpu"]], ["Số CPU logic (WSL)", env["logical_cpus"]], ["RAM (WSL)", env["memory_total"]],
                                         ["Hệ điều hành", env["platform"]], ["Python", env["python"]], ["Java", env["java"]],
                                         ["Spark", results["engines"]["spark"]], ["DuckDB", results["engines"]["duckdb"]],
                                         ["Dữ liệu", f"{env['input_files']} file Parquet, {env['input_bytes'] / 2**20:.1f} MiB, "
                                                     f"{results['result_rows']['q1_scan_summary'][0][0]:,} bài"]]), "",
        "## Thời gian (giây, median)", "",
        f"![Median query time](figures/{figure.name})", "",
        *table(["Luồng", "Truy vấn", "Spark", "DuckDB", "Spark / DuckDB", "Khớp"],
               [[r["threads"], r["query"], f"{r['spark_median_seconds']:.3f}", f"{r['duckdb_median_seconds']:.4f}",
                 f"{r['spark_to_duckdb_ratio']:.1f}×", "có" if r["results_match"] else "KHÔNG"] for r in results["summary"]]), "",
        "Thời gian khởi động engine (giây): " + "; ".join(
            f"{s['threads']} luồng — Spark {s['spark']:.1f}, DuckDB {s['duckdb']:.3f}" for s in results["startup_seconds"]) + ". "
        "Cấu hình Spark thứ hai tái sử dụng JVM đã khởi động (PySpark giữ tiến trình JVM sau `spark.stop()`), nên chỉ đo "
        "thời gian tạo SparkContext mới; con số của cấu hình đầu tiên mới phản ánh chi phí khởi động đầy đủ.", "",
        "## Diễn giải", "",
        "- Đây là phép đo trên **một máy, Spark ở chế độ local**. Kết quả không mô tả hiệu năng của Spark trên cụm nhiều máy.",
        "- Dữ liệu (~55 MiB nén, 1,7 triệu dòng) vừa trong RAM của một máy. Ở quy mô này, chi phí cố định của Spark "
        "(lập kế hoạch, lên lịch task, JVM, shuffle) chiếm phần lớn thời gian; DuckDB là engine nhúng, vector hóa, không có chi phí đó.",
        "- Spark có lợi thế khi dữ liệu vượt quá bộ nhớ một máy, khi cần chạy phân tán, chịu lỗi, hoặc tích hợp MLlib — như "
        "pipeline huấn luyện của dự án. DuckDB phù hợp cho truy vấn tương tác và dashboard trên dữ liệu đã thu gọn.",
        "- Các truy vấn được viết bằng SQL chung cho cả hai dialect (ví dụ tính giờ/thứ bằng số học trên `created_utc`), "
        "nên không dùng hàm đặc thù có thể nhanh hơn trên từng engine.", "",
        "## Tái lập", "", "```bash",
        "JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \\",
        f"  .venv/bin/python -m askreddit.benchmark --threads {' '.join(str(t) for t in configs)} --run-id <run-id-mới>", "```", "",
    ]
    summary = output / "summary_vi.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return [figure, summary]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/askreddit_clean")
    parser.add_argument("--report-root", default="reports/benchmark")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("spark_vs_duckdb_%Y%m%d"))
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--spark-memory", default="2g")
    parser.add_argument("--duckdb-memory", default="2GB")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = (Path(args.report_root) / args.run_id).resolve()
    if not args.render_only:
        if args.repeats < 1 or args.warmup < 0 or any(t < 1 for t in args.threads):
            parser.error("repeats must be >= 1, warmup >= 0 and threads >= 1")
        results = run_benchmark(Path(args.input).resolve(), output, args.threads, args.repeats, args.warmup,
                                args.spark_memory, args.duckdb_memory)
        if not results["all_results_match"]:
            LOG.error("Engines disagree on at least one query; see results.json")
    print("\n".join(str(p) for p in render(output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
