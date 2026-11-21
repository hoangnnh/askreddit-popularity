"""Offline, resumable full-data Fang/Chua experiment. No ingestion or downloads."""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from askreddit.clean import CLEAN_SCHEMA, create_spark, file_fingerprint, guard_paths, write_json
from askreddit.evaluation import comparison_csv, confusion_figure
from askreddit.features import (FEATURE_COLUMNS, fit_preprocessing, transform_features,
                               validate_feature_sources, with_normalized_title)
from askreddit.labeling import (SPLITS, VERSION, apply_labels, assign_splits, class_distributions,
                               fit_thresholds, split_intervals)
from askreddit.training import choose_candidate, fit_candidate, load_model, timed_evaluation

LOG = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inventory(path: Path):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [file_fingerprint(p) for p in files]


class Run:
    def __init__(self, config: dict, run_id: str, resume: bool):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Run ID must contain only letters, numbers, underscores and hyphens")
        if config["config_version"] != 1 or config["algorithm_version"] != VERSION:
            raise ValueError("Unsupported configuration or labeling algorithm version")
        split_intervals(config["split_dates"])
        validate_feature_sources(config["features"]["feature_sources"])
        ids = [c["id"] for c in config["candidates"]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", s) for s in ids):
            raise ValueError("Candidate IDs must be unique and filesystem safe")
        self.config = config
        self.input = Path(config["input"]).resolve()
        self.data = (Path(config["data_root"]) / run_id).resolve()
        self.artifacts = (Path(config["artifact_root"]) / run_id).resolve()
        self.reports = (Path(config["report_root"]) / run_id).resolve()
        self.input_files = sorted(self.input.rglob("*.parquet"))
        if not self.input_files or not (self.input / "_SUCCESS").is_file():
            raise ValueError("A completed local cleaned Parquet dataset is required")
        identity = {"config": config, "input_files": [file_fingerprint(p) for p in self.input_files],
                    "python_version": platform.python_version()}
        self.identity_sha = digest(identity)
        if resume:
            prior = json.loads((self.reports / "identity.json").read_text())
            if prior["identity_sha256"] != self.identity_sha:
                raise ValueError("Cannot resume: input hashes, configuration or Python version changed")
        else:
            guard_paths(self.input_files, [self.data, self.artifacts, self.reports])
            for path in (self.data, self.artifacts, self.reports):
                path.mkdir(parents=True, exist_ok=False)
            write_json(self.reports / "identity.json", {**identity, "identity_sha256": self.identity_sha})
            write_json(self.reports / "config.json", config)
            write_json(self.reports / "search_plan.json", {
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidates": config["candidates"], "selection": config["selection"],
                "test_used_for_tuning": False, "training_sample_policy": "All training rows; no dataset subsampling. RF uses its declared internal bagging rate.",
            })

    def receipt(self, stage):
        return self.reports / "receipts" / f"{stage}.json"

    def source_hashes(self, modules):
        return [file_fingerprint(Path(__file__).with_name(name + ".py")) for name in modules]

    def completed(self, stage, modules):
        receipt = self.receipt(stage)
        if not receipt.exists():
            return False
        saved = json.loads(receipt.read_text())
        if saved["identity_sha256"] != self.identity_sha or saved["source_files"] != self.source_hashes(modules):
            raise ValueError(f"Stage {stage} is incompatible with the current inputs/configuration/code")
        for artifact in saved["artifacts"]:
            if inventory(Path(artifact["path"])) != artifact["files"]:
                raise ValueError(f"Stage {stage} artifact changed or is incomplete: {artifact['path']}")
        LOG.info("Verified and reused completed stage %s", stage)
        return True

    def finish(self, stage, paths, modules):
        write_json(self.receipt(stage), {
            "identity_sha256": self.identity_sha, "source_files": self.source_hashes(modules),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": [{"path": str(p), "files": inventory(p)} for p in paths],
        })

    def events(self, phase, **details):
        LOG.info("%s %s", phase, details)
        with (self.reports / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, **details}) + "\n")


PREPARE_MODULES = ["experiment", "labeling", "features"]
FEATURE_MODULES = ["experiment", "features"]
TRAIN_MODULES = ["experiment", "training", "evaluation", "labeling"]


def verify_with_duckdb(run, fitted, distributions):
    import duckdb
    connection = duckdb.connect(config={"threads": 2, "memory_limit": "1GB", "autoinstall_known_extensions": False, "autoload_known_extensions": False})
    paths = [str(p) for p in run.input_files]
    intervals = split_intervals(run.config["split_dates"])
    medians = []
    lower = 2
    try:
        for step in fitted["steps"]:
            n, median = connection.execute(
                "SELECT count(*), median(score) FROM read_parquet(?) WHERE created_utc >= ? AND created_utc < ? AND score >= ?",
                [paths, *intervals["train"], lower]).fetchone()
            # Observed source scores are small integers, so these .0/.5 medians are exact in binary64.
            expected = Fraction(step["median_numerator"], step["median_denominator"])
            if Fraction(str(median)) != expected or n != step["remainder_records"]:
                raise ValueError("Independent median verification disagrees with Spark")
            medians.append({"median": str(median), "records": n})
            lower = step["effective_upper_exclusive"]
        clauses = " ".join(f"WHEN score < {int(b)} THEN {k}" for k, b in enumerate(fitted["effective_boundaries"]))
        observed = {}
        for split, bounds in intervals.items():
            rows = connection.execute(
                f"SELECT CASE {clauses} ELSE 7 END AS label, count(*) FROM read_parquet(?) WHERE created_utc >= ? AND created_utc < ? GROUP BY 1 ORDER BY 1",
                [paths, *bounds]).fetchall()
            actual = dict(rows)
            expected = {r["label"]: r["count"] for r in distributions[split]["labels"]}
            if {k: actual.get(k, 0) for k in range(8)} != expected:
                raise ValueError(f"Independent class count verification failed for {split}")
            observed[split] = actual
        return {"status": "passed", "engine": "DuckDB", "medians": medians, "class_counts": observed}
    finally:
        connection.close()


def prepare_stage(spark, run):
    if run.completed("prepare", PREPARE_MODULES):
        return
    started = perf_counter()
    run.events("prepare_started")
    raw = spark.read.parquet(str(run.input))
    if dict(raw.dtypes) != dict(CLEAN_SCHEMA):
        raise ValueError(f"Unexpected cleaned schema: {raw.dtypes}")
    for path in run.input_files:
        if dict(spark.read.parquet(str(path)).dtypes) != dict(CLEAN_SCHEMA[:6]):
            raise ValueError(f"Unexpected physical file schema: {path}")
    data = with_normalized_title(assign_splits(raw, run.config["split_dates"])).persist(StorageLevel.DISK_ONLY)
    try:
        stats = data.agg(F.count("*").alias("records"), F.countDistinct("id").alias("distinct_ids"),
            F.min("created_utc").alias("min_created_utc"), F.max("created_utc").alias("max_created_utc"),
            F.min("score").alias("min_score"), F.max("score").alias("max_score"),
            *[F.count(F.when(F.col(c).isNull(), 1)).alias("null_" + c) for c, _ in CLEAN_SCHEMA],
            F.count(F.when(F.col("score") == 0, 1)).alias("score_zero"),
            F.count(F.when(F.col("score") == 1, 1)).alias("score_one"),
            F.count(F.when(F.col("score") < 0, 1)).alias("negative_scores"),
            F.count(F.when(F.col("split").isNull(), 1)).alias("outside_split_intervals"),
            F.count(F.when((F.col("subreddit") != "AskReddit") | ~F.col("id").rlike("^[a-z0-9]+$") |
                           (F.col("normalized_title") == ""), 1)).alias("invalid_required_fields"),
            F.count(F.when((F.col("year") != F.year(F.timestamp_seconds("created_utc"))) |
                           (F.col("month") != F.month(F.timestamp_seconds("created_utc"))), 1)).alias("invalid_partitions"),
        ).first().asDict()
        if stats["records"] != stats["distinct_ids"] or any(v for k, v in stats.items() if k.startswith("null_")) or any(
            stats[k] for k in ("outside_split_intervals", "invalid_required_fields", "invalid_partitions")):
            raise ValueError(f"Cleaned-data invariants failed: {stats}")
        stats["splits"] = [r.asDict() for r in data.groupBy("split").agg(F.count("*").alias("records"),
            F.min("created_utc").alias("min_created_utc"), F.max("created_utc").alias("max_created_utc")).orderBy("split").collect()]
        if len(stats["splits"]) != 3:
            raise ValueError("Every chronological split must be nonempty")
        stats["unique_ids_within_and_across_splits"] = True  # Global uniqueness is the stronger check.
        fitted = fit_thresholds(data.filter(F.col("split") == "train"))
        fitted.update(input_fingerprint=run.identity_sha, split_dates=run.config["split_dates"])
        fitted["historical_diagnostic_matches"] = fitted["effective_boundaries"] == [2, 3, 4, 6, 11, 46, 271]
        if not fitted["historical_diagnostic_matches"]:
            LOG.warning("Fitted thresholds differ from the historical diagnostic: %s", fitted["effective_boundaries"])
        labeled = apply_labels(data, fitted)
        distributions = class_distributions(labeled)
        if sum(d["total"] for d in distributions.values()) != stats["records"] or any(
            r["count"] == 0 for r in distributions["train"]["labels"]):
            raise ValueError("Label counts do not reconcile or training is missing a class")
        # Aggregate overlap; never collect titles or discard repeated titles.
        titles = data.groupBy("normalized_title").agg(*[
            F.count(F.when(F.col("split") == split, 1)).alias(split) for split in SPLITS]).persist(StorageLevel.DISK_ONLY)
        try:
            pairs = {}
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                shared = titles.filter((F.col(a) > 0) & (F.col(b) > 0))
                pairs[a + "__" + b] = shared.agg(F.count("*").alias("shared_distinct_titles"),
                    F.coalesce(F.sum(a), F.lit(0)).alias("left_records"), F.coalesce(F.sum(b), F.lit(0)).alias("right_records")).first().asDict()
            train_titles = titles.filter(F.col("train") > 0).select("normalized_title").withColumn("title_seen_in_train", F.lit(True))
            labeled = labeled.join(train_titles, "normalized_title", "left").fillna({"title_seen_in_train": False})
            target = run.data / "labeled"
            labeled.repartition(run.config["spark"]["write_partitions"], "split", "month").write.mode("errorifexists").option(
                "compression", "zstd").partitionBy("split", "year", "month").parquet(str(target))
        finally:
            titles.unpersist()
        written = spark.read.parquet(str(target))
        if class_distributions(written) != distributions or written.groupBy("score").agg(F.countDistinct("label").alias("n")).filter("n > 1").count():
            raise ValueError("Labeled read-back/counts/equal-score consistency check failed")
        independent = verify_with_duckdb(run, fitted, distributions)
        for name, content in (("input_validation", stats), ("label_boundaries", fitted),
                               ("class_distributions", distributions), ("title_overlap", pairs),
                               ("independent_verification", independent)):
            write_json(run.reports / f"{name}.json", content)
        if [file_fingerprint(p) for p in run.input_files] != json.loads((run.reports / "identity.json").read_text())["input_files"]:
            raise ValueError("Input files changed while preparing labels")
        run.finish("prepare", [target, *[run.reports / (n + ".json") for n in (
            "input_validation", "label_boundaries", "class_distributions", "title_overlap", "independent_verification")]], PREPARE_MODULES)
        run.events("prepare_completed", elapsed_seconds=perf_counter() - started, records=stats["records"], boundaries=fitted["effective_boundaries"])
    finally:
        data.unpersist()


def features_stage(spark, run):
    if not run.completed("prepare", PREPARE_MODULES):
        raise ValueError("Complete prepare before feature fitting")
    if run.completed("features", FEATURE_MODULES):
        return
    run.events("features_started")
    data = spark.read.parquet(str(run.data / "labeled"))
    started = perf_counter()
    model, metadata = fit_preprocessing(data.filter(F.col("split") == "train"), run.config["features"])
    metadata["fitting_seconds"] = perf_counter() - started
    path = run.artifacts / "preprocessing"
    model.write().save(str(path))
    model = PipelineModel.load(str(path))
    counts = json.loads((run.reports / "class_distributions.json").read_text())
    times = {}
    for split in SPLITS:
        started = perf_counter()
        transformed = transform_features(data.filter(F.col("split") == split), model, run.config["features"])
        target = run.data / "features" / split
        transformed.repartition(run.config["spark"]["shuffle_partitions"], "id").write.mode("errorifexists").option("compression", "zstd").parquet(str(target))
        written = spark.read.parquet(str(target))
        actual = written.count()
        if actual != counts[split]["total"]:
            raise ValueError(f"Feature row count mismatch in {split}")
        times[split] = {"records": actual, "transformation_and_write_seconds": perf_counter() - started}
        run.events("features_split_completed", split=split, **times[split])
    metadata["transformation"] = times
    write_json(run.reports / "features.json", metadata)
    run.finish("features", [path, run.data / "features", run.reports / "features.json"], FEATURE_MODULES)


def train_stage(spark, run, candidate_id=None):
    if not run.completed("features", FEATURE_MODULES):
        raise ValueError("Complete features before training")
    counts = [r["count"] for r in json.loads((run.reports / "class_distributions.json").read_text())["train"]["labels"]]
    metadata = json.loads((run.reports / "features.json").read_text())
    train = spark.read.parquet(str(run.data / "features/train"))
    validation = spark.read.parquet(str(run.data / "features/validation"))
    candidates = [c for c in run.config["candidates"] if candidate_id is None or c["id"] == candidate_id]
    if not candidates:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    for spec in candidates:
        name = spec["id"]
        if run.completed("candidate_" + name, TRAIN_MODULES):
            continue
        run.events("candidate_started", candidate=name, configuration=spec)
        try:
            model, info = fit_candidate(train, spec, counts, run.config["seed"])
        except Exception as error:
            if spec["kind"] != "rf":
                raise
            # RF is explicitly conditional on available resources. Keep the precise
            # failure; never substitute a smoke/sample run or invented RF metrics.
            path = run.reports / "candidates" / (name + ".json")
            write_json(path, {"candidate": spec, "status": "failed_optional_comparison", "error": str(error)})
            run.finish("candidate_" + name, [path], TRAIN_MODULES)
            run.events("optional_candidate_failed", candidate=name, error=str(error))
            continue
        info.setdefault("feature_dimension", metadata["dimensions"].get(spec.get("features"), 0))
        saved = []
        if model is not None:
            path = run.artifacts / "models" / name
            model.write().save(str(path))
            model = load_model(spec, path)
            saved.append(path)
        validation_metrics = timed_evaluation(validation, spec, model, info)
        result = {"candidate": spec, "status": "completed", "training": info, "validation": validation_metrics}
        path = run.reports / "candidates" / (name + ".json")
        write_json(path, result)
        saved.append(path)
        run.finish("candidate_" + name, saved, TRAIN_MODULES)
        run.events("candidate_completed", candidate=name, training_seconds=info["training_seconds"],
                   validation_macro_f1=validation_metrics["macro_f1"])
        del model
        spark.catalog.clearCache()


def evaluate_stage(spark, run):
    if run.completed("evaluation", TRAIN_MODULES):
        return
    results = []
    for spec in run.config["candidates"]:
        if not run.completed("candidate_" + spec["id"], TRAIN_MODULES):
            raise ValueError(f"Candidate incomplete; test is still sealed: {spec['id']}")
        result = json.loads((run.reports / "candidates" / (spec["id"] + ".json")).read_text())
        if result["status"] == "completed":
            results.append(result)
        elif spec["kind"] != "rf":
            raise ValueError(f"Required candidate failed: {spec['id']}")
    selected = choose_candidate(results)
    selection_path = run.reports / "selection.json"
    if selection_path.exists():
        prior = json.loads(selection_path.read_text())
        if prior["selected_candidate"] != selected or prior["validation_results_sha256"] != digest(results):
            raise ValueError("Frozen validation selection changed")
    else:
        write_json(selection_path, {"selected_candidate": selected, "criterion": run.config["selection"],
            "validation_results_sha256": digest(results), "frozen_before_test": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    run.events("selection_frozen", candidate=selected)
    test = spark.read.parquet(str(run.data / "features/test"))
    rows = []
    for result in results:
        spec, info = result["candidate"], result["training"]
        path = run.reports / "test" / (spec["id"] + ".json")
        if run.completed("test_" + spec["id"], TRAIN_MODULES):
            metrics = json.loads(path.read_text())
        else:
            model = load_model(spec, run.artifacts / "models" / spec["id"])
            metrics = timed_evaluation(test, spec, model, info, unseen_slice=(spec["id"] == selected))
            write_json(path, metrics)
            run.finish("test_" + spec["id"], [path], TRAIN_MODULES)
            del model
        for split, scores in (("validation", result["validation"]), ("test", metrics)):
            rows.append({"candidate": spec["id"], "split": split, "selected": spec["id"] == selected,
                         **scores, "training_seconds": info["training_seconds"], "feature_dimension": info["feature_dimension"]})
            confusion_figure(scores, run.reports / "figures" / f"{spec['id']}_{split}.png", f"{spec['id']} — {split}")
        run.events("test_candidate_completed", candidate=spec["id"], macro_f1=metrics["macro_f1"])
    selected_spec = next(c for c in run.config["candidates"] if c["id"] == selected)
    selected_path = run.artifacts / "selected"
    if not selected_path.exists():
        selected_path.mkdir()
        write_json(selected_path / "manifest.json", {"candidate": selected_spec,
            "preprocessing": str(run.artifacts / "preprocessing"), "model": str(run.artifacts / "models" / selected) if selected_spec["kind"] != "majority" else None,
            "feature_metadata": str(run.reports / "features.json"), "label_mapping": str(run.reports / "label_boundaries.json"),
            "training": next(r["training"] for r in results if r["candidate"]["id"] == selected)})
    comparison_csv(rows, run.reports / "model_comparison.csv")
    write_json(run.reports / "model_comparison.json", {"selected_candidate": selected, "rows": rows})
    current = [file_fingerprint(p) for p in run.input_files]
    if current != json.loads((run.reports / "identity.json").read_text())["input_files"]:
        raise ValueError("Cleaned inputs changed during the experiment")
    write_json(run.reports / "completed.json", {"status": "completed", "selected_candidate": selected,
        "full_training_rows_used": True, "input_files_unchanged": True, "test_used_for_tuning": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    run.finish("evaluation", [selection_path, run.reports / "model_comparison.csv", run.reports / "model_comparison.json",
                              run.reports / "completed.json", run.reports / "figures", selected_path], TRAIN_MODULES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fang_chua_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=["prepare", "features", "train", "evaluate", "all"], default="all")
    parser.add_argument("--candidate", help="Train one predefined candidate, only with --stage train")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = spark = None
    try:
        if args.candidate and args.stage != "train":
            raise ValueError("--candidate requires --stage train")
        config = json.loads(Path(args.config).read_text())
        run = Run(config, args.run_id, args.resume)
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        # Use persistent filesystem spill space instead of this host's small RAM-backed /tmp.
        scratch = run.artifacts / "spark_tmp"
        scratch.mkdir(exist_ok=True)
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch)
        spark = create_spark(config["spark"]["driver_memory"], config["spark"]["shuffle_partitions"])
        run.events("stage_started", stage=args.stage, spark_version=spark.version)
        if args.stage in ("prepare", "all"):
            prepare_stage(spark, run)
        if args.stage in ("features", "all"):
            features_stage(spark, run)
        if args.stage in ("train", "all"):
            train_stage(spark, run, args.candidate)
        if args.stage in ("evaluate", "all"):
            evaluate_stage(spark, run)
        run.events("stage_completed", stage=args.stage)
    except BaseException as error:
        if run is not None:
            path = run.reports / "failures" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
            with suppress(Exception):
                write_json(path, {"stage": args.stage, "candidate": args.candidate, "error": str(error),
                                  "status": "failed", "partial_artifacts_are_not_complete": True})
        LOG.exception("Experiment stage failed")
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
