"""Binary run identity, train-quantile labels, train-fitted features and chi-square n-gram selection.

Stage receipts hash only the modules each stage depends on, so later fixes to the
training or evaluation code never invalidate completed labels/features.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.feature import CountVectorizerModel, VectorAssembler, VectorSlicer
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import functions as F

from askreddit.binary_labeling import (VERSION, apply_binary_labels, fit_binary_thresholds,
                                       group_distributions, group_sql, label_rules)
from askreddit.clean import CLEAN_SCHEMA, file_fingerprint, guard_paths, write_json
from askreddit.experiment import Run, digest
from askreddit.features import (STRUCTURED_COLUMNS, TIME_COLUMNS, base_features, fit_preprocessing,
                                validate_feature_sources, with_normalized_title)
from askreddit.labeling import SPLITS, assign_splits, split_intervals

PREPARE_MODULES = ["binary_prepare", "binary_labeling", "labeling", "features"]
FEATURE_MODULES = ["binary_prepare", "features"]
TITLE_COLUMNS = [c for c in STRUCTURED_COLUMNS if c not in TIME_COLUMNS]
TITLE_INDICES = [STRUCTURED_COLUMNS.index(c) for c in TITLE_COLUMNS]
# *_title_* columns drop the posting-time inputs: the title is the only predictor.
FEATURE_COLUMNS = ("time_features", "structured_features", "text_features", "combined_features", "selected_features",
                   "title_structured_features", "combined_title_features", "selected_title_features")
CANDIDATE_KINDS = {"majority", "lr", "gbt"}


class BinaryRun(Run):
    """Same receipts/events as the eight-class Run, with binary validation rules."""

    def __init__(self, config: dict, run_id: str, resume: bool):  # noqa: D107 - deliberately not calling Run.__init__
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Run ID must contain only letters, numbers, underscores and hyphens")
        if config["config_version"] != 1 or config["algorithm_version"] != VERSION:
            raise ValueError("Unsupported configuration or binary labeling version")
        split_intervals(config["split_dates"])
        label_rules(config.get("labeling"))
        validate_feature_sources(config["features"]["feature_sources"])
        ids = [c["id"] for c in config["candidates"]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", s) for s in ids):
            raise ValueError("Candidate IDs must be unique and filesystem safe")
        for spec in config["candidates"]:
            if spec["kind"] not in CANDIDATE_KINDS or (spec["kind"] != "majority" and spec["features"] not in FEATURE_COLUMNS):
                raise ValueError(f"Unsupported candidate: {spec}")
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
                "chi_square": config["chi_square"], "test_used_for_tuning": False,
                "training_sample_policy": "All low/high training rows; middle group excluded by the label rule. GBT uses its declared internal subsampling rate.",
            })


def verify_with_duckdb(run, fitted, distributions):
    """Recompute quantiles and group counts from the cleaned Parquet with a different engine."""
    import duckdb
    connection = duckdb.connect(config={"threads": 2, "memory_limit": "1GB", "autoinstall_known_extensions": False,
                                        "autoload_known_extensions": False})
    paths = [str(p) for p in run.input_files]
    intervals = split_intervals(run.config["split_dates"])
    minimum = label_rules(fitted.get("rules"))["population_min_score"]
    population = "" if minimum is None else f" AND score >= {int(minimum)}"
    try:
        median, upper, total = connection.execute(f"""
            WITH h AS (SELECT score, count(*) AS c FROM read_parquet(?) WHERE created_utc >= ? AND created_utc < ?{population} GROUP BY score),
                 cum AS (SELECT score, sum(c) OVER (ORDER BY score) AS cum, sum(c) OVER () AS n FROM h)
            SELECT min(score) FILTER (WHERE cum * 2 >= n), min(score) FILTER (WHERE cum * 4 >= 3 * n), max(n) FROM cum""",
            [paths, *intervals["train"]]).fetchone()
        if (int(median), int(upper), int(total)) != (fitted["median"], fitted["p75"], fitted.get("population_records", fitted["train_records"])):
            raise ValueError("Independent quantile verification disagrees with Spark")
        observed = {}
        for split, bounds in intervals.items():
            rows = connection.execute(
                f"SELECT {group_sql(fitted)}, count(*) "
                "FROM read_parquet(?) WHERE created_utc >= ? AND created_utc < ? GROUP BY 1", [paths, *bounds]).fetchall()
            actual = dict(rows)
            expected = {g: v["count"] for g, v in distributions[split]["groups"].items()}
            if {g: actual.get(g, 0) for g in expected} != expected:
                raise ValueError(f"Independent group count verification failed for {split}")
            observed[split] = actual
        return {"status": "passed", "engine": "DuckDB", "median": int(median), "p75": int(upper),
                "population_records": int(total), "group_counts": observed}
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
    data = with_normalized_title(assign_splits(raw, run.config["split_dates"])).persist(StorageLevel.DISK_ONLY)
    try:
        stats = data.agg(F.count("*").alias("records"), F.countDistinct("id").alias("distinct_ids"),
            F.min("score").alias("min_score"), F.max("score").alias("max_score"),
            *[F.count(F.when(F.col(c).isNull(), 1)).alias("null_" + c) for c, _ in CLEAN_SCHEMA],
            F.count(F.when(F.col("split").isNull(), 1)).alias("outside_split_intervals"),
            F.count(F.when(F.col("normalized_title") == "", 1)).alias("empty_titles")).first().asDict()
        if stats["records"] != stats["distinct_ids"] or any(v for k, v in stats.items() if k.startswith("null_")) or \
                stats["outside_split_intervals"] or stats["empty_titles"]:
            raise ValueError(f"Cleaned-data invariants failed: {stats}")
        fitted = fit_binary_thresholds(data.filter(F.col("split") == "train"), run.config.get("labeling"))
        expected = run.config["labeling"]["expected_train_thresholds"]
        fitted.update(input_fingerprint=run.identity_sha, split_dates=run.config["split_dates"],
                      matches_eda_expectation=(fitted["median"], fitted["p75"]) == (expected["median"], expected["p75"]))
        labeled = apply_binary_labels(data, fitted)
        distributions = group_distributions(labeled)
        if sum(d["total"] for d in distributions.values()) != stats["records"]:
            raise ValueError("Group counts do not reconcile with the input")
        # Label-free flag for the unseen-title slice; uses every train title.
        train_titles = data.filter(F.col("split") == "train").select("normalized_title").distinct() \
            .withColumn("title_seen_in_train", F.lit(True))
        kept = labeled.filter(F.col("label").isNotNull()).join(train_titles, "normalized_title", "left") \
            .fillna({"title_seen_in_train": False})
        target = run.data / "labeled"
        kept.repartition(run.config["spark"]["write_partitions"], "split", "month").write.mode("errorifexists").option(
            "compression", "zstd").partitionBy("split", "year", "month").parquet(str(target))
        written = spark.read.parquet(str(target))
        counts = {(r.split, r.popularity_group): r["count"] for r in written.groupBy("split", "popularity_group").count().collect()}
        if counts != {(s, g): distributions[s]["groups"][g]["count"] for s in SPLITS for g in ("low", "high")} or \
                written.groupBy("score").agg(F.countDistinct("label").alias("n")).filter("n > 1").count():
            raise ValueError("Labeled read-back/counts/equal-score consistency check failed")
        independent = verify_with_duckdb(run, fitted, distributions)
        for name, content in (("input_validation", stats), ("label_thresholds", fitted),
                              ("class_distributions", distributions), ("independent_verification", independent)):
            write_json(run.reports / f"{name}.json", content)
        run.finish("prepare", [target, *[run.reports / (n + ".json") for n in (
            "input_validation", "label_thresholds", "class_distributions", "independent_verification")]], PREPARE_MODULES)
        run.events("prepare_completed", elapsed_seconds=perf_counter() - started, median=fitted["median"], p75=fitted["p75"],
                   train_counts=fitted["train_counts"])
    finally:
        data.unpersist()


def chi_square_selection(train, vocabulary, config, k):
    """Top-k vocabulary terms by chi-square statistic on binary term presence (train rows only).

    Spark's ChiSqSelector ranks by p-value; on 1M+ rows many p-values underflow to 0
    and tie, so ranking by the statistic (df = 1 for every binary term) keeps the order
    meaningful. Both use the same MLlib ChiSquareTest.
    """
    presence = CountVectorizerModel.from_vocabulary(vocabulary, inputCol="terms", outputCol="term_presence", binary=True)
    frame = presence.transform(base_features(train, config)).select("id", "terms", "term_presence") \
        .join(train.select("id", "label"), "id").persist(StorageLevel.DISK_ONLY)
    try:
        started = perf_counter()
        rows = ChiSquareTest.test(frame, "term_presence", "label", flatten=True).collect()
        if len(rows) != len(vocabulary):
            raise ValueError("Chi-square test did not return one row per vocabulary term")
        ranked = sorted(((int(r.featureIndex), float(r.statistic), float(r.pValue), int(r.degreesOfFreedom)) for r in rows),
                        key=lambda t: (-t[1], t[0]))
        selected = ranked[:k]
        terms = [vocabulary[i] for i, *_ in selected]
        documents = {r.term: (r.documents, r.high_documents) for r in frame.select(
            "label", F.explode(F.array_distinct("terms")).alias("term")).filter(F.col("term").isin(terms))
            .groupBy("term").agg(F.count("*").alias("documents"), F.sum("label").alias("high_documents")).collect()}
        base_rate = frame.agg(F.avg("label")).first()[0]
        return {
            "method": "MLlib ChiSquareTest on binary train term presence; top-k by statistic, ties by vocabulary index",
            "num_top_features": k, "train_positive_rate": base_rate, "seconds": perf_counter() - started,
            "selected_indices": [i for i, *_ in selected],
            "selected_terms": [{"rank": r + 1, "index": i, "term": vocabulary[i], "statistic": stat, "p_value": pv,
                                "degrees_of_freedom": dof, "train_documents": int(documents[vocabulary[i]][0]),
                                "high_rate_when_present": float(documents[vocabulary[i]][1]) / documents[vocabulary[i]][0]}
                               for r, (i, stat, pv, dof) in enumerate(selected)],
        }
    finally:
        frame.unpersist()


def features_stage(spark, run):
    if not run.completed("prepare", PREPARE_MODULES):
        raise ValueError("Complete prepare before feature fitting")
    if run.completed("features", FEATURE_MODULES):
        return
    run.events("features_started")
    config = run.config["features"]
    data = spark.read.parquet(str(run.data / "labeled"))
    train = data.filter(F.col("split") == "train")
    started = perf_counter()
    model, metadata = fit_preprocessing(train, config)
    metadata["fitting_seconds"] = perf_counter() - started
    path = run.artifacts / "preprocessing"
    model.write().save(str(path))
    model = PipelineModel.load(str(path))
    vocabulary = list(model.stages[0].vocabulary)
    chi = chi_square_selection(train, vocabulary, config, run.config["chi_square"]["num_top_features"])
    run.events("chi_square_completed", seconds=chi["seconds"], top_terms=[t["term"] for t in chi["selected_terms"][:10]])
    selection = PipelineModel(stages=[
        VectorSlicer(inputCol="text_features", outputCol="selected_text", indices=chi["selected_indices"]),
        VectorAssembler(inputCols=["selected_text", "structured_features"], outputCol="selected_features"),
        VectorSlicer(inputCol="structured_features", outputCol="title_structured_features", indices=TITLE_INDICES),
        VectorAssembler(inputCols=["text_features", "title_structured_features"], outputCol="combined_title_features"),
        VectorAssembler(inputCols=["selected_text", "title_structured_features"], outputCol="selected_title_features")])
    selection_path = run.artifacts / "selection"
    selection.write().save(str(selection_path))
    selection = PipelineModel.load(str(selection_path))
    counts = json.loads((run.reports / "class_distributions.json").read_text())
    times = {}
    for split in SPLITS:
        started = perf_counter()
        frame = data.filter(F.col("split") == split)
        predictors = selection.transform(model.transform(base_features(frame, config))).select("id", *FEATURE_COLUMNS)
        joined = predictors.join(frame.select("id", "label", "split", "title_seen_in_train"), "id")
        target = run.data / "features" / split
        joined.repartition(run.config["spark"]["shuffle_partitions"], "id").write.mode("errorifexists") \
            .option("compression", "zstd").parquet(str(target))
        actual = spark.read.parquet(str(target)).count()
        if actual != counts[split]["kept"]:
            raise ValueError(f"Feature row count mismatch in {split}")
        times[split] = {"records": actual, "transformation_and_write_seconds": perf_counter() - started}
        run.events("features_split_completed", split=split, **times[split])
    metadata["dimensions"]["selected_features"] = len(chi["selected_indices"]) + metadata["dimensions"]["structured_features"]
    metadata["dimensions"]["title_structured_features"] = len(TITLE_COLUMNS)
    metadata["dimensions"]["combined_title_features"] = metadata["dimensions"]["text_features"] + len(TITLE_COLUMNS)
    metadata["dimensions"]["selected_title_features"] = len(chi["selected_indices"]) + len(TITLE_COLUMNS)
    metadata["title_columns"] = TITLE_COLUMNS
    metadata["dimensions"].pop("rf_features", None)
    metadata["chi_square"] = chi
    metadata["transformation"] = times
    metadata["fitted_rows"] = "Low/high training rows only (excluded and middle groups removed before fitting)."
    write_json(run.reports / "features.json", metadata)
    run.finish("features", [path, selection_path, run.data / "features", run.reports / "features.json"], FEATURE_MODULES)
