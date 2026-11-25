"""Export a binary run's best logistic-regression model for Spark-free inference and verify parity.

Takes the validation-selected LR candidate of a binary run (``selection.json`` →
``best_by_family.lr``; tree models stay offline, as the proposal specifies), writes
``model.json`` + ``model.npz`` into a fresh directory, and compares NumPy and Spark
probabilities on sampled validation/test posts plus Unicode edge cases. The export
is refused unless every probability agrees within the tolerance.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from askreddit.binary_inference import BinaryPopularityModel
from askreddit.clean import create_spark, utc_seconds, write_json
from askreddit.export_model import EDGE_TITLES

LOG = logging.getLogger(__name__)
TOLERANCE = 1e-6


def summary(metrics: dict) -> dict:
    keys = ("records", "positives", "positive_rate", "auc_roc", "pr_auc", "threshold", "precision", "recall", "f1",
            "accuracy", "balanced_accuracy", "confusion_matrix")
    return {k: metrics[k] for k in keys}


THRESHOLD_POLICIES = {
    "validation_max_f1": "Frozen validation threshold maximizing F1 of the high class (the run's predeclared policy).",
    "probability_0_5": "Probability 0.5, the natural cut for a model trained with balanced class weights. Chosen for the demo after the run's results were viewed, because the max-F1 threshold labels almost every post high when the signal is weak.",
}


def export(spark, run_id: str, output: Path, artifact_root: Path, report_root: Path, sample_size: int, seed: int,
           candidate: str | None = None, threshold_policy: str = "validation_max_f1") -> dict:
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from askreddit.features import STRUCTURED_COLUMNS, base_features

    reports, artifacts = report_root / run_id, artifact_root / run_id
    selection = json.loads((reports / "selection.json").read_text())
    config = json.loads((reports / "config.json").read_text())
    candidate_id = candidate or selection["best_by_family"]["lr"]
    spec = next(c for c in config["candidates"] if c["id"] == candidate_id)
    if spec["kind"] != "lr":
        raise ValueError(f"Only logistic-regression candidates can be exported; {candidate_id} is {spec['kind']}")
    preprocessing = PipelineModel.load(str(artifacts / "preprocessing"))
    chi_selection = PipelineModel.load(str(artifacts / "selection"))
    model = LogisticRegressionModel.load(str(artifacts / "models" / candidate_id))
    if model.numClasses != 2:
        raise ValueError("Expected a binary logistic-regression model")
    stages = preprocessing.stages
    vocabulary, idf, scaler = list(stages[0].vocabulary), stages[1].idf.toArray(), stages[4]
    thresholds = json.loads((reports / "label_thresholds.json").read_text())
    if threshold_policy not in THRESHOLD_POLICIES:
        raise ValueError(f"Unknown threshold policy: {threshold_policy}")
    candidate_report = json.loads((reports / "candidates" / f"{candidate_id}.json").read_text())
    validation = candidate_report["validation"]
    weights = candidate_report["training"]["class_weights"]
    test = json.loads((reports / "test" / f"{candidate_id}.json").read_text())
    if threshold_policy == "probability_0_5":
        validation, test = ({**m, **m["at_probability_0_5"]} for m in (validation, test))
    metadata = {
        "format_version": 1, "task": "binary", "run_id": run_id, "candidate": spec, "features": spec["features"],
        "selected_overall": selection["selected_candidate"], "feature_dimension": int(model.numFeatures),
        "threshold": 0.5 if threshold_policy == "probability_0_5" else float(selection["thresholds"][candidate_id]),
        "threshold_policy": threshold_policy, "threshold_policy_description": THRESHOLD_POLICIES[threshold_policy],
        "validation_max_f1_threshold": float(selection["thresholds"][candidate_id]),
        "vocabulary": vocabulary, "structured_columns": STRUCTURED_COLUMNS, "feature_config": config["features"],
        "split_dates": config["split_dates"],
        "labels": {"population": thresholds.get("population", "all train posts"), "low": thresholds["low_rule"],
                   "high": thresholds["high_rule"], "median": thresholds["median"], "p75": thresholds["p75"]},
        "metrics": {"validation": summary(validation), "test": summary(test)},
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": ["title"] + (["created_utc (UTC)"] if spec["features"] in ("time_features", "structured_features", "combined_features") else []),
        "excluded_inputs": ["score", "selftext", "id", "label"],
        "probability_note": "Trained with balanced class weights: raw probabilities are relative scores. "
                            "calibrated_probability() adds prior_logit_offset = log(w_low / w_high) to undo the weighting.",
        "class_weights": weights, "prior_logit_offset": float(np.log(weights[0] / weights[1])),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "model.npz", coefficients=model.coefficients.toArray(), intercept=np.array(model.intercept),
                        idf=idf, std=scaler.std.toArray())
    (output / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    start = utc_seconds(config["split_dates"]["validation"][0])
    clean = spark.read.parquet(str(Path(config["input"]).resolve())).filter(F.col("created_utc") >= start)
    total = clean.count()
    sample = clean.sample(fraction=min(1.0, 2.0 * sample_size / total), seed=seed).orderBy("id").limit(sample_size) \
        .select("id", "title", "created_utc", "score")
    edges = spark.createDataFrame([(f"edge{i}", t, start + 3600 * 7 * i, None) for i, t in enumerate(EDGE_TITLES)],
                                  "id string, title string, created_utc long, score long")
    rows = sample.unionByName(edges)
    transformed = chi_selection.transform(preprocessing.transform(base_features(rows, config["features"])))
    scored = model.setFeaturesCol(spec["features"]).transform(transformed) \
        .select("id", "title", "created_utc", vector_to_array("probability")[1].alias("p")) \
        .join(rows.select("id", "score"), "id").collect()  # base_features keeps only id/title/created_utc
    local = BinaryPopularityModel.load(output)
    differences, flips = [], []
    for row in scored:
        actual = local.predict_proba(row.title, row.created_utc if local.uses_time else None)
        differences.append(abs(actual - row.p))
        if (actual >= metadata["threshold"]) != (row.p >= metadata["threshold"]):
            flips.append({"id": row.id, "title": row.title, "spark": row.p, "numpy": actual})
    from askreddit.binary_labeling import is_high, is_low, model_cuts
    rules, low_cut, high_cut = model_cuts(thresholds)
    kept = [r for r in scored if r.score is not None and (rules["population_min_score"] is None or r.score >= rules["population_min_score"])
            and (is_low(r.score, low_cut, rules["low_rule"]) or is_high(r.score, high_cut, rules["high_rule"]))]
    calibration = {"records": len(kept), "actual_high_rate": float(np.mean([is_high(r.score, high_cut, rules["high_rule"]) for r in kept])),
                   "mean_raw_probability": float(np.mean([r.p for r in kept])),
                   "mean_calibrated_probability": float(np.mean([local.calibrated_probability(r.title, r.created_utc if local.uses_time else None) for r in kept]))} if kept else None
    verification = {"calibration_on_sample": calibration, "rows_compared": len(scored), "edge_cases": len(EDGE_TITLES), "sampled_from": "validation+test interval",
                    "max_abs_probability_difference": max(differences), "tolerance": TOLERANCE,
                    "label_flips_at_frozen_threshold": len(flips), "flip_examples": flips[:10],
                    "passed": not flips and max(differences) <= TOLERANCE}
    write_json(output / "verification.json", verification)
    if not verification["passed"]:
        raise ValueError(f"Spark-free inference disagrees with Spark: {verification}")
    return verification


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate", help="LR candidate to export (default: best LR by validation AUC)")
    parser.add_argument("--threshold-policy", choices=sorted(THRESHOLD_POLICIES), default="validation_max_f1")
    parser.add_argument("--output", help="Fresh directory (default artifacts/demo/RUN)")
    parser.add_argument("--artifact-root", default="artifacts/experiments")
    parser.add_argument("--report-root", default="reports/experiments")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=212)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output or f"artifacts/demo/{args.run_id}").resolve()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = None
    try:
        spark = create_spark("2g", 8)
        result = export(spark, args.run_id, output, Path(args.artifact_root).resolve(), Path(args.report_root).resolve(),
                        args.sample_size, args.seed, args.candidate, args.threshold_policy)
        LOG.info("Exported %s; parity %s", output, result)
    except BaseException:
        LOG.exception("Export failed")
        return 1
    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
