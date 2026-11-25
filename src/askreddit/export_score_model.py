"""Export the validation-selected stage-2 score regression for Spark-free inference and verify parity."""

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

from askreddit.clean import create_spark, utc_seconds, write_json
from askreddit.export_model import EDGE_TITLES
from askreddit.score_inference import ScoreModel

LOG = logging.getLogger(__name__)
TOLERANCE = 1e-9


def export(spark, run_id: str, output: Path, artifact_root: Path, report_root: Path, sample_size: int, seed: int) -> dict:
    from pyspark.ml import PipelineModel
    from pyspark.ml.regression import LinearRegressionModel
    from pyspark.sql import functions as F
    from askreddit.features import STRUCTURED_COLUMNS, base_features

    reports, artifacts = report_root / run_id, artifact_root / run_id
    config = json.loads((reports / "config.json").read_text())
    selected = json.loads((reports / "selection.json").read_text())["selected_candidate"]
    spec = next(c for c in config["candidates"] if c["id"] == selected)
    if spec["kind"] != "ridge":
        raise ValueError(f"Selected candidate {selected} is not a ridge model")
    preprocessing = PipelineModel.load(str(artifacts / "preprocessing"))
    title_pipeline = PipelineModel.load(str(artifacts / "title_features"))
    model = LinearRegressionModel.load(str(artifacts / "models" / selected))
    candidate = json.loads((reports / "candidates" / f"{selected}.json").read_text())
    test = json.loads((reports / "test" / f"{selected}.json").read_text())
    keys = ("records", "rmse_log", "mae_log", "r2_log", "spearman", "median_abs_error_score", "interval_coverage", "auc_high_vs_low")
    metadata = {
        "format_version": 1, "task": "score_regression", "run_id": run_id, "candidate": spec, "features": spec["features"],
        "feature_dimension": int(model.numFeatures), "threshold": 0.0, "vocabulary": list(preprocessing.stages[0].vocabulary),
        "structured_columns": STRUCTURED_COLUMNS, "feature_config": config["features"], "min_score": config["min_score"],
        "population": f"posts with score >= {config['min_score']}", "label": "log(1 + score), score observed ~36 h after posting",
        "residual_quantiles": candidate["validation"]["residual_quantiles"],
        "prediction_percentiles": candidate["validation"]["prediction_percentiles"],
        "metrics": {"validation": {k: candidate["validation"][k] for k in keys}, "test": {k: test[k] for k in keys}},
        "inputs": ["title"], "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "model.npz", coefficients=model.coefficients.toArray(), intercept=np.array(model.intercept),
                        idf=preprocessing.stages[1].idf.toArray(), std=preprocessing.stages[4].std.toArray())
    (output / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    start = utc_seconds(config["split_dates"]["validation"][0])
    clean = spark.read.parquet(str(Path(config["input"]).resolve())).filter((F.col("created_utc") >= start) & (F.col("score") >= config["min_score"]))
    total = clean.count()
    sample = clean.sample(fraction=min(1.0, 2.0 * sample_size / total), seed=seed).orderBy("id").limit(sample_size).select("id", "title", "created_utc")
    edges = spark.createDataFrame([(f"edge{i}", t, start) for i, t in enumerate(EDGE_TITLES)], "id string, title string, created_utc long")
    rows = sample.unionByName(edges)
    scored = model.setFeaturesCol(spec["features"]).transform(title_pipeline.transform(preprocessing.transform(
        base_features(rows, config["features"])))).select("title", "prediction").collect()
    local = ScoreModel.load(output)
    difference = max(abs(local.margin(r.title) - r.prediction) for r in scored)
    verification = {"rows_compared": len(scored), "edge_cases": len(EDGE_TITLES), "max_abs_log_prediction_difference": difference,
                    "tolerance": TOLERANCE, "passed": difference <= TOLERANCE}
    write_json(output / "verification.json", verification)
    if not verification["passed"]:
        raise ValueError(f"Spark-free inference disagrees with Spark: {verification}")
    return verification


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", help="Fresh directory (default artifacts/demo/RUN)")
    parser.add_argument("--artifact-root", default="artifacts/experiments")
    parser.add_argument("--report-root", default="reports/experiments")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=212)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = None
    try:
        spark = create_spark("2g", 8)
        result = export(spark, args.run_id, Path(args.output or f"artifacts/demo/{args.run_id}").resolve(),
                        Path(args.artifact_root).resolve(), Path(args.report_root).resolve(), args.sample_size, args.seed)
        LOG.info("Exported; parity %s", result)
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
