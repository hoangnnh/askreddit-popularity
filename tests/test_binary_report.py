"""The binary report renders figures and Markdown from saved JSON alone."""

import json

import numpy as np

from askreddit.binary_evaluation import best_f1_threshold, evaluate_scores
from askreddit.binary_experiment import cyclic_effects
from askreddit.binary_report import build
from askreddit.features import STRUCTURED_COLUMNS, TIME_COLUMNS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_report_builds_from_synthetic_run(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    candidates = [{"id": "majority", "kind": "majority"},
                  {"id": "time_lr", "kind": "lr", "features": "time_features"},
                  {"id": "combined_lr", "kind": "lr", "features": "combined_features"},
                  {"id": "gbt_d5", "kind": "gbt", "features": "selected_features"},
                  {"id": "gbt_d7", "kind": "gbt", "features": "selected_features"}]
    reports = tmp_path / "run"
    write(reports / "config.json", {"candidates": candidates, "disclosure_vi": "Thử nghiệm."})
    write(reports / "label_thresholds.json", {"median": 3, "p75": 4, "quantile_definition": "inverse ECDF",
        "population": "train posts with score >= 2", "low_rule": "score < 3", "high_rule": "score >= 4"})
    write(reports / "independent_verification.json", {"status": "passed"})
    write(reports / "class_distributions.json", {s: {"total": 10, "kept": 8, "positive_rate_among_kept": 0.25,
        "groups": {g: {"count": c} for g, c in (("excluded", 5), ("low", 6), ("middle", 2), ("high", 2))}} for s in ("train", "validation", "test")})
    write(reports / "features.json", {"chi_square": {"method": "chi2", "num_top_features": 2, "train_positive_rate": 0.17,
        "selected_terms": [{"rank": 1, "term": "u:why", "statistic": 10.0, "train_documents": 5, "high_rate_when_present": 0.1}]}})
    effects = cyclic_effects({c: 0.1 for c in TIME_COLUMNS}, {c: 1.0 for c in TIME_COLUMNS})
    terms = [{"term": f"t{i}", "kind": "u", "coefficient": float(i)} for i in range(1, 4)]
    write(reports / "analysis.json", {
        "logistic_regression": {"time_lr": {"time_effects": effects},
                                "combined_lr": {"top_positive": terms, "top_negative": [{**t, "coefficient": -t["coefficient"]} for t in terms],
                                                "structured": {c: 0.1 for c in STRUCTURED_COLUMNS}, "time_effects": effects}},
        "gradient_boosted_trees": {"gbt_d5": {"top_features": [{"feature": "hour_sin", "importance": 0.6}, {"feature": "why", "importance": 0.4}],
                                              "structured_importance_share": 0.6, "time_importance_share": 0.6}}})
    write(reports / "selection.json", {"selected_candidate": "combined_lr", "best_by_family": {"lr": "combined_lr", "gbt": "gbt_d5"}})
    for spec in candidates:
        if spec["id"] == "gbt_d7":
            write(reports / "candidates/gbt_d7.json", {"candidate": spec, "status": "failed_optional_comparison", "error": "OOM"})
            continue
        p = np.full(300, 0.5) if spec["kind"] == "majority" else np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
        threshold = 0.5 if spec["kind"] == "majority" else best_f1_threshold(y, p)
        metrics = evaluate_scores(y, p, threshold, seen=rng.random(300) < 0.5)
        write(reports / "candidates" / f"{spec['id']}.json", {"candidate": spec, "status": "completed",
              "threshold": {"value": threshold}, "validation": metrics})
        write(reports / "test" / f"{spec['id']}.json", metrics)
    path = build(reports)
    text = path.read_text()
    assert "`combined_lr` ⭐" in text and "lỗi" in text and "AUC test" in text and "score ≤ 1" in text
    for name in ("roc_test", "confusion_test", "lr_top_ngrams", "gbt_importance", "time_effects"):
        assert (reports / "figures" / f"{name}.png").stat().st_size > 1000
