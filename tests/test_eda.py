"""EDA definitions, UTC time handling, train-only associations and end-to-end outputs."""

import json
import os
from pathlib import Path
import sys

import pytest

from askreddit.clean import create_spark, utc_seconds
from askreddit.eda import (OUTPUT_FILES, analysis_frame, bucket_labels, quantiles_from_frequencies, run_eda)
from askreddit.eda_report import FIGURES, render_all

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/fang_chua_v2.json").read_text())
TITLES = ["What is your favorite food?", "Why do cats purr?", "If you could fly, where would you go?",
          "[Serious] How do you cope with stress?", "Would you rather have X or Y?", "I just moved, what now",
          "Redditors, which movie is overrated?"]


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = create_spark("1g", 2)
    yield session
    session.stop()


def test_exact_quantiles_and_validation():
    pairs = [(10, 2), (0, 5), (1, 3)]
    assert quantiles_from_frequencies(pairs, (0.5, 0.6, 0.8, 0.81, 1)) == {"0.5": 0, "0.6": 1, "0.8": 1, "0.81": 10, "1": 10}
    assert quantiles_from_frequencies([(7, 1)], (0.001,)) == {"0.001": 7}
    with pytest.raises(ValueError):
        quantiles_from_frequencies([], (0.5,))
    with pytest.raises(ValueError):
        quantiles_from_frequencies(pairs, (0,))


def test_bucket_labels():
    assert bucket_labels((0, 4, 7, 8)) == ["0–3", "4–6", "7", "≥8"]
    assert bucket_labels((0.0, 0.1), integer=False) == ["0.00–0.10", "≥0.10"]


def fixture_rows():
    start, end = utc_seconds("2025-01-01T00:00:00Z"), utc_seconds("2026-01-01T00:00:00Z")
    rows = []
    for i in range(700):
        created = start + (end - start - 1) * i // 699
        score = (i * 37) % 600 if i % 3 else i % 2
        rows.append((f"t{i:04d}", "AskReddit", TITLES[i % len(TITLES)], "" if i % 4 else "some body text",
                     created, score))
    return rows


def write_fixture(spark, path):
    frame = spark.createDataFrame(fixture_rows(), "id string, subreddit string, title string, selftext string, created_utc long, score long")
    from pyspark.sql import functions as F
    frame = frame.withColumns({"year": F.year(F.timestamp_seconds("created_utc")), "month": F.month(F.timestamp_seconds("created_utc"))})
    frame.write.partitionBy("year", "month").parquet(str(path))


def test_utc_time_and_title_patterns(spark):
    monday = utc_seconds("2025-01-06T00:00:00Z")
    rows = [("a", "AskReddit", "What would you do?", "", monday, 5, 2025, 1),
            ("b", "AskReddit", "if you could live anywhere?", "body", monday - 1, 0, 2025, 1),
            ("c", "AskReddit", "(serious) Why is my code slow", "", monday + 13 * 3600, 1, 2025, 1)]
    raw = spark.createDataFrame(rows, "id string, subreddit string, title string, selftext string, created_utc long, score long, year int, month int")
    result = {r.id: r for r in analysis_frame(raw, CONFIG).collect()}
    assert (result["a"].weekday, result["a"].hour) == (0, 0)
    assert (result["b"].weekday, result["b"].hour) == (6, 23)
    assert result["c"].hour == 13 and result["c"].serious_tag and result["c"].starts_why and result["c"].first_person
    assert result["a"].starts_what and result["a"].addresses_reader and not result["a"].starts_if
    assert result["b"].starts_if and result["b"].contains_if_you_could and not result["b"].title_only
    assert result["a"].title_only and result["a"].question_present == 1 and result["c"].question_count == 0
    assert result["a"].split == "train" and result["a"].char_bucket == 0 and result["a"].word_bucket == 1


def test_end_to_end_outputs_and_train_only_associations(spark, tmp_path):
    source = tmp_path / "clean"
    write_fixture(spark, source)
    output = tmp_path / "eda"
    run = run_eda(spark, source, CONFIG, output, "configs/fang_chua_v2.json")
    assert run["records"] == 700 and run["association_scope"] == "train"
    data = {name: json.loads((output / f"{name}.json").read_text()) for name in OUTPUT_FILES}
    assert data["profile"]["records"] == 700 and data["profile"]["ids_unique"]
    assert sum(m["records"] for m in data["monthly_volume"]["months"]) == 700
    assert len(data["monthly_volume"]["months"]) == 12
    labels = data["label_feasibility"]
    assert sum(d["total"] for d in labels["class_distributions"].values()) == 700
    train_records = labels["class_distributions"]["train"]["total"]
    assert data["associations"]["patterns"]["overall"]["records"] == train_records
    assert sum(b["records"] for b in data["associations"]["title_characters"]) == train_records
    assert sum(cell["records"] for cell in data["posting_time"]["train_weekday_hour"]) == train_records
    assert sum(h["records_all_year"] for h in data["posting_time"]["by_hour"]) == 700
    expected_what = sum(1 for i, row in enumerate(fixture_rows()) if row[3] is not None and row[2].startswith("What")
                        and row[4] < 1759276800)
    what = next(p for p in data["associations"]["patterns"]["patterns"] if p["pattern"] == "starts_what")
    assert what["records"] == expected_what
    assert data["score_distribution"]["all"]["records"] == 700
    assert sum(c for _, c in data["score_frequencies"]["all"]) == 700
    with pytest.raises(ValueError):
        run_eda(spark, source, CONFIG, output)
    paths = render_all(output)
    assert [p.stem for p in paths[:-1]] == list(FIGURES) and all(p.stat().st_size > 0 for p in paths)
    summary = (output / "summary_vi.md").read_text()
    assert "Phân tích khám phá dữ liệu" in summary and "700" in summary
