"""Headless smoke test of the Streamlit demo against the generated reports and exported model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/demo.json").read_text())
REQUIRED = [ROOT / CONFIG["eda_dir"] / "profile.json", ROOT / CONFIG["experiment_dir"] / "completed.json",
            ROOT / CONFIG["model_dir"] / "model.json", ROOT / CONFIG["benchmark_dir"] / "results.json",
            ROOT / CONFIG["binary_experiment_dir"] / "completed.json", ROOT / CONFIG["binary_model_dir"] / "model.json",
            ROOT / CONFIG["engagement_model_dir"] / "model.json", ROOT / CONFIG["score_model_dir"] / "model.json",
            ROOT / CONFIG["score_experiment_dir"] / "completed.json", ROOT / CONFIG["dedup_report"]]


@pytest.mark.skipif(not all(p.exists() for p in REQUIRED), reason="Generated reports or exported model are not present")
def test_app_renders_every_tab_and_predicts():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert not app.warning, [w.value for w in app.warning]  # every data source was found
    app.text_input[0].input("What's the most useless talent you have?")
    next(b for b in app.button if b.label == "Dự đoán").click()
    app.run()
    assert not app.exception, app.exception
    assert any("phổ biến" in s.value for s in app.success)
    assert len(app.text_input) == 1  # title only: no posting-time inputs anywhere
    assert any("Khả năng có tương tác" in m.label for m in app.metric)
    assert any("Khả năng viral" in m.label for m in app.metric)
    for radio in app.radio:
        radio.set_value("test").run()
    assert not app.exception
