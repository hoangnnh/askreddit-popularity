"""Streamlit demo: AskReddit 2025 analysis, model evaluation and Spark-free prediction.

Run from the repository root:
    .venv/bin/streamlit run app/streamlit_app.py

Reads saved reports (EDA, experiments, benchmark) and the exported model listed
in configs/demo.json. It never starts Spark and never reads the score of a new
question: predictions use only the title and the UTC posting time.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from askreddit.inference import PopularityModel  # noqa: E402

BLUE, ORANGE = "#257de0", "#f47230"
CONFIG = json.loads((ROOT / "configs" / "demo.json").read_text())
WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


@st.cache_data
def read_json(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource
def load_model(relative: str):
    path = ROOT / relative
    return PopularityModel.load(path) if (path / "model.json").exists() else None


def image(relative: str, caption: str | None = None):
    path = ROOT / relative
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa có hình `{relative}`.")


def pct(value):
    return f"{100 * value:.2f}%"


def missing(what: str, command: str):
    st.warning(f"Chưa có {what}. Tạo bằng lệnh:")
    st.code(command, language="bash")


st.set_page_config(page_title="AskReddit Popularity", page_icon="📊", layout="wide")
st.title("Dự đoán độ phổ biến câu hỏi r/AskReddit")
st.caption("IE212 — Nhóm 14. Dữ liệu: các submission r/AskReddit năm 2025 (Arctic Shift, bản Parquet open-index/arctic). "
           "Mô hình chỉ dùng **tiêu đề** và **thời điểm đăng (UTC)**; score chỉ dùng để tạo nhãn.")

eda_dir, run_dir = CONFIG["eda_dir"], CONFIG["experiment_dir"]
profile = read_json(f"{eda_dir}/profile.json")
tabs = st.tabs(["Tổng quan dữ liệu", "Tiêu đề & thời gian", "Đánh giá mô hình", "Dự đoán", "Benchmark Spark vs DuckDB"])

with tabs[0]:
    if profile is None:
        missing("kết quả EDA", "PYTHONPATH=src .venv/bin/python -m askreddit.eda --output reports/eda")
    else:
        scores, labels = read_json(f"{eda_dir}/score_distribution.json"), read_json(f"{eda_dir}/label_feasibility.json")
        monthly = read_json(f"{eda_dir}/monthly_volume.json")
        cols = st.columns(4)
        cols[0].metric("Bài đã làm sạch", f"{profile['records']:,}")
        cols[1].metric("Score ≤ 1", pct(scores["all"]["share_le_1"]))
        cols[2].metric("Score cao nhất", f"{scores['all']['max']:,}")
        cols[3].metric("Bài trung bình mỗi ngày", f"{monthly['daily_summary']['mean']:,.0f}")
        st.caption(f"Thời gian phủ: {profile['coverage_utc'][0][:10]} → {profile['coverage_utc'][1][:10]} (UTC). "
                   f"ID duy nhất: {'có' if profile['ids_unique'] else 'không'}. Bài chỉ có tiêu đề: {pct(profile['title_only_share'])}.")
        left, right = st.columns(2)
        with left:
            st.subheader("Số bài theo tháng")
            frame = pd.DataFrame([{"Tháng": f"{m['month']:02d}", "Số bài": m["records"]} for m in monthly["months"]])
            st.bar_chart(frame, x="Tháng", y="Số bài", color=BLUE)
        with right:
            st.subheader("Score 0–20 (% tổng số bài)")
            total = scores["all"]["records"]
            frame = pd.DataFrame([{"Score": r["score"], "% bài": 100 * r["records"] / total} for r in scores["small_scores"]])
            st.bar_chart(frame, x="Score", y="% bài", color=BLUE)
        st.subheader("Tám mức độ phổ biến (ngưỡng fit trên train)")
        rows = []
        for mapping in labels["thresholds"]["mapping"]:
            k, low, high = mapping["label"], mapping["lower_inclusive"], mapping["upper_exclusive"]
            text = f"≤ {high - 1}" if low is None else f"≥ {low}" if high is None else (str(low) if high == low + 1 else f"{low}–{high - 1}")
            row = {"Nhãn": k, "Score": text}
            for split, name in (("train", "Train"), ("validation", "Validation"), ("test", "Test")):
                item = labels["class_distributions"][split]["labels"][k]
                row[name] = f"{item['count']:,} ({item['percentage']:.2f}%)"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        image(f"{eda_dir}/figures/score_distribution.png", "Phân phối score và đuôi trên (log–log)")

with tabs[1]:
    timing, associations = read_json(f"{eda_dir}/posting_time.json"), read_json(f"{eda_dir}/associations.json")
    if timing is None:
        missing("kết quả EDA", "PYTHONPATH=src .venv/bin/python -m askreddit.eda --output reports/eda")
    else:
        st.caption("Các tỉ lệ độ phổ biến chỉ tính trên **train** (01/01–30/09/2025). Đây là quan hệ mô tả, không phải nhân quả.")
        left, right = st.columns(2)
        with left:
            st.subheader("Số bài theo giờ UTC (cả năm)")
            frame = pd.DataFrame([{"Giờ UTC": h["label"], "Số bài": h["records_all_year"]} for h in timing["by_hour"]])
            st.bar_chart(frame, x="Giờ UTC", y="Số bài", color=BLUE)
        with right:
            st.subheader("% score ≥ 2 theo giờ UTC (train)")
            frame = pd.DataFrame([{"Giờ UTC": h["label"], "% score ≥ 2": 100 * h["train"]["share_score_ge_2"]}
                                  for h in timing["by_hour"] if h["train"]])
            st.bar_chart(frame, x="Giờ UTC", y="% score ≥ 2", color=BLUE)
        image(f"{eda_dir}/figures/weekday_hour_heatmap.png", "Thứ × giờ UTC trên train")
        st.subheader("Mẫu câu hỏi trong tiêu đề (train)")
        overall = associations["patterns"]["overall"]
        frame = pd.DataFrame([{"Mẫu": p["pattern"], "Số bài": p["records"], "% score ≥ 2": round(100 * p["share_score_ge_2"], 2),
                               "Lift (score ≥ 2)": round(p["lift_score_ge_2"], 3), "% nhãn 6–7": round(100 * p["share_label_ge_6"], 3),
                               "Lift (nhãn 6–7)": round(p["lift_label_ge_6"], 3)}
                              for p in associations["patterns"]["patterns"] if p["records"]]).sort_values("Lift (score ≥ 2)", ascending=False)
        st.caption(f"Toàn train: {pct(overall['share_score_ge_2'])} có score ≥ 2; {pct(overall['share_label_ge_6'])} thuộc nhãn 6–7.")
        st.dataframe(frame, hide_index=True, width="stretch")
        image(f"{eda_dir}/figures/length_popularity.png", "Độ dài tiêu đề và độ phổ biến (train)")

with tabs[2]:
    comparison, completed = read_json(f"{run_dir}/model_comparison.json"), read_json(f"{run_dir}/completed.json")
    config = read_json(f"{run_dir}/config.json")
    if comparison is None or completed is None:
        missing("kết quả thí nghiệm", "PYTHONPATH=src .venv/bin/python -m askreddit.experiment --config configs/fang_chua_v2.json --run-id <run>")
    else:
        selected = completed["selected_candidate"]
        test = read_json(f"{run_dir}/test/{selected}.json")
        baseline = next(r for r in comparison["rows"] if r["candidate"] == "majority" and r["split"] == "test")
        st.markdown(f"Run **{Path(run_dir).name}** — mô hình được chọn bằng macro-F1 trên validation: **{selected}**.")
        if config and config.get("disclosure_vi"):
            st.info(config["disclosure_vi"])
        cols = st.columns(4)
        cols[0].metric("Macro-F1 (test)", f"{test['macro_f1']:.4f}", f"{test['macro_f1'] - baseline['macro_f1']:+.4f} so với baseline")
        cols[1].metric("Balanced accuracy (test)", f"{test['balanced_accuracy']:.4f}", f"{test['balanced_accuracy'] - baseline['balanced_accuracy']:+.4f}")
        cols[2].metric("Mean threshold F1 (test)", f"{test['mean_threshold_f1']:.4f}")
        cols[3].metric("Accuracy (test)", pct(test["accuracy"]), f"{100 * (test['accuracy'] - baseline['accuracy']):+.1f} điểm %", delta_color="off")
        st.caption("Baseline luôn đoán lớp 0 đạt accuracy ~70% nhưng macro-F1 chỉ ~0,10. Với 8 lớp mất cân bằng, macro-F1 là thước đo chính.")
        split = st.radio("Split", ["validation", "test"], horizontal=True, key="split")
        frame = pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Macro-F1": r["macro_f1"],
                               "Balanced acc.": r["balanced_accuracy"], "Accuracy": r["accuracy"], "Weighted-F1": r["weighted_f1"],
                               "Mean threshold F1": r["mean_threshold_f1"], "Số chiều": r["feature_dimension"],
                               "Train (giây)": round(r["training_seconds"], 1)} for r in comparison["rows"] if r["split"] == split])
        st.dataframe(frame.sort_values("Macro-F1", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Macro-F1", "Balanced acc.", "Accuracy", "Weighted-F1", "Mean threshold F1")})
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Từng lớp trên test")
            st.dataframe(pd.DataFrame([{"Nhãn": r["label"], "Precision": round(r["precision"], 4), "Recall": round(r["recall"], 4),
                                        "F1": round(r["f1"], 4), "Support": r["support"]} for r in test["per_class"]]),
                         hide_index=True, width="stretch")
        with right:
            image(f"{run_dir}/figures/{selected}_test.png", "Confusion matrix trên test")
        previous = CONFIG.get("previous_experiment_dir")
        if previous and read_json(f"{previous}/model_comparison.json"):
            with st.expander(f"So sánh với run trước ({Path(previous).name})"):
                rows = read_json(f"{previous}/model_comparison.json")["rows"]
                st.dataframe(pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Split": r["split"],
                                            "Macro-F1": round(r["macro_f1"], 4), "Accuracy": round(r["accuracy"], 4)} for r in rows]),
                             hide_index=True, width="stretch")

with tabs[3]:
    model = load_model(CONFIG["model_dir"])
    if model is None:
        missing("mô hình xuất cho demo", f"PYTHONPATH=src .venv/bin/python -m askreddit.export_model --run-id {Path(run_dir).name}")
    else:
        meta = model.metadata
        st.markdown(f"Mô hình **{meta['candidate']['id']}** (run `{meta['run_id']}`), suy luận bằng NumPy — không khởi động Spark. "
                    "Kết quả khớp với Spark trên mẫu kiểm tra (xem `verification.json`).")
        with st.form("predict"):
            title = st.text_input("Tiêu đề câu hỏi (tiếng Anh)", "What's a small habit that completely changed your life?")
            c1, c2 = st.columns(2)
            day = c1.date_input("Ngày đăng (UTC)", date(2025, 12, 15))
            moment = c2.time_input("Giờ đăng (UTC)", time(15, 0))
            submitted = st.form_submit_button("Dự đoán")
        if submitted and title.strip():
            created = int(datetime.combine(day, moment, tzinfo=timezone.utc).timestamp())
            result = model.predict(title, created)
            probabilities = result["probabilities"]
            st.success(f"Nhãn dự đoán: **{result['label']}** ({result['score_range']}) — xác suất {pct(probabilities[result['label']])}. "
                       f"Đăng vào {WEEKDAYS_VI[datetime.fromtimestamp(created, timezone.utc).weekday()]}, {moment.strftime('%H:%M')} UTC "
                       f"({(moment.hour + 7) % 24:02d}:{moment.minute:02d} giờ Việt Nam).")
            frame = pd.DataFrame([{"Nhãn": f"{m['label']}: {m['score_range']}", "Xác suất (%)": 100 * p}
                                  for m, p in zip(meta["label_mapping"], probabilities)])
            st.bar_chart(frame, x="Nhãn", y="Xác suất (%)", color=BLUE)
            st.caption("Mô hình được huấn luyện với trọng số lớp cân bằng, nên xác suất các lớp hiếm bị đẩy lên so với tần suất thực; "
                       "hãy đọc đây là mức độ tương đối, không phải xác suất đã hiệu chỉnh.")
            st.subheader("Đặc trưng đóng góp nhiều nhất vào nhãn dự đoán")
            st.dataframe(pd.DataFrame(model.contributions(title, created, result["label"], top=10)),
                         hide_index=True, width="stretch")
        terms = model.top_terms(7, 20)
        if terms:
            with st.expander("Từ/cụm từ có hệ số tương đối cao nhất cho nhãn 7 (score ≥ 271)"):
                st.dataframe(pd.DataFrame(terms), hide_index=True, width="stretch")
        st.caption("Liên hệ giữa từ ngữ, giờ đăng và độ phổ biến là quan hệ thống kê; không có nghĩa là đổi tiêu đề sẽ làm bài phổ biến hơn.")

with tabs[4]:
    benchmark_dir = CONFIG.get("benchmark_dir")
    results = read_json(f"{benchmark_dir}/results.json") if benchmark_dir else None
    if results is None:
        missing("kết quả benchmark", "PYTHONPATH=src .venv/bin/python -m askreddit.benchmark --threads 2 4")
    else:
        st.markdown("Cùng một đoạn SQL, cùng bộ Parquet, cùng số luồng. Spark chạy local trên một máy.")
        st.caption(f"Kết quả hai engine khớp nhau: {'có' if results['all_results_match'] else 'KHÔNG'}. "
                   f"CPU: {results['environment']['cpu']}; Spark {results['engines']['spark']}; DuckDB {results['engines']['duckdb']}.")
        frame = pd.DataFrame([{"Luồng": r["threads"], "Truy vấn": r["query"], "Spark (s)": round(r["spark_median_seconds"], 3),
                               "DuckDB (s)": round(r["duckdb_median_seconds"], 4), "Spark / DuckDB": round(r["spark_to_duckdb_ratio"], 1)}
                              for r in results["summary"]])
        st.dataframe(frame, hide_index=True, width="stretch")
        image(f"{benchmark_dir}/figures/median_times.png", "Median thời gian truy vấn (thang log)")
