"""Streamlit demo: AskReddit 2025 analysis, binary popularity model, eight-level extension and benchmark.

Run from the repository root:
    .venv/bin/streamlit run app/streamlit_app.py

Reads saved reports (EDA, experiments, benchmark) and the exported models listed
in configs/demo.json. It never starts Spark and never reads the score of a new
question. The binary model (registered proposal) uses only the title; the
eight-level extension also uses the UTC posting time.
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
from askreddit.binary_inference import BinaryPopularityModel  # noqa: E402
from askreddit.inference import PopularityModel  # noqa: E402
from askreddit.score_inference import ScoreModel  # noqa: E402

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


@st.cache_resource
def load_binary_model(relative: str):
    path = ROOT / relative
    return BinaryPopularityModel.load(path) if (path / "model.json").exists() else None


@st.cache_resource
def load_score_model(relative: str):
    path = ROOT / relative
    return ScoreModel.load(path) if (path / "model.json").exists() else None


def term(name: str) -> str:
    return name[2:] if name[:2] in ("u:", "b:") else name


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
           "Bài toán chính: phân loại **phổ biến cao / thấp** chỉ từ **tiêu đề**; score chỉ dùng để tạo nhãn.")

eda_dir, run_dir = CONFIG["eda_dir"], CONFIG["experiment_dir"]
binary_dir, reference_dirs = CONFIG["binary_experiment_dir"], CONFIG.get("binary_reference_dirs", [])
engagement_dir, score_dir = CONFIG["engagement_experiment_dir"], CONFIG["score_experiment_dir"]
profile = read_json(f"{eda_dir}/profile.json")
tabs = st.tabs(["Tổng quan dữ liệu", "Tiêu đề & thời gian", "Mô hình cao / thấp", "Dự đoán score (2 bước)", "Dự đoán",
                "Mở rộng: 8 mức (Fang/Chua)", "Benchmark Spark vs DuckDB"])

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
                   f"ID duy nhất: {'có' if profile['ids_unique'] else 'không'}. Bài chỉ có tiêu đề: {pct(profile['title_only_share'])}. "
                   "Score là giá trị Arctic Shift lấy lại khoảng 36 giờ sau khi đăng.")
        dedup = read_json(CONFIG.get("dedup_report", ""))
        if dedup:
            st.subheader("Bỏ bài trùng tiêu đề (dữ liệu dùng cho mô hình)")
            cols = st.columns(4)
            cols[0].metric("Sau khi bỏ trùng", f"{dedup['output_records']:,}")
            cols[1].metric("Tiêu đề lặp lại", f"{dedup['repeated_titles']:,}", f"{dedup['postings_in_repeated_titles']:,} bài", delta_color="off")
            cols[2].metric("Bài trùng bị bỏ", f"{dedup['non_best_postings_dropped']:,}")
            cols[3].metric("Placeholder bị bỏ", f"{dedup['placeholder_titles_dropped']:,}")
            st.caption("Mỗi tiêu đề chuẩn hóa chỉ giữ bài có score cao nhất trong năm 2025 (score của câu hỏi khi \"gặp may\" nhất). "
                       "Cùng một câu hỏi có thể được 1 điểm ở lần đăng này và hàng nghìn điểm ở lần khác.")
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
        binary_thresholds = read_json(f"{binary_dir}/label_thresholds.json")
        binary_groups = read_json(f"{binary_dir}/class_distributions.json")
        if binary_thresholds and binary_groups:
            st.subheader("Nhãn phổ biến cao / thấp (ngưỡng fit trên train)")
            st.markdown(f"Tập tính phân vị: **{binary_thresholds['population']}** — trung vị {binary_thresholds['median']}, "
                        f"p75 {binary_thresholds['p75']}. **Cao:** {binary_thresholds['high_rule']}. **Thấp:** {binary_thresholds['low_rule']}. "
                        "Nhóm giữa và bài ngoài tập bị loại.")
            names = {"excluded": "Loại (score ≤ 1)", "low": "Thấp", "middle": "Giữa (loại)", "high": "Cao"}
            st.dataframe(pd.DataFrame([{"Split": split, **{names[g]: f"{d['groups'][g]['count']:,} ({d['groups'][g]['percentage_of_split']:.1f}%)"
                                                            for g in names if g in d["groups"]}}
                                       for split, d in binary_groups.items()]), hide_index=True, width="stretch")
        image(f"{eda_dir}/figures/score_distribution.png", "Phân phối score và đuôi trên (log–log)")


def eight_level_tables():
    labels = read_json(f"{eda_dir}/label_feasibility.json")
    if labels:
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
    comparison, completed = read_json(f"{binary_dir}/model_comparison.json"), read_json(f"{binary_dir}/completed.json")
    config = read_json(f"{binary_dir}/config.json")
    if comparison is None or completed is None:
        missing("kết quả mô hình nhị phân", f"PYTHONPATH=src .venv/bin/python -m askreddit.binary_experiment --config configs/binary_v4.json --run-id {Path(binary_dir).name}")
    else:
        selected = completed["selected_candidate"]
        best = completed.get("best_by_family", {})
        test = read_json(f"{binary_dir}/test/{selected}.json")
        st.markdown(f"Run **{Path(binary_dir).name}** — chọn bằng **AUC-ROC trên validation** (10–11/2025): **{selected}**. "
                    "Ngưỡng xác suất tối đa F1 lớp *cao* trên validation, cố định trước khi chấm test (12/2025). Lớp dương = phổ biến cao.")
        if config and config.get("disclosure_vi"):
            st.info(config["disclosure_vi"])
        cols = st.columns(5)
        cols[0].metric("AUC-ROC (test)", f"{test['auc_roc']:.4f}", f"{test['auc_roc'] - 0.5:+.4f} so với ngẫu nhiên")
        cols[1].metric("PR-AUC (test)", f"{test['pr_auc']:.4f}", f"{test['pr_auc'] - test['positive_rate']:+.4f} so với ngẫu nhiên")
        cols[2].metric("F1 lớp cao (test)", f"{test['f1']:.4f}", f"{test['f1'] - test['all_high_f1_reference']:+.4f} so với luôn đoán cao")
        cols[3].metric("Precision (test)", f"{test['precision']:.4f}")
        cols[4].metric("Recall (test)", f"{test['recall']:.4f}")
        st.caption(f"Tỉ lệ bài cao trên test: {pct(test['positive_rate'])}. Baseline lớp đa số (luôn đoán thấp) có AUC 0,5 và F1 = 0; "
                   f"luôn đoán cao cho F1 = {test['all_high_f1_reference']:.4f}.")
        split = st.radio("Split", ["validation", "test"], horizontal=True, key="binary_split")
        frame = pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Họ": r["kind"], "AUC-ROC": r["auc_roc"],
                               "PR-AUC": r["pr_auc"], "Ngưỡng": r["threshold"], "Precision": r["precision"], "Recall": r["recall"],
                               "F1": r["f1"], "Số chiều": r["feature_dimension"], "Train (giây)": round(r["training_seconds"], 1)}
                              for r in comparison["rows"] if r["split"] == split])
        st.dataframe(frame.sort_values("AUC-ROC", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("AUC-ROC", "PR-AUC", "Ngưỡng", "Precision", "Recall", "F1")})
        left, right = st.columns(2)
        with left:
            image(f"{binary_dir}/figures/roc_test.png", "Đường ROC trên test")
        with right:
            image(f"{binary_dir}/figures/confusion_test.png", f"Ma trận nhầm lẫn trên test — LR tốt nhất ({best.get('lr')}) và GBT tốt nhất ({best.get('gbt')})")
        st.subheader("N-gram quan trọng và feature importance")
        left, right = st.columns(2)
        with left:
            image(f"{binary_dir}/figures/lr_top_ngrams.png", "Hệ số Logistic Regression: xanh = đẩy về cao, đỏ = đẩy về thấp")
        with right:
            image(f"{binary_dir}/figures/gbt_importance.png", "Feature importance của GBT (top-k n-gram chọn bằng chi-square + cấu trúc tiêu đề)")
        st.subheader("Ảnh hưởng của thời điểm đăng (mô hình ablation chỉ dùng giờ/thứ)")
        image(f"{binary_dir}/figures/time_effects.png", "Thay đổi log-odds theo giờ và thứ (UTC). Mô hình chính không dùng thời gian.")
        for reference_dir in reference_dirs:
            reference = read_json(f"{reference_dir}/model_comparison.json")
            ref_config = read_json(f"{reference_dir}/config.json") or {}
            if reference:
                with st.expander(f"Run tham chiếu {Path(reference_dir).name}"):
                    if ref_config.get("disclosure_vi"):
                        st.caption(ref_config["disclosure_vi"])
                    st.dataframe(pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Split": r["split"],
                                                "AUC-ROC": round(r["auc_roc"], 4), "PR-AUC": round(r["pr_auc"], 4), "F1": round(r["f1"], 4)}
                                               for r in reference["rows"]]), hide_index=True, width="stretch")
                    image(f"{reference_dir}/figures/lr_top_ngrams.png", "Hệ số LR của run tham chiếu")

with tabs[3]:
    engagement, score_cmp = read_json(f"{engagement_dir}/model_comparison.json"), read_json(f"{score_dir}/model_comparison.json")
    population = read_json(f"{score_dir}/population.json")
    if engagement is None or score_cmp is None:
        missing("kết quả dự đoán score", "PYTHONPATH=src .venv/bin/python -m askreddit.score_regression --run-id <run>")
    else:
        st.markdown("Dự đoán score **hai bước**, chỉ từ tiêu đề, trên dữ liệu đã bỏ trùng. Score là giá trị **khoảng 36 giờ sau khi đăng** "
                    "(Arctic Shift lấy lại score sau ~36 giờ; với AskReddit đây gần như là score cuối cùng).")
        st.markdown("**Bước 1 — bài có tương tác không?** Phân loại score ≥ 2 với score ≤ 1 trên mọi bài. "
                    "Nhóm score ≤ 1 gồm cả bài bị mod gỡ, nên bước này học nhiều về luật của r/AskReddit.")
        st.dataframe(pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Split": r["split"],
                                    "AUC-ROC": round(r["auc_roc"], 4), "PR-AUC": round(r["pr_auc"], 4)}
                                   for r in engagement["rows"]]), hide_index=True, width="stretch")
        st.markdown("**Bước 2 — nếu có tương tác, score khoảng bao nhiêu?** Hồi quy ridge trên log(1 + score), các bài score ≥ 2.")
        if population:
            st.caption(f"Số bài score ≥ 2: train {population['population_records']['train']:,}, validation "
                       f"{population['population_records']['validation']:,}, test {population['population_records']['test']:,}.")
        frame = pd.DataFrame([{"Mô hình": r["candidate"] + (" ✅" if r["selected"] else ""), "Split": r["split"],
                               "RMSE (log)": r["rmse_log"], "MAE (log)": r["mae_log"], "R² (log)": r["r2_log"], "Spearman": r["spearman"],
                               "Sai số tuyệt đối trung vị (điểm)": r["median_abs_error_score"],
                               "Phủ khoảng 80%": r["interval_coverage"], "AUC cao vs thấp": r["auc_high_vs_low"]}
                              for r in score_cmp["rows"]])
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in frame.columns if c not in ("Mô hình", "Split")})
        st.caption("Spearman đo mô hình xếp hạng các tiêu đề đúng thứ tự đến đâu (dự đoán *tương đối*). "
                   "R² thấp là bình thường: cùng một câu hỏi có thể được 2 hay 2.000 điểm.")
        left, right = st.columns(2)
        with left:
            image(f"{score_dir}/figures/deciles_test.png", "Score thực tế theo nhóm thập phân của score dự đoán (test)")
        with right:
            image(f"{score_dir}/figures/top_ngrams.png", "N-gram làm tăng / giảm score dự đoán")

with tabs[4]:
    binary_model = load_binary_model(CONFIG["binary_model_dir"])
    engagement_model = load_binary_model(CONFIG["engagement_model_dir"])
    score_model = load_score_model(CONFIG["score_model_dir"])
    if binary_model is None or engagement_model is None or score_model is None:
        missing("mô hình xuất cho demo", "PYTHONPATH=src .venv/bin/python -m askreddit.export_binary_model --run-id <run>; "
                "PYTHONPATH=src .venv/bin/python -m askreddit.export_score_model --run-id <run>")
    else:
        st.markdown("Chỉ cần nhập **tiêu đề**. Ba mô hình Logistic/Ridge suy luận bằng NumPy (không khởi động Spark), "
                    "đã kiểm tra khớp với Spark (`verification.json`).")
        with st.form("predict"):
            title = st.text_input("Tiêu đề câu hỏi (tiếng Anh)", "What song instantly makes you happy?")
            submitted = st.form_submit_button("Dự đoán")
        if submitted and title.strip():
            engaged = engagement_model.calibrated_probability(title)
            estimate = score_model.predict_score(title)
            group = binary_model.predict(title)
            low, high = estimate["interval"]
            st.success(f"Nhóm phổ biến (đề cương): **phổ biến {'CAO' if group['label'] == 'high' else 'THẤP'}** "
                       f"(điểm {group['probability_high']:.3f}, ngưỡng {group['threshold']:.2f}).")
            cols = st.columns(3)
            cols[0].metric("Khả năng có tương tác (score ≥ 2)", f"{100 * engaged:.0f}%")
            cols[1].metric("Nếu có tương tác: score dự kiến", f"≈ {estimate['score']:.0f}", f"khoảng {low:.0f}–{high:.0f} (80%)", delta_color="off")
            cols[2].metric("Xếp hạng tương đối", f"cao hơn {estimate['relative_percentile']}%", "so với các bài có tương tác", delta_color="off")
            st.caption("Score đo khoảng 36 giờ sau khi đăng. Khoảng 80% lấy từ sai số trên validation; "
                       "các bài viral (hàng nghìn điểm) nằm ngoài khoảng này vì chúng phụ thuộc thời điểm và vài vote đầu, không chỉ tiêu đề.")
            st.subheader("Từ ngữ đẩy score lên / xuống")
            st.dataframe(pd.DataFrame([{"Đặc trưng": term(c["feature"]), "Đóng góp vào log(1 + score)": round(c["contribution"], 4),
                                        "Hướng": "↑ score" if c["contribution"] > 0 else "↓ score"}
                                       for c in score_model.contributions(title, top=10)]), hide_index=True, width="stretch")
        terms = score_model.top_terms(20)
        if terms["high"]:
            with st.expander("Từ/cụm từ có hệ số lớn nhất trong mô hình score"):
                left, right = st.columns(2)
                left.markdown("**Tăng score**")
                left.dataframe(pd.DataFrame([{"N-gram": term(t["term"]), "Hệ số": round(t["coefficient"], 3)} for t in terms["high"]]),
                               hide_index=True, width="stretch")
                right.markdown("**Giảm score**")
                right.dataframe(pd.DataFrame([{"N-gram": term(t["term"]), "Hệ số": round(t["coefficient"], 3)} for t in terms["low"]]),
                                hide_index=True, width="stretch")
        st.caption("Liên hệ giữa từ ngữ và độ phổ biến là quan hệ thống kê; không có nghĩa là đổi tiêu đề sẽ làm bài phổ biến hơn.")

with tabs[5]:
    st.info("Phần mở rộng ngoài đề cương: tám mức độ phổ biến theo Fang và cộng sự (2016) và Chua (2021). "
            "Mô hình này dùng cả tiêu đề và giờ đăng UTC.")
    eight_level_tables()
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

with tabs[5]:
    st.caption("Mô hình tám mức dùng cả giờ đăng UTC nên không còn form dự đoán trong demo (demo chỉ nhập tiêu đề). "
               "Kết quả đánh giá ở trên vẫn giữ nguyên.")

with tabs[6]:
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
