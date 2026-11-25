"""Figures and a Vietnamese results summary for a completed binary run (reads saved JSON only; no Spark)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]  # categorical slots 1-4, fixed order
POSITIVE, NEGATIVE = "#257de0", "#ec4647"  # diverging poles (blue = toward high, red = toward low)
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
FAMILY_VI = {"majority": "Baseline lớp đa số", "lr": "Logistic Regression (L2)", "gbt": "Gradient Boosted Trees"}
FEATURES_VI = {"time_features": "giờ + thứ (UTC)", "structured_features": "độ dài + dấu hỏi + chữ hoa + giờ/thứ",
               "text_features": "TF-IDF 1–2-gram", "combined_features": "TF-IDF + đặc trưng cấu trúc",
               "selected_features": "top-k n-gram (chi-square) + đặc trưng cấu trúc",
               "title_structured_features": "độ dài + dấu hỏi + chữ hoa (không thời gian)",
               "combined_title_features": "TF-IDF + cấu trúc tiêu đề (không thời gian)",
               "selected_title_features": "top-k n-gram (chi-square) + cấu trúc tiêu đề (không thời gian)"}
STRUCTURED_VI = {"title_characters": "Số ký tự", "title_words": "Số từ", "question_count": "Số dấu hỏi",
                 "question_present": "Có dấu hỏi", "uppercase_ratio": "Tỉ lệ chữ hoa", "hour_sin": "Giờ (sin)",
                 "hour_cos": "Giờ (cos)", "weekday_sin": "Thứ (sin)", "weekday_cos": "Thứ (cos)"}
WEEKDAYS_VI = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def load(path: Path):
    return json.loads(path.read_text())


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}%"


def num(value, digits=4):
    return f"{value:.{digits}f}"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def roc_figure(plt, curves, destination):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Ngẫu nhiên (AUC 0.5)")
    for color, (label, curve, auc) in zip(SERIES, curves):
        ax.plot(curve["fpr"], curve["tpr"], color=color, linewidth=2, label=f"{label} (AUC {auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Tỉ lệ dương tính giả (FPR)", ylabel="Tỉ lệ dương tính thật (Recall)",
           title="Đường ROC trên tập test (12/2025)")
    style(ax)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def confusion_figure(plt, panels, destination):
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#f4f9ff", "#cce3fd", "#84b9f3", "#338ced", "#1b64be"])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), constrained_layout=True, squeeze=False)
    for ax, (title, metrics) in zip(axes[0], panels):
        matrix = metrics["confusion_matrix"]
        rows = [[v / sum(r) if sum(r) else 0.0 for v in r] for r in matrix]
        ax.imshow(rows, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]:,}\n({rows[i][j]:.1%})", ha="center", va="center", fontsize=11,
                        color="white" if rows[i][j] > 0.6 else TEXT)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Thấp", "Cao"], yticklabels=["Thấp", "Cao"],
               xlabel="Dự đoán", ylabel="Thực tế", title=f"{title}\nngưỡng {metrics['threshold']:.3f}")
        ax.tick_params(colors=MUTED, labelcolor=TEXT)
    fig.suptitle("Ma trận nhầm lẫn trên tập test (% theo hàng)", color=TEXT)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def coefficient_figure(plt, entry, title, destination, top=15):
    # barh draws index 0 at the bottom: strongest negative at the bottom, strongest positive at the top.
    items = entry["top_negative"][:top] + entry["top_positive"][:top][::-1]
    labels = [i["term"] for i in items]
    values = [i["coefficient"] for i in items]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(items) + 1.4), constrained_layout=True)
    ax.barh(range(len(items)), values, color=[POSITIVE if v > 0 else NEGATIVE for v in values], height=0.7)
    ax.set_yticks(range(len(items)), labels, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set(xlabel="Hệ số (log-odds trên TF-IDF chuẩn hóa L2)", title=title)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def importance_figure(plt, entry, title, destination, top=20):
    items = entry["top_features"][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(items) + 1.4), constrained_layout=True)
    ax.barh(range(len(items)), [i["importance"] for i in items], color=SERIES[0], height=0.7)
    ax.set_yticks(range(len(items)), [STRUCTURED_VI.get(i["feature"], i["feature"]) for i in items], fontsize=9)
    ax.set(xlabel="Feature importance (tổng = 1)", title=title)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def time_figure(plt, series, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, gridspec_kw={"width_ratios": [24, 9]})
    for color, (label, effects) in zip(SERIES, series):
        axes[0].plot(range(24), effects["hour_utc"], color=color, linewidth=2, marker="o", markersize=4, label=label)
        axes[1].plot(range(7), effects["weekday_monday0"], color=color, linewidth=2, marker="o", markersize=4, label=label)
    axes[0].set(xticks=range(0, 24, 3), xlabel="Giờ đăng (UTC)", ylabel="Thay đổi log-odds so với trung bình",
                title="Ảnh hưởng của giờ đăng")
    axes[1].set(xticks=range(7), xlabel="Thứ (UTC)", title="Ảnh hưởng của thứ")
    axes[1].set_xticklabels(WEEKDAYS_VI)
    for ax in axes:
        ax.axhline(0, color=MUTED, linewidth=1)
        style(ax)
    axes[0].legend(frameon=False, labelcolor=TEXT)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(header))) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def build(reports: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = load(reports / "config.json")
    thresholds = load(reports / "label_thresholds.json")
    distributions = load(reports / "class_distributions.json")
    features = load(reports / "features.json")
    selection = load(reports / "selection.json")
    analysis = load(reports / "analysis.json")
    verification = load(reports / "independent_verification.json")
    results = {}
    for spec in config["candidates"]:
        candidate = load(reports / "candidates" / f"{spec['id']}.json")
        if candidate["status"] == "completed":
            candidate["test"] = load(reports / "test" / f"{spec['id']}.json")
        results[spec["id"]] = candidate
    selected = selection["selected_candidate"]
    best_lr, best_gbt = selection["best_by_family"].get("lr"), selection["best_by_family"].get("gbt")
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)

    curves = [(cid, results[cid]["test"]["roc_curve"], results[cid]["test"]["auc_roc"])
              for cid in dict.fromkeys(c for c in ("time_lr", "structured_lr", "title_structured_lr", best_lr, best_gbt) if c and c in results
                                       and results[c]["status"] == "completed")]
    roc_figure(plt, curves[-4:], figures / "roc_test.png")
    panels = [(f"{best_lr} (max-F1 trên validation)", results[best_lr]["test"])] if best_lr else []
    panels += [(f"{cid} (xác suất 0,5)", results[cid]["test"]["at_probability_0_5"]) for cid in (best_lr, best_gbt) if cid]
    confusion_figure(plt, panels, figures / "confusion_test.png")
    lr_entry = analysis["logistic_regression"].get(best_lr, {})
    if "top_positive" in lr_entry:
        coefficient_figure(plt, lr_entry, f"N-gram ảnh hưởng mạnh nhất — {best_lr}", figures / "lr_top_ngrams.png")
    if best_gbt:
        importance_figure(plt, analysis["gradient_boosted_trees"][best_gbt], f"Feature importance — {best_gbt}",
                          figures / "gbt_importance.png")
    time_series = [(cid, analysis["logistic_regression"][cid]["time_effects"]) for cid in ("time_lr", best_lr)
                   if cid in analysis["logistic_regression"] and "time_effects" in analysis["logistic_regression"][cid]]
    if time_series:
        time_figure(plt, time_series, figures / "time_effects.png")

    md = [f"# Kết quả bài toán nhị phân — run `{reports.name}`", "",
          "Báo cáo sinh tự động bằng `python -m askreddit.binary_report` từ các file JSON cùng thư mục.", "",
          f"> **Công bố:** {config['disclosure_vi']}", "",
          "## 1. Gán nhãn (chỉ dùng tập train)", "",
          f"- Tập tính phân vị: **{thresholds.get('population', 'all train posts')}**. Trung vị = **{thresholds['median']}**, "
          f"phân vị 75% = **{thresholds['p75']}** ({thresholds['quantile_definition']}). "
          f"DuckDB kiểm tra độc lập: **{verification['status']}**.",
          f"- **Phổ biến cao:** {thresholds['high_rule']}. **Phổ biến thấp:** {thresholds['low_rule']}. "
          "Nhóm giữa (và bài ngoài tập tính phân vị, nếu có) bị loại khỏi huấn luyện và đánh giá.",
          f"- {config.get('labeling', {}).get('rationale_vi', 'Vì 71% bài train có score ≤ 1 nên không cắt được đúng 25% cao nhất mà không tách các bài cùng điểm; quy tắc > p75 giữ nhóm cao nằm hẳn trên ranh giới phân vị.')}", "",
          table(["Split", "Tổng", "Loại (ngoài tập)", "Thấp", "Giữa (loại)", "Cao", "Tỉ lệ cao trong dữ liệu giữ lại"],
                [[s, f"{d['total']:,}", f"{d['groups'].get('excluded', {}).get('count', 0):,}", f"{d['groups']['low']['count']:,}",
                  f"{d['groups']['middle']['count']:,}", f"{d['groups']['high']['count']:,}", pct(d["positive_rate_among_kept"])]
                 for s, d in distributions.items()]), "",
          "## 2. Chọn N-gram bằng chi-square (train)", "",
          f"{features['chi_square']['method']}. Chọn **{features['chi_square']['num_top_features']}** n-gram cho GBT "
          f"(tỉ lệ cao toàn train: {pct(features['chi_square']['train_positive_rate'])}).", "",
          table(["Hạng", "N-gram", "Chi²", "Số bài train", "Tỉ lệ cao khi có n-gram"],
                [[t["rank"], f"`{t['term'][2:]}`", f"{t['statistic']:,.0f}", f"{t['train_documents']:,}",
                  pct(t["high_rate_when_present"])] for t in features["chi_square"]["selected_terms"][:15]]), "",
          "## 3. So sánh mô hình", "",
          f"Chọn mô hình bằng **{'PR-AUC' if config.get('selection', {}).get('criterion') == 'validation_pr_auc' else 'AUC-ROC'} trên validation** "
          f"(10–11/2025). Cột *Thiết kế* ghi cách xử lý mất cân bằng (trọng số lớp hoặc undersampling lớp thấp chỉ trên train). "
          f"Ngưỡng xác suất của từng mô hình là ngưỡng tối đa "
          f"F1 lớp cao trên validation, cố định trước khi chấm test (12/2025). Lớp dương = phổ biến cao.", ""]
    rows = []
    for cid, r in results.items():
        spec = r["candidate"]
        if r["status"] != "completed":
            rows.append([f"`{cid}`", FAMILY_VI[spec["kind"]], FEATURES_VI.get(spec.get("features"), "—"), "", "lỗi", "", "", "", "", "", "", ""])
            continue
        v, t = r["validation"], r["test"]
        mark = " ⭐" if cid == selected else ""
        rows.append([f"`{cid}`{mark}", FAMILY_VI[spec["kind"]], FEATURES_VI.get(spec.get("features"), "—"),
                     r.get("training", {}).get("design", "balanced weights") if spec["kind"] != "majority" else "—",
                     num(v["auc_roc"]), num(v["pr_auc"]), num(t["auc_roc"]), num(t["pr_auc"]), num(t["threshold"], 3),
                     num(t["precision"]), num(t["recall"]), num(t["f1"])])
    md += [table(["Mô hình", "Họ", "Đặc trưng", "Thiết kế", "AUC val", "PR-AUC val", "AUC test", "PR-AUC test", "Ngưỡng", "Precision test",
                  "Recall test", "F1 test"], rows), ""]
    test_ref = next(r["test"] for r in results.values() if r["status"] == "completed")
    md += ["Cùng các mô hình tại **ngưỡng 0,5** — điểm cắt tự nhiên khi huấn luyện với trọng số lớp cân bằng "
           "(các số này do mã đánh giá ghi sẵn cho mọi ứng viên, không tính lại sau khi xem test):", "",
           table(["Mô hình", "Balanced acc. val", "Balanced acc. test", "Precision test", "Recall test", "F1 test", "% đoán cao (test)"],
                 [[f"`{cid}`" + (" ⭐" if cid == selected else ""), num(r["validation"]["at_probability_0_5"]["balanced_accuracy"]),
                   num(r["test"]["at_probability_0_5"]["balanced_accuracy"]), num(r["test"]["at_probability_0_5"]["precision"]),
                   num(r["test"]["at_probability_0_5"]["recall"]), num(r["test"]["at_probability_0_5"]["f1"]),
                   pct(r["test"]["at_probability_0_5"]["predicted_positive_rate"], 1)]
                  for cid, r in results.items() if r["status"] == "completed"]), ""]
    sel = results[selected]["test"]
    if sel["recall"] > 0.85 and sel["f1"] - sel["all_high_f1_reference"] < 0.02:
        md += [f"> **Lưu ý về ngưỡng.** Ngưỡng tối đa F1 trên validation ({num(sel['threshold'], 3)}) gần như đoán \"cao\" cho mọi bài "
               f"(recall {num(sel['recall'], 3)}): F1 {num(sel['f1'])} chỉ nhỉnh hơn mức luôn đoán cao ({num(sel['all_high_f1_reference'])}). "
               "Khi lớp dương chiếm gần 40% và tín hiệu yếu, F1 tối đa đạt được bằng cách đoán cao gần hết. Vì vậy AUC-ROC/PR-AUC "
               "(không phụ thuộc ngưỡng) là thước đo chính, và kết quả phân lớp được đọc tại ngưỡng 0,5. "
               "Việc nhấn mạnh ngưỡng 0,5 được quyết định sau khi xem kết quả run này.", ""]
    md += [f"Mốc tham chiếu trên test: tỉ lệ lớp cao = {pct(test_ref['positive_rate'])} (= PR-AUC ngẫu nhiên); "
           f"luôn dự đoán \"cao\" cho F1 = {num(test_ref['all_high_f1_reference'])}; baseline lớp đa số (luôn \"thấp\") "
           f"có AUC 0.5 và F1 = 0.", "",
           f"**Mô hình được chọn:** `{selected}`. Tốt nhất mỗi họ: LR = `{best_lr}`, GBT = `{best_gbt}`.", "",
           "![ROC test](figures/roc_test.png)", "", "![Ma trận nhầm lẫn](figures/confusion_test.png)", ""]
    sel_test = results[selected]["test"]
    if "calibration" in sel_test:
        c = sel_test["calibration"]
        md += [f"Hiệu chỉnh xác suất của `{selected}` trên test (cộng {num(c['prior_logit_offset'], 3)} vào logit để bỏ ảnh hưởng của "
               f"trọng số/undersampling): xác suất trung bình {num(c['mean_calibrated_probability'])} so với tỉ lệ thật "
               f"{num(c['positive_rate'])} (trước hiệu chỉnh {num(c['mean_raw_probability'])}); Brier {num(c['brier_calibrated'])} "
               f"(trước {num(c['brier_raw'])}).", ""]
    if "unseen_train_titles" in sel_test:
        u = sel_test["unseen_train_titles"]
        md += [f"Trên {u['records']:,} bài test có tiêu đề chưa từng xuất hiện trong train, `{selected}` đạt AUC "
               f"{num(u['auc_roc'])}, F1 {num(u['f1'])} (toàn test: AUC {num(sel_test['auc_roc'])}, F1 {num(sel_test['f1'])}).", ""]
    if "top_positive" in lr_entry:
        md += ["## 4. N-gram quan trọng (hệ số Logistic Regression)", "",
               f"Hệ số của `{best_lr}`; dương = đẩy về phổ biến cao, âm = đẩy về phổ biến thấp. Đây là quan hệ dự đoán, "
               "không phải nhân quả.", "",
               table(["Hạng", "Đẩy về CAO", "Hệ số", "Đẩy về THẤP", "Hệ số"],
                     [[i + 1, f"`{p['term']}`", num(p["coefficient"], 3), f"`{n['term']}`", num(n["coefficient"], 3)]
                      for i, (p, n) in enumerate(zip(lr_entry["top_positive"][:15], lr_entry["top_negative"][:15]))]), "",
               "![Top n-gram](figures/lr_top_ngrams.png)", "",
               "Hệ số đặc trưng cấu trúc (trên mỗi độ lệch chuẩn train):", "",
               table(["Đặc trưng", "Hệ số"], [[STRUCTURED_VI.get(k, k), num(v, 4)] for k, v in lr_entry["structured"].items()]), ""]
    if best_gbt:
        g = analysis["gradient_boosted_trees"][best_gbt]
        md += ["## 5. Feature importance (GBT)", "",
               f"`{best_gbt}`: đặc trưng cấu trúc chiếm {pct(g['structured_importance_share'])} tổng importance, "
               f"riêng giờ/thứ chiếm {pct(g['time_importance_share'])}.", "",
               table(["Hạng", "Đặc trưng", "Importance"], [[i + 1, STRUCTURED_VI.get(f["feature"], f"`{f['feature']}`"),
                                                             num(f["importance"])] for i, f in enumerate(g["top_features"][:15])]), "",
               "![GBT importance](figures/gbt_importance.png)", ""]
    if time_series:
        md += ["## 6. Ảnh hưởng của thời điểm đăng", ""]
        for cid, effects in time_series:
            hours = effects["hour_utc"]
            days = effects["weekday_monday0"]
            md.append(f"- `{cid}`: theo giờ, log-odds dao động {num(max(hours) - min(hours), 3)} "
                      f"(cao nhất {hours.index(max(hours))}h, thấp nhất {hours.index(min(hours))}h UTC); "
                      f"theo thứ dao động {num(max(days) - min(days), 3)} (cao nhất {WEEKDAYS_VI[days.index(max(days))]}, "
                      f"thấp nhất {WEEKDAYS_VI[days.index(min(days))]}).")
        if "time_lr" in results and results["time_lr"]["status"] == "completed":
            md.append(f"- Chỉ dùng giờ/thứ: AUC test {num(results['time_lr']['test']['auc_roc'])}, gần mức ngẫu nhiên 0.5. "
                      "Thời điểm đăng chỉ có tín hiệu yếu so với nội dung tiêu đề.")
        md += ["", "![Ảnh hưởng thời gian](figures/time_effects.png)", ""]
    train_groups = distributions["train"]["groups"]
    md += ["## 7. Hạn chế", "",
           f"- Bỏ nhóm giữa ({train_groups['middle']['count']:,} bài train) làm bài toán dễ hơn thực tế: mô hình không phải phân biệt các bài sát ranh giới."]
    if train_groups.get("excluded", {}).get("count"):
        md.append(f"- Bỏ {train_groups['excluded']['count']:,} bài train có score ≤ 1: mô hình trả lời câu hỏi \"trong các bài đã có tương tác "
                  "(score ≥ 2), bài nào phổ biến\", chưa áp dụng trực tiếp cho một bài vừa đăng chưa biết có bị gỡ hay không.")
    md += [
           "- Score là giá trị lưu trong archive, không phải score sau một khoảng thời gian cố định; bài bị gỡ có thể mang score thấp.",
           "- Tháng test 12/2025 đã được dùng để chấm các run trước, nên test không còn hoàn toàn chưa nhìn thấy.",
           "- Chưa chạy: trigram, thí nghiệm loại từ dừng, giờ miền Đông Hoa Kỳ (từ vựng hiện tại là 1–2-gram, giữ từ dừng, giờ UTC).",
           "- Chi-square chọn n-gram trên toàn bộ train trước khi huấn luyện GBT; ngưỡng và lựa chọn mô hình chỉ dùng validation.", ""]
    path = reports / "results_vi.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-root", default="reports/experiments")
    args = parser.parse_args(argv)
    print(build(Path(args.report_root) / args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
