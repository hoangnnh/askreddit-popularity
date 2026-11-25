"""Render EDA figures and the Vietnamese summary from saved JSON; never starts Spark."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e6e6e6", "#bdbdbd"
BLUE, ORANGE, AQUA, LIGHT_BLUE = "#257de0", "#f47230", "#00b894", "#9dc8f7"
SEQUENTIAL = ["#cce3fd", "#b6d5f8", "#9dc8f7", "#84b9f3", "#6aabf1", "#519ced", "#338ced",
              "#257de0", "#2370cd", "#1b64be", "#1757ac", "#0f4a96", "#0e3d7e"]
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
PATTERN_LABELS = {
    "starts_what": "Starts with “what”", "starts_why": "Starts with “why”", "starts_how": "Starts with “how”",
    "starts_who": "Starts with “who”", "starts_which": "Starts with “which”",
    "starts_when_where": "Starts with “when/where”", "starts_yes_no_verb": "Starts with yes/no verb",
    "starts_if": "Starts with “if”", "contains_if_you_could": "Contains “if you could”",
    "contains_would_you_rather": "Contains “would you rather”", "addresses_reader": "Addresses reader (you/your)",
    "first_person": "First person (I/my/me)", "serious_tag": "[Serious] tag", "nsfw_mention": "Mentions NSFW",
    "mentions_reddit": "Mentions Reddit/redditors",
}
PATTERN_LABELS_VI = {
    "starts_what": "Bắt đầu bằng “what”", "starts_why": "Bắt đầu bằng “why”", "starts_how": "Bắt đầu bằng “how”",
    "starts_who": "Bắt đầu bằng “who”", "starts_which": "Bắt đầu bằng “which”",
    "starts_when_where": "Bắt đầu bằng “when/where”", "starts_yes_no_verb": "Bắt đầu bằng động từ câu hỏi có/không",
    "starts_if": "Bắt đầu bằng “if”", "contains_if_you_could": "Chứa “if you could”",
    "contains_would_you_rather": "Chứa “would you rather”", "addresses_reader": "Nói với người đọc (you/your)",
    "first_person": "Ngôi thứ nhất (I/my/me)", "serious_tag": "Thẻ [Serious]", "nsfw_mention": "Nhắc tới NSFW",
    "mentions_reddit": "Nhắc tới Reddit/redditors",
}
FIGURES = ("monthly_volume", "score_distribution", "label_distribution", "title_length", "posting_time",
           "weekday_hour_heatmap", "title_patterns", "length_popularity")


def load(root: Path) -> dict:
    names = ("run", "profile", "monthly_volume", "score_distribution", "score_frequencies", "title_characteristics",
             "posting_time", "associations", "label_feasibility")
    return {name: json.loads((root / f"{name}.json").read_text()) for name in names}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def bars(axis, labels, values, color=BLUE, reference=None, reference_label=None, horizontal=False):
    positions = range(len(values))
    if horizontal:
        axis.barh(positions, values, color=color, height=0.7, edgecolor="white", linewidth=1)
        axis.set_yticks(list(positions), labels)
        axis.grid(axis="x", visible=True)
        axis.grid(axis="y", visible=False)
        if reference is not None:
            axis.axvline(reference, color=INK, lw=1, ls=(0, (4, 3)), label=reference_label)
    else:
        axis.bar(positions, values, color=color, width=0.75, edgecolor="white", linewidth=1)
        axis.set_xticks(list(positions), labels)
        if reference is not None:
            axis.axhline(reference, color=INK, lw=1, ls=(0, (4, 3)), label=reference_label)
    thousands(axis.xaxis if horizontal else axis.yaxis)
    if reference is not None and not horizontal:
        axis.set_ylim(0, max(max(values), reference) * 1.18)  # headroom keeps the legend off the bars
    if reference_label:
        axis.legend(loc="lower right" if horizontal else "upper right", fontsize=8.5)


def thousands(axis):
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"))


def save(fig, path: Path):
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def train_share(data, key="share_score_ge_2"):
    return data["associations"]["patterns"]["overall"][key]


def fig_monthly_volume(data, path):
    plt = pyplot()
    monthly = data["monthly_volume"]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in monthly["daily"]]
    counts = [c for _, c in monthly["daily"]]
    rolling = [sum(counts[max(0, i - 3):i + 4]) / len(counts[max(0, i - 3):i + 4]) for i in range(len(counts))]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.6), constrained_layout=True)
    top.plot(dates, counts, color=LIGHT_BLUE, lw=1, label="Daily posts")
    top.plot(dates, rolling, color=BLUE, lw=2, label="Centered 7-day mean")
    for boundary, name in ((datetime(2025, 10, 1), "validation →"), (datetime(2025, 12, 1), "test →")):
        top.axvline(boundary, color=MUTED, lw=1, ls=(0, (2, 2)))
        top.text(boundary, max(counts), " " + name, color=MUTED, fontsize=8, va="top")
    top.set(title="Posts per UTC day, 2025", ylabel="Posts")
    thousands(top.yaxis)
    top.set_ylim(bottom=0)
    top.legend(loc="lower left", fontsize=8.5)
    months = monthly["months"]
    bars(bottom, [MONTHS[m["month"] - 1] for m in months], [100 * m["share_score_ge_2"] for m in months])
    bottom.set(title="Share of posts with score ≥ 2 by month (archived score)", ylabel="% of posts")
    return save(fig, path)


def fig_score_distribution(data, path):
    plt = pyplot()
    summary = data["score_distribution"]
    total = summary["all"]["records"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    small = summary["small_scores"]
    bars(left, [str(r["score"]) if r["score"] % 2 == 0 else "" for r in small], [100 * r["records"] / total for r in small])
    left.set(title="Scores 0–20", xlabel="Archived score", ylabel="% of all posts")
    pairs = sorted(data["score_frequencies"]["all"])
    remaining, xs, ys = total, [], []
    for value, count in pairs:
        if value >= 1:
            xs.append(value)
            ys.append(remaining / total)
        remaining -= count
    right.step(xs, ys, where="post", color=BLUE, lw=2)
    right.set(xscale="log", yscale="log", title="Upper tail: P(score ≥ s)", xlabel="s (log scale)", ylabel="Share of posts (log scale)")
    right.grid(axis="x", visible=True)
    for boundary in data["label_feasibility"]["thresholds"]["effective_boundaries"]:
        right.axvline(boundary, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        right.text(boundary, 1.0, f" {boundary}", color=MUTED, fontsize=7.5, va="top", rotation=90)
    right.text(0.98, 0.95, "dotted: label boundaries", transform=right.transAxes, ha="right", color=MUTED, fontsize=8)
    return save(fig, path)


def interval_text(row):
    low, high = row["lower_inclusive"], row["upper_exclusive"]
    if low is None:
        return f"≤{high - 1}"
    if high is None:
        return f"≥{low}"
    return str(low) if high == low + 1 else f"{low}–{high - 1}"


def fig_label_distribution(data, path):
    plt = pyplot()
    labels = data["label_feasibility"]
    distributions = labels["class_distributions"]
    fig, axis = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    width = 0.26
    for offset, split in zip((-width, 0, width), ("train", "validation", "test")):
        values = [r["percentage"] for r in distributions[split]["labels"]]
        axis.bar([k + offset for k in range(8)], values, width=width, color=SPLIT_COLORS[split],
                 edgecolor="white", linewidth=1, label=split)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    axis.set_ylim(0.3, 120)
    axis.set_xticks(range(8), [f"{r['label']}\nscore {interval_text(r)}" for r in labels["thresholds"]["mapping"]])
    axis.set(title="Eight-class label distribution by chronological split (train-fitted thresholds)",
             ylabel="% of split (log scale)")
    axis.legend(ncols=3, loc="upper right")
    return save(fig, path)


def fig_title_length(data, path):
    plt = pyplot()
    title = data["title_characteristics"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    binned = {}
    for value, count in title["title_characters"]["frequencies"]:
        binned[value // 5 * 5] = binned.get(value // 5 * 5, 0) + count
    total = sum(binned.values())
    left.bar(list(binned), [100 * c / total for c in binned.values()], width=4.4, align="edge", color=BLUE)
    median = title["title_characters"]["quantiles"]["0.5"]
    left.axvline(median, color=INK, lw=1, ls=(0, (4, 3)), label=f"median {median}")
    left.set(title="Title length (characters, 5-character bins)", xlabel="Characters", ylabel="% of posts")
    left.legend(loc="upper right")
    words = [(v, c) for v, c in title["title_words"]["frequencies"] if v <= 60]
    total = sum(c for _, c in title["title_words"]["frequencies"])
    right.bar([v for v, _ in words], [100 * c / total for _, c in words], width=0.8, color=BLUE)
    median = title["title_words"]["quantiles"]["0.5"]
    right.axvline(median, color=INK, lw=1, ls=(0, (4, 3)), label=f"median {median}")
    right.set(title="Title length (feature-tokenizer words, ≤60 shown)", xlabel="Words", ylabel="% of posts")
    right.legend(loc="upper right")
    return save(fig, path)


def fig_posting_time(data, path):
    plt = pyplot()
    timing = data["posting_time"]
    overall = 100 * train_share(data)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for row, key, xlabel in ((0, "by_hour", "UTC hour"), (1, "by_weekday", "UTC weekday")):
        items = timing[key]
        names = [i["label"] if key == "by_weekday" or int(i["label"]) % 3 == 0 else "" for i in items]
        bars(axes[row][0], names, [i["records_all_year"] for i in items])
        axes[row][0].set(title=f"Posts by {xlabel} — all of 2025", xlabel=xlabel, ylabel="Posts")
        bars(axes[row][1], names, [100 * i["train"]["share_score_ge_2"] if i["train"] else 0 for i in items],
             reference=overall, reference_label="train overall")
        axes[row][1].set(title=f"Share with score ≥ 2 by {xlabel} — train", xlabel=xlabel, ylabel="% of posts")
    return save(fig, path)


def fig_heatmap(data, path):
    plt = pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    cells = data["posting_time"]["train_weekday_hour"]
    grids = {key: [[float("nan")] * 24 for _ in range(7)] for key in ("records", "share_score_ge_2")}
    for cell in cells:
        grids["records"][cell["weekday"]][cell["hour"]] = cell["records"]
        grids["share_score_ge_2"][cell["weekday"]][cell["hour"]] = 100 * cell["share_score_ge_2"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), constrained_layout=True)
    for axis, key, title, unit in ((axes[0], "records", "Train posts by weekday × UTC hour", "Posts"),
                                   (axes[1], "share_score_ge_2", "Train share with score ≥ 2 by weekday × UTC hour", "% of posts")):
        image = axis.imshow(grids[key], cmap=cmap, aspect="auto")
        axis.set_xticks(range(24), [f"{h:02d}" for h in range(24)])
        axis.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axis.grid(False)
        axis.set(title=title, xlabel="UTC hour")
        fig.colorbar(image, ax=axis, label=unit, shrink=0.9)
    return save(fig, path)


def fig_title_patterns(data, path):
    plt = pyplot()
    section = data["associations"]["patterns"]
    items = sorted((p for p in section["patterns"] if p["records"]), key=lambda p: p["share_score_ge_2"])
    names = [f"{PATTERN_LABELS[p['pattern']]}  (n={p['records']:,})" for p in items]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 0.42 * len(items) + 1.6), sharey=True, constrained_layout=True)
    bars(left, names, [100 * p["share_score_ge_2"] for p in items], horizontal=True,
         reference=100 * section["overall"]["share_score_ge_2"], reference_label="all train posts")
    left.set(title="Share with score ≥ 2 — train", xlabel="% of posts with the pattern")
    bars(right, names, [100 * p["share_label_ge_6"] for p in items], horizontal=True,
         reference=100 * section["overall"]["share_label_ge_6"], reference_label="all train posts")
    right.set(title="Share in labels 6–7 (score ≥ 46) — train", xlabel="% of posts with the pattern")
    return save(fig, path)


def fig_length_popularity(data, path):
    plt = pyplot()
    associations = data["associations"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for row, key, unit in ((0, "title_characters", "Title characters"), (1, "title_words", "Title words")):
        buckets = associations[key]
        names = [b["bucket"] for b in buckets]
        for column, measure, title in ((0, "share_score_ge_2", "Share with score ≥ 2"), (1, "share_label_ge_6", "Share in labels 6–7")):
            bars(axes[row][column], names, [100 * b[measure] for b in buckets],
                 reference=100 * train_share(data, measure), reference_label="train overall")
            axes[row][column].set(title=f"{title} by {unit.lower()} — train", xlabel=unit, ylabel="% of posts")
            axes[row][column].tick_params(axis="x", labelsize=8)
    return save(fig, path)


def render_figures(root: Path, data: dict) -> list[Path]:
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    functions = (fig_monthly_volume, fig_score_distribution, fig_label_distribution, fig_title_length,
                 fig_posting_time, fig_heatmap, fig_title_patterns, fig_length_popularity)
    return [function(data, folder / f"{name}.png") for name, function in zip(FIGURES, functions)]


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value):
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(header))) + " |"]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def key_findings(data: dict) -> list[str]:
    """Data-driven headline bullets; every number comes from the saved JSON."""
    scores, associations, timing = data["score_distribution"]["all"], data["associations"], data["posting_time"]
    labels = data["label_feasibility"]
    overall = associations["patterns"]["overall"]
    minimum = max(1, overall["records"] // 1000)  # ignore patterns covering < 0.1% of train posts
    ranked = sorted((p for p in associations["patterns"]["patterns"] if p["records"] >= minimum and p["lift_score_ge_2"] is not None),
                    key=lambda p: p["lift_score_ge_2"])
    worst, best = ranked[0], ranked[-1]
    questions = {b["bucket"]: b for b in associations["question_marks"]}
    hours = [h for h in timing["by_hour"] if h["train"]]
    high, low = (max(hours, key=lambda h: h["train"]["share_score_ge_2"]), min(hours, key=lambda h: h["train"]["share_score_ge_2"]))
    weekdays = [d["train"]["share_score_ge_2"] for d in timing["by_weekday"] if d["train"]]
    spearman = data["associations"]["correlations"]["spearman_with_score"]
    two = labels["score_2_share_of_upper_remainder"]["pooled_train"]["share_eq_2"]
    return ["## Nhận xét chính", "",
            f"- **Score rất lệch:** {pct(scores['share_le_1'])} bài có score ≤ 1, median {scores['quantiles']['0.5']}, "
            f"p99 = {scores['quantiles']['0.99']}, max = {scores['max']:,}. Accuracy vì vậy không phải thước đo phù hợp.",
            f"- **Từ ngữ của tiêu đề liên hệ rõ nhất với độ phổ biến (train):** mẫu tốt nhất là {PATTERN_LABELS_VI[best['pattern']]} "
            f"({pct(best['share_score_ge_2'])} có score ≥ 2, lift {best['lift_score_ge_2']:.2f}); kém nhất là "
            f"{PATTERN_LABELS_VI[worst['pattern']]} ({pct(worst['share_score_ge_2'])}, lift {worst['lift_score_ge_2']:.2f}), "
            f"so với {pct(overall['share_score_ge_2'])} toàn train."
            + (f" Tiêu đề không có dấu hỏi chỉ đạt {pct(questions['0']['share_score_ge_2'])}." if "0" in questions else ""),
            "- **Giả thuyết (chưa kiểm chứng):** các mẫu kém nhất trùng với quy định của r/AskReddit (không hỏi lời khuyên cá nhân, "
            "tiêu đề phải là câu hỏi), nên score lưu trữ có thể phản ánh cả việc bài bị gỡ, không chỉ mức quan tâm của người đọc.",
            f"- **Thời điểm đăng chỉ liên hệ yếu:** theo giờ UTC, tỉ lệ score ≥ 2 dao động từ {pct(low['train']['share_score_ge_2'])} "
            f"({low['label']}:00) đến {pct(high['train']['share_score_ge_2'])} ({high['label']}:00), ngược chiều với lượng bài; "
            f"theo thứ chỉ từ {pct(min(weekdays))} đến {pct(max(weekdays))}.",
            f"- **Tương quan đơn biến yếu:** Spearman với score từ {min(spearman.values()):+.3f} đến {max(spearman.values()):+.3f}.",
            f"- **Nhãn tám lớp khả thi nhưng mong manh:** mọi lớp đều khác rỗng; lớp 1 (score = 2) tồn tại chỉ vì "
            f"{pct(two, 1)} bài score ≥ 2 có score đúng bằng 2 (ngưỡng là 50%).", ""]


def render_summary(data: dict) -> str:
    profile, monthly, scores = data["profile"], data["monthly_volume"], data["score_distribution"]
    title, timing, associations = data["title_characteristics"], data["posting_time"], data["associations"]
    labels, run = data["label_feasibility"], data["run"]
    overall = associations["patterns"]["overall"]
    nulls = {k: v for k, v in profile["null_counts"].items() if v}
    lines = [
        "# Phân tích khám phá dữ liệu (EDA) — r/AskReddit 2025", "",
        f"Báo cáo này được sinh tự động bằng `python -m askreddit.eda` từ các file JSON cùng thư mục. Spark tính mọi số liệu "
        f"trên toàn bộ **{profile['records']:,}** bài đã làm sạch (thời gian chạy {run['elapsed_seconds']:.0f} giây, Spark {run['spark_version']}).",
        "",
        "**Phạm vi:** các bảng khối lượng, độ phủ và phân phối score mô tả cả năm 2025. Các phân tích *liên hệ* giữa đặc điểm "
        "tiêu đề, thời điểm đăng và độ phổ biến chỉ dùng **tập train** (01/01/2025 – 30/09/2025 UTC), để validation và test "
        "không ảnh hưởng tới quyết định mô hình. Đây là quan hệ mô tả, **không phải quan hệ nhân quả**. Score là giá trị lưu "
        "trong archive, không phải score sau một khoảng thời gian cố định.", "",
        "Hai thước đo độ phổ biến dùng xuyên suốt: **tỉ lệ score ≥ 2** (nhãn ≥ 1) và **tỉ lệ nhãn 6–7** "
        f"(score ≥ {labels['thresholds']['effective_boundaries'][5]}, khoảng 1% bài phổ biến nhất).", "",
        *key_findings(data), "",
        "## 1. Hồ sơ dữ liệu", "",
        f"- Số bài: **{profile['records']:,}**; ID {'duy nhất' if profile['ids_unique'] else '**không** duy nhất'} "
        f"({profile['distinct_ids']:,} ID khác nhau).",
        f"- Thời gian phủ (UTC): {profile['coverage_utc'][0]} → {profile['coverage_utc'][1]}.",
        f"- Giá trị null: {'không có ở cột nào' if not nulls else nulls}. Tiêu đề rỗng: {profile['blank_titles']:,}.",
        f"- Score từ {profile['score_range'][0]:,} đến {profile['score_range'][1]:,}.",
        f"- Bài chỉ có tiêu đề (selftext rỗng): **{profile['title_only_posts']:,}** ({pct(profile['title_only_share'], 4)}); "
        f"chỉ {profile['records'] - profile['title_only_posts']:,} bài có nội dung vì r/AskReddit không cho phép nội dung bài.",
        f"- Tiêu đề chuẩn hóa khác nhau: {profile['normalized_titles']['distinct_normalized_titles']:,}; "
        f"{profile['normalized_titles']['records_with_repeated_title']:,} bài "
        f"({pct(profile['normalized_titles']['share_records_with_repeated_title'])}) có tiêu đề xuất hiện hơn một lần; "
        f"tiêu đề lặp nhiều nhất xuất hiện {profile['normalized_titles']['max_repeats_of_one_title']:,} lần.", "",
        "Schema: " + ", ".join(f"`{c['column']}` ({c['type']})" for c in profile["schema"]) + ".", "",
        "Năm tiêu đề chuẩn hóa lặp lại nhiều nhất:", "",
        *table(["Tiêu đề chuẩn hóa", "Số bài"], [[f"`{t['normalized_title'][:80]}`", f"{t['records']:,}"]
                                                for t in profile["normalized_titles"]["top_repeated"][:5]]), "",
        "## 2. Khối lượng bài theo thời gian", "",
        "![Khối lượng bài theo ngày và tỉ lệ score ≥ 2 theo tháng](figures/monthly_volume.png)", "",
    ]
    months = monthly["months"]
    busiest, quietest = max(months, key=lambda m: m["records"]), min(months, key=lambda m: m["records"])
    daily = monthly["daily_summary"]
    lines += [
        f"Trung bình **{daily['mean']:,.0f}** bài/ngày (median {daily['median']:,}, thấp nhất {daily['min']:,}, cao nhất {daily['max']:,}). "
        f"Tháng nhiều bài nhất là {busiest['month']:02d}/2025 ({busiest['records']:,} bài), ít nhất là "
        f"{quietest['month']:02d}/2025 ({quietest['records']:,} bài). Đủ 12 tháng, {daily['days']} ngày có dữ liệu.", "",
        *table(["Tháng", "Số bài", "Bài/ngày", "% score ≤ 1", "% score ≥ 2", "% nhãn 6–7", "Median score", "% chỉ tiêu đề"],
               [[f"{m['month']:02d}/{m['year']}", f"{m['records']:,}", f"{m['posts_per_day']:,.0f}", pct(m["share_score_le_1"]),
                 pct(m["share_score_ge_2"]), pct(m["share_label_ge_6"]), num(m["median_score"]), pct(m["share_title_only"])] for m in months]),
        "", "Cột độ phổ biến của tháng 10–12 chỉ để mô tả; không dùng cho quyết định mô hình.", "",
        "## 3. Phân phối score", "",
        "![Phân phối score và đuôi trên](figures/score_distribution.png)", "",
    ]
    whole = scores["all"]
    lines += [
        f"Score tập trung rất mạnh ở giá trị thấp: score = 0 chiếm **{pct(whole['zero_share'])}**, score = 1 chiếm "
        f"**{pct(whole['one_share'])}**, tổng cộng **{pct(whole['share_le_1'])}** bài có score ≤ 1. Đồng thời đuôi trên rất dài: "
        f"giá trị lớn nhất là {whole['max']:,}, có {whole['distinct_values']:,} giá trị score khác nhau, và trung bình "
        f"({whole['mean']:.2f}) lớn hơn nhiều so với median ({whole['quantiles']['0.5']}). Vì vậy accuracy không phù hợp làm "
        "thước đo chính, và các ngưỡng nhãn cần dựa trên phân vị thay vì khoảng đều.", "",
        *table(["Phân vị", *[f"p{float(p) * 100:g}" for p in whole["quantiles"]]],
               [["Score (cả năm)", *whole["quantiles"].values()]]), "",
        *table(["Score ≥", "Số bài", "Tỉ lệ"], [[f"{t['score_at_least']:,}", f"{t['records']:,}", pct(t["share"], 3)] for t in whole["tail"]]), "",
        "Theo split (định nghĩa phân vị: nghịch đảo hàm phân phối thực nghiệm, chính xác):", "",
        *table(["Split", "Số bài", "% score ≤ 1", "Median", "p90", "p99", "Max"],
               [[s, f"{d['records']:,}", pct(d["share_le_1"]), d["quantiles"]["0.5"], d["quantiles"]["0.9"],
                 d["quantiles"]["0.99"], f"{d['max']:,}"] for s, d in scores["by_split"].items()]), "",
        "## 4. Đặc điểm tiêu đề", "",
        "![Độ dài tiêu đề](figures/title_length.png)", "",
    ]
    chars, words = title["title_characters"]["quantiles"], title["title_words"]["quantiles"]
    lines += [
        f"Tiêu đề có median **{chars['0.5']} ký tự** (p5–p95: {chars['0.05']}–{chars['0.95']}) và **{words['0.5']} từ** "
        f"(p5–p95: {words['0.05']}–{words['0.95']}). **{pct(title['question_present_share'])}** tiêu đề có dấu hỏi. "
        "Các định nghĩa giống hệt đặc trưng của mô hình (NFKC, gộp khoảng trắng, tokenizer của mô hình).", "",
        f"Bảng mẫu câu hỏi trên tập train (toàn bộ train: {pct(overall['share_score_ge_2'])} có score ≥ 2, "
        f"{pct(overall['share_label_ge_6'])} thuộc nhãn 6–7). *Lift* = tỉ lệ của nhóm / tỉ lệ toàn train; các mẫu có thể chồng lên nhau.", "",
    ]
    patterns = sorted(associations["patterns"]["patterns"], key=lambda p: -(p["lift_score_ge_2"] or 0))
    lines += [*table(["Mẫu", "Số bài train", "% train", "% score ≥ 2", "Lift", "% nhãn 6–7", "Lift", "Median score"],
                     [[PATTERN_LABELS_VI[p["pattern"]], f"{p['records']:,}", pct(p["share_of_train"]), pct(p["share_score_ge_2"]),
                       "—" if p["lift_score_ge_2"] is None else f"{p['lift_score_ge_2']:.2f}", pct(p["share_label_ge_6"]),
                       "—" if p["lift_label_ge_6"] is None else f"{p['lift_label_ge_6']:.2f}",
                       "—" if p["median_score"] is None else num(p["median_score"])] for p in patterns]), "",
              "![Mẫu câu hỏi và độ phổ biến](figures/title_patterns.png)", "",
              "## 5. Thời điểm đăng (UTC)", "",
              "![Theo giờ và thứ](figures/posting_time.png)", "",
              "![Thứ × giờ](figures/weekday_hour_heatmap.png)", ""]
    hours = [h for h in timing["by_hour"] if h["train"]]
    peak, trough = max(timing["by_hour"], key=lambda h: h["records_all_year"]), min(timing["by_hour"], key=lambda h: h["records_all_year"])
    best, worst = max(hours, key=lambda h: h["train"]["share_score_ge_2"]), min(hours, key=lambda h: h["train"]["share_score_ge_2"])
    lines += [
        f"Cả năm, giờ có nhiều bài nhất là **{peak['label']}:00 UTC** ({peak['records_all_year']:,} bài), ít nhất là "
        f"**{trough['label']}:00 UTC** ({trough['records_all_year']:,} bài). Trên train, tỉ lệ score ≥ 2 cao nhất lúc "
        f"**{best['label']}:00** ({pct(best['train']['share_score_ge_2'])}) và thấp nhất lúc **{worst['label']}:00** "
        f"({pct(worst['train']['share_score_ge_2'])}). Giờ Việt Nam = giờ UTC + 7. Phần lớn người dùng Reddit ở Bắc Mỹ "
        "(UTC−5 đến UTC−8), nên khi diễn giải cần quy đổi khung giờ UTC sang các múi giờ đó.", "",
        *table(["Thứ", "Số bài cả năm", "Số bài train", "% score ≥ 2", "% nhãn 6–7", "Median score"],
               [[d["label"], f"{d['records_all_year']:,}", f"{d['train']['records']:,}", pct(d["train"]["share_score_ge_2"]),
                 pct(d["train"]["share_label_ge_6"]), num(d["train"]["median_score"])] for d in timing["by_weekday"] if d["train"]]), "",
        "## 6. Liên hệ giữa đặc điểm và độ phổ biến (train)", "",
        "![Độ dài tiêu đề và độ phổ biến](figures/length_popularity.png)", "",
    ]

    def bucket_table(key, heading):
        return [f"**{heading}**", "", *table(["Nhóm", "Số bài", "% score ≥ 2", "% nhãn 6–7", "Nhãn TB", "Median score"],
                [[b["bucket"], f"{b['records']:,}", pct(b["share_score_ge_2"]), pct(b["share_label_ge_6"]),
                  f"{b['mean_label']:.3f}", num(b["median_score"])] for b in associations[key]]), ""]
    lines += bucket_table("title_characters", "Độ dài tiêu đề (ký tự)")
    lines += bucket_table("title_words", "Độ dài tiêu đề (từ)")
    lines += bucket_table("question_marks", "Số dấu hỏi")
    body = associations["title_only"]
    if min(b["records"] for b in body) >= 100:
        lines += bucket_table("title_only", "Có nội dung (selftext) hay chỉ có tiêu đề")
    else:
        lines += [f"**Có nội dung hay chỉ có tiêu đề:** chỉ {sum(b['records'] for b in body if b['bucket'] == 'has body'):,} bài train có selftext "
                  "(r/AskReddit không cho phép nội dung bài), nên không so sánh nhóm này; selftext gần như không mang thông tin.", ""]
    lines += bucket_table("uppercase_ratio", "Tỉ lệ chữ hoa trên chữ cái")
    correlations = associations["correlations"]
    lines += ["**Tương quan với score (train)**", "",
              *table(["Đặc trưng", "Spearman với score", "Pearson với log(1+score)"],
                     [[k, f"{v:+.4f}", f"{correlations['pearson_with_log1p_score'][k]:+.4f}"]
                      for k, v in correlations["spearman_with_score"].items()]), "",
              ("Các hệ số tương quan đơn biến đều nhỏ (|ρ| < 0,1)"
               + (" và cùng âm: tiêu đề dài hơn, nhiều dấu hỏi hơn hoặc nhiều chữ hoa hơn đi kèm score hơi thấp hơn. "
                  if all(v < 0 for v in correlations["spearman_with_score"].values()) else ". ")
               + "Không có đặc trưng đơn lẻ nào giải thích được nhiều về score; tín hiệu dự đoán, nếu có, nằm ở từ ngữ cụ thể "
               "(xem bảng mẫu câu hỏi) và mức độ yếu này phù hợp với kết quả mô hình."
               if all(abs(v) < 0.1 for v in correlations["spearman_with_score"].values()) else
               "Xem độ lớn của từng hệ số trước khi diễn giải; đây là quan hệ đơn biến, không kiểm soát các yếu tố khác."), "",
              "## 7. Mức sẵn sàng của nhãn tám lớp", "",
              "![Phân phối nhãn theo split](figures/label_distribution.png)", ""]
    thresholds = labels["thresholds"]
    rows = []
    for mapping in thresholds["mapping"]:
        k = mapping["label"]
        rows.append([k, interval_text(mapping), *[f"{labels['class_distributions'][s]['labels'][k]['count']:,} "
                     f"({labels['class_distributions'][s]['labels'][k]['percentage']:.2f}%)" for s in ("train", "validation", "test")]])
    lines += [
        f"Ngưỡng fit trên train bằng quy trình median phần dư phía trên (Fang và cộng sự, 2016): median chính xác "
        f"**{', '.join(s['median_exact'] for s in thresholds['steps'])}**, biên nguyên **{thresholds['effective_boundaries']}**. "
        f"Mọi lớp đều khác rỗng ở cả ba split: **{'có' if labels['all_classes_nonempty'] else 'KHÔNG'}**.", "",
        *table(["Nhãn", "Score", "Train", "Validation", "Test"], rows), "",
        "Số bài train có score đúng bằng một biên (luôn vào lớp cao hơn; các score bằng nhau không bao giờ bị tách):", "",
        *table(["Biên", "Số bài train tại biên", "Nhãn nhận"],
               [[b["boundary"], f"{b['train_records_at_boundary_score']:,}", b["assigned_label"]] for b in labels["boundary_ties"]]), "",
        "Độ ổn định: fit lại cùng quy trình trên từng tháng train (chỉ để chẩn đoán, không phải ánh xạ dùng cho mô hình). "
        "Bước chia đầu tiên chỉ tạo được lớp 1 khi **dưới một nửa** số bài có score ≥ 2 có score đúng bằng 2; nếu không, median = 2 "
        "và lớp 1 rỗng (quy trình báo lỗi thay vì tạo lớp rỗng).", "",
        *table(["Tháng", "Biên fit riêng tháng", "% score = 2 trong các bài score ≥ 2"],
               [[f"{s['month']:02d}/2025", s["boundaries"] or "thoái hóa: median phần dư = 2, lớp 1 rỗng", pct(share["share_eq_2"], 1)]
                for s, share in zip(labels["monthly_train_threshold_stability"], labels["score_2_share_of_upper_remainder"]["by_train_month"])]), "",
        f"Gộp cả giai đoạn train, tỉ lệ này là **{pct(labels['score_2_share_of_upper_remainder']['pooled_train']['share_eq_2'], 1)}** — "
        "chỉ vừa dưới 50%, nên median đầu tiên bằng 3 và lớp 1 = {score 2} tồn tại. Biên đầu (2, 3, 4, 6, 11) ổn định giữa các tháng; "
        "hai biên trên (46, 271) dao động theo tháng. Đây là điểm cần nêu khi trình bày: ánh xạ nhãn phụ thuộc giai đoạn fit.", "",
        "Baseline luôn dự đoán lớp 0 (tính giải tích từ số lượng lớp):", "",
        *table(["Split", "Accuracy", "Macro-F1", "Balanced accuracy"],
               [[s, pct(b["accuracy"]), f"{b['macro_f1']:.4f}", f"{b['balanced_accuracy']:.3f}"] for s, b in labels["analytical_majority_baseline"].items()]), "",
        "## 8. Giới hạn diễn giải", "",
        "- Score là giá trị lưu trong archive tại thời điểm thu thập; không có thời điểm quan sát thống nhất cho mọi bài.",
        "- Mọi liên hệ là mô tả, không kiểm soát các yếu tố khác (chủ đề, tác giả, lượng người online). Không suy ra rằng đổi tiêu đề hay giờ đăng sẽ làm bài phổ biến hơn.",
        "- Các mẫu câu hỏi được nhận diện bằng biểu thức chính quy đơn giản trên tiêu đề viết thường và có thể chồng lên nhau.",
        "- Giờ và thứ tính theo UTC; người dùng ở các múi giờ khác nhau trải nghiệm cùng một giờ UTC khác nhau.",
        "- Phân tích liên hệ chỉ dùng train; validation/test chỉ xuất hiện trong bảng khối lượng, phân phối score và số lượng nhãn.", "",
        "## Tái lập", "",
        "```bash", "JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \\",
        f"  .venv/bin/python -m askreddit.eda --config {run['config_path'] or 'configs/fang_chua_v2.json'} --output reports/eda_reproduction", "",
        "# Chỉ vẽ lại hình và báo cáo từ JSON đã lưu (không cần Spark):",
        "PYTHONPATH=src .venv/bin/python -m askreddit.eda --render-only --output reports/eda", "```", "",
    ]
    return "\n".join(lines)


def render_all(root: Path) -> list[Path]:
    data = load(root)
    paths = render_figures(root, data)
    summary = root / "summary_vi.md"
    summary.write_text(render_summary(data), encoding="utf-8")
    return [*paths, summary]
