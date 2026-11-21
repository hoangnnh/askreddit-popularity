"""Render a Vietnamese results report from a completed, machine-readable experiment."""

import argparse
import json
from pathlib import Path


def render(root: Path) -> str:
    def read(name):
        return json.loads((root / name).read_text())
    completed = read("completed.json")
    if completed["status"] != "completed":
        raise ValueError("A completed full-data experiment is required")
    config, labels, distributions = read("config.json"), read("label_boundaries.json"), read("class_distributions.json")
    features, comparison = read("features.json"), read("model_comparison.json")
    overlap, inputs = read("title_overlap.json"), read("input_validation.json")
    selected = completed["selected_candidate"]
    test = read(f"test/{selected}.json")
    selected_result = read(f"candidates/{selected}.json")
    baseline = next(r for r in comparison["rows"] if r["candidate"] == "majority" and r["split"] == "test")
    lines = [
        "# Kết quả dự đoán tám mức độ phổ biến của bài đăng AskReddit", "",
        f"Thí nghiệm đã chạy trên toàn bộ {inputs['records']:,} bài đã làm sạch. "
        f"Mô hình được chọn bằng macro-F1 tám lớp trên validation: **{selected}**. "
        f"Trên test, macro-F1 đạt **{test['macro_f1']:.4f}**, balanced accuracy "
        f"**{test['balanced_accuracy']:.4f}**, accuracy **{test['accuracy']:.4f}**.", "",
        "## Phương pháp và phạm vi diễn giải", "",
        "Quy tắc nhãn lấy từ [Fang, Cheng và Ostendorf (2016), mục 3](https://aclanthology.org/W16-6209.pdf), "
        "được [Chua (2021), mục 4.2](https://aclanthology.org/2021.acl-srw.10.pdf) sử dụng lại: "
        "score ≤ 1 thuộc lớp 0; dùng sáu median liên tiếp của phần dư phía trên để tạo lớp 1–6, phần cuối thuộc lớp 7. "
        "Giá trị bằng nhau luôn có cùng nhãn. Metric mean_threshold_f1 là trung bình bảy binary F1 tại các ngưỡng "
        "nhãn ≥ 1,…,7; nó khác macro-F1 tám lớp.", "",
        "Chia thời gian, fit ngưỡng và tiền xử lý chỉ trên train, TF-IDF, LR/RF, cách làm trơn trọng số và chọn mô hình "
        "là thiết kế của dự án. Fang nghiên cứu comments và có dùng ngữ cảnh phản hồi về sau; dự án này chỉ dùng "
        "title và thời điểm đăng của submissions. Target là score ghi trong archive, không phải score sau 24 giờ. "
        "Không có metadata thời điểm quan sát score để xác lập cùng một khoảng tích lũy cho mọi bài. "
        "Kết quả dự đoán không chứng minh quan hệ nhân quả của cách đặt tiêu đề hay giờ đăng.", "",
        "## Dữ liệu và nhãn thực tế", "",
        "Các khoảng UTC là nửa kín: train [2025-01-01, 2025-10-01), validation [2025-10-01, 2025-12-01), "
        "test [2025-12-01, 2026-01-01). Không chia ngẫu nhiên, không loại bài trùng tiêu đề qua các split.", "",
        f"Median train chính xác: **{', '.join(s['median_exact'] for s in labels['steps'])}**. "
        f"Các biên nguyên: **{labels['effective_boundaries']}**. "
        "DuckDB đã kiểm chứng độc lập median và số lượng từng lớp ở cả ba split.", "",
        "| Nhãn | Khoảng score | Train | Validation | Test |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in labels["mapping"]:
        k, lo, hi = row["label"], row["lower_inclusive"], row["upper_exclusive"]
        interval = f"≤ {hi-1}" if lo is None else f"≥ {lo}" if hi is None else str(lo) if hi == lo + 1 else f"{lo}–{hi-1}"
        counts = [distributions[s]["labels"][k] for s in ("train", "validation", "test")]
        lines.append(f"| {k} | {interval} | " + " | ".join(f"{r['count']:,} ({r['percentage']:.2f}%)" for r in counts) + " |")
    lines += ["| Tổng | | " + " | ".join(f"{distributions[s]['total']:,}" for s in ("train", "validation", "test")) + " |", "",
              f"Có {overlap['train__test']['shared_distinct_titles']:,} tiêu đề chuẩn hóa chung giữa train và test, "
              f"tương ứng {overlap['train__test']['right_records']:,} bài test. "
              f"Tập test chưa thấy tiêu đề trong train gồm {test['unseen_train_titles']['records']:,} bài.", "",
              "## So sánh mô hình", "",
              "Trọng số train tỷ lệ với `(N/(8*n_k))**alpha` và được chuẩn hóa để trung bình trên bản ghi train bằng 1. "
              "Validation/test giữ tần suất tự nhiên; không áp dụng trọng số train vào metric. "
              "Chọn macro-F1 validation lớn nhất, làm tròn 12 chữ số rồi xử lý hòa bằng ID ứng viên tăng dần. "
              "Lựa chọn đã được ghi cố định trước khi dự đoán test; các kết quả test không thay đổi lựa chọn.", ""]
    for split, heading in (("validation", "Validation"), ("test", "Test sau khi chốt lựa chọn")):
        lines += [f"### {heading}", "", "| Mô hình | Macro-F1 | Balanced acc. | Accuracy | Weighted-F1 | Mean threshold F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in comparison["rows"]:
            if row["split"] != split:
                continue
            name = f"**{row['candidate']}**" if row["selected"] else row["candidate"]
            lines.append(f"| {name} | " + " | ".join(f"{row[k]:.4f}" for k in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "mean_threshold_f1")) + " |")
        lines.append("")
    lines += [f"So với baseline lớp phổ biến nhất, macro-F1 test thay đổi từ {baseline['macro_f1']:.4f} "
              f"thành {test['macro_f1']:.4f}, còn accuracy thay đổi từ {baseline['accuracy']:.2%} "
              f"thành {test['accuracy']:.2%}. Đây là đánh đổi theo tiêu chí macro-F1 đã đăng ký, "
              "không phải cải thiện đồng thời mọi metric. Chất lượng phân biệt tám lớp vẫn thấp. "
              "Việc một số cấu hình chỉ dự đoán lớp 0 không chứng minh title hoặc thời điểm đăng không có tín hiệu dự đoán.", "",
              "### Chi tiết từng lớp của mô hình được chọn trên test", "",
              "| Nhãn | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in test["per_class"]:
        lines.append(f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']:,} |")
    lines.append("")
    unseen = test["unseen_train_titles"]
    lines += [f"Với mô hình đã chọn, tập con tiêu đề chưa thấy trong train đạt macro-F1 **{unseen['macro_f1']:.4f}**, "
              f"balanced accuracy **{unseen['balanced_accuracy']:.4f}**, accuracy **{unseen['accuracy']:.4f}**, "
              f"weighted-F1 **{unseen['weighted_f1']:.4f}**, mean_threshold_f1 **{unseen['mean_threshold_f1']:.4f}**.", "",
              "Precision, recall, F1 và support từng lớp, cùng hai dạng confusion matrix, nằm trong JSON của mỗi mô hình. "
              "Mọi macro average dùng đủ tám lớp; các phép chia không xác định đóng góp 0.", "",
              f"![Confusion matrices của mô hình chọn trên test](figures/{selected}_test.png)", "",
              "## Tài nguyên và kiểm tra", "",
              f"Fit tiền xử lý mất {features['fitting_seconds']:.2f} giây. "
              f"Số chiều: {features['dimensions']}. "
              f"Mô hình được chọn huấn luyện trên {selected_result['training']['training_records']:,} bản ghi train.", "",
              "| Mô hình | Thời gian train (giây) | Số chiều | Chạm giới hạn vòng lặp LR |",
              "| --- | ---: | ---: | --- |"]
    failures = []
    for spec in config["candidates"]:
        record = read(f"candidates/{spec['id']}.json")
        if record["status"] != "completed":
            failures.append(record)
            continue
        info = record["training"]
        lines.append(f"| {spec['id']} | {info['training_seconds']:.2f} | {info['feature_dimension']} | {info.get('reached_iteration_limit', 'N/A')} |")
    lines += ["", "Thời gian biến đổi và ghi đặc trưng theo split: " + "; ".join(
        f"{split}: {info['transformation_and_write_seconds']:.2f} giây" for split, info in features["transformation"].items()) + ".", "",
        "Đã qua kiểm tra schema/trường bắt buộc, ID duy nhất trong và giữa split, biên UTC, nhãn theo score nhất quán, "
        "đối soát số lượng, median/count độc lập, train-only preprocessing, loại score khỏi danh sách đặc trưng và "
        "định nghĩa hai metric F1. Các estimator đều nhận toàn bộ train; RF dùng bagging nội bộ đã khai báo. "
        "SHA-256 đầu vào không thay đổi.", "",
        "Máy có khoảng 8 GB RAM; Spark chạy local[2] với heap 2 GB, cache/spill trên đĩa. "
        "Từ vựng và RF được giới hạn trong cấu hình. Các mô hình LR chạm giới hạn vòng lặp, nếu có, "
        "được đánh dấu ở bảng trên; không coi đó là bằng chứng đã hội tụ. PyArrow/SciPy không cần cho đường xử lý này. "
        "Không cài đặt hay tải dữ liệu, thư viện, hoặc mô hình pretrained.", ""]
    lines += ["Runtime báo không tải được native `libblas.so.3`, nhưng Spark vẫn hoàn tất bằng đường xử lý có sẵn. "
              "Nếu muốn bổ sung tăng tốc native trên Arch Linux, lệnh thủ công tùy chọn là `sudo pacman -S blas`; "
              "lệnh này chưa được chạy và không cần để tái lập đường xử lý hiện tại.", ""]
    if failures:
        lines += ["RF là so sánh tùy tài nguyên. Ứng viên chưa chạy thành công được ghi rõ, không có metric thay thế:", ""]
        lines += [f"- {r['candidate']['id']}: `{r['error']}`" for r in failures]
        lines.append("")
    lines += ["## Kết quả lưu và lệnh tái lập", "",
        "Dữ liệu nhãn/đặc trưng: `data/experiments/" + root.name + "/`. "
        "Tiền xử lý/mô hình: `artifacts/experiments/" + root.name + "/`. "
        "Các bảng đầy đủ: [model_comparison.csv](model_comparison.csv), [class_distributions.json](class_distributions.json), "
        "[label_boundaries.json](label_boundaries.json), [selection.json](selection.json).", "",
        "```bash", "JAVA_HOME=/usr/lib/jvm/java-21-openjdk SPARK_LOCAL_IP=127.0.0.1 PYTHONPATH=src \\",
        "  .venv/bin/python -m askreddit.experiment \\",
        "  --config configs/fang_chua_v1.json --run-id fang_chua_reproduction", "",
        "PYTHONPATH=src .venv/bin/python -m askreddit.report_results \\",
        "  --run-id fang_chua_reproduction", "```", "",
        "Dùng run ID mới cho một lần chạy mới; dùng `--resume` chỉ khi cần tái sử dụng các artifact đã được xác minh tương thích.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-root", default="reports/experiments")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.report_root) / args.run_id
    report = render(root)
    output = Path(args.output) if args.output else root / "results_vi.md"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
