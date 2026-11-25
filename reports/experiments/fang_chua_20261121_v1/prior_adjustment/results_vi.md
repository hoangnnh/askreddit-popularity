# Điều chỉnh prior sau huấn luyện — run `fang_chua_20261121_v1`

> **Lưu ý:** Phân tích này được thiết kế ngày 2026-11-25 sau khi đã xem kết quả validation VÀ test của run này. τ chỉ được chọn bằng macro-F1 trên validation, nhưng test không còn là hold-out hoàn toàn chưa nhìn thấy theo nghĩa chặt.

Mô hình đã lưu không được huấn luyện lại. Điểm lớp s_k(x) (margin của LR, log xác suất của RF) được trừ τ·log π_k, với π là phân phối nhãn của tập train ([Menon và cộng sự, 2021](https://arxiv.org/abs/2007.07314)). Mỗi ứng viên chọn một τ theo macro-F1 tám lớp trên validation, lưới τ từ -1.0 đến 2.0 (61 giá trị). τ = 0 tái tạo chính xác ma trận nhầm lẫn đã lưu của validation và test (đã kiểm tra). Test chỉ được chấm tại τ = 0 và tại τ đã chọn.

Ứng viên tốt nhất sau điều chỉnh (theo validation): **combined_a05_r003**, τ = **0.5**. Macro-F1 trên test: 0.1033 → **0.1245** (baseline majority 0.1033).
Lựa chọn gốc của run: `combined_a1_r003`.

| Ứng viên | τ* | Val macro-F1 (τ=0 → τ*) | Test macro-F1 (τ=0 → τ*) | Test balanced acc. (τ=0 → τ*) | Test accuracy (τ=0 → τ*) |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_lr | 1.0 | 0.1037 → 0.1131 | 0.1033 → 0.1107 | 0.1250 → 0.1322 | 0.7046 → 0.3777 |
| text_lr | 1.0 | 0.1037 → 0.1273 | 0.1033 → 0.1238 | 0.1250 → 0.1630 | 0.7046 → 0.4465 |
| combined_a0_r003 | 0.85 | 0.1037 → 0.1222 | 0.1033 → 0.1202 | 0.1250 → 0.1345 | 0.7046 → 0.5808 |
| combined_a0_r010 | 0.85 | 0.1037 → 0.1208 | 0.1033 → 0.1193 | 0.1250 → 0.1303 | 0.7046 → 0.6323 |
| combined_a05_r003 | 0.5 | 0.1037 → 0.1308 | 0.1033 → 0.1245 | 0.1250 → 0.1563 | 0.7046 → 0.3374 |
| combined_a05_r010 | 0.5 | 0.1037 → 0.1299 | 0.1033 → 0.1246 | 0.1250 → 0.1533 | 0.7046 → 0.3386 |
| combined_a1_r003 | 0.0 | 0.1198 → 0.1198 | 0.1135 → 0.1135 | 0.1889 → 0.1889 | 0.3579 → 0.3579 |
| combined_a1_r010 | 0.0 | 0.1193 → 0.1193 | 0.1135 → 0.1135 | 0.1882 → 0.1882 | 0.3586 → 0.3586 |
| rf_compact | 0.4 | 0.1037 → 0.1243 | 0.1033 → 0.1238 | 0.1250 → 0.1348 | 0.7046 → 0.6412 |

Chi tiết: [results.json](results.json), [validation_sweep.json](validation_sweep.json), [comparison.csv](comparison.csv), [plan.json](plan.json).
