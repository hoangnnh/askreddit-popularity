# Kết quả bài toán nhị phân — run `binary_20261125_v5`

Báo cáo sinh tự động bằng `python -m askreddit.binary_report` từ các file JSON cùng thư mục.

> **Công bố:** Cấu hình v5 thiết kế ngày 2026-11-25 sau v4. Giống hệt v4 (chỉ xét bài score ≥ 2, thấp = dưới trung vị, cao = 25% cao nhất, mô hình chỉ dùng tiêu đề) nhưng huấn luyện trên dữ liệu đã bỏ trùng tiêu đề (mỗi tiêu đề chuẩn hóa giữ một bài có score cao nhất; xem reports/data_quality_dedup/run.json). Việc chọn bài giữ lại dùng score cả năm 2025, kể cả các tháng validation/test; không tiêu đề nào nằm ở hai tập. Tháng test 12/2025 đã được dùng ở các run trước; mô hình chỉ được chọn bằng validation.

## 1. Gán nhãn (chỉ dùng tập train)

- Tập tính phân vị: **train posts with score >= 2**. Trung vị = **3**, phân vị 75% = **4** (Inverse empirical CDF: the smallest train score s with F(s) >= q, from exact integer counts.). DuckDB kiểm tra độc lập: **passed**.
- **Phổ biến cao:** score >= 4 (p75 of train posts with score >= 2). **Phổ biến thấp:** score < 3 (median of train posts with score >= 2). Nhóm giữa (và bài ngoài tập tính phân vị, nếu có) bị loại khỏi huấn luyện và đánh giá.
- Trong tập score ≥ 2 của train, trung vị = 3 và p75 = 4. Lấy thấp = score < trung vị (49,1%, gần 50% nhất) và cao = score ≥ p75 (28,7%, gần 25% nhất) — đúng nghĩa đen "dưới trung vị" và "25% cao nhất" trong đề cương.

| Split | Tổng | Loại (ngoài tập) | Thấp | Giữa (loại) | Cao | Tỉ lệ cao trong dữ liệu giữ lại |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 122,787 | 86,108 | 16,958 | 8,382 | 11,339 | 40.07% |
| train | 1,203,860 | 858,185 | 169,365 | 76,832 | 99,478 | 37.00% |
| validation | 232,113 | 164,016 | 31,531 | 15,264 | 21,302 | 40.32% |

## 2. Chọn N-gram bằng chi-square (train)

MLlib ChiSquareTest on binary train term presence; top-k by statistic, ties by vocabulary index. Chọn **128** n-gram cho GBT (tỉ lệ cao toàn train: 37.00%).

| Hạng | N-gram | Chi² | Số bài train | Tỉ lệ cao khi có n-gram |
| --- | ---: | ---: | ---: | ---: |
| 1 | `or` | 705 | 24,923 | 29.27% |
| 2 | `and` | 689 | 38,829 | 31.05% |
| 3 | `what’s` | 451 | 42,187 | 41.58% |
| 4 | `how` | 370 | 34,610 | 32.34% |
| 5 | `instantly` | 344 | 2,286 | 55.64% |
| 6 | `trump` | 291 | 918 | 64.16% |
| 7 | `what’s a` | 277 | 11,111 | 44.46% |
| 8 | `something` | 222 | 18,867 | 42.06% |
| 9 | `what’s something` | 201 | 7,279 | 44.92% |
| 10 | `with` | 197 | 18,148 | 32.14% |
| 11 | `in` | 187 | 43,200 | 34.09% |
| 12 | `for` | 184 | 25,890 | 33.13% |
| 13 | `that instantly` | 182 | 948 | 58.12% |
| 14 | `epstein` | 172 | 299 | 73.58% |
| 15 | `some` | 160 | 8,057 | 30.31% |

## 3. So sánh mô hình

Chọn mô hình bằng **AUC-ROC trên validation** (10–11/2025). Ngưỡng xác suất của từng mô hình là ngưỡng tối đa F1 lớp cao trên validation, cố định trước khi chấm test (12/2025). Lớp dương = phổ biến cao.

| Mô hình | Họ | Đặc trưng | AUC val | AUC test | PR-AUC test | Ngưỡng | Precision test | Recall test | F1 test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority` | Baseline lớp đa số | — | 0.5000 | 0.5000 | 0.4007 | 0.500 | 0.0000 | 0.0000 | 0.0000 |
| `time_lr` | Logistic Regression (L2) | giờ + thứ (UTC) | 0.5193 | 0.5161 | 0.4090 | 0.472 | 0.4007 | 1.0000 | 0.5722 |
| `title_structured_lr` | Logistic Regression (L2) | độ dài + dấu hỏi + chữ hoa (không thời gian) | 0.5479 | 0.5533 | 0.4414 | 0.245 | 0.4007 | 0.9996 | 0.5721 |
| `text_lr_r1e5` | Logistic Regression (L2) | TF-IDF 1–2-gram | 0.6152 | 0.5989 | 0.4906 | 0.346 | 0.4182 | 0.9210 | 0.5752 |
| `combined_title_lr_r1e4` | Logistic Regression (L2) | TF-IDF + cấu trúc tiêu đề (không thời gian) | 0.6120 | 0.5981 | 0.4900 | 0.396 | 0.4145 | 0.9354 | 0.5745 |
| `combined_title_lr_r1e5` ⭐ | Logistic Regression (L2) | TF-IDF + cấu trúc tiêu đề (không thời gian) | 0.6155 | 0.5999 | 0.4915 | 0.343 | 0.4182 | 0.9244 | 0.5759 |
| `gbt_d5` | Gradient Boosted Trees | top-k n-gram (chi-square) + cấu trúc tiêu đề (không thời gian) | 0.5844 | 0.5781 | 0.4684 | 0.306 | 0.4027 | 0.9923 | 0.5729 |

Cùng các mô hình tại **ngưỡng 0,5** — điểm cắt tự nhiên khi huấn luyện với trọng số lớp cân bằng (các số này do mã đánh giá ghi sẵn cho mọi ứng viên, không tính lại sau khi xem test):

| Mô hình | Balanced acc. val | Balanced acc. test | Precision test | Recall test | F1 test | % đoán cao (test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority` | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| `time_lr` | 0.5116 | 0.5157 | 0.4158 | 0.5192 | 0.4618 | 50.0% |
| `title_structured_lr` | 0.5326 | 0.5397 | 0.4322 | 0.6542 | 0.5205 | 60.7% |
| `text_lr_r1e5` | 0.5825 | 0.5709 | 0.4733 | 0.5542 | 0.5105 | 46.9% |
| `combined_title_lr_r1e4` | 0.5800 | 0.5687 | 0.4681 | 0.5718 | 0.5148 | 48.9% |
| `combined_title_lr_r1e5` ⭐ | 0.5838 | 0.5716 | 0.4738 | 0.5564 | 0.5118 | 47.1% |
| `gbt_d5` | 0.5592 | 0.5571 | 0.4548 | 0.5752 | 0.5080 | 50.7% |

> **Lưu ý về ngưỡng.** Ngưỡng tối đa F1 trên validation (0.343) gần như đoán "cao" cho mọi bài (recall 0.924): F1 0.5759 chỉ nhỉnh hơn mức luôn đoán cao (0.5722). Khi lớp dương chiếm gần 40% và tín hiệu yếu, F1 tối đa đạt được bằng cách đoán cao gần hết. Vì vậy AUC-ROC/PR-AUC (không phụ thuộc ngưỡng) là thước đo chính, và kết quả phân lớp được đọc tại ngưỡng 0,5. Việc nhấn mạnh ngưỡng 0,5 được quyết định sau khi xem kết quả run này.

Mốc tham chiếu trên test: tỉ lệ lớp cao = 40.07% (= PR-AUC ngẫu nhiên); luôn dự đoán "cao" cho F1 = 0.5722; baseline lớp đa số (luôn "thấp") có AUC 0.5 và F1 = 0.

**Mô hình được chọn:** `combined_title_lr_r1e5`. Tốt nhất mỗi họ: LR = `combined_title_lr_r1e5`, GBT = `gbt_d5`.

![ROC test](figures/roc_test.png)

![Ma trận nhầm lẫn](figures/confusion_test.png)

Trên 28,297 bài test có tiêu đề chưa từng xuất hiện trong train, `combined_title_lr_r1e5` đạt AUC 0.5999, F1 0.5759 (toàn test: AUC 0.5999, F1 0.5759).

## 4. N-gram quan trọng (hệ số Logistic Regression)

Hệ số của `combined_title_lr_r1e5`; dương = đẩy về phổ biến cao, âm = đẩy về phổ biến thấp. Đây là quan hệ dự đoán, không phải nhân quả.

| Hạng | Đẩy về CAO | Hệ số | Đẩy về THẤP | Hệ số |
| --- | ---: | ---: | ---: | ---: |
| 1 | `trump` | 3.585 | `or` | -2.112 |
| 2 | `nsfw` | 2.332 | `how` | -1.542 |
| 3 | `epstein` | 2.125 | `why is` | -1.541 |
| 4 | `conservatives` | 2.005 | `and` | -1.439 |
| 5 | `sex` | 1.921 | `vs` | -1.406 |
| 6 | `congress` | 1.839 | `what` | -1.382 |
| 7 | `depression` | 1.727 | `ai` | -1.374 |
| 8 | `instantly` | 1.719 | `for` | -1.367 |
| 9 | `compliment` | 1.708 | `in` | -1.315 |
| 10 | `sexual` | 1.659 | `story` | -1.294 |
| 11 | `americans` | 1.643 | `where` | -1.289 |
| 12 | `naked` | 1.642 | `and are` | -1.284 |
| 13 | `politicians` | 1.550 | `shoes` | -1.269 |
| 14 | `you` | 1.515 | `prank` | -1.247 |
| 15 | `attractive` | 1.508 | `drama` | -1.229 |

![Top n-gram](figures/lr_top_ngrams.png)

Hệ số đặc trưng cấu trúc (trên mỗi độ lệch chuẩn train):

| Đặc trưng | Hệ số |
| --- | ---: |
| Số dấu hỏi | -0.0153 |
| Có dấu hỏi | 0.0033 |
| Số ký tự | -0.2160 |
| Số từ | 0.1610 |
| Tỉ lệ chữ hoa | -0.0210 |

## 5. Feature importance (GBT)

`gbt_d5`: đặc trưng cấu trúc chiếm 16.76% tổng importance, riêng giờ/thứ chiếm 0.00%.

| Hạng | Đặc trưng | Importance |
| --- | ---: | ---: |
| 1 | Số ký tự | 0.0799 |
| 2 | Tỉ lệ chữ hoa | 0.0551 |
| 3 | `you` | 0.0387 |
| 4 | `what` | 0.0331 |
| 5 | `instantly` | 0.0320 |
| 6 | `and` | 0.0269 |
| 7 | Số từ | 0.0253 |
| 8 | `something` | 0.0239 |
| 9 | `the` | 0.0200 |
| 10 | `for` | 0.0174 |
| 11 | `story` | 0.0162 |
| 12 | `what’s` | 0.0157 |
| 13 | `sex` | 0.0156 |
| 14 | `best` | 0.0153 |
| 15 | `that` | 0.0144 |

![GBT importance](figures/gbt_importance.png)

## 6. Ảnh hưởng của thời điểm đăng

- `time_lr`: theo giờ, log-odds dao động 0.210 (cao nhất 6h, thấp nhất 18h UTC); theo thứ dao động 0.026 (cao nhất T3, thấp nhất T7).
- Chỉ dùng giờ/thứ: AUC test 0.5161, gần mức ngẫu nhiên 0.5. Thời điểm đăng chỉ có tín hiệu yếu so với nội dung tiêu đề.

![Ảnh hưởng thời gian](figures/time_effects.png)

## 7. Hạn chế

- Bỏ nhóm giữa (76,832 bài train) làm bài toán dễ hơn thực tế: mô hình không phải phân biệt các bài sát ranh giới.
- Bỏ 858,185 bài train có score ≤ 1: mô hình trả lời câu hỏi "trong các bài đã có tương tác (score ≥ 2), bài nào phổ biến", chưa áp dụng trực tiếp cho một bài vừa đăng chưa biết có bị gỡ hay không.
- Score là giá trị lưu trong archive, không phải score sau một khoảng thời gian cố định; bài bị gỡ có thể mang score thấp.
- Tháng test 12/2025 đã được dùng để chấm các run trước, nên test không còn hoàn toàn chưa nhìn thấy.
- Chưa chạy: trigram, thí nghiệm loại từ dừng, giờ miền Đông Hoa Kỳ (từ vựng hiện tại là 1–2-gram, giữ từ dừng, giờ UTC).
- Chi-square chọn n-gram trên toàn bộ train trước khi huấn luyện GBT; ngưỡng và lựa chọn mô hình chỉ dùng validation.
