# ISE Training Test - Context và kế hoạch thực hiện

## 1. Mục đích của tài liệu

Tài liệu này là nguồn ngữ cảnh chung cho toàn bộ quá trình giải bài toán. Mỗi
phase có thể được thực hiện trong một cuộc trò chuyện mới. Khi bắt đầu cuộc trò
chuyện, yêu cầu agent đọc toàn bộ file này, kiểm tra các artifact hiện có và chỉ
thực hiện phase được chỉ định.

Mục tiêu làm việc:

- Tối ưu **Macro F1 Score** trên test/private leaderboard.
- Mục tiêu vận hành ban đầu: OOF Macro F1 từ `0.40` trở lên.
- Hoàn thành một submission hợp lệ trong khoảng 5 giờ.
- Giữ giải pháp đủ rõ ràng để có thể hiểu và trình bày lại trong hai ngày.
- Không có phương pháp nào có thể bảo đảm thứ hạng top 2-top 3; mọi quyết định
  phải dựa trên validation, không dựa vào lời hứa về leaderboard.

## 2. Tóm tắt bài toán và dữ liệu đã xác minh

Đây là bài toán supervised single-label multiclass classification:

- `train.csv`: 51.452 dòng, 17 cột.
- `test.csv`: 21.947 dòng, 16 cột.
- Tổng cộng: 73.399 `track_id` duy nhất.
- 15 feature đầu vào.
- 112 lớp, target là `track_genre`.
- `track_genre` thực tế là số nguyên liên tục từ 0 đến 111.
- `genre_mapping.csv` ánh xạ `genre_id` sang tên genre để diễn giải.
- Submission phải có hai cột `track_id,track_genre` và dùng genre ID dạng số.
- Metric chính thức theo README là Macro F1, không phải accuracy hoặc recall.

Các nhóm feature:

- Thông tin cơ bản: `popularity`, `duration_ms`, `explicit`.
- Âm thanh: `danceability`, `energy`, `loudness`, `speechiness`,
  `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
- Nhạc lý/categorical: `key`, `mode`, `time_signature`.
- `track_id` chỉ dùng để ghép kết quả, không dùng làm feature dự báo.

Chất lượng dữ liệu đã xác minh:

- Không có NaN hoặc infinity.
- Không có `track_id` trùng trong từng tập hoặc giao nhau giữa train và test.
- Số mẫu mỗi lớp không cân bằng: ít nhất 51, nhiều nhất 700, trung vị 488.
- Train được sắp thành từng khối theo target; tuyệt đối không chia validation
  theo vị trí dòng.
- Có 585 nhóm feature trùng hoàn toàn trong train, gồm 1.988 dòng; tất cả bản
  trùng trong một nhóm có cùng nhãn.
- Không có bộ 15 feature trùng hoàn toàn giữa train và test.
- Có một dòng train có `duration_ms = 0`.
- Có 99 dòng train và 58 dòng test có đồng thời `tempo`, `danceability`,
  `speechiness`, `valence`, `time_signature` bằng 0. Cần giữ một indicator cho
  trạng thái này thay vì tự động xóa dòng.
- Phân phối biên train/test khá giống nhau; chưa thấy covariate shift lớn.

Một số lớp hiếm:

- `reggaeton`: 51.
- `reggae`: 61.
- `indie`: 85.
- `edm`: 90.

Một số lớp lớn:

- `tango`: 700.
- `sleep`: 699.
- `study`: 699.

## 3. Nguyên tắc làm việc bắt buộc

### Môi trường

Workspace sử dụng `.venv` với Python 3.14. Agent phải gọi interpreter tường
minh và kiểm tra trước khi chạy code:

```bash
.venv/bin/python --version
.venv/bin/python -c "import sys; print(sys.executable)"
```

Không dùng pip toàn cục hoặc `sudo pip`. Các package chính hiện có:

- numpy 2.5.1
- pandas 3.0.5
- scipy 1.18.0
- scikit-learn 1.9.0
- xgboost 3.4.1
- PyTorch CPU 2.13.0

CatBoost và LightGBM hiện chưa được cài. Không cài thêm trong luồng 5 giờ trừ
khi người dùng chủ động duyệt thay đổi.

### Quy tắc chống leakage

- Không dùng `track_id` làm feature.
- Fit preprocessing và mọi thống kê theo target chỉ trên training fold.
- Các dòng có 15 feature giống hệt nhau phải ở cùng một fold.
- Mọi model và mọi thử nghiệm phải dùng cùng một bộ fold đã lưu.
- Không nhìn test để chọn feature, hyperparameter, class weight hoặc ensemble
  weight.
- Không dùng artist/album/track metadata bên ngoài nếu chưa xác nhận luật cho
  phép external data.

### Quy tắc thí nghiệm

- Metric quyết định duy nhất: OOF Macro F1.
- Accuracy, Macro Recall, per-class precision/recall/F1 và log-loss chỉ dùng để
  chẩn đoán.
- Mỗi lần chỉ thay đổi một nhóm yếu tố có chủ đích.
- Ghi lại cấu hình, runtime, điểm từng fold, trung bình và độ lệch chuẩn.
- Không giữ thay đổi chỉ vì một fold tốt hơn; ưu tiên mean Macro F1 ổn định.
- Một cải tiến rất nhỏ dưới khoảng 0,003 phải được xem xét cùng độ biến động
  giữa các fold trước khi kết luận.
- Dùng `random_state=42` trừ khi đang kiểm tra độ nhạy theo seed.

### Quyền điều khiển của người dùng

Sau mỗi phase, agent phải báo cáo:

1. Những gì đã chạy hoặc thay đổi.
2. Artifact được tạo và đường dẫn.
3. Thời gian chạy.
4. Macro F1 từng fold, mean và standard deviation nếu đã train model.
5. Phân tích ngắn gọn kết quả.
6. Khuyến nghị rõ ràng: `GIỮ`, `BỎ` hoặc `THỬ TIẾP`.

Không tự động chuyển sang phase lớn tiếp theo nếu người dùng chưa yêu cầu.

## 4. Cấu trúc artifact đề xuất

Các phase nên dùng một notebook hoặc script rõ ràng, không sửa các CSV gốc.
Cấu trúc dự kiến:

```text
ISE_TRAINNING_TEST_23-8-2026/
├── CONTEXT.md
├── README.md
├── train.csv
├── test.csv
├── genre_mapping.csv
├── sample_submission.csv
├── 01_validation_and_baseline.ipynb
├── 02_xgboost_and_features.ipynb
├── 03_ensemble_and_submission.ipynb
├── artifacts/
│   ├── validation_folds.csv
│   ├── experiment_log.csv
│   ├── oof_extratrees.npy
│   ├── test_extratrees.npy
│   ├── oof_xgboost.npy
│   ├── test_xgboost.npy
│   └── metrics_*.json
└── submissions/
    └── submission_*.csv
```

Các file xác suất có thể lớn và thường không nên commit vào Git. Trước khi tạo
artifact, kiểm tra `.gitignore` và giữ dữ liệu nguồn nguyên vẹn.

## 5. Kế hoạch 5 giờ

| Khoảng thời gian | Công việc |
|---|---|
| 00:00-00:20 | Phase 1: validation chống leakage |
| 00:20-00:50 | Phase 2: Extra Trees baseline |
| 00:50-01:15 | Phase 3: feature engineering tối giản |
| 01:15-02:45 | Phase 4: XGBoost screening và 3-fold OOF |
| 02:45-03:25 | Phase 5: xử lý mất cân bằng/decision adjustment |
| 03:25-03:55 | Phase 6: ensemble xác suất |
| 03:55-04:20 | Phase 7: phân tích lỗi và quyết định cuối |
| 04:20-04:40 | Phase 8: tạo và kiểm tra submission |
| 04:40-05:00 | Dự phòng runtime, lỗi file hoặc submission |

Runtime thực tế có thể thay đổi do XGBoost multiclass phải dự đoán 112 lớp.
Nếu XGBoost chạy lâu, giảm số cấu hình thử nghiệm trước khi giảm chất lượng
validation hoặc bỏ kiểm tra submission.

## 6. Phase 1 - Validation chống leakage

### Mục tiêu

Tạo một cách chia train cố định, đáng tin cậy và dùng chung cho mọi thí nghiệm.

### Công việc

1. Đọc `train.csv` và `test.csv`.
2. Xác định:
   - `TARGET = "track_genre"`.
   - `ID_COLUMN = "track_id"`.
   - `FEATURES` là 15 cột test trừ `track_id`.
3. Tạo `group_id` bằng hash toàn bộ 15 feature.
4. Dùng `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.
5. Xác minh:
   - Mọi dòng được gán đúng một fold.
   - Mỗi validation fold có đủ 112 lớp.
   - Không có `group_id` xuất hiện ở cả train và validation trong cùng split.
   - Phân phối lớp giữa các fold hợp lý.
6. Lưu `artifacts/validation_folds.csv` gồm `track_id,fold`.
7. Định nghĩa hàm metric Macro F1 cố định với `labels=np.arange(112)` và
   `zero_division=0`.

### Đầu ra bắt buộc

- `artifacts/validation_folds.csv`.
- Bảng số dòng và số lớp theo fold.
- Kiểm tra leakage đều pass.

### Tiêu chí hoàn thành

Không model nào được train trước khi phase này hoàn thành.

### Prompt mở cuộc trò chuyện mới

> Đọc toàn bộ `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md`. Thực hiện duy nhất
> Phase 1 - Validation chống leakage. Giải thích từng bước trước hoặc ngay sau
> khi thực hiện, tạo artifact được yêu cầu, báo cáo kiểm tra và dừng ở checkpoint
> để tôi quyết định.

## 7. Phase 2 - Extra Trees baseline

### Mục tiêu

Tạo baseline phi tuyến nhanh, dễ hiểu và có OOF probability để làm mốc so sánh.

### Cấu hình xuất phát

```python
ExtraTreesClassifier(
    n_estimators=600,
    max_features=0.8,
    min_samples_leaf=2,
    class_weight=None,
    n_jobs=-1,
    random_state=42,
)
```

### Công việc

1. Dùng đúng fold từ Phase 1.
2. Train ba model, mỗi fold một lần làm validation.
3. Lưu OOF probability theo đúng thứ tự train.
4. Predict test ở mỗi fold và lấy trung bình xác suất.
5. Báo cáo:
   - Macro F1 từng fold, mean và standard deviation.
   - Accuracy và Macro Recall để chẩn đoán.
   - 15 genre có F1 thấp nhất.
   - Runtime train và predict.
6. Lưu model metadata/cấu hình và xác suất.

### Đầu ra dự kiến

- `artifacts/oof_extratrees_raw.npy`.
- `artifacts/test_extratrees_raw.npy`.
- Metrics JSON và một dòng trong `experiment_log.csv`.

### Quyết định checkpoint

- Giữ Extra Trees làm thành viên ensemble nếu xác suất và CV ổn định.
- Không tune rộng; chỉ thử `min_samples_leaf` hoặc `max_features` nếu có dấu
  hiệu overfit rõ ràng.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và kiểm tra artifact Phase 1.
> Thực hiện duy nhất Phase 2 - Extra Trees baseline trên đúng fold đã lưu. Giải
> thích model ở mức tôi có thể học lại, lưu OOF/test probability, báo cáo Macro
> F1 từng fold và dừng ở checkpoint.

## 8. Phase 3 - Feature engineering có kiểm soát

### Mục tiêu

Thêm một số biểu diễn có ý nghĩa miền, dễ giải thích và kiểm chứng riêng biệt.

### Feature ứng viên

```text
duration_log       = log1p(duration_ms)
key_sin            = sin(2*pi*key/12)
key_cos            = cos(2*pi*key/12)
tempo_normalized   = tempo đưa về vùng 80-160 bằng half/double time
audio_zero_flag    = tempo == 0 và time_signature == 0
energy_loudness    = energy * loudness
dance_valence      = danceability * valence
acoustic_instru    = acousticness * instrumentalness
```

Luôn giữ feature gốc bên cạnh feature mới, trừ khi validation chứng minh việc
loại bỏ có lợi.

### Công việc

1. Viết một hàm feature engineering dùng giống nhau cho train/test.
2. Kiểm tra shape, dtype, NaN và infinity sau biến đổi.
3. Chạy Extra Trees trên đúng fold để so sánh `raw` và `engineered`.
4. Không thay hyperparameter model trong cùng thí nghiệm so sánh feature.

### Tiêu chí giữ

- Macro F1 tăng đủ rõ so với biến động fold; hoặc
- Giúp các lớp khó mà không làm mean Macro F1 giảm đáng kể.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và kết quả Phase 2. Thực hiện
> duy nhất Phase 3. Giải thích ý nghĩa từng feature, so sánh raw với engineered
> trên cùng fold và cùng Extra Trees config, lưu kết quả rồi dừng để tôi quyết
> định giữ/bỏ từng nhóm feature.

## 9. Phase 4 - XGBoost multiclass

### Mục tiêu

Huấn luyện mô hình chính có khả năng học quan hệ phi tuyến và tương tác giữa ít
feature trên bài toán 112 lớp.

### Cấu hình xuất phát

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=112,
    tree_method="hist",
    n_estimators=350,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=5.0,
    n_jobs=-1,
    random_state=42,
)
```

Phải dùng `multi:softprob` để giữ xác suất 112 lớp phục vụ ensemble và decision
adjustment. Không dùng `multi:softmax` cho pipeline chính.

### Cách tiết kiệm thời gian

1. Chạy một fold với cấu hình xuất phát để đo runtime.
2. Chỉ screening tối đa ba cấu hình:
   - XGB-A: `max_depth=6`.
   - XGB-B: `max_depth=8`.
   - XGB-C: `max_depth=7` với bộ feature tốt nhất Phase 3.
3. Không grid search rộng.
4. Chọn một cấu hình rồi chạy đủ ba fold.

### Đầu ra dự kiến

- `artifacts/oof_xgboost_*.npy`.
- `artifacts/test_xgboost_*.npy`.
- Metrics từng fold, mean/std, runtime và cấu hình đầy đủ.

### Tiêu chí quyết định

Chọn bằng OOF Macro F1. Nếu hai cấu hình rất gần nhau, ưu tiên cấu hình ổn định
hơn và dễ giải thích hơn.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` cùng kết quả các phase trước.
> Thực hiện Phase 4 - XGBoost theo chiến lược screening có giới hạn thời gian.
> Đo runtime một fold trước, không grid search rộng, báo cáo Macro F1 và dừng ở
> checkpoint trước khi chạy biến thể tốn thời gian khác.

## 10. Phase 5 - Mất cân bằng và điều chỉnh quyết định

### Mục tiêu

Cải thiện Macro F1 bằng cách giảm thiên lệch về lớp lớn mà không làm precision
của lớp hiếm sụp đổ.

### Phương pháp ưu tiên: prior correction trên OOF

Với xác suất `p_k` và tần suất lớp `freq_k`:

```text
adjusted_score_k = p_k / (freq_k ** beta)
prediction = argmax(adjusted_score)
```

Thử `beta` trên OOF, ví dụ từ 0,00 đến 0,80 theo bước 0,05 hoặc 0,10. Không dùng
test để chọn beta.

### Phương pháp thứ hai: sample weight mềm

Chỉ train thêm nếu còn thời gian và OOF cho thấy lớp hiếm cần hỗ trợ:

```text
weight_i = (median_class_count / count[y_i]) ** alpha
```

Thử ít giá trị `alpha`, ưu tiên `0.3`, `0.5`, `0.7`. Tính class count từ training
fold. Không dùng full inverse weighting mặc định.

### Nguyên tắc

- So sánh unweighted, prior correction và weighted model độc lập.
- Không chồng weighting mạnh với beta mạnh mà không kiểm chứng.
- Chọn duy nhất theo OOF Macro F1.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và nạp OOF probability tốt
> nhất. Thực hiện Phase 5, bắt đầu bằng prior correction không cần retrain. Cho
> tôi bảng beta so với Macro F1 và per-class effects. Chỉ đề xuất weighted
> retraining nếu dữ liệu chứng minh cần thiết, rồi dừng ở checkpoint.

## 11. Phase 6 - Ensemble xác suất

### Mục tiêu

Kết hợp sai số khác nhau của Extra Trees và XGBoost.

### Công thức

```text
p_ensemble = w * p_xgboost + (1-w) * p_extratrees
```

Thử `w` từ 0,5 đến 1,0, sau đó áp dụng beta nhỏ nếu Phase 5 cho thấy có lợi.

### Công việc

1. Xác minh class order và row order của các file xác suất giống nhau.
2. Tune `w` chỉ trên OOF.
3. Với mỗi `w`, thử một tập beta giới hạn.
4. Báo cáo bề mặt kết quả `w x beta` và chọn vùng ổn định, không chỉ một điểm
   nhọn có thể overfit.
5. Tạo test ensemble bằng đúng `w` và beta đã chọn trên OOF.

### Tiêu chí giữ

Ensemble phải tốt hơn XGBoost đơn trên OOF Macro F1 đủ rõ hoặc ổn định hơn giữa
các fold.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và kiểm tra OOF/test
> probabilities của Extra Trees và XGBoost. Thực hiện Phase 6, xác minh alignment
> trước khi blend, tune w và beta chỉ bằng OOF Macro F1, báo cáo vùng tham số ổn
> định rồi dừng để tôi chọn cấu hình cuối.

## 12. Phase 7 - Phân tích lỗi và quyết định cuối

### Mục tiêu

Hiểu model đang sai ở đâu và quyết định có đủ lý do để thử thêm một thay đổi
trước deadline hay không.

### Báo cáo cần có

- Confusion matrix theo OOF.
- 20 genre có F1 thấp nhất và 20 genre cao nhất.
- Các cặp nhãn nhầm lẫn lớn nhất bằng tên từ `genre_mapping.csv`.
- Quan hệ giữa class count và per-class F1.
- Tần suất nhãn prediction so với tần suất nhãn thật.
- So sánh XGBoost, Extra Trees và ensemble theo từng lớp.

### Quyết định

- Nếu còn ít hơn 40 phút: khóa cấu hình và chuyển sang submission.
- Chỉ tạo specialist model nếu một nhóm nhỏ confusion chiếm phần lỗi lớn, còn
  đủ thời gian và có cách validation rõ ràng.
- Genre mapping chỉ dùng để diễn giải hoặc tạo tín hiệu family phụ; không ghép
  tên genre vào feature của từng row.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và kết quả ensemble tốt nhất.
> Thực hiện duy nhất Phase 7: phân tích lỗi OOF bằng tên genre, chỉ ra nguồn lỗi
> lớn nhất và đưa khuyến nghị có căn cứ là khóa model hay thử thêm. Không tự ý
> train mô hình mới trước khi tôi quyết định.

## 13. Phase 8 - Tạo, kiểm tra và nộp submission

### Mục tiêu

Tạo submission hợp lệ từ cấu hình đã khóa, không thay đổi model ở phase này.

### Công việc

1. Dùng test probability đã average qua ba fold.
2. Áp dụng đúng ensemble weight và beta đã chọn bằng OOF.
3. `argmax` để nhận genre ID 0-111.
4. Tạo đúng hai cột:

   ```text
   track_id,track_genre
   ```

5. Kiểm tra:
   - Đúng 21.947 dòng.
   - ID cùng thứ tự với `test.csv`.
   - ID duy nhất, không thiếu hoặc thừa.
   - Target là integer và nằm trong 0-111.
   - Không NaN/infinity.
   - Có nhiều hơn một nhãn được dự đoán.
   - File đọc lại bằng pandas không thay đổi shape/dtype ngoài dự kiến.
6. Lưu tên file chứa timestamp hoặc điểm OOF, ví dụ:

   ```text
   submissions/submission_xgb_extra_oof_0xxxx.csv
   ```

### Đầu ra

- Một submission chính đã kiểm tra.
- Một bản ghi đầy đủ về model/config/OOF score tạo ra submission.
- Nếu tạo submission dự phòng, nó phải xuất phát từ cấu hình OOF tốt thứ hai,
  không phải một biến thể ngẫu nhiên.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và cấu hình cuối đã được duyệt.
> Thực hiện duy nhất Phase 8: tạo submission, chạy toàn bộ sanity checks, báo cáo
> đường dẫn và thống kê nhãn. Không thay model hoặc chọn tham số bằng test.

## 14. Phase 9 tùy chọn - Học sâu và cải tiến sau submission

Phase này không nằm trên critical path 5 giờ. Mục tiêu là hiểu sâu giải pháp
trong hai ngày và chuẩn bị phiên bản sau.

Các chủ đề theo thứ tự:

1. Hiểu Macro F1 từ confusion matrix và per-class precision/recall.
2. Hiểu Extra Trees: random split, bagging, bias/variance.
3. Hiểu XGBoost: boosting tuần tự, learning rate, depth và regularization.
4. Hiểu OOF prediction, CV bagging và vì sao không được tune bằng test.
5. Hiểu probability ensemble và prior correction.
6. Xây per-genre statistical profile bằng median/IQR và class likelihood.
7. Thử similarity-to-genre features, fit hoàn toàn bên trong từng fold.
8. Phân cụm confusion để tạo genre family mềm.
9. Chỉ sau đó mới cân nhắc One-vs-Rest, specialist model hoặc multi-task MLP.

### Prompt mở cuộc trò chuyện mới

> Đọc `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md` và toàn bộ experiment log. Dạy
> lại cho tôi Phase 9 theo kết quả thực tế của dự án, bắt đầu từ Macro F1 và OOF.
> Mỗi khái niệm phải gắn với code hoặc artifact đã tạo, chưa chạy cải tiến mới
> nếu tôi chưa yêu cầu.

## 15. Các hướng không ưu tiên trong deadline 5 giờ

- Neural network lớn: chỉ có 15 feature và không có embedding giàu thông tin.
- KNN: nhạy với scale và khó trong vùng genre chồng lấn.
- RBF-SVM: tốn chi phí cho 51.452 mẫu và 112 lớp, xác suất khó hiệu chỉnh.
- 112 One-vs-Rest XGBoost: chi phí lớn, score giữa các model cần calibration.
- Hard hierarchy: lỗi family loại bỏ luôn genre đúng.
- Pseudo-label test: rủi ro khuếch đại lỗi khi chưa có baseline mạnh.
- External Spotify metadata: có thể vi phạm luật hoặc tinh thần đề.
- Grid search rộng: không phù hợp deadline CPU và không tăng hiểu biết tương
  ứng.

## 16. Nhật ký trạng thái

Mỗi cuộc trò chuyện hoàn thành phase phải cập nhật phần này bằng kết quả ngắn
gọn và đường dẫn artifact. Không ghi dự đoán hoặc kết quả chưa kiểm chứng.

- [x] Phase 1 - Validation chống leakage. Hoàn thành ngày 2026-08-23 bằng
  `01_validation_and_baseline.ipynb`. Đã lưu
  `artifacts/validation_folds.csv`: fold 0/1/2 có lần lượt
  17.152/17.150/17.150 dòng, mỗi fold đủ 112 lớp, group overlap bằng 0 và sai
  lệch tỷ lệ lớp lớn nhất so với 1/3 là 1,0929%. Runtime notebook khoảng 11 giây.
- [ ] Phase 2 - Extra Trees baseline.
- [ ] Phase 3 - Feature engineering.
- [ ] Phase 4 - XGBoost multiclass.
- [ ] Phase 5 - Mất cân bằng/decision adjustment.
- [ ] Phase 6 - Ensemble.
- [ ] Phase 7 - Phân tích lỗi.
- [ ] Phase 8 - Submission.
- [ ] Phase 9 - Học sâu/cải tiến sau submission.

## 17. Prompt chung khi bắt đầu một cuộc trò chuyện mới

Sao chép prompt sau và thay `<PHASE>`:

> Chúng ta đang làm dự án ISE Training Test. Hãy đọc toàn bộ
> `ISE_TRAINNING_TEST_23-8-2026/CONTEXT.md`, kiểm tra trạng thái và artifact hiện
> có, sau đó thực hiện duy nhất `<PHASE>`. Tuân thủ `.venv`, chống leakage và dùng
> OOF Macro F1 làm metric quyết định. Trước thay đổi lớn hãy giải thích mục đích;
> cuối phase cập nhật nhật ký trong CONTEXT.md, báo cáo kết quả và dừng để tôi
> quyết định phase tiếp theo.
