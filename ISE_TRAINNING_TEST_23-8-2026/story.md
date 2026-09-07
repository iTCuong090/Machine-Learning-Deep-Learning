# Từ chiếc cân đúng đến 0.403 Public: câu chuyện thật của bài toán 112 thể loại nhạc

Đây không phải câu chuyện về một mô hình “thần kỳ”. Điểm tốt nhất được ghi lại trong repo đến từ việc sửa lần lượt ba thứ ít hào nhoáng hơn: **cách đo**, **cách kết hợp các góc nhìn**, và **cách biến xác suất thành quyết định phù hợp với Macro F1**.

Kết quả cuối cùng được chọn là [`submission_iter3_prior_alpha_0p30.csv`](./submission_iter3_prior_alpha_0p30.csv): ensemble 5-fold của XGBoost, LightGBM và Extra Trees, với trọng số `0.40 / 0.25 / 0.35`, prior alpha `0.30` và threshold gamma `0.75`. Nhật ký trong repo ghi nhận file này đạt **Public Macro F1 = 0.403 trên 51% test**. Không có Private score hay thứ hạng cuối trong repo, nên câu chuyện dừng đúng ở điều có thể chứng minh: đây là ứng viên cuối được chọn, không phải lời khẳng định top bao nhiêu.

## 1. Đề bài nhỏ ở số cột, lớn ở số lớp

[`README.md`](./README.md) mô tả một bài toán multiclass tưởng như gọn: dự đoán genre của một track Spotify từ 15 thuộc tính như popularity, duration, danceability, energy, loudness, acousticness, tempo, key, mode và time signature. Nhưng dữ liệu có:

- 51.452 dòng train và 21.947 dòng test;
- 73.399 `track_id` duy nhất;
- 112 lớp, mã liên tục từ 0 đến 111;
- lớp nhỏ nhất chỉ có 51 mẫu, lớp lớn nhất có 700 mẫu, trung vị là 488.

Điều quyết định toàn bộ chiến thuật là metric **Macro F1**. Mỗi genre có một lá phiếu ngang nhau, dù là `reggaeton` với 51 mẫu hay `tango` với 700 mẫu. Vì vậy, một model đoán tốt các lớp đông nhưng làm biến mất vài lớp hiếm vẫn bị phạt mạnh. Accuracy được giữ lại để chẩn đoán, nhưng không được dùng để chọn phương án.

`track_id` cũng bị loại khỏi feature ngay từ đầu. Nó chỉ còn một nhiệm vụ: giữ đúng thứ tự khi đóng gói submission. Repo không dùng tên ca sĩ, album, tên bài hát hay dữ liệu Spotify bên ngoài.

## 2. Thành tựu đầu tiên không phải điểm số, mà là một chiếc cân đáng tin

Ngày 23/8/2026, dự án khởi đầu bằng [`CONTEXT.md`](./CONTEXT.md) và [`01_validation_and_baseline.ipynb`](./01_validation_and_baseline.ipynb), được commit ở `8003a77` lúc 18:06. Kế hoạch đặt mục tiêu OOF Macro F1 từ `0.40`, thời gian khoảng năm giờ và một nguyên tắc: chưa được tin bất kỳ model nào trước khi validation chống leakage hoàn tất.

Có hai cái bẫy thật trong dữ liệu.

Thứ nhất, train được xếp thành các khối theo target. Nếu chia theo vị trí dòng, validation sẽ không đại diện cho dữ liệu chưa thấy.

Thứ hai, 15 feature không hoàn toàn duy nhất. Notebook validation đếm được:

- 50.049 vector feature khác nhau trên 51.452 dòng;
- 585 nhóm có từ hai dòng trở lên;
- 1.988 dòng nằm trong các nhóm trùng;
- tương đương 1.403 dòng lặp “ngoài bản đầu tiên”.

Tất cả dòng trong cùng một nhóm trùng đều có cùng nhãn. Xóa chúng sẽ làm mất tín hiệu về tần suất; để các bản sao rơi sang hai phía train/validation lại khiến cây có thể nhớ một mẫu giống hệt. Lựa chọn thực tế là hash toàn bộ 15 feature thành `group_id`, rồi dùng `StratifiedGroupKFold`:

- `Stratified` giữ tỷ lệ của 112 lớp gần cân bằng giữa các fold;
- `Group` khóa mọi bản sao giống hệt vào cùng một fold.

Lượt validation độc lập đầu tiên dùng 3 fold và seed 42, sinh ra [`artifacts/validation_folds.csv`](./artifacts/validation_folds.csv). Ba validation fold có 17.152, 17.150 và 17.150 dòng; mỗi fold đủ 112 lớp; số group giao nhau giữa train và validation bằng 0; sai lệch tỷ lệ lớp lớn nhất so với `1/3` chỉ 1,0929%.

Đây là quyết định quan trọng nhất của cả bài. Nếu chiếc cân sai, mọi chữ số phía sau chỉ là ảo giác.

### Một ngã rẽ cần nói thật

Kế hoạch trong `CONTEXT.md` yêu cầu mọi phase dùng lại đúng artifact seed 42. Nhưng notebook lời giải sau đó không đọc file fold này; nó tự tạo lại group split với seed `20260823`. Năm vòng leaderboard tiếp theo cũng dùng seed `20260823`. Như vậy, lời hứa “dùng đúng một file fold từ đầu đến cuối” đã không được thực hiện nguyên văn.

Điều vẫn được giữ là nguyên tắc cốt lõi: mọi model trong từng chuỗi so sánh dùng chung split, mọi duplicate group ở cùng một phía, mọi fold đủ 112 lớp và test không tham gia chọn split. Đây là một sai khác về quy trình tái lập cần ghi nhận, không phải bằng chứng leakage.

## 3. Kế hoạch tám phase đã biến thành một notebook chạy trọn đường

`CONTEXT.md` dự kiến tám phase tách biệt: validation, Extra Trees, feature engineering, XGBoost, xử lý mất cân bằng, ensemble, phân tích lỗi và submission. Nhật ký trạng thái của file này chỉ đánh dấu Phase 1 hoàn thành. Thực tế triển khai đã rẽ sang một đường nhanh hơn: toàn bộ phần train–ensemble–calibration được gom vào [`ISE_music_genre_final_solution.ipynb`](./ISE_music_genre_final_solution.ipynb).

Bản đầu của notebook cuối được commit ở `5917e44` lúc 19:51, chỉ gần hai giờ sau commit khởi đầu. Nó có 18 cell, tiêu đề khi ấy là *“Từ 0.36 đến 0.417 — nhật ký đi tìm một submission đủ tin cậy”*, và đã chứa output của cả pipeline. Chính snapshot Git này khớp với CSV 0.402 còn trên filesystem tốt hơn các output Colab được chạy lại về sau.

Nói cách khác, repo không đi tuần tự qua đủ tám notebook như bản kế hoạch. Nó giữ triết lý của kế hoạch, nhưng triển khai bằng một all-in-one notebook để kịp có submission chạy lại được.

## 4. Không chọn model theo danh tiếng

Nhật ký screening trong notebook ghi lại các thử nghiệm trên một fold:

| Thử nghiệm được ghi lại | Macro F1 screening | Quyết định lúc đó |
|---|---:|---|
| Extra Trees, `leaf=1` | 0.36413 | Bỏ cấu hình |
| Extra Trees, `leaf=2` | 0.36434 | Giữ làm nguồn diversity |
| Extra Trees, class weight balanced | 0.36173 | Bỏ |
| XGBoost raw | 0.39328 | Giữ làm model chính |
| XGBoost raw + prior | 0.39864 | Giữ ý tưởng prior correction |
| XGBoost engineered + prior | 0.40003 | Giữ feature engineering |
| XGBoost class weight nhẹ | 0.39834 | Bỏ |
| LightGBM categorical + prior | 0.39386 | Giữ làm góc nhìn thứ ba |
| CatBoost CPU | ước tính hơn 3 giờ/lượt | Bỏ vì không phù hợp thời gian Colab |

Không phải mọi log screening thô còn tồn tại thành artifact độc lập, nên bảng này là nhật ký được kể lại trong notebook, không mạnh bằng các ma trận OOF 5-fold còn nguyên. Dù vậy, code cuối cho thấy các kết luận của screening đã thật sự đi vào pipeline:

- không dùng `class_weight="balanced"`;
- không dùng CatBoost;
- dùng XGBoost làm model đơn mạnh nhất;
- giữ LightGBM và Extra Trees dù điểm đơn thấp hơn;
- giữ xác suất của cả ba thay vì chỉ lấy nhãn cứng.

### XGBoost: model chính, nhưng được cho một cách nhìn phù hợp hơn

XGBoost dùng `multi:softprob`, không dùng `softmax`, vì pipeline còn cần đủ 112 xác suất để blend, sửa prior và calibrate threshold. Cấu hình chạy đầy đủ có tối đa 700 cây, depth 7, learning rate 0.06, regularization và early stopping.

Nó không nhận nguyên xi 15 cột. `key` được biểu diễn bằng one-hot, sin/cos trên vòng 12 nốt và circle of fifths; `key × mode` trở thành 24 trạng thái; time signature được one-hot. Các interaction như `danceability × energy`, `danceability × valence`, `acousticness × instrumentalness`, `tempo × danceability` và cờ `audio_missing` đưa vào những quan hệ âm học mà một split đơn lẻ khó thấy ngay. Kết quả là 15 cột thô trở thành 67 feature cho XGBoost.

Tất cả phép biến đổi này là phép tính theo từng dòng. Không có scaler, mean, quantile hay target encoding học từ test.

### LightGBM: không bắt chước XGBoost

LightGBM giữ đúng 15 cột, nhưng đánh dấu `explicit`, `key`, `mode` và `time_signature` là categorical với domain cố định. Ý đồ không phải tạo một bản XGBoost thứ hai, mà là cho ensemble một kiểu chia cây khác. Model có tối đa 1.200 vòng và early stopping, nhờ đó không cần đoán trước số cây tối ưu cho từng fold.

### Extra Trees: yếu hơn nhưng không vô dụng

Extra Trees dùng 15 feature thô, 120 cây, `min_samples_leaf=2`. Nó rẻ, ngẫu nhiên hơn và sai theo cách khác boosting. Repo không giữ nó vì điểm đơn cao; repo giữ nó vì xác suất bổ sung diversity. Đây cũng là lý do trọng số của nó không tỷ lệ tuyến tính với điểm standalone.

Trong snapshot 3-fold đầu tiên, mean Macro F1 của ba model là:

| Model | Mean Macro F1 3-fold |
|---|---:|
| Extra Trees | 0.365562 |
| LightGBM | 0.396678 |
| XGBoost | 0.403965 |

XGBoost đã vượt mục tiêu vận hành `0.40`, nhưng câu chuyện chưa dừng ở model đơn.

## 5. Ba giám khảo và một lớp quyết định dành riêng cho Macro F1

Ba ma trận xác suất được blend theo:

```text
0.40 × XGBoost + 0.25 × LightGBM + 0.35 × Extra Trees
```

XGBoost nhận trọng số lớn nhất vì mạnh nhất. LightGBM bổ sung boosting với categorical handling. Extra Trees nhận 0.35 vì diversity. Notebook mô tả đây là một vùng plateau, không phải một nghiệm nhiều chữ số được tối ưu đến kiệt quệ; repo hiện không còn một grid artifact đủ để tuyên bố bộ trọng số này là tối ưu duy nhất.

Sau blend là prior correction:

```text
adjusted_score[k] = probability[k] / class_prior[k] ** alpha
```

Với `alpha=0.35`, lớp hiếm được nâng vừa phải thay vì dùng inverse class weight mạnh ngay trong lúc train. Quan trọng hơn, prior của mỗi validation fold chỉ được tính từ phần train của fold đó. Ensemble cộng prior đạt mean 3-fold **0.413009**, cao hơn từng model đơn; global OOF là **0.413640**.

Nhưng prior chỉ sửa thiên lệch chung theo tần suất. Mỗi genre vẫn có một vùng precision–recall khác nhau. Pipeline vì thế học 112 threshold riêng, mỗi threshold chọn điểm F1 nhị phân tốt nhất của một lớp. Để không tự chấm trên chính dữ liệu đã học threshold, mỗi held-out fold được xử lý như sau:

1. học 112 threshold từ hai fold còn lại;
2. áp threshold lên fold chưa tham gia học;
3. dùng `gamma` để làm mềm hoặc tăng mức ảnh hưởng của threshold;
4. chọn gamma theo mean của ba held-out score.

Gamma `0.8` nằm trong một vùng khá phẳng quanh `0.7–0.9`. Nó đưa mean từ **0.413009 lên 0.416984**, tức `+0.003975`, và cả ba fold đều được chấm bằng threshold học ở nơi khác. Đây là thay đổi có bằng chứng mạnh nhất trong pipeline 3-fold.

Sau cùng, threshold được fit trên toàn bộ OOF, còn test chỉ đi qua công thức đã khóa. Ở lượt tạo file này, không có phân phối dự đoán test nào được dùng để chỉnh alpha, gamma hay trọng số.

## 6. Phong bì đầu tiên và cú tăng 0.399 → 0.402

Trước notebook hoàn chỉnh từng có một script `build_submission_from_scratch.py`. Script ấy không còn trong repo, và CSV nó tạo ra đã bị ghi đè vì dùng cùng tên `submission_final_macro_f1_ensemble.csv`. Những gì còn lại là bản phục dựng [`preliminary_0399_reconstruction.json`](./artifacts/leaderboard_loop/preliminary_0399_reconstruction.json), dựa trên log hội thoại và điểm Public người dùng báo lại:

- ensemble ba model, trọng số `0.40 / 0.25 / 0.35`;
- prior alpha `0.35`;
- chưa có class threshold;
- dự đoán 111/112 lớp, thiếu `indie` (ID 56);
- Public Macro F1 `0.399`.

Sau khi thêm cross-fold per-class threshold với gamma `0.8`, Public tăng lên **0.402**. File 0.402 còn hiện hữu là [`submission_final_macro_f1_ensemble.csv`](./submission_final_macro_f1_ensemble.csv). Nó có đúng 21.947 ID nhưng tự nhiên chỉ dự đoán 111/112 lớp, thiếu `house` (ID 53).

Đã có lúc người giải cân nhắc ép một dòng thành lớp còn thiếu. Ý tưởng đó bị bỏ vì “coverage repair” không kích hoạt trên các held-out fold, tức không có bằng chứng validation rằng sửa tay test sẽ tốt hơn. Lượt 5-fold sau đó tự dự đoán đủ 112 lớp mà Public vẫn đứng ở 0.402. Vì thế cú tăng 0.399 → 0.402 không đến từ việc làm bảng phân phối trông đủ lớp; nó đến từ việc dời decision boundary hợp lý hơn.

Trước khi ghi CSV, code còn khóa đúng hai cột `track_id,track_genre`, đủ 21.947 dòng, đúng thứ tự ID của test và sample submission, ID duy nhất, nhãn integer trong `0..111`. Một model tốt vẫn có thể nhận điểm 0 vì sai format; repo không để lỗi vận hành đó xảy ra.

## 7. Đêm 5-fold: tiến trình bị ngắt nhưng thí nghiệm không mất

Sau mốc 0.402, mục tiêu là cho mỗi model học 80% train thay vì 66,7%. [`leaderboard_round1_5fold.py`](./leaderboard_round1_5fold.py) được viết để lưu OOF và test probability sau từng model, từng fold. Ý đồ rất thực dụng: các vòng chỉnh decision layer sau đó có thể tái sử dụng xác suất, không phải train lại ba model chỉ để đổi một con số.

Dấu thời gian artifact cho thấy tiến trình đầu đã chạy xong các fold index 0–2 và tới LightGBM của fold index 3, tức fold thứ tư, thì bị gián đoạn. [`leaderboard_round1_5fold_resume.py`](./leaderboard_round1_5fold_resume.py) xuất hiện sau đó, kiểm tra cả file test probability lẫn tổng xác suất OOF để phân biệt checkpoint hoàn chỉnh với file dở. Nó chỉ train phần còn thiếu: XGBoost của fold thứ tư và cả ba model của fold thứ năm. Báo cáo [`training_summary_5fold_resumed.json`](./artifacts/leaderboard_loop/training_summary_5fold_resumed.json) ghi trạng thái `complete_after_resume` và runtime phần resume khoảng 273 giây.

Các artifact 5-fold hiện còn đủ sáu ma trận OOF/test. Chúng đều hữu hạn, có đúng shape `51_452 × 112` hoặc `21_947 × 112`, và tổng xác suất mỗi dòng gần 1. Điểm base model trong [`base_metrics_5fold_resumed.csv`](./artifacts/leaderboard_loop/base_metrics_5fold_resumed.csv) là:

| Model | Mean Macro F1 5-fold | Std |
|---|---:|---:|
| Extra Trees | 0.375793 | 0.009784 |
| LightGBM | 0.409597 | 0.009738 |
| XGBoost | 0.414589 | 0.006482 |

Việc checkpoint không trực tiếp làm điểm cao hơn, nhưng nó giữ được tính tái lập và cho phép vòng lặp tiếp tục sau sự cố thay vì bắt đầu lại hoặc trộn artifact của hai split khác nhau.

## 8. Năm lần nhích, mỗi lần chỉ đổi một thứ

Sau khi khóa OOF/test probability 5-fold, các script sau chỉ thao tác ở decision layer. Report của lượt 1–4 vẫn mang trạng thái `awaiting_score`; lượt kế tiếp mới ghi nhận Public của lượt trước. Report lượt 5 được cập nhật thành `scored` sau phản hồi cuối. Các file `leaderboard_history_after_round*.csv` còn giữ những snapshot trung gian ấy. Đây là dấu vết tốt cho thấy kết quả không được viết ngược vào một bảng duy nhất sau khi đã biết hết đáp án.

| Mốc | Thay đổi duy nhất | Cross-fold Macro F1 | Std | Dòng test đổi | Public F1 |
|---|---|---:|---:|---:|---:|
| Sơ bộ | 3-fold ensemble + prior, chưa threshold | 0.413009 | 0.003898 | Không còn CSV để đối chiếu | **0.399** |
| Gốc | Thêm per-class threshold, gamma 0.80 | 0.416984 | 0.002760 | Không còn bản sơ bộ | **0.402** |
| Lượt 1 | `n_splits: 3 → 5` | 0.425578 | 0.006953 | 2.803 (12,77%) | **0.402** |
| Lượt 2 | `gamma: 0.80 → 0.75` | 0.426203 | 0.006487 | 165 (0,75%) | **0.402** |
| Lượt 3 | `prior alpha: 0.35 → 0.30` | 0.426462 | 0.006087 | 347 (1,58%) | **0.403** |
| Lượt 4 | `prior alpha: 0.300 → 0.275` | 0.426591 | 0.006689 | 325 (1,48%) | **0.403** |
| Lượt 5 | Chuyển 0.025 weight LightGBM sang Extra Trees | **0.427082** | 0.007791 | 623 (2,84%) | **0.402** |

Hai mốc 3-fold và chuỗi 5-fold không được so tuyệt đối như cùng một thí nghiệm, vì số fold và lượng train mỗi model khác nhau. Từ lượt 1 đến lượt 5, fold, seed, feature và xác suất base model mới thật sự được giữ cố định.

### Lượt 1: nhiều dữ liệu train hơn không bảo đảm Public tăng

Năm fold làm cả ba base model và ensemble OOF tốt hơn. Ensemble calibrated lên `0.425578`, và 2.803 nhãn test thay đổi so với file gốc. Public vẫn là `0.402`. Đây không phải bằng chứng 5-fold vô ích; nó là bằng chứng OOF tốt hơn chưa chắc sửa đúng nhóm lỗi đang chi phối 51% Public.

### Lượt 2: gamma đang ở trên plateau

Đổi gamma `0.80 → 0.75` tăng cross-fold chỉ `0.000625` và đổi 165/21.947 dòng. Public vẫn làm tròn ở `0.402`. Tiếp tục dò `0.76`, `0.77`, `0.78` sẽ tạo cảm giác tối ưu nhưng rất dễ thành overfit vào vài chữ số OOF hoặc Public.

### Lượt 3: recipe cũ đã nâng lớp hiếm hơi quá tay

Giảm alpha `0.35 → 0.30` nghĩa là vẫn bù mất cân bằng, nhưng bớt khuếch đại prior của lớp hiếm. Chỉ 347 dòng test đổi nhãn; cross-fold tăng nhẹ và std giảm; Public lần đầu lên **0.403**. Đây là tín hiệu hợp lý rằng correction cũ hơi mạnh, chứ không phải bằng chứng nên xóa prior hoàn toàn.

### Lượt 4: đi thêm nửa bước và gặp trần

Alpha `0.275` cho mean nhỉnh hơn `0.000129`, nhưng std tăng từ `0.006087` lên `0.006689` và Public vẫn `0.403`. Mức chênh nhỏ hơn rất nhiều ngưỡng thận trọng `0.003` đã ghi trong `CONTEXT.md`. Đó là plateau, không phải chiến thắng mới.

### Lượt 5: OOF cao nhất lại là lựa chọn tệ hơn

Chuyển 0.025 trọng số từ LightGBM sang Extra Trees đưa mean lên **0.427082**, cao nhất toàn chuỗi. Nhưng std cũng lên **0.007791**, 623 dòng test bị đổi và Public rơi xuống `0.402`. Đây là phản ví dụ quan trọng nhất trong repo: chọn đúng dòng có mean OOF lớn nhất vẫn có thể là quyết định sai khi mức tăng chỉ `0.000490` và độ ổn định xấu đi.

## 9. Vì sao đáp án cuối quay lại lượt 3

Lượt 3 và lượt 4 cùng Public `0.403`. Lượt 4 có mean cao hơn một lượng rất nhỏ, nhưng lượt 3 có bốn ưu điểm:

1. std thấp hơn: `0.006087` so với `0.006689`;
2. alpha `0.30` là một giá trị tròn, dễ giải thích và tái lập;
3. nó xuất hiện sớm hơn, nên chịu ít vòng phản hồi Public hơn;
4. lượt 5 cho thấy đuổi theo cực đại OOF mỏng có thể làm Public giảm.

Vì Public chỉ chấm 51% test còn Private mới quyết định thứ hạng, lựa chọn bảo thủ là quay về [`submission_iter3_prior_alpha_0p30.csv`](./submission_iter3_prior_alpha_0p30.csv), không phải giữ file có OOF cao nhất.

Đó cũng là ý nghĩa thật của “đạt điểm cao” trong repo này: không phải tăng bằng mọi giá, mà là đạt vượt mục tiêu `0.40` bằng một pipeline có lý do, biết dừng ở vùng ổn định và không hy sinh 49% chưa nhìn thấy để săn thêm `0.001` trên Public.

## 10. Những gì repo dạy rõ hơn chính con số 0.403

**Validation là một phần của model.** Group split không tạo thêm feature, nhưng nó quyết định mọi kết luận sau có đáng tin hay không.

**Macro F1 cần một decision layer riêng.** Model học xác suất tốt chưa chắc argmax mặc định đã cân bằng precision–recall cho 112 lớp. Prior correction và per-class threshold tạo cú tăng Public rõ nhất mà không thay kiến trúc model.

**Diversity có giá trị, nhưng không phải giấy phép gán weight tùy ý.** Extra Trees yếu hơn vẫn giúp blend; đến khi tăng weight của nó thêm 0.025, OOF nhích nhưng variance và Public xấu đi.

**Checkpoint cũng là một lựa chọn ML.** Lưu OOF/test probability theo fold giúp phục hồi một lượt train bị ngắt, kiểm tra alignment và thử decision rule mới với chi phí thấp.

**Leaderboard là phép đo, không phải tập train thứ hai.** Chuỗi năm lượt có dùng Public làm tín hiệu thực nghiệm, nên nguy cơ leaderboard overfitting là có thật. Quyết định dừng và quay lại lượt 3 quan trọng không kém bất kỳ tuning nào.

## 11. Ghi chú kiểm chứng: đâu là sự kiện, đâu là lời kể lại

Để không biến hậu truyện thành huyền thoại, các giới hạn chứng cứ cần được giữ nguyên:

- Các shape dữ liệu, duplicate, fold audit, code model, ma trận probability, metric 5-fold và sáu CSV hiện hữu có thể kiểm tra trực tiếp trong workspace.
- Dữ liệu, artifact và CSV đều bị `.gitignore` loại khỏi Git. Chúng là bằng chứng filesystem hiện có, còn notebook/script và lịch sử chỉnh sửa mới là bằng chứng được version control bảo toàn.
- Các Public score `0.399–0.403` được ghi trong notebook và [`leaderboard_history_final.csv`](./artifacts/leaderboard_loop/leaderboard_history_final.csv). Repo không có ảnh chụp hay API export từ leaderboard, nên đây là điểm do người dùng báo và dự án ghi lại, không phải bằng chứng bên thứ ba.
- CSV 0.399 và script tạo nó không còn. Mốc này chỉ có reconstruction JSON, vì vậy độ chắc chắn thấp hơn các mốc còn file thật.
- `CONTEXT.md` là kế hoạch ban đầu, không phải nhật ký hoàn công: nó còn ghi LightGBM chưa được cài và chỉ đánh dấu Phase 1, trong khi notebook cuối đã cài/dùng LightGBM và hoàn tất submission.
- Bản notebook hiện tại đã được biên tập nhiều lần sau cuộc thi: từ 18 cell ở commit `5917e44` thành 77 cell ở `3cbb03d` ngày 25/8. Một lượt chạy Colab muộn nhúng output `0.406936 → 0.411997` và 112/112 predicted classes, không khớp snapshot 3-fold gốc `0.413009 → 0.416984` cũng như CSV 0.402 đang có. Câu chuyện này ưu tiên snapshot Git gốc, file CSV thật và artifact theo từng vòng; output muộn chỉ được xem là một lần tái chạy ở môi trường khác.
- Một cell hiện tại in “tổng số nhóm trùng lặp: 0”, nhưng biểu thức trong cell thực ra đếm nhóm **trùng feature nhưng khác nhãn**. Con số đúng là 0 nhóm nhãn mâu thuẫn; số nhóm feature trùng là 585, như notebook validation độc lập đã ghi.
- Không có Private score hoặc bảng xếp hạng cuối. Vì vậy không nên suy diễn `0.403 Public` thành một thứ hạng cụ thể.

## 12. Dấu vết thời gian còn lại

Các giờ của commit đến từ Git; các giờ tạo artifact/CSV là mtime hiện còn trên filesystem và có thể thay đổi nếu file được sao chép.

| Thời điểm | Dấu vết |
|---|---|
| 23/8, 17:55–18:06 | Artifact validation được tạo; commit `8003a77` thêm kế hoạch và notebook chống leakage |
| 23/8, 19:44–19:51 | CSV ensemble 3-fold 0.402 được ghi; commit `5917e44` lưu notebook lời giải 18 cell |
| 23/8, 19:58–20:14 | Train 5-fold; tiến trình gián đoạn và hoàn tất bằng script resume |
| 23/8, 20:15–20:28 | Lần lượt tạo năm CSV từ đổi số fold, gamma, alpha và blend weight |
| 23/8, 20:48 | Hoàn thiện reconstruction mốc 0.399 và lịch sử Public |
| 23/8, 23:19 | Commit `06f7c9f` đưa bảy script vòng leaderboard vào Git |
| 23–25/8 | Nhiều commit Colab biến notebook chạy thi thành tài liệu giảng giải 77 cell |

Nếu phải tóm câu chuyện trong một câu: **điểm 0.403 không đến từ việc đổi sang model phức tạp hơn, mà từ việc xây một validation không tự lừa mình, giữ ba sai số khác nhau, hiệu chỉnh quyết định đúng với Macro F1, rồi đủ tỉnh táo quay khỏi nghiệm OOF cao nhất khi bằng chứng ngoài mẫu không ủng hộ nó.**
