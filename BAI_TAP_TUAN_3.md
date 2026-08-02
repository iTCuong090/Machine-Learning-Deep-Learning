# Bài tập tuần 3 — Nền tảng NumPy và Machine Learning từ số 0

> Tài liệu này dành cho người chưa học Machine Learning (ML) hoặc Deep Learning (DL). Mục tiêu không phải chỉ là chạy được mã, mà là hiểu dữ liệu đi qua mô hình như thế nào, vì sao công thức đúng và cách phát hiện khi chương trình sai.

## Mục lục

0. [Pitching Bootcamp — cách học trên Kaggle](#0-pitching-bootcamp--cách-học-trên-kaggle)
1. [Bức tranh tổng thể](#1-bức-tranh-tổng-thể)
2. [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
3. [Mean, Variance và Standard Deviation](#3-mean-variance-và-standard-deviation)
4. [Dot product, matrix multiplication và cross product](#4-dot-product-matrix-multiplication-và-cross-product)
5. [Vectorized Affine Transformation](#5-vectorized-affine-transformation)
6. [Feature Standardization](#6-feature-standardization)
7. [Regression Metrics](#7-regression-metrics)
8. [Stable Sigmoid và Binary Cross-Entropy](#8-stable-sigmoid-và-binary-cross-entropy)
9. [Confusion Matrix và Classification Metrics](#9-confusion-matrix-và-classification-metrics)
10. [Lựa chọn Classification Threshold](#10-lựa-chọn-classification-threshold)
11. [Univariate Linear Regression bằng Gradient Descent](#11-univariate-linear-regression-bằng-gradient-descent)
12. [Multiple Linear Regression — Vectorized](#12-multiple-linear-regression--vectorized)
13. [Logistic Regression from Scratch](#13-logistic-regression-from-scratch)
14. [Housing Price Prediction trên Kaggle](#14-housing-price-prediction-trên-kaggle)
15. [Titanic trên Kaggle](#15-titanic-trên-kaggle)
16. [Checklist nộp bài](#16-checklist-nộp-bài)
17. [Thuật ngữ và tài liệu tham khảo](#17-thuật-ngữ-và-tài-liệu-tham-khảo)
18. [Thẻ phòng thủ cho 13 bài tập](#18-thẻ-phòng-thủ-cho-13-bài-tập)
19. [Ngân hàng câu hỏi xoáy và mock interview](#19-ngân-hàng-câu-hỏi-xoáy-và-mock-interview)
20. [Code drills và checklist trước pitching](#20-code-drills-và-checklist-trước-pitching)

---

## 0. Pitching Bootcamp — cách học trên Kaggle

Theo ngày hệ thống của tài liệu, buổi pitching là **Thứ Ba, 04/08/2026**. Thời gian ngắn nên mục tiêu không phải thuộc từng dòng code; mục tiêu là có thể **tái dựng ý tưởng, kiểm tra shape, giải thích lựa chọn và sửa lỗi trước mặt người hỏi**.

### 0.1 Thao tác notebook trên Kaggle

1. Vào Kaggle → **Code** → **New Notebook** → **File > Import Notebook**, chọn `BAI_TAP_TUAN_3.ipynb`.
2. Trong **Session options**, chọn CPU; 13 bài này không cần GPU/TPU.
3. Với House Prices/Titanic, bấm **Add Input**, tìm đúng competition và gắn dữ liệu. Notebook tự dò các đường dẫn `/kaggle/input/house-prices-advanced-regression-techniques` và `/kaggle/input/titanic`.
4. Chạy cell thiết lập đầu tiên, sau đó chạy từng cell bằng `Shift + Enter`. Không bấm Run all ngay trong lần học đầu.
5. Sau mỗi phiên, chọn **Save Version**. Đặt tên có ý nghĩa, ví dụ `v03-standardization-understood`, thay vì chỉ `version 3`.
6. Trước buổi pitching, restart session rồi **Run All** một lần. Một notebook chỉ chạy đúng nhờ biến còn sót từ lần chạy trước chưa được coi là tái lập được.

Nên tạo hai bản trên Kaggle:

- **Learning copy**: được phép viết note, thử sai, thêm print và làm hỏng code.
- **Pitch copy**: sạch, chạy từ đầu đến cuối, chỉ giữ output/đồ thị cần trình bày.

Kaggle session có thể hết hạn; file tạo trong vùng làm việc tạm thời không phải lúc nào cũng tồn tại sau khi session đóng. Hãy Save Version và tải submission/notebook quan trọng về máy.

### 0.2 Chu trình học sáu bước cho mỗi bài

Không học bằng cách liên tục bấm Run. Dùng vòng lặp sau:

```text
1. Recall     Đóng lời giải, nói lại mục tiêu và công thức.
2. Predict    Viết trước shape/output hoặc dự đoán loss/metric.
3. Run        Chạy cell và so với dự đoán.
4. Break      Cố tình tạo một lỗi: sai axis, sai shape, std=0, lr quá lớn...
5. Repair     Sửa lỗi và giải thích vì sao bản sửa đúng.
6. Teach-back Nói trong 60–90 giây như đang dạy một người khác.
```

Mỗi bài chỉ được đánh dấu “đã học” khi đạt cả bốn tầng:

| Mức | Bạn làm được gì? |
|---:|---|
| 0 | Chưa nhận ra khái niệm |
| 1 | Nhận ra khi nhìn lời giải |
| 2 | Giải thích được nhưng chưa tự code được |
| 3 | Tự code được từ chữ ký hàm và test |
| 4 | Trả lời được biến thể, trade-off và sửa code lỗi |

Mục tiêu trước pitching: tất cả bài đạt ít nhất mức `3`; các bài 3–11 đạt mức `4` vì đây là vùng dễ bị hỏi code/toán sâu.

### 0.3 Khung trả lời khi bị hỏi dồn

Dùng cấu trúc **K–S–C–K** trong 30–60 giây:

1. **Kết luận**: trả lời trực tiếp một câu.
2. **Shape/công thức**: nêu phép toán và shape quan trọng.
3. **Cơ chế**: giải thích vì sao, trade-off hoặc giả định.
4. **Kiểm chứng**: chỉ vào test, metric, đồ thị hoặc edge case trong notebook.

Ví dụ, câu “Vì sao phải standardize?”:

> Standardization đưa các feature về scale tương đương để gradient descent hội tụ ổn định hơn. Tôi fit `mean/std` theo `axis=0` trên train, rồi dùng cùng thống kê cho validation/test để tránh leakage. Feature có `std=0` được ép về 0. Bằng chứng là đồ thị cùng learning rate: bản scaled giảm loss, bản raw dao động hoặc phân kỳ.

Nếu chưa chắc, hãy nói rõ giả định thay vì đoán: “Nếu anh đang hỏi population variance thì NumPy dùng `ddof=0`; sample variance sẽ dùng `ddof=1`.” Cách này tốt hơn trả lời dài nhưng không đúng trọng tâm.

### 0.4 Lịch ôn tăng tốc đến Thứ Ba

Không học liên tục nhiều giờ. Dùng block 50 phút học + 10 phút nghỉ, mỗi block phải có sản phẩm quan sát được.

| Thời điểm | Nội dung | Sản phẩm bắt buộc |
|---|---|---|
| Chủ Nhật | Bài 1–5: NumPy, shape, affine, standardization, regression metrics | Tự code lại 5 hàm và nói 1 phút/bài |
| Chủ Nhật tối | Bài 6–8: sigmoid/BCE, confusion matrix, threshold | Stress test `±1000`, bảng TP/TN/FP/FN, giải thích imbalance |
| Thứ Hai sáng | Bài 9–11: ba mô hình gradient descent | Tự viết gradient, vẽ loss, giải thích ba learning rates |
| Thứ Hai chiều | Bài 12–13: House Prices và Titanic | Chạy pipeline, lưu local metric/submission thật |
| Thứ Hai tối | Mock pitching 2 vòng | Vòng 1 mở notebook; vòng 2 không nhìn tài liệu; ghi câu trả lời yếu |
| Thứ Ba trước buổi | Ôn thẻ phòng thủ và 10 câu ngẫu nhiên | Không học chủ đề mới; kiểm tra notebook Run All |

Ưu tiên theo rủi ro nếu thiếu thời gian: **shape/broadcasting → leakage → numerical stability → gradient → metric/threshold → Kaggle workflow**.

### 0.5 Nhật ký học ngay trong notebook

Sau mỗi bài, thêm một Markdown cell theo mẫu:

```markdown
#### Nhật ký Bài ...
- Tôi có thể giải thích mà không nhìn: ...
- Shape/công thức cốt lõi: ...
- Lỗi tôi vừa tạo và sửa: ...
- Câu tôi vẫn trả lời yếu: ...
- Mức hiện tại (0–4): ...
```

Không ghi “đã hiểu”. Hãy ghi bằng chứng cụ thể như “tự viết được gradient `X.T @ error / m` và giải thích shape `(n,m)@(m,)->(n,)`”.

---

## 1. Bức tranh tổng thể

### 1.1 Machine Learning là gì?

Trong lập trình thông thường, ta viết trực tiếp các quy tắc biến đầu vào thành đầu ra. Trong supervised machine learning (học có giám sát), ta đưa cho máy nhiều cặp:

```text
đặc trưng X  ──>  mô hình có tham số  ──>  dự đoán ŷ
                                          │
nhãn thật y ───────── so sánh bằng loss ──┘
                         │
                  cập nhật tham số
```

- **Sample/observation**: một mẫu dữ liệu, ví dụ một ngôi nhà hoặc một hành khách.
- **Feature**: thuộc tính dùng để dự đoán, ví dụ diện tích hoặc tuổi.
- **Target/label**: giá trị cần dự đoán, ví dụ giá nhà hoặc sống sót.
- **Model**: hàm có tham số học từ dữ liệu.
- **Training**: quá trình điều chỉnh tham số để loss giảm.
- **Inference/prediction**: dùng tham số đã học để dự đoán dữ liệu mới.

Hai loại bài toán trong tuần này:

| Loại | Target | Ví dụ | Đầu ra |
|---|---|---|---|
| Regression | Giá trị liên tục | Giá nhà | `183500.0` |
| Binary classification | Một trong hai lớp | Sống sót/không | xác suất rồi nhãn `0`/`1` |

### 1.2 Quy ước shape

Tài liệu dùng quy ước:

- `m`: số samples.
- `n`: số features đầu vào.
- `k`: số đầu ra.
- `X.shape == (m, n)`: mỗi hàng là một sample, mỗi cột là một feature.
- `W.shape == (n, k)`: ma trận trọng số.
- `b.shape == (k,)`: bias.
- `y.shape == (m,)`: target một chiều.

Ví dụ 3 ngôi nhà, mỗi nhà có 2 đặc trưng:

```python
X = np.array([
    [50.0, 2.0],   # diện tích, số phòng của nhà 1
    [80.0, 3.0],
    [120.0, 4.0],
])
# X.shape là (3, 2)
```

### 1.3 Vectorization là gì?

Vectorization là biểu diễn phép tính trên cả mảng thay vì tự viết vòng lặp qua từng sample/feature.

```python
# Không vectorized
result = []
for row in X:
    result.append(2 * row + 1)

# Vectorized
result = 2 * X + 1
```

NumPy thực hiện phần lặp trong mã đã biên dịch, nên cách vectorized thường ngắn hơn, ít lỗi shape hơn và nhanh hơn rõ rệt trên dữ liệu lớn. “Không dùng vòng lặp” trong bài tập thường có nghĩa là không viết `for` qua từng sample hoặc feature; vòng lặp qua epochs hoặc danh sách threshold vẫn hợp lý.

---

## 2. Chuẩn bị môi trường

Python 3.10+ là đủ. Với 11 bài đầu, chỉ cần NumPy và Matplotlib; hai bài Kaggle dùng thêm pandas và scikit-learn để xây baseline thực tế.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib pandas scikit-learn jupyter kaggle
```

Có thể học trong file `.py` hoặc Jupyter Notebook. Đầu mỗi notebook:

```python
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
```

Ba thao tác kiểm tra cần tạo thành thói quen:

```python
print(X.shape)             # kích thước
print(X.dtype)             # kiểu dữ liệu
print(np.isfinite(X).all()) # có NaN/inf không?
```

> Không đặt tên file là `numpy.py`, `pandas.py`, `sklearn.py` hoặc `matplotlib.py`, vì file của bạn sẽ che khuất thư viện thật khi `import`.

---

## 3. Mean, Variance và Standard Deviation

Đề chính thức: [Mean, Var, and Std — HackerRank](https://www.hackerrank.com/challenges/np-mean-var-and-std/problem)

### 3.1 Trực giác và công thức

Với dãy số \(x_1, x_2, \ldots, x_m\):

**Mean (trung bình)** mô tả tâm của dữ liệu:

$$
\mu = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

**Variance (phương sai)** đo độ phân tán bình phương quanh mean:

$$
\mathrm{Var}(x) = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu)^2
$$

**Standard deviation (độ lệch chuẩn)** đưa độ phân tán về cùng đơn vị với dữ liệu:

$$
\sigma = \sqrt{\mathrm{Var}(x)}
$$

Ví dụ `[1, 3, 5]` có mean `3`. Các độ lệch là `[-2, 0, 2]`; bình phương là `[4, 0, 4]`; variance là `8/3`; std là `sqrt(8/3)`.

NumPy mặc định dùng **population variance**, tức chia cho `m` (`ddof=0`). Một số công thức thống kê mẫu chia cho `m - 1` (`ddof=1`); HackerRank cần mặc định của NumPy.

### 3.2 Hiểu `axis`

Cho:

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])  # shape (2, 3)
```

Hãy hiểu `axis` là **chiều bị thu gọn**:

| Lệnh | Chiều bị thu gọn | Kết quả | Shape |
|---|---|---|---|
| `A.mean(axis=0)` | hàng | mean theo từng cột: `[2.5, 3.5, 4.5]` | `(3,)` |
| `A.mean(axis=1)` | cột | mean theo từng hàng: `[2., 5.]` | `(2,)` |
| `A.mean()` | mọi chiều | `3.5` | scalar |

Trong ML, vì hàng là samples và cột là features, `X.mean(axis=0)` tính mean riêng cho từng feature.

### 3.3 Lời giải HackerRank

Đề yêu cầu mean theo `axis=1`, variance theo `axis=0`, và std của toàn bộ mảng.

```python
import numpy as np

n, m = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(n)])

print(np.mean(a, axis=1))
print(np.var(a, axis=0))
print(round(np.std(a), 11))
```

`round(..., 11)` tránh khác biệt định dạng số thực trên test của HackerRank. Vòng lặp ở đây chỉ dùng để **đọc input**, không phải cài phép toán ML.

### 3.4 Tự kiểm tra

```python
A = np.array([[1, 2], [3, 4]])

assert np.allclose(np.mean(A, axis=1), [1.5, 3.5])
assert np.allclose(np.var(A, axis=0), [1.0, 1.0])
assert np.isclose(np.std(A), np.sqrt(1.25))
```

Lỗi thường gặp:

- Đảo `axis=0` và `axis=1`.
- Dùng `ddof=1` trong khi đề cần `ddof=0`.
- Dữ liệu là số nguyên không gây vấn đề cho `np.mean`, nhưng khi tự tạo output bằng `np.empty_like` có thể vô tình làm mất phần thập phân.

---

## 4. Dot product, matrix multiplication và cross product

Đề chính thức: [Dot and Cross — HackerRank](https://www.hackerrank.com/challenges/np-dot-and-cross/problem)

### 4.1 Ba phép toán dễ bị nhầm

**Element-wise multiplication** nhân từng vị trí tương ứng:

```python
A * B
```

Hai mảng phải cùng shape hoặc broadcast được.

**Dot product của hai vector** tạo một scalar:

$$
a \cdot b = \sum_i a_i b_i
$$

```python
np.dot(np.array([1, 2]), np.array([3, 4]))  # 11
```

**Matrix multiplication** lấy tích vô hướng giữa từng hàng của `A` và từng cột của `B`:

$$
C_{ij}=\sum_{r=1}^{n}A_{ir}B_{rj}
$$

Nếu `A.shape == (m, n)` và `B.shape == (n, k)`, thì `A @ B` có shape `(m, k)`. Hai kích thước “ở giữa” phải bằng nhau:

```text
(m, n) @ (n, k) -> (m, k)
     └────┘
      phải khớp
```

**Cross product** là một phép khác, thường dành cho vector 3 chiều và cho ra vector vuông góc với hai vector đầu vào. Tên bài HackerRank là “Dot and Cross”, nhưng task cụ thể chỉ yêu cầu matrix product.

### 4.2 Ví dụ

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)
# [[ 5 12]
#  [21 32]]

print(A @ B)
# [[19 22]
#  [43 50]]
```

Phần tử đầu của `A @ B` là `1*5 + 2*7 = 19`, không phải `1*5 = 5`.

### 4.3 Lời giải HackerRank

```python
import numpy as np

n = int(input())
A = np.array([list(map(int, input().split())) for _ in range(n)])
B = np.array([list(map(int, input().split())) for _ in range(n)])

print(A @ B)
```

`np.dot(A, B)` cũng đúng với hai ma trận 2-D, nhưng toán tử `@` thể hiện ý định nhân ma trận rõ nhất.

### 4.4 Tự kiểm tra shape trước khi nhân

```python
assert A.ndim == 2 and B.ndim == 2
assert A.shape[1] == B.shape[0]
C = A @ B
assert C.shape == (A.shape[0], B.shape[1])
```

---

## 5. Vectorized Affine Transformation

### 5.1 Affine transformation là gì?

Một lớp tuyến tính trong ML/DL thường tính:

$$
Z = XW + b
$$

Tên “affine” chính xác hơn “linear” vì có thêm `b`. Với:

- `X`: `(m, n)`
- `W`: `(n, k)`
- `X @ W`: `(m, k)`
- `b`: `(k,)`
- `Z`: `(m, k)`

`b` được cộng vào **mọi hàng** nhờ broadcasting:

```text
X @ W                   b                  Z
(m, k)                (k,)               (m, k)
[[z11, z12],      + [b1, b2]      = [[z11+b1, z12+b2],
 [z21, z22],                            [z21+b1, z22+b2],
 ...]                                     ...]
```

### 5.2 Cài đặt

```python
import numpy as np

def affine(X, W, b):
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    if X.ndim != 2 or W.ndim != 2 or b.ndim != 1:
        raise ValueError("X, W, b phải lần lượt có ndim 2, 2, 1")
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"Không thể nhân shape {X.shape} với {W.shape}")
    if W.shape[1] != b.shape[0]:
        raise ValueError(f"W tạo {W.shape[1]} outputs nhưng b có {b.shape[0]} phần tử")

    return X @ W + b
```

### 5.3 Kiểm thử `k=1` và `k>1`

```python
X = np.array([[1, 2], [3, 4], [5, 6]])

# k = 1: lưu ý W vẫn là (2, 1), b là (1,)
W1 = np.array([[10], [1]])
b1 = np.array([0.5])
Z1 = affine(X, W1, b1)
assert Z1.shape == (3, 1)
assert np.allclose(Z1[:, 0], [12.5, 34.5, 56.5])

# k = 2
W2 = np.array([[1, 0], [0, 1]])
b2 = np.array([10, 20])
Z2 = affine(X, W2, b2)
assert Z2.shape == (3, 2)
assert np.allclose(Z2, [[11, 22], [13, 24], [15, 26]])
```

Lỗi shape nguy hiểm nhất là dùng `b.shape == (k, 1)`. Shape đó không broadcast theo hàng với `(m, k)` như mong muốn. Hãy dùng `(k,)` hoặc, khi cần thể hiện rõ hàng, `(1, k)`.

---

## 6. Feature Standardization

### 6.1 Vì sao cần chuẩn hóa?

Giả sử hai feature là diện tích khoảng `50–300` m² và số phòng khoảng `1–6`. Gradient theo feature diện tích có thể lớn hơn hẳn chỉ vì đơn vị đo. Gradient descent phải đi trong một mặt loss kéo dài, dẫn đến dao động hoặc cần learning rate rất nhỏ.

Standardization biến mỗi feature thành:

$$
x' = \frac{x - \mu}{\sigma}
$$

Sau biến đổi, feature không hằng thường có mean gần `0`, std gần `1`. Đây không phải là “biến mọi giá trị về khoảng 0–1”; đó là min-max scaling, một phép khác.

### 6.2 Cài đặt

```python
import numpy as np

def fit_standardizer(X_train):
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train phải có shape (m, n)")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    return mean, std


def transform(X, mean, std):
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    if X.ndim != 2 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Shape mong đợi: X (m,n), mean (n,), std (n,)")
    if X.shape[1] != mean.size or mean.shape != std.shape:
        raise ValueError("Số features của X, mean và std phải khớp")

    zero_std = std == 0
    safe_std = np.where(zero_std, 1.0, std)
    standardized = (X - mean) / safe_std
    standardized[:, zero_std] = 0.0
    return standardized
```

Với feature hằng, ta không học được một scale có ý nghĩa. Mã vừa thay mẫu số `0` bằng `1` để tránh phép chia cho 0, vừa đặt toàn bộ cột tương ứng về `0.0`; do đó yêu cầu vẫn đúng ngay cả khi validation/test chứa một giá trị khác thường ở cột này.

### 6.3 Tự kiểm tra

```python
X_train = np.array([
    [1, 10, 7],
    [2, 20, 7],
    [3, 30, 7],
], dtype=float)
X_valid = np.array([[4, 40, 999]], dtype=float)

mean, std = fit_standardizer(X_train)
X_train_s = transform(X_train, mean, std)
X_valid_s = transform(X_valid, mean, std)

assert np.allclose(X_train_s.mean(axis=0), [0, 0, 0])
assert np.allclose(X_train_s[:, :2].std(axis=0), [1, 1])
assert np.allclose(X_train_s[:, 2], 0)
assert np.allclose(X_valid_s[:, 2], 0)
assert X_valid_s.shape == X_valid.shape
```

### 6.4 Vì sao chỉ fit trên train?

Không được tính mean/std trên toàn bộ dataset vì khi đó thông tin phân phối của validation/test đã đi vào quá trình huấn luyện; đây là **data leakage**. Kết quả đánh giá sẽ lạc quan hơn khả năng thật khi mô hình gặp dữ liệu hoàn toàn mới. Ta `fit` mean/std trên train một lần, rồi dùng chính các thống kê đó để `transform` train, validation và test.

Quy trình đúng:

```python
mean, std = fit_standardizer(X_train)
X_train_s = transform(X_train, mean, std)
X_valid_s = transform(X_valid, mean, std)  # không fit lại
X_test_s  = transform(X_test, mean, std)    # không fit lại
```

---

## 7. Regression Metrics

Cho sai số của sample `i` là \(e_i = y_i - \hat y_i\).

### 7.1 Ý nghĩa các metric

**MAE — Mean Absolute Error**:

$$
\mathrm{MAE}=\frac{1}{m}\sum_i|y_i-\hat y_i|
$$

Dễ diễn giải, cùng đơn vị với target và ít nhạy với outlier hơn MSE.

**MSE — Mean Squared Error**:

$$
\mathrm{MSE}=\frac{1}{m}\sum_i(y_i-\hat y_i)^2
$$

Phạt lỗi lớn mạnh hơn vì bình phương; đơn vị bị bình phương.

**RMSE — Root Mean Squared Error**:

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
$$

Cùng đơn vị với target nhưng vẫn nhạy với lỗi lớn.

**R² — coefficient of determination**:

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
$$

- `1`: dự đoán hoàn hảo.
- `0`: ngang với baseline luôn đoán mean của `y_true`.
- `< 0`: tệ hơn baseline đó. R² âm là hợp lệ, không phải lỗi chương trình.

### 7.2 Cài đặt từ đầu

```python
import numpy as np

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Dữ liệu không được rỗng")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải cùng shape")

    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return float(mae), float(mse), float(rmse), float(r2)
```

Khi `y_true` là hằng số, R² theo công thức có mẫu số 0. Ở đây ta quy ước dự đoán hoàn hảo trả `1.0`, còn dự đoán không hoàn hảo trả `0.0`; cần ghi rõ quy ước này trong báo cáo.

### 7.3 Tự kiểm tra bằng tay

```python
y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 5])

mae, mse, rmse, r2 = regression_metrics(y_true, y_pred)
assert np.isclose(mae, 2 / 3)
assert np.isclose(mse, 4 / 3)
assert np.isclose(rmse, np.sqrt(4 / 3))
assert np.isclose(r2, -1.0)
```

Không đánh giá trên train rồi coi đó là khả năng tổng quát hóa. Metric quan trọng phải được tính trên validation/test chưa dùng để học tham số.

---

## 8. Stable Sigmoid và Binary Cross-Entropy

### 8.1 Sigmoid

Logistic regression tạo ra logit \(z\), là một số bất kỳ, rồi sigmoid ánh xạ nó vào khoảng `(0, 1)`:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

- `z = 0` → xác suất `0.5`.
- `z` rất dương → gần `1`.
- `z` rất âm → gần `0`.

Công thức trực tiếp `1 / (1 + np.exp(-z))` làm `np.exp(1000)` overflow khi `z=-1000`. Ta tách hai miền nhưng vẫn dùng các phép toán vectorized:

```python
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    positive = z >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)

    return out.item() if out.ndim == 0 else out
```

Ở nhánh âm ta dùng \(e^z/(1+e^z)\), tương đương về toán học nhưng không cần tính \(e^{-z}\) rất lớn. Không nên viết hai công thức bên trong `np.where`, vì NumPy thường vẫn tính cả hai nhánh trước khi chọn và có thể phát cảnh báo overflow.

### 8.2 Binary Cross-Entropy (BCE)

Với nhãn \(y_i\in\{0,1\}\) và xác suất \(p_i\):

$$
\mathrm{BCE}=-\frac{1}{m}\sum_i[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

Nếu dự đoán đúng và tự tin, loss nhỏ. Nếu dự đoán sai nhưng rất tự tin, loss rất lớn. Vì `log(0) = -inf`, phải chặn xác suất khỏi chính xác `0` và `1`.

```python
def binary_cross_entropy(y, probability):
    y = np.asarray(y, dtype=float).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)

    if y.size == 0 or y.shape != probability.shape:
        raise ValueError("y và probability phải cùng shape và không rỗng")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BCE nhị phân yêu cầu y chỉ gồm 0 và 1")
    if not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("Xác suất phải nằm trong [0, 1]")

    eps = np.finfo(float).eps
    p = np.clip(probability, eps, 1.0 - eps)
    loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(loss)
```

`np.log1p(-p)` tính `log(1-p)` chính xác hơn khi `p` nhỏ. Nếu API cho phép truyền thẳng **logits**, công thức `mean(logaddexp(0, z) - y*z)` còn ổn định hơn; nhưng bài này yêu cầu input là probability.

### 8.3 Stress test

```python
z = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
p = sigmoid(z)
loss = binary_cross_entropy(np.array([0, 0, 1, 1, 1]), p)

assert np.isfinite(p).all()
assert np.isfinite(loss)
assert np.all((p >= 0) & (p <= 1))
assert np.isclose(p[2], 0.5)
```

---

## 9. Confusion Matrix và Classification Metrics

Giả sử lớp dương (positive) là `1`, lớp âm (negative) là `0`:

| | Dự đoán 1 | Dự đoán 0 |
|---|---:|---:|
| Thật 1 | TP — đúng dương | FN — bỏ sót dương |
| Thật 0 | FP — báo động giả | TN — đúng âm |

Các metric:

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

Trong những mẫu được dự đoán dương, bao nhiêu mẫu thật sự dương?

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

Trong những mẫu thật sự dương, tìm được bao nhiêu?

$$
F1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

### 9.1 Cài đặt

```python
import numpy as np

def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải cùng shape và không rỗng")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true chỉ được chứa 0 và 1")
    if not np.all((y_pred == 0) | (y_pred == 1)):
        raise ValueError("y_pred chỉ được chứa 0 và 1")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return tp, tn, fp, fn, float(accuracy), float(precision), float(recall), float(f1)
```

### 9.2 Tự kiểm tra

```python
y_true = np.array([1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0])

result = classification_metrics(y_true, y_pred)
tp, tn, fp, fn, accuracy, precision, recall, f1 = result

assert (tp, tn, fp, fn) == (2, 2, 1, 1)
assert np.isclose(accuracy, 4 / 6)
assert np.isclose(precision, 2 / 3)
assert np.isclose(recall, 2 / 3)
assert np.isclose(f1, 2 / 3)

# Không có dự đoán positive: precision phải là 0.0, không NaN
assert classification_metrics([0, 1], [0, 0])[5] == 0.0
```

---

## 10. Lựa chọn Classification Threshold

Mô hình classification thường trả xác suất. Threshold biến xác suất thành nhãn:

$$
\hat y = \begin{cases}
1 & p \ge t\\
0 & p < t
\end{cases}
$$

### 10.1 Cài đặt

```python
def find_best_threshold(y_true, probabilities):
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if y_true.size == 0 or y_true.shape != probabilities.shape:
        raise ValueError("Hai input phải cùng shape và không rỗng")
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("probabilities phải nằm trong [0, 1]")

    thresholds = np.arange(5, 96, 5) / 100.0
    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        f1 = classification_metrics(y_true, y_pred)[-1]
        # Chỉ thay khi lớn hơn, nên nếu hòa sẽ giữ threshold nhỏ hơn đã gặp trước.
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
```

Vòng lặp này đi qua đúng 19 thresholds, không đi qua samples hoặc features, nên hoàn toàn hợp lý. `np.arange(5, 96, 5) / 100` tránh các bất ngờ tích lũy số thực của `np.arange(0.05, 1.0, 0.05)`.

### 10.2 Vì sao `0.5` không luôn tốt nhất?

`0.5` chỉ là quy tắc mặc định. Threshold tốt phụ thuộc vào tỷ lệ lớp, độ hiệu chỉnh xác suất, metric mục tiêu và chi phí sai lầm. Trong sàng lọc bệnh, bỏ sót người bệnh (FN) có thể đắt hơn báo động giả (FP), nên ta có thể hạ threshold để tăng recall; trong lọc spam quan trọng, ta có thể tăng threshold để tránh xóa nhầm thư thật.

Threshold phải được chọn trên **validation set**, không chọn trên test set. Nếu tối ưu trên test, test đã tham gia quyết định mô hình và không còn là đánh giá khách quan.

### 10.3 Vì sao accuracy gây hiểu nhầm khi mất cân bằng?

Giả sử 1.000 giao dịch chỉ có 10 gian lận. Mô hình luôn đoán “không gian lận” có accuracy `990/1000 = 99%`, nhưng `recall = 0%`: nó không tìm được ca gian lận nào. Vì vậy cần xem confusion matrix, precision, recall, F1 và chọn metric theo chi phí thực tế.

### 10.4 Tự kiểm tra tie-break

```python
y = np.array([0, 1])
p = np.array([0.01, 0.99])

# Mọi threshold từ 0.05 đến 0.95 đều hoàn hảo; phải chọn 0.05.
assert np.isclose(find_best_threshold(y, p), 0.05)
```

---

## 11. Univariate Linear Regression bằng Gradient Descent

“Univariate” nghĩa là chỉ có một feature `x`. Mô hình:

$$
\hat y^{(i)} = wx^{(i)} + b
$$

- `w` (weight/slope): `x` tăng 1 đơn vị thì dự đoán thay đổi bao nhiêu.
- `b` (bias/intercept): dự đoán khi `x = 0`.

Ví dụ `x` là diện tích và `y` là giá nhà. Mô hình tìm một đường thẳng gần các điểm dữ liệu nhất.

### 11.1 Cost function

Theo đề:

$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2
$$

Hệ số `1/2` giúp đạo hàm của bình phương gọn hơn; nó không thay đổi vị trí cực tiểu. Với \(e_i=\hat y_i-y_i\), gradient là:

$$
\frac{\partial J}{\partial w}=\frac{1}{m}\sum_i e_ix_i
$$

$$
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_i e_i
$$

Gradient chỉ hướng tăng nhanh nhất của loss. Vì muốn giảm loss, ta đi ngược lại:

$$
w \leftarrow w-\alpha\frac{\partial J}{\partial w}, \qquad
b \leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

`α` là learning rate; `epoch` là một lần dùng toàn bộ training set để cập nhật tham số.

### 11.2 Cài đặt

```python
import numpy as np

def fit_linear_regression_1d(x, y, learning_rate, epochs):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size == 0 or x.shape != y.shape:
        raise ValueError("x và y phải cùng shape và không rỗng")
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate và epochs phải dương")

    m = x.size
    w = 0.0
    b = 0.0

    def cost():
        error = w * x + b - y
        return np.mean(error ** 2) / 2.0

    # Phần tử đầu là loss trước khi cập nhật; sau đó mỗi epoch thêm một loss.
    cost_history = [float(cost())]

    for _ in range(epochs):
        y_pred = w * x + b
        error = y_pred - y

        dw = np.dot(x, error) / m
        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db
        cost_history.append(float(cost()))

    return float(w), float(b), np.asarray(cost_history)
```

Vòng lặp qua `epochs` là cần thiết vì mỗi cập nhật phụ thuộc vào tham số của bước trước. Bên trong mỗi epoch, dự đoán và gradient cho tất cả samples đều được vectorized.

### 11.3 Thí nghiệm ba learning rate

Dùng dữ liệu có `x` đã chuẩn hóa để ảnh hưởng của learning rate dễ quan sát:

```python
rng = np.random.default_rng(42)
x_raw = np.linspace(0, 10, 100)
y = 3.0 * x_raw + 2.0 + rng.normal(0, 1, size=x_raw.size)
x = (x_raw - x_raw.mean()) / x_raw.std()

histories = {}
for lr in [0.001, 0.1, 2.1]:
    w, b, history = fit_linear_regression_1d(
        x, y, learning_rate=lr, epochs=200
    )
    histories[lr] = history
    print(f"lr={lr:<5} w={w:>10.4f} b={b:>10.4f} "
          f"first={history[0]:.4g} last={history[-1]:.4g}")

plt.figure(figsize=(8, 5))
for lr, history in histories.items():
    plt.plot(history, label=f"learning_rate={lr}")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Cost J (log scale)")
plt.title("Ảnh hưởng của learning rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Nhận xét dự kiến trên dữ liệu này:

- `0.001`: loss giảm đều nhưng chậm; cần nhiều epochs.
- `0.1`: hội tụ nhanh và ổn định.
- `2.1`: thường dao động với biên độ tăng và phân kỳ. Loss có thể lớn đến mức `inf`; khi đó đồ thị/log và cảnh báo NumPy chính là bằng chứng phân kỳ.

Không có ba learning rate “đúng cho mọi dataset”. Scale của `x` thay đổi thì khoảng learning rate ổn định cũng đổi; đó là một lý do cần standardization.

### 11.4 Vẽ đường hồi quy và kiểm tra điều kiện đề

```python
w, b, cost_history = fit_linear_regression_1d(x, y, 0.1, 200)
assert np.isfinite(cost_history).all()
assert cost_history[-1] < cost_history[0]

plt.scatter(x, y, s=18, alpha=0.7, label="Dữ liệu")
plt.plot(x, w * x + b, color="red", label="Đường dự đoán")
plt.xlabel("x đã standardize")
plt.ylabel("y")
plt.legend()
plt.show()
```

Nếu loss tăng:

1. kiểm tra dấu cập nhật có phải `-=` không;
2. kiểm tra gradient có chia `m` không;
3. giảm learning rate;
4. standardize `x`;
5. kiểm tra dữ liệu có NaN/inf không.

---

## 12. Multiple Linear Regression — Vectorized

Với nhiều features, mô hình trở thành:

$$
\hat y=Xw+b
$$

Với `X: (m,n)`, `w: (n,)`, `b: scalar`, NumPy broadcast `b` và trả `y_pred: (m,)`.

Cost vẫn là:

$$
J(w,b)=\frac{1}{2m}\|Xw+b-y\|_2^2
$$

Gradient dạng vectorized:

$$
\nabla_w J=\frac{1}{m}X^T(\hat y-y), \qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_i(\hat y_i-y_i)
$$

Kiểm tra shape:

```text
X.T       @ error  -> dw
(n, m)    @ (m,)   -> (n,)
```

### 12.1 Class hoàn chỉnh

Class dưới đây có tùy chọn standardization để ta so sánh công bằng. Mean/std chỉ được fit trong `fit` từ `X` train.

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000, standardize=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.standardize = standardize

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X phải là (m,n), y phải là (m,) và cùng số samples")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate và epochs phải dương")

        self.n_features_in_ = X.shape[1]
        if self.standardize:
            self.mean_, self.std_ = fit_standardizer(X)
            X_work = transform(X, self.mean_, self.std_)
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.ones(X.shape[1])
            X_work = X

        m, n = X_work.shape
        self.w_ = np.zeros(n, dtype=float)
        self.b_ = 0.0
        self.cost_history_ = []

        # Lưu loss ban đầu.
        initial_error = X_work @ self.w_ + self.b_ - y
        self.cost_history_.append(float(np.mean(initial_error ** 2) / 2.0))

        for _ in range(self.epochs):
            error = X_work @ self.w_ + self.b_ - y
            dw = X_work.T @ error / m
            db = np.mean(error)

            self.w_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db

            new_error = X_work @ self.w_ + self.b_ - y
            self.cost_history_.append(float(np.mean(new_error ** 2) / 2.0))

        return self

    def predict(self, X):
        if not hasattr(self, "w_"):
            raise RuntimeError("Phải gọi fit trước predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X có số features không đúng")

        X_work = (transform(X, self.mean_, self.std_)
                  if self.standardize else X)
        return X_work @ self.w_ + self.b_
```

Không có vòng lặp qua samples hoặc features. Vòng lặp qua epochs vẫn cần thiết.

### 12.2 Chia train/validation đúng cách

Ta có thể tự tạo index để chưa phụ thuộc scikit-learn:

```python
rng = np.random.default_rng(7)
m = 500
X = np.column_stack([
    rng.normal(100_000, 20_000, m),  # feature scale rất lớn
    rng.normal(5, 1, m),             # feature scale nhỏ
    rng.normal(0, 0.01, m),          # feature scale rất nhỏ
])
y = X @ np.array([0.02, 4.0, -100.0]) + 10 + rng.normal(0, 20, m)

indices = rng.permutation(m)
split = int(0.8 * m)
train_idx, valid_idx = indices[:split], indices[split:]
X_train, X_valid = X[train_idx], X[valid_idx]
y_train, y_valid = y[train_idx], y[valid_idx]
```

Tách dữ liệu **trước** khi standardize. Class sẽ chỉ học mean/std từ `X_train`.

### 12.3 So sánh có và không standardization

```python
raw_model = LinearRegressionGD(
    learning_rate=0.01, epochs=300, standardize=False
).fit(X_train, y_train)

scaled_model = LinearRegressionGD(
    learning_rate=0.01, epochs=300, standardize=True
).fit(X_train, y_train)

plt.figure(figsize=(8, 5))
plt.plot(raw_model.cost_history_, label="Không standardize")
plt.plot(scaled_model.cost_history_, label="Có standardize")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Với cùng learning rate, phiên bản raw có thể tăng thành `inf` ngay vì feature đầu cỡ `100000`, trong khi phiên bản standardized thường giảm ổn định. Nếu muốn raw model hội tụ, phải dùng learning rate rất nhỏ, khiến các hướng có scale nhỏ học cực chậm. Trong báo cáo, hãy chụp đồ thị, ghi learning rate/epochs và mô tả hiện tượng quan sát được thay vì chỉ viết “standardization tốt hơn”.

### 12.4 Đánh giá

```python
y_pred = scaled_model.predict(X_valid)
mae, mse, rmse, r2 = regression_metrics(y_valid, y_pred)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

assert np.isfinite(scaled_model.cost_history_).all()
assert scaled_model.cost_history_[-1] < scaled_model.cost_history_[0]
```

Các giá trị metric chỉ có ý nghĩa khi đi kèm ngữ cảnh đơn vị target, cách chia dữ liệu và baseline. Ví dụ RMSE giá nhà `20 triệu` có thể tốt hoặc tệ tùy thị trường và độ phân tán giá.

---

## 13. Logistic Regression from Scratch

Dù tên chứa “Regression”, logistic regression là mô hình **classification**. Nó gồm:

$$
z=Xw+b
$$

$$
p=\sigma(z)
$$

$$
\hat y=\mathbf{1}[p\ge t]
$$

Trong đó `p` là xác suất lớp `1`, còn `t` là threshold.

### 13.1 Vì sao không dùng MSE?

Với sigmoid, BCE phù hợp với mô hình xác suất Bernoulli và cho gradient rất gọn:

$$
\nabla_w J=\frac{1}{m}X^T(p-y), \qquad
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_i(p_i-y_i)
$$

Đây gần giống linear regression, nhưng prediction đi qua sigmoid và loss là BCE.

### 13.2 Class hoàn chỉnh

```python
import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, epochs=1000, standardize=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.standardize = standardize

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X phải là (m,n), y phải là (m,) và cùng số samples")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Logistic regression nhị phân yêu cầu y là 0/1")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate và epochs phải dương")

        self.n_features_in_ = X.shape[1]
        if self.standardize:
            self.mean_, self.std_ = fit_standardizer(X)
            X_work = transform(X, self.mean_, self.std_)
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.ones(X.shape[1])
            X_work = X

        m, n = X_work.shape
        self.w_ = np.zeros(n, dtype=float)
        self.b_ = 0.0
        self.loss_history_ = []

        # Loss ban đầu với p=0.5.
        self.loss_history_.append(
            binary_cross_entropy(y, sigmoid(X_work @ self.w_ + self.b_))
        )

        for _ in range(self.epochs):
            probabilities = sigmoid(X_work @ self.w_ + self.b_)
            error = probabilities - y

            dw = X_work.T @ error / m
            db = np.mean(error)

            self.w_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db

            new_probabilities = sigmoid(X_work @ self.w_ + self.b_)
            self.loss_history_.append(
                binary_cross_entropy(y, new_probabilities)
            )

        self.loss_history_ = np.asarray(self.loss_history_)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "w_"):
            raise RuntimeError("Phải gọi fit trước predict_proba")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X có số features không đúng")

        X_work = (transform(X, self.mean_, self.std_)
                  if self.standardize else X)
        return sigmoid(X_work @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold phải nằm trong [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)
```

Loss lý tưởng giảm tổng thể; không nhất thiết giảm ở **mọi** iteration nếu learning rate lớn. Nếu loss tăng/dao động mạnh, giảm learning rate và kiểm tra standardization.

### 13.3 Huấn luyện trên dữ liệu mẫu

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 2))
true_logits = 2.0 * X[:, 0] - 1.0 * X[:, 1] - 0.3
probability = sigmoid(true_logits)
y = rng.binomial(1, probability)

indices = rng.permutation(len(y))
split = int(0.8 * len(y))
train_idx, valid_idx = indices[:split], indices[split:]
X_train, X_valid = X[train_idx], X[valid_idx]
y_train, y_valid = y[train_idx], y[valid_idx]

model = LogisticRegressionGD(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

assert np.isfinite(model.loss_history_).all()
assert model.loss_history_[-1] < model.loss_history_[0]

plt.plot(model.loss_history_)
plt.xlabel("Iteration")
plt.ylabel("Binary cross-entropy")
plt.title("Logistic Regression training loss")
plt.grid(alpha=0.3)
plt.show()
```

### 13.4 Chọn threshold và báo cáo metric

```python
valid_probability = model.predict_proba(X_valid)
best_threshold = find_best_threshold(y_valid, valid_probability)
y_pred = model.predict(X_valid, threshold=best_threshold)

tp, tn, fp, fn, accuracy, precision, recall, f1 = (
    classification_metrics(y_valid, y_pred)
)

print(f"Best threshold: {best_threshold:.2f}")
print("Confusion matrix (rows=true, columns=pred):")
print(np.array([[tn, fp], [fn, tp]]))
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
```

Không dùng validation metrics để tiếp tục chỉnh mô hình vô hạn rồi vẫn gọi đó là kết quả cuối. Trong project nhỏ, có thể tách train/validation/test; trong project ít dữ liệu, dùng cross-validation để kết quả ổn định hơn.

---

## 14. Housing Price Prediction trên Kaggle

Competition chính thức: [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Đây là bài toán regression: dự đoán cột `SalePrice` từ thông tin ngôi nhà. Kaggle đánh giá bằng RMSE giữa logarithm của giá dự đoán và giá thật, nên sai lệch theo tỷ lệ quan trọng hơn sai lệch tuyệt đối thuần túy.

### 14.1 Quy trình Kaggle từ đầu

1. Tạo tài khoản Kaggle và mở trang competition.
2. Đọc `Overview`, `Evaluation`, `Data` và `Rules`; bấm **Join Competition** và chấp nhận rules.
3. Tải `train.csv`, `test.csv`, `sample_submission.csv` từ tab **Data**.
4. Chia `train.csv` thành train/validation cục bộ. `test.csv` của Kaggle không có target và không dùng để chọn mô hình.
5. Khám phá dữ liệu, xây preprocessing, huấn luyện trên local train, đánh giá trên local validation.
6. Khi đã chốt quy trình, fit lại trên toàn bộ `train.csv`.
7. Dự đoán `test.csv`, tạo file đúng schema của `sample_submission.csv`.
8. Vào **Submit Predictions**, upload CSV, ghi lại public score và mô tả phiên bản.

Cũng có thể dùng CLI sau khi cài `kaggle` và cấu hình API token trong trang Settings của tài khoản:

```bash
kaggle competitions download \
  -c house-prices-advanced-regression-techniques \
  -p data/house-prices

kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f submission_house_prices.csv \
  -m "Ridge + median imputation + one-hot + log target"
```

Bạn vẫn phải join và chấp nhận rules trên web trước. Không đưa API token vào Git; nếu dùng file credential, giữ file đó ngoài repository và giới hạn quyền truy cập theo hướng dẫn của Kaggle.

### 14.2 Hiểu ba tập dữ liệu

| Tập | Có `SalePrice`? | Mục đích |
|---|---:|---|
| Local train | Có | Học preprocessing và tham số mô hình |
| Local validation | Có | So sánh mô hình/hyperparameter |
| Kaggle test | Không | Tạo submission; Kaggle giữ nhãn thật |

`sample_submission.csv` không phải đáp án. Nó chỉ cho biết đúng tên cột, thứ tự và số hàng cần nộp.

### 14.3 Khảo sát dữ liệu (EDA) tối thiểu

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/house-prices")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(train_df.shape, test_df.shape)
print(train_df.head())
print(train_df["SalePrice"].describe())
print(train_df.isna().mean().sort_values(ascending=False).head(15))
print(train_df.dtypes.value_counts())

train_df["SalePrice"].hist(bins=40)
plt.title("Phân phối SalePrice")
plt.show()

np.log1p(train_df["SalePrice"]).hist(bins=40)
plt.title("Phân phối log(1 + SalePrice)")
plt.show()
```

Cần trả lời trong báo cáo:

- Có bao nhiêu rows/features?
- Target có lệch phải hoặc outlier không?
- Cột nào thiếu nhiều?
- Cột nào numeric, cột nào categorical?
- `Id` là định danh hay feature có ý nghĩa?

`Id` thường được giữ để tạo submission nhưng loại khỏi feature. Không xóa ngay mọi cột có missing: trong dữ liệu nhà ở, missing đôi khi mang nghĩa “không có garage/basement”; baseline dưới đây impute để đơn giản, còn phiên bản nâng cao nên đọc `data_description.txt` và xử lý theo ngữ nghĩa.

### 14.4 Baseline đúng quy trình

Baseline dùng:

- numeric missing → median;
- numeric features → standardization;
- categorical missing → giá trị phổ biến nhất;
- categorical features → one-hot encoding;
- target → `log1p(SalePrice)`;
- model → Ridge regression, tức linear regression có regularization L2.

`Pipeline` rất quan trọng: mọi median, mean, std và danh mục one-hot chỉ được fit từ local train, ngăn data leakage.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = train_df.drop(columns=["SalePrice", "Id"])
y_price = train_df["SalePrice"].to_numpy(dtype=float)
y_log = np.log1p(y_price)
X_kaggle_test = test_df.drop(columns=["Id"])

X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

house_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=10.0)),
])

house_model.fit(X_train, y_train_log)
valid_pred_log = house_model.predict(X_valid)
valid_pred_price = np.maximum(0.0, np.expm1(valid_pred_log))
y_valid_price = np.expm1(y_valid_log)

log_rmse = np.sqrt(np.mean((valid_pred_log - y_valid_log) ** 2))
mae, mse, rmse, r2 = regression_metrics(y_valid_price, valid_pred_price)

print(f"Local log-RMSE: {log_rmse:.5f}")
print(f"MAE           : {mae:,.0f}")
print(f"RMSE          : {rmse:,.0f}")
print(f"R²            : {r2:.4f}")
```

Vì metric Kaggle là log-RMSE, dùng `log_rmse` để chọn mô hình chính. MAE, RMSE và R² trên đơn vị giá thật vẫn nên báo cáo để diễn giải thực tế.

### 14.5 Chọn và tạo thêm feature

Baseline dùng tất cả features sau preprocessing. Có thể thử các feature tổng hợp dựa trên hiểu biết miền:

```python
def add_house_features(df):
    df = df.copy()
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["HouseAgeAtSale"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    return df

X = add_house_features(train_df.drop(columns=["SalePrice", "Id"]))
X_kaggle_test = add_house_features(test_df.drop(columns=["Id"]))
```

Sau khi thêm feature, phải chạy lại phần xác định `numeric_features`, chia train/validation và fit pipeline. So sánh trên cùng split/random seed; nếu thay cả split lẫn feature cùng lúc, ta không biết nguyên nhân score đổi.

### 14.6 Fit toàn bộ train và tạo submission

Chỉ làm bước này sau khi đã chốt pipeline/hyperparameter bằng validation:

```python
# X và X_kaggle_test phải đã qua cùng hàm add_house_features hoặc cùng để nguyên.
house_model.fit(X, y_log)
test_pred_log = house_model.predict(X_kaggle_test)
test_pred_price = np.maximum(0.0, np.expm1(test_pred_log))

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_pred_price,
})

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert submission.columns.tolist() == sample.columns.tolist()
assert len(submission) == len(sample)
assert np.isfinite(submission["SalePrice"]).all()

submission.to_csv("submission_house_prices.csv", index=False)
print(submission.head())
```

`index=False` rất quan trọng; nếu quên, pandas thêm một cột index ngoài schema và Kaggle có thể từ chối file.

### 14.7 Mẫu bảng kết quả cần ghi

| Phiên bản | Features | Model | Local log-RMSE | Local MAE | Local R² | Kaggle score |
|---|---|---|---:|---:|---:|---:|
| V1 | all, basic preprocessing | Ridge α=10 | điền kết quả | điền | điền | điền sau submit |
| V2 | + TotalSF, age, bathroom | Ridge α=10 | điền | điền | điền | điền |

Không chép một score mẫu rồi coi là kết quả của mình. Score phụ thuộc code, dữ liệu và submission thực tế; hãy lưu screenshot hoặc đường dẫn submission cùng ngày chạy.

---

## 15. Titanic trên Kaggle

Competition chính thức: [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

Đây là binary classification: dự đoán `Survived` (`1` sống sót, `0` không sống sót). Kaggle dùng **accuracy**. File train chính thức có nhãn; file test không có `Survived` và cần dự đoán cho từng `PassengerId`.

### 15.1 Khảo sát dữ liệu

```python
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/titanic")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(train_df.shape, test_df.shape)
print(train_df.head())
print(train_df["Survived"].value_counts())
print(train_df["Survived"].value_counts(normalize=True))
print(train_df.isna().sum().sort_values(ascending=False))
print(pd.crosstab(train_df["Sex"], train_df["Survived"], normalize="index"))
print(pd.crosstab(train_df["Pclass"], train_df["Survived"], normalize="index"))
```

Các cột ban đầu đáng chú ý:

| Feature | Ý nghĩa | Xử lý baseline |
|---|---|---|
| `Pclass` | hạng vé, đại diện mức kinh tế-xã hội | categorical |
| `Sex` | giới tính | categorical |
| `Age` | tuổi, có missing | median |
| `SibSp`, `Parch` | người thân đi cùng | numeric và tạo `FamilySize` |
| `Fare` | giá vé, test có thể missing | median |
| `Embarked` | cảng lên tàu, có missing | most frequent + one-hot |
| `Name` | chứa danh xưng | rút `Title`, không dùng nguyên chuỗi |
| `Cabin` | thiếu nhiều | baseline bỏ qua; có thể tạo `HasCabin` |
| `Ticket` | mã có cấu trúc phức tạp | baseline bỏ qua |
| `PassengerId` | định danh submission | không dùng làm feature baseline |

### 15.2 Feature engineering không dùng target

```python
def add_titanic_features(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    title = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    df["Title"] = title
    return df

train_features_df = add_titanic_features(train_df)
test_features_df = add_titanic_features(test_df)
```

Feature engineering phải dùng quy tắc có thể áp dụng giống hệt cho train/test. Không tạo feature từ `Survived`; đó là leakage trực tiếp.

### 15.3 Baseline Logistic Regression

Ta dùng implementation scikit-learn cho pipeline Kaggle vì nó xử lý tốt output one-hot và có regularization. Đây là bước ứng dụng; để chứng minh hiểu thuật toán, Bài 13 vẫn phải nộp class tự cài đặt.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = [
    "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"
]
categorical_features = ["Pclass", "Sex", "Embarked", "Title"]
selected_features = numeric_features + categorical_features

X = train_features_df[selected_features]
y = train_features_df["Survived"].to_numpy()
X_kaggle_test = test_features_df[selected_features]

# stratify giữ gần nguyên tỷ lệ sống sót ở cả hai tập.
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

titanic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

titanic_model.fit(X_train, y_train)
valid_probability = titanic_model.predict_proba(X_valid)[:, 1]
```

### 15.4 Đánh giá threshold `0.5` và threshold tối ưu F1

```python
def print_classification_report_from_scratch(y_true, y_pred, name):
    tp, tn, fp, fn, accuracy, precision, recall, f1 = (
        classification_metrics(y_true, y_pred)
    )
    print(f"\n{name}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print(f"accuracy={accuracy:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f}")
    return accuracy, f1

y_pred_05 = (valid_probability >= 0.5).astype(int)
print_classification_report_from_scratch(y_valid, y_pred_05, "threshold=0.50")

best_f1_threshold = find_best_threshold(y_valid, valid_probability)
y_pred_best_f1 = (valid_probability >= best_f1_threshold).astype(int)
print_classification_report_from_scratch(
    y_valid, y_pred_best_f1, f"best F1 threshold={best_f1_threshold:.2f}"
)
```

Kaggle Titanic chấm accuracy, còn hàm bài 8 tối ưu F1. Do đó threshold tốt nhất theo F1 chưa chắc tạo Kaggle score tốt nhất. Để mục tiêu local khớp competition, có thể tìm threshold có validation accuracy cao nhất và khi hòa chọn threshold nhỏ nhất:

```python
thresholds = np.arange(5, 96, 5) / 100.0
accuracy_scores = np.array([
    classification_metrics(
        y_valid, (valid_probability >= threshold).astype(int)
    )[4]
    for threshold in thresholds
])
best_accuracy_threshold = float(thresholds[np.argmax(accuracy_scores)])
print("Best validation-accuracy threshold:", best_accuracy_threshold)
```

Ở đây vòng lặp/list comprehension chỉ qua 19 thresholds. `np.argmax` trả vị trí đầu tiên, nên tự động chọn threshold nhỏ nhất khi hòa.

Vì validation Titanic nhỏ, threshold tối ưu có thể thay đổi theo split. Báo cáo cả `0.5` và threshold được chọn, dùng cross-validation nếu muốn kết luận đáng tin hơn, và không dò threshold theo Kaggle public leaderboard.

### 15.5 Fit lại và tạo submission

```python
# Sau khi chốt feature/model/threshold, fit lại bằng toàn bộ train có nhãn.
titanic_model.fit(X, y)
test_probability = titanic_model.predict_proba(X_kaggle_test)[:, 1]
test_prediction = (test_probability >= best_accuracy_threshold).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_prediction,
})

sample = pd.read_csv(DATA_DIR / "gender_submission.csv")
assert submission.columns.tolist() == sample.columns.tolist()
assert len(submission) == len(sample)
assert set(submission["Survived"].unique()).issubset({0, 1})

submission.to_csv("submission_titanic.csv", index=False)
print(submission.head())
```

Submit bằng giao diện web hoặc CLI:

```bash
kaggle competitions submit \
  -c titanic \
  -f submission_titanic.csv \
  -m "Logistic regression + Title + FamilySize + local threshold"
```

### 15.6 Cách trình bày báo cáo Titanic

Một báo cáo đủ rõ nên có:

1. mục tiêu và metric của competition;
2. kích thước dữ liệu, tỷ lệ lớp, missing values;
3. lý do chọn/bỏ từng nhóm feature;
4. quy tắc imputation, encoding, standardization;
5. cách chia train/validation và random seed;
6. confusion matrix, accuracy, precision, recall, F1;
7. so sánh threshold `0.5` và threshold đã chọn;
8. bảng các thử nghiệm và Kaggle score thật;
9. lỗi gặp phải, giới hạn và hướng cải thiện.

| Phiên bản | Features | Model | Threshold | Val Accuracy | Val F1 | Kaggle score |
|---|---|---|---:|---:|---:|---:|
| V1 | Pclass, Sex, Age, family, Fare, Embarked | Logistic Regression | 0.50 | điền | điền | điền |
| V2 | V1 + Title + HasCabin | Logistic Regression | điền | điền | điền | điền |

Đừng coi public leaderboard là validation set. Thử quá nhiều lần rồi chọn theo public score sẽ overfit leaderboard, tương tự việc nhìn đáp án test trong lúc làm bài.

---

## 16. Checklist nộp bài

### Bài 1–2

- [ ] Chạy đúng sample HackerRank.
- [ ] Hiểu và giải thích được `axis=0`, `axis=1`.
- [ ] Phân biệt `*`, `np.dot`/`@`, và `np.cross`.

### Bài 3–8

- [ ] Các hàm không lặp qua samples/features.
- [ ] `affine` đúng shape khi `k=1` và `k>1`.
- [ ] Standardizer chỉ fit trên train; feature `std=0` thành 0 trên train.
- [ ] Metrics trả Python `float` và xử lý mẫu số 0 theo yêu cầu.
- [ ] Sigmoid/BCE hữu hạn với logits trong `[-1000, 1000]`.
- [ ] Threshold thử đúng `0.05, 0.10, ..., 0.95`; tie chọn nhỏ nhất.
- [ ] Có giải thích data leakage, threshold và class imbalance.

### Bài 9–11

- [ ] Gradient tự tính, không dùng model/metrics bị cấm.
- [ ] Phép tính theo samples/features được vectorized.
- [ ] Lưu loss trước và trong huấn luyện; loss cuối nhỏ hơn loss đầu ở cấu hình hội tụ.
- [ ] Có đồ thị loss ghi rõ trục, tiêu đề và learning rate.
- [ ] Thử ít nhất ba learning rates và nhận xét dựa trên đồ thị/số liệu.
- [ ] Multiple regression so sánh cùng dữ liệu/cấu hình có và không standardization.
- [ ] Logistic regression báo cáo confusion matrix và đủ bốn metrics.

### Bài 12–13

- [ ] Notebook/script chạy lại được từ đầu tới cuối.
- [ ] Có EDA, preprocessing, feature selection/engineering và lý do.
- [ ] Không fit preprocessing trên validation/Kaggle test.
- [ ] Có local validation và metric đúng với competition.
- [ ] Submission đúng số hàng/tên cột, không có index thừa/NaN/inf.
- [ ] Ghi Kaggle score thật, mô tả submission và bằng chứng.
- [ ] Không commit API token hoặc dữ liệu nếu rules không cho phép phân phối lại.

---

## 17. Thuật ngữ và tài liệu tham khảo

### 17.1 Thuật ngữ nhanh

| Thuật ngữ | Nghĩa ngắn |
|---|---|
| train set | dữ liệu dùng để học tham số |
| validation set | dữ liệu dùng để chọn mô hình/hyperparameter/threshold |
| test set | dữ liệu chỉ dùng cho đánh giá cuối |
| parameter | `w`, `b` được mô hình học |
| hyperparameter | learning rate, epochs, Ridge `alpha`, threshold do ta chọn |
| loss/cost | đại lượng mô hình cố giảm trong training |
| metric | đại lượng dùng báo cáo/chọn mô hình |
| epoch | một lần xử lý toàn bộ train set |
| gradient | đạo hàm chỉ hướng loss tăng nhanh nhất |
| broadcasting | quy tắc NumPy mở rộng shape tương thích khi tính toán |
| data leakage | thông tin ngoài train lọt vào quá trình học/quyết định |
| overfitting | khớp train/validation đã xem nhưng tổng quát hóa kém |
| baseline | giải pháp đầu tiên đơn giản, đúng quy trình để làm mốc |

### 17.2 Thứ tự học đề xuất

Không nên bắt đầu ngay bằng Kaggle. Học theo chuỗi phụ thuộc:

```text
NumPy + shape + axis
        ↓
affine + broadcasting
        ↓
standardization + metrics
        ↓
gradient descent tuyến tính
        ↓
sigmoid + BCE + classification metrics
        ↓
logistic regression + threshold
        ↓
pipeline dữ liệu và Kaggle
```

Mỗi ngày, hãy tự gõ lại một phần nhỏ, dự đoán shape/kết quả trước khi chạy, rồi cố tình tạo một lỗi để đọc thông báo. Hiểu được vì sao một `assert` thất bại có giá trị học tập hơn việc chép một notebook chạy đúng nhưng không giải thích được.

### 17.3 Liên kết chính thức

- [HackerRank — Mean, Var, and Std](https://www.hackerrank.com/challenges/np-mean-var-and-std/problem)
- [HackerRank — Dot and Cross](https://www.hackerrank.com/challenges/np-dot-and-cross/problem)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Kaggle Public API documentation](https://www.kaggle.com/docs/api)
- [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

---

## 18. Thẻ phòng thủ cho 13 bài tập

Bảng này là “bản đồ phòng thủ”. Trước buổi pitching, hãy có thể che cột bên phải và tự nói lại từng hàng.

| Bài tập | Một câu phải nói được | Shape/công thức phải nhớ | Bẫy dễ bị hỏi | Bằng chứng nên mở trong notebook |
|---:|---|---|---|---|
| 1 | `axis` là chiều bị thu gọn | `X.mean(axis=0) -> (n,)` | `ddof=0` và `ddof=1` | ví dụ ma trận 2×3 |
| 2 | `@` khác nhân từng phần tử `*` | `(m,n)@(n,k)->(m,k)` | tên đề có cross nhưng task dùng matrix product | tính tay một phần tử |
| 3 | Affine là matrix product cộng bias | `XW+b`, output `(m,k)` | `b=(k,)` khác `(k,1)` | test `k=1` và `k>1` |
| 4 | Fit thống kê chỉ trên train | `(X-mean)/std` theo cột | leakage và `std=0` | cột hằng thành toàn 0 |
| 5 | Mỗi regression metric nhấn mạnh loại lỗi khác nhau | MAE, MSE, RMSE, R² | R² có thể âm; target hằng | ví dụ tính tay `[1,2,3]` |
| 6 | Sigmoid/BCE phải ổn định số | sigmoid tách nhánh; probability được clip | `np.where` vẫn có thể tính cả hai nhánh | stress test `±1000` |
| 7 | Metric phụ thuộc lớp positive | TP/TN/FP/FN và bốn công thức | mẫu số 0; đảo FP/FN | confusion matrix 2×2 |
| 8 | Threshold là quyết định theo mục tiêu, không phải hằng số tự nhiên | `p>=t`; quét 0.05…0.95 | tie-break; tuning trên test | trường hợp hòa chọn 0.05 |
| 9 | GD đi ngược gradient của cost | `dw=mean(error*x)`, `db=mean(error)` | dấu cập nhật, hệ số `1/2m`, learning rate | ba đường loss |
| 10 | Gradient nhiều chiều là phép nhân ma trận | `dw=X.T@error/m` | vòng lặp epochs được phép; samples/features không | raw vs standardized loss |
| 11 | Logistic regression là classifier xác suất | `p=sigmoid(Xw+b)`, `dw=X.T@(p-y)/m` | vì sao BCE; loss hữu hạn nhưng không nhất thiết giảm từng bước | loss và confusion matrix |
| 12 | Kaggle test không phải local validation | log-target + preprocessing pipeline | leakage, `Id`, schema submission | local metric và CSV |
| 13 | Metric competition quyết định cách chọn threshold | Titanic chấm accuracy | F1 threshold chưa chắc tốt cho accuracy | so sánh 0.5/F1/accuracy threshold |

### 18.1 Bài 1 — Mean, Variance, Standard Deviation

**Pitch 30 giây:** Mean mô tả tâm, variance đo độ phân tán bình phương và standard deviation là căn của variance nên trở lại đơn vị gốc. Với dữ liệu ML `(m,n)`, `axis=0` thu gọn samples và trả một thống kê cho mỗi feature. NumPy mặc định dùng population variance `ddof=0`, đúng yêu cầu HackerRank.

**Nếu bị hỏi xoáy:** Outlier tác động mạnh tới variance/std vì sai lệch bị bình phương. Standard deviation không âm; bằng 0 khi feature hằng.

### 18.2 Bài 2 — Dot and Cross

**Pitch 30 giây:** `A * B` nhân từng vị trí; `A @ B` lấy tích vô hướng giữa hàng của A và cột của B. Điều kiện là kích thước trong khớp nhau: `(m,n)@(n,k)->(m,k)`. Cross product là phép khác; tên challenge nhắc cả hai nhưng task yêu cầu matrix product.

**Nếu bị hỏi xoáy:** `np.dot` và `@` giống nhau cho hai ma trận 2-D, nhưng hành vi với array nhiều chiều khác; `@` thể hiện ý định nhân ma trận rõ hơn.

### 18.3 Bài 3 — Vectorized Affine

**Pitch 30 giây:** Tôi tính `Z=X@W+b` một lần cho toàn bộ batch. `X@W` có shape `(m,k)` và `b=(k,)` được broadcast theo từng hàng. Không có vòng lặp qua samples/features; tôi kiểm tra shape đầu vào để lỗi xuất hiện sớm và dễ hiểu.

**Nếu bị hỏi xoáy:** Với `k=1`, phải giữ `W=(n,1)`, `b=(1,)` để output là `(m,1)`; nếu dùng `W=(n,)`, output sẽ thành `(m,)` và làm thay đổi API.

### 18.4 Bài 4 — Feature Standardization

**Pitch 30 giây:** Tôi fit mean/std theo từng cột chỉ từ train và dùng lại chúng cho validation/test. Làm vậy đưa feature về scale tương đương, giúp GD ổn định, đồng thời tránh leakage. Cột có std bằng 0 được ép toàn 0 vì không có scale hữu ích để học.

**Nếu bị hỏi xoáy:** Không phải mô hình nào cũng cần scaling như nhau; tree-based models thường ít nhạy với scale, còn GD, k-NN, SVM và regularization thường nhạy.

### 18.5 Bài 5 — Regression Metrics

**Pitch 30 giây:** MAE dễ diễn giải và ít nhạy outlier hơn; MSE/RMSE phạt lỗi lớn mạnh hơn; RMSE trở lại đơn vị target. R² so mô hình với baseline đoán mean: `1` hoàn hảo, `0` ngang baseline và có thể âm nếu tệ hơn baseline.

**Nếu bị hỏi xoáy:** Không dùng R² một mình; nó không nói sai số là bao nhiêu đơn vị và không tự bảo đảm mô hình phù hợp.

### 18.6 Bài 6 — Stable Sigmoid và BCE

**Pitch 30 giây:** Công thức sigmoid trực tiếp overflow tại logit âm rất lớn. Tôi tách nhánh theo dấu: nhánh dương dùng `exp(-z)`, nhánh âm dùng `exp(z)`. BCE clip probability khỏi đúng 0/1 trước khi lấy log, nên chạy hữu hạn với `z` từ `-1000` tới `1000`.

**Nếu bị hỏi xoáy:** Viết hai biểu thức trong `np.where` chưa chắc an toàn vì NumPy có thể tính cả hai nhánh trước khi chọn. Nếu nhận logits trực tiếp, `logaddexp` là cách còn ổn định hơn.

### 18.7 Bài 7 — Confusion Matrix và Metrics

**Pitch 30 giây:** Tôi chọn nhãn `1` là positive rồi đếm TP, TN, FP, FN bằng boolean masks. Accuracy đo tổng thể; precision trả lời “dự đoán positive đúng bao nhiêu”; recall trả lời “positive thật tìm được bao nhiêu”; F1 cân bằng precision/recall. Mẫu số 0 trả `0.0` theo yêu cầu.

**Nếu bị hỏi xoáy:** Đổi lớp nào là positive sẽ đổi precision/recall và ý nghĩa FP/FN, dù accuracy không đổi.

### 18.8 Bài 8 — Classification Threshold

**Pitch 30 giây:** Tôi quét 19 thresholds từ 0.05 đến 0.95 trên validation set, tính F1 và chỉ cập nhật khi điểm lớn hơn, nên khi hòa giữ threshold nhỏ nhất. `0.5` không luôn tốt vì class balance, calibration, metric và chi phí FP/FN khác nhau.

**Nếu bị hỏi xoáy:** Không chọn threshold trên test hoặc public leaderboard; làm vậy biến dữ liệu đánh giá thành dữ liệu tuning.

### 18.9 Bài 9 — Linear Regression 1-D bằng GD

**Pitch 30 giây:** Model là `ŷ=wx+b`, cost là nửa MSE. Đạo hàm là `dw=mean((ŷ-y)x)` và `db=mean(ŷ-y)`; cập nhật đi ngược gradient. Tôi lưu loss ban đầu và sau mỗi epoch, rồi so ba learning rates để chỉ ra hội tụ chậm, ổn định và phân kỳ.

**Nếu bị hỏi xoáy:** Hệ số `1/2` không đổi nghiệm tối ưu; nó triệt tiêu hệ số `2` khi đạo hàm bình phương.

### 18.10 Bài 10 — Multiple Linear Regression

**Pitch 30 giây:** Mở rộng một feature thành `X:(m,n)`, `w:(n,)`; prediction là `X@w+b`. Gradient vectorized `X.T@error/m` có shape `(n,)`. Tôi so cùng learning rate có/không standardization để tách đúng tác động của scaling.

**Nếu bị hỏi xoáy:** Vòng lặp qua epochs là cần vì bước sau phụ thuộc bước trước; điều bị cấm là vòng lặp thủ công qua từng sample/feature bên trong gradient.

### 18.11 Bài 11 — Logistic Regression

**Pitch 30 giây:** Logistic regression lấy affine logit rồi qua sigmoid để ra xác suất. BCE phù hợp với Bernoulli và gradient trở thành `X.T@(p-y)/m`. `predict_proba` tách khỏi `predict` để threshold có thể thay đổi theo mục tiêu.

**Nếu bị hỏi xoáy:** Loss giảm không bảo đảm accuracy tăng ở mọi bước; loss tối ưu xác suất liên tục, còn accuracy dùng quyết định rời rạc theo threshold.

### 18.12 Bài 12 — Housing Price Prediction

**Pitch 30 giây:** Tôi tách local train/validation trước, dùng pipeline để impute, scale và one-hot chỉ fit trên train. Vì competition chấm log-RMSE, tôi học `log1p(SalePrice)`, nhưng vẫn báo cáo MAE/RMSE/R² ở đơn vị giá thật. Sau khi chốt model, tôi fit lại toàn bộ train và tạo CSV đúng schema.

**Nếu bị hỏi xoáy:** `sample_submission` chỉ là schema, không phải ground truth. `Id` dùng để ghép submission, thường không dùng làm feature baseline.

### 18.13 Bài 13 — Titanic

**Pitch 30 giây:** Tôi tạo `FamilySize`, `IsAlone`, `HasCabin`, `Title`, rồi xử lý missing/one-hot trong pipeline. Validation dùng stratify để giữ tỷ lệ lớp. Tôi báo cáo confusion matrix và bốn metrics; vì Kaggle chấm accuracy, threshold chọn theo F1 chỉ là phân tích phụ, không mặc định là threshold submission.

**Nếu bị hỏi xoáy:** Feature engineering phải dùng quy tắc áp dụng giống nhau cho train/test và không được sử dụng `Survived`; nếu không sẽ leakage.

---

## 19. Ngân hàng câu hỏi xoáy và mock interview

### 19.1 Cách luyện

Chạy cell sau để lấy 10 câu ngẫu nhiên. Trả lời thành tiếng, tối đa 45 giây/câu, chưa xem đáp án. Sau đó gọi `reveal_pitch_answers(mock_ids)` và tự chấm:

- `0`: sai hoặc im lặng;
- `1`: đúng kết luận nhưng thiếu công thức/shape;
- `2`: đúng, có cơ chế và edge case;
- `3`: đúng, ngắn, có bằng chứng từ notebook.

```python
PITCH_BANK = [
    (1, "axis=0 trên X shape (m,n) trả shape gì?", "Shape (n,), một thống kê cho mỗi feature."),
    (1, "Vì sao std có cùng đơn vị còn variance thì không?", "Variance bình phương sai lệch nên có đơn vị bình phương; std lấy căn nên trở lại đơn vị gốc."),
    (1, "NumPy var mặc định chia m hay m-1?", "Chia m, ddof=0; muốn sample variance thường dùng ddof=1."),
    (2, "Phân biệt A*B và A@B.", "* là element-wise có broadcasting; @ là matrix multiplication và yêu cầu inner dimensions khớp."),
    (2, "(5,3)@(3,2) cho shape gì và vì sao?", "(5,2); 3 là chiều được cộng lại trong dot products."),
    (2, "Tên đề Dot and Cross, vì sao code không dùng cross?", "Task cụ thể yêu cầu matrix product; cross chỉ là phần giới thiệu API."),
    (3, "Vì sao b shape (k,) broadcast đúng với (m,k)?", "NumPy so từ chiều cuối: k khớp k và chiều thiếu được mở rộng qua m hàng."),
    (3, "k=1 thì output nên là (m,) hay (m,1)?", "Theo contract đề là (m,1); phải giữ W=(n,1), b=(1,)."),
    (3, "Affine có phải linear transformation không?", "Có bias thì chính xác là affine; chỉ linear khi b=0."),
    (4, "Tại sao fit mean/std toàn dataset là leakage?", "Validation/test distribution đã ảnh hưởng preprocessing học được, làm đánh giá lạc quan."),
    (4, "Feature std=0 xử lý thế nào?", "Dùng safe denominator và ép cột đó về 0 để tránh chia 0 và bỏ feature không có biến thiên."),
    (4, "Standardization có đổi thứ tự các mẫu theo một feature không?", "Không nếu std>0; đó là phép affine đơn điệu tăng."),
    (5, "Metric nào nhạy outlier hơn: MAE hay RMSE?", "RMSE vì sai số bị bình phương trước khi lấy trung bình/căn."),
    (5, "R² âm có phải bug?", "Không; mô hình có thể tệ hơn baseline luôn dự đoán mean."),
    (5, "R² có đơn vị không?", "Không; nó là tỷ lệ tương đối giữa hai tổng bình phương."),
    (6, "Sigmoid trực tiếp hỏng ở z=-1000 như thế nào?", "Nó cần exp(1000), gây overflow; tách công thức theo dấu tránh số mũ dương khổng lồ."),
    (6, "Vì sao clip probability trong BCE?", "Tránh log(0)=-inf và duy trì loss hữu hạn."),
    (6, "np.where có chắc tránh overflow không?", "Không; các biểu thức hai nhánh thường được tính trước khi chọn."),
    (7, "FP và FN khác nhau thế nào?", "FP dự đoán 1 khi thật 0; FN dự đoán 0 khi thật 1."),
    (7, "Không dự đoán positive nào thì precision bằng gì?", "Theo quy ước bài là 0.0 vì TP+FP=0."),
    (7, "Accuracy có đổi nếu đổi nhãn nào là positive không?", "Không; precision/recall và tên FP/FN thì đổi."),
    (8, "Tại sao 0.5 không luôn tốt nhất?", "Metric, class prior, calibration và chi phí FP/FN có thể yêu cầu điểm vận hành khác."),
    (8, "Vì sao chọn threshold trên validation, không trên test?", "Threshold là hyperparameter; tuning trên test làm mất đánh giá độc lập."),
    (8, "Hai threshold hòa F1 xử lý thế nào?", "Duyệt tăng dần và chỉ cập nhật khi lớn hơn; giữ threshold nhỏ nhất."),
    (9, "Tại sao cập nhật dùng dấu trừ?", "Gradient chỉ hướng tăng nhanh nhất; đi ngược để giảm cost."),
    (9, "Vì sao cost có 1/(2m)?", "Chia m để scale không phụ thuộc số mẫu; 1/2 làm đạo hàm gọn."),
    (9, "Loss tăng thì kiểm tra gì trước?", "Dấu gradient, chia m, learning rate, feature scale và NaN/inf."),
    (10, "Chứng minh shape của dw.", "X.T là (n,m), error là (m,), tích cho (n,), đúng shape w."),
    (10, "Vì sao vẫn có loop qua epochs?", "Mỗi update phụ thuộc tham số bước trước; chỉ phép tính samples/features cần vectorized."),
    (10, "Scaling tác động gì lên bề mặt loss?", "Giảm độ kéo dài theo các trục, giúp một learning rate tiến ổn định theo nhiều hướng."),
    (11, "Vì sao logistic regression là classification?", "Nó mô hình hóa log-odds/xác suất lớp rồi threshold thành nhãn."),
    (11, "Gradient BCE + sigmoid vì sao là p-y?", "Đạo hàm sigmoid và log loss triệt tiêu, để lại sai số xác suất p-y."),
    (11, "Loss giảm nhưng F1 giảm có thể xảy ra không?", "Có; BCE và F1 tối ưu mục tiêu khác nhau, F1 còn phụ thuộc threshold."),
    (12, "Kaggle test.csv dùng để chọn model được không?", "Không có ground truth và không nên dùng public leaderboard như validation."),
    (12, "Vì sao House Prices dùng log1p target?", "Khớp log-RMSE, giảm lệch phải và tập trung vào sai số tương đối."),
    (12, "Pipeline ngăn leakage thế nào?", "Các transformer được fit chỉ khi model.fit trên local train, rồi tái dùng khi transform validation."),
    (13, "Titanic chấm accuracy, có nên dùng best-F1 threshold không?", "Không mặc định; phải chọn theo objective đã xác định và báo cáo trade-off."),
    (13, "Vì sao stratify khi split?", "Giữ tỷ lệ lớp gần nhau, làm so sánh ổn định hơn trên dữ liệu nhỏ."),
    (13, "Title có phải leakage không?", "Không nếu chỉ trích từ Name bằng cùng quy tắc và không dùng Survived; nó vẫn có rủi ro overfit cần validation."),
]


def new_pitch_mock(n_questions=10, seed=None):
    if not 1 <= n_questions <= len(PITCH_BANK):
        raise ValueError("n_questions không hợp lệ")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(PITCH_BANK), size=n_questions, replace=False)
    print("Bắt đầu đồng hồ — tối đa 45 giây/câu:\n")
    for order, idx in enumerate(ids, 1):
        exercise, question, _ = PITCH_BANK[int(idx)]
        print(f"{order:02}. [Bài {exercise}] {question}")
    return ids


def reveal_pitch_answers(ids):
    print("\nĐáp án tham khảo:\n")
    for order, idx in enumerate(ids, 1):
        exercise, question, answer = PITCH_BANK[int(idx)]
        print(f"{order:02}. [Bài {exercise}] {question}\n    → {answer}\n")


# Đổi seed hoặc dùng seed=None để lấy đề mới mỗi lần.
mock_ids = new_pitch_mock(n_questions=10, seed=42)
# Sau khi đã trả lời thành tiếng, chạy dòng dưới trong một cell mới:
# reveal_pitch_answers(mock_ids)
```

### 19.2 Kỹ thuật hỏi tiếp kiểu “đáp xoay”

Sau mọi câu trả lời, tự ép mình trả lời thêm ba lớp:

1. **Nếu bỏ điều kiện đó thì sao?** Ví dụ bỏ clip → `log(0)`; bỏ scaling → GD có thể phân kỳ.
2. **Cho tôi shape cụ thể.** Không nói “ma trận phù hợp”; hãy nói `(m,n)@(n,k)->(m,k)`.
3. **Chứng minh bằng test nào?** Nêu normal case, edge case và failure case.

Những câu nối chuỗi có xác suất cao:

- `axis=0` liên quan gì đến standardization?
- `X@W+b` xuất hiện lại ở linear và logistic regression thế nào?
- Tại sao sigmoid ổn định nhưng BCE vẫn cần clip?
- Loss dùng để train khác metric dùng để báo cáo ra sao?
- Standardization sai thời điểm dẫn tới leakage trong Kaggle thế nào?
- Vì sao threshold tốt nhất theo F1 không chắc tốt nhất theo accuracy?
- Nếu loss cuối thấp hơn loss đầu, đã đủ kết luận mô hình tốt chưa?

### 19.3 Kịch bản mở đầu pitching trong 90 giây

> Em xem 13 bài như một pipeline thống nhất chứ không phải 13 đáp án rời. Bài 1–4 xây nền NumPy: axis, matrix multiplication, affine và chuẩn hóa. Bài 5–8 xây cách đánh giá regression/classification và biến xác suất thành quyết định ổn định. Bài 9–11 dùng các khối đó để tự cài linear và logistic regression bằng gradient descent vectorized. Cuối cùng, House Prices và Titanic đưa toàn bộ quy trình vào dữ liệu thật: split trước preprocessing, chống leakage, đánh giá local, fit lại toàn bộ train và tạo submission đúng schema. Trong phần trình bày em sẽ dùng shape, loss curve, edge-case tests và metric validation làm bằng chứng cho từng quyết định.

Không học thuộc nguyên văn. Hãy giữ cấu trúc và thay bằng giọng nói tự nhiên của bạn.

---

## 20. Code drills và checklist trước pitching

### 20.1 Cách dùng code drills

Các hàm dưới đây cố ý để `raise NotImplementedError`. Khi Run All, tests bị tắt nên notebook không lỗi. Trong phiên luyện riêng:

1. đổi `RUN_CODE_DRILLS = True`;
2. tự cài lại mà không nhìn lời giải phía trên;
3. chạy tests;
4. nếu sai, đọc assertion/shape rồi sửa trước khi mở lời giải.

```python
RUN_CODE_DRILLS = False
```

#### Drill Bài 1–4

```python
def drill_statistics(X):
    """Return mean axis=1, variance axis=0, overall std."""
    raise NotImplementedError


def drill_matrix_product(A, B):
    """Return matrix product, not element-wise product."""
    raise NotImplementedError


def drill_affine(X, W, b):
    """Return XW+b with output shape (m,k)."""
    raise NotImplementedError


def drill_standardizer(X_train, X):
    """Fit mean/std on X_train and transform X; zero-std columns become zero."""
    raise NotImplementedError


if RUN_CODE_DRILLS:
    A = np.array([[1., 2.], [3., 4.]])
    row_mean, col_var, total_std = drill_statistics(A)
    assert np.allclose(row_mean, [1.5, 3.5])
    assert np.allclose(col_var, [1., 1.])
    assert np.isclose(total_std, np.sqrt(1.25))
    assert np.array_equal(drill_matrix_product(A, A), [[7, 10], [15, 22]])

    X_d = np.array([[1., 2.], [3., 4.]])
    W_d = np.array([[1.], [2.]])
    assert drill_affine(X_d, W_d, np.array([0.5])).shape == (2, 1)

    train_d = np.array([[1., 7.], [2., 7.], [3., 7.]])
    transformed_d = drill_standardizer(train_d, np.array([[4., 999.]]))
    assert transformed_d.shape == (1, 2)
    assert transformed_d[0, 1] == 0.0
    print("PASS drills 1–4")
```

#### Drill Bài 5–8

```python
def drill_regression_metrics(y_true, y_pred):
    """Return MAE, MSE, RMSE, R2 without sklearn.metrics."""
    raise NotImplementedError


def drill_sigmoid_and_bce(z, y):
    """Return stable probabilities and finite BCE."""
    raise NotImplementedError


def drill_classification_metrics(y_true, y_pred):
    """Return TP,TN,FP,FN,accuracy,precision,recall,F1."""
    raise NotImplementedError


def drill_best_threshold(y_true, probabilities):
    """Try 0.05..0.95; maximize F1; smallest threshold wins ties."""
    raise NotImplementedError


if RUN_CODE_DRILLS:
    metrics_d = drill_regression_metrics([1, 2, 3], [1, 2, 5])
    assert np.allclose(metrics_d[:3], [2/3, 4/3, np.sqrt(4/3)])
    assert np.isclose(metrics_d[3], -1.0)

    probability_d, bce_d = drill_sigmoid_and_bce(
        np.array([-1000., 0., 1000.]), np.array([0, 1, 1])
    )
    assert np.isfinite(probability_d).all() and np.isfinite(bce_d)

    cls_d = drill_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert cls_d[:4] == (1, 1, 1, 1)
    assert np.isclose(drill_best_threshold([0, 1], [0.01, 0.99]), 0.05)
    print("PASS drills 5–8")
```

#### Drill Bài 9–13

```python
def drill_linear_gradients_1d(x, y, w, b):
    """Return dw, db for J=(1/2m)*sum((wx+b-y)^2)."""
    raise NotImplementedError


def drill_linear_gradients_nd(X, y, w, b):
    """Return vectorized dw:(n,), db:scalar."""
    raise NotImplementedError


def drill_logistic_gradients(X, y, w, b):
    """Return vectorized BCE gradients using stable sigmoid."""
    raise NotImplementedError


def drill_validate_house_submission(ids, prices):
    """Return True only when lengths match and all prices are finite/non-negative."""
    raise NotImplementedError


def drill_choose_titanic_threshold(y_true, probabilities, metric="accuracy"):
    """Choose threshold 0.05..0.95 for the requested metric."""
    raise NotImplementedError


if RUN_CODE_DRILLS:
    x_d = np.array([-1., 0., 1.])
    y_d = 2 * x_d + 1
    dw_1d, db_1d = drill_linear_gradients_1d(x_d, y_d, 0., 0.)
    assert np.isscalar(dw_1d) and np.isscalar(db_1d)

    X_d = np.column_stack([x_d, x_d ** 2])
    w_d = np.zeros(2)
    dw_nd, db_nd = drill_linear_gradients_nd(X_d, y_d, w_d, 0.)
    assert np.asarray(dw_nd).shape == (2,) and np.isscalar(db_nd)

    dw_log, db_log = drill_logistic_gradients(
        X_d, np.array([0, 1, 1]), w_d, 0.
    )
    assert np.asarray(dw_log).shape == (2,) and np.isfinite(db_log)
    assert drill_validate_house_submission([1, 2], [100000., 200000.]) is True

    threshold_d = drill_choose_titanic_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], metric="accuracy"
    )
    assert 0.05 <= threshold_d <= 0.95
    print("PASS drills 9–13")
```

### 20.2 Checklist “hot seat” trước khi vào pitching

- [ ] Tôi có thể viết 11 API cốt lõi từ chữ ký hàm mà không copy.
- [ ] Tôi luôn nói shape trước khi nói phép nhân ma trận/gradient.
- [ ] Tôi phân biệt loss dùng để train và metric dùng để báo cáo.
- [ ] Tôi giải thích được data leakage bằng một ví dụ cụ thể.
- [ ] Tôi có thể vẽ và điền TP/TN/FP/FN trong 20 giây.
- [ ] Tôi giải thích được ba edge cases: `std=0`, mẫu số metric bằng 0, logit `±1000`.
- [ ] Tôi có thể tự đạo hàm `dw`, `db` cho linear regression.
- [ ] Tôi nhớ gradient logistic là `X.T@(p-y)/m` và giải thích nguồn gốc.
- [ ] Tôi có đồ thị loss của ba learning rates và nhận xét bằng số liệu thật.
- [ ] Tôi có local metrics và Kaggle score thật cho House Prices/Titanic, hoặc nói rõ phần nào chưa chạy.
- [ ] Tôi biết vì sao House dùng log-RMSE còn Titanic dùng accuracy.
- [ ] Tôi đã restart session và Run All bản pitch ít nhất một lần.
- [ ] Tôi đã làm hai mock interviews, ghi lại ít nhất năm câu trả lời yếu và sửa chúng.

### 20.3 Khi bị bí trong lúc trả lời

Không im lặng quá lâu và không bịa. Dùng quy trình:

1. nhắc lại câu hỏi để xác nhận phạm vi;
2. nêu giả định về shape/lớp positive/metric;
3. dựng một ví dụ 2×2 hoặc ba samples;
4. suy ra từ công thức;
5. nói cách kiểm chứng bằng `assert`.

Ví dụ: “Em chưa nhớ chính xác giá trị số, nhưng với `X:(m,n)` và error `(m,)`, gradient của `w` phải có shape `(n,)`; vì vậy phép vectorized hợp lý là `X.T @ error / m`. Em sẽ kiểm chứng bằng finite-difference gradient nếu cần.” Đây là tư duy kỹ thuật tốt hơn việc đoán.

---

## Kết luận

Các bài tuần 3 tạo thành một hệ thống thống nhất: `X @ W + b` sinh dự đoán; standardization giúp tối ưu ổn định; loss hướng dẫn gradient descent; metric đo chất lượng; threshold biến xác suất thành quyết định; pipeline ngăn leakage khi đưa toàn bộ quy trình vào dữ liệu thật. Khi hoàn thành, bạn chưa cần biết Deep Learning, nhưng đã có gần như toàn bộ khối xây dựng toán học và kỹ thuật để hiểu một neuron/lớp dense cơ bản.
