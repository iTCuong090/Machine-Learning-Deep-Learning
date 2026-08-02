# Bài tập tuần 3 — Nền tảng NumPy và Machine Learning từ số 0

> Tài liệu này dành cho người chưa học Machine Learning (ML) hoặc Deep Learning (DL). Mục tiêu không phải chỉ là chạy được mã, mà là hiểu dữ liệu đi qua mô hình như thế nào, vì sao công thức đúng và cách phát hiện khi chương trình sai.

## Mục lục

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

    safe_std = np.where(std == 0, 1.0, std)
    return (X - mean) / safe_std
```

Với feature hằng, mọi giá trị train đều bằng mean; thay mẫu số `0` bằng `1` làm `(x - mean) / 1 = 0`. Nếu dữ liệu validation/test của feature đó khác hằng số đã thấy ở train, kết quả không nhất thiết là 0 — điều này hợp lý vì đó là giá trị mới lệch khỏi train.

### 6.3 Tự kiểm tra

```python
X_train = np.array([
    [1, 10, 7],
    [2, 20, 7],
    [3, 30, 7],
], dtype=float)
X_valid = np.array([[4, 40, 7]], dtype=float)

mean, std = fit_standardizer(X_train)
X_train_s = transform(X_train, mean, std)
X_valid_s = transform(X_valid, mean, std)

assert np.allclose(X_train_s.mean(axis=0), [0, 0, 0])
assert np.allclose(X_train_s[:, :2].std(axis=0), [1, 1])
assert np.allclose(X_train_s[:, 2], 0)
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
