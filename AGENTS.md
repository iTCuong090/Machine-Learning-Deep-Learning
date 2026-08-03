# Hướng dẫn làm việc trong workspace

## Môi trường Python bắt buộc

- Workspace dùng môi trường ảo cục bộ `.venv` với Python 3.14.
- Không cài package bằng pip toàn cục và không dùng `sudo pip`.
- Agent phải gọi trực tiếp `.venv/bin/python` và `.venv/bin/python -m pip` trong lệnh tự động; không được giả định shell đã chạy `source`.
- Trước khi chạy code, kiểm tra interpreter bằng:

  ```bash
  .venv/bin/python --version
  .venv/bin/python -c "import sys; print(sys.executable)"
  ```

Nếu `.venv` bị xóa, có thể dựng lại không cần quyền quản trị. Máy hiện tại
thiếu module hệ thống `ensurepip`, vì vậy dùng pip hệ thống làm bootstrap duy
nhất; lệnh `--python .venv` vẫn cài pip vào môi trường ảo, không cài toàn cục:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install --upgrade pip setuptools wheel
```

## Cài đặt và cập nhật dependency

Bộ Data Science/ML cơ bản:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

PyTorch CPU cho Deep Learning:

```bash
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  -r requirements-dl.txt
```

Khi cần thêm package:

1. Cài bằng `.venv/bin/python -m pip install <package>`.
2. Ghi dependency trực tiếp vào `requirements.txt`, hoặc vào `requirements-dl.txt` nếu chỉ dành cho Deep Learning.
3. Không ghi toàn bộ output của `pip freeze` đè lên các file requirement; chỉ thêm dependency mà dự án chủ động sử dụng.

## Chạy notebook và script

Người dùng có thể kích hoạt môi trường khi làm việc tương tác:

```bash
source .venv/bin/activate
python -m jupyter lab
```

Agent nên dùng đường dẫn tường minh:

```bash
.venv/bin/python script.py
.venv/bin/python -m jupyter lab
```

Trong Jupyter, chọn kernel có interpreter kết thúc bằng `.venv/bin/python`.

Trong sandbox có home chỉ-đọc, agent đặt cache trong workspace trước khi dùng
Matplotlib/Jupyter để tránh ghi ra ngoài phạm vi cho phép:

```bash
mkdir -p .cache/matplotlib .cache/jupyter
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export JUPYTER_CONFIG_DIR="$PWD/.cache/jupyter"
```

## Quy ước ML/Data Science

- Đặt random seed khi kết quả cần tái lập.
- Fit preprocessing chỉ trên tập train để tránh data leakage.
- Kiểm tra `shape`, `dtype`, NaN/inf và metric trước khi kết luận mô hình đúng.
- Ưu tiên lưu dữ liệu lớn, checkpoint và artifact ngoài Git; các đường dẫn phổ biến đã có trong `.gitignore`.
- PyTorch CPU là backend DL mặc định. Chỉ đổi sang CUDA sau khi xác nhận máy có GPU/driver phù hợp và dùng đúng wheel index.
- TensorFlow chưa được đưa vào requirements vì hiện không có wheel tương thích Python 3.14 trên PyPI.

## Kiểm tra nhanh môi trường

```bash
.venv/bin/python -c "import numpy, pandas, scipy, sklearn, matplotlib, torch; print('environment OK'); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```
