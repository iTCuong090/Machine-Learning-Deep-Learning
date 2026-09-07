BÀI 1 - Predictive Maintenance: Baseline, Data Mismatch & Multi-task Learning
Dữ liệu: Machine Predictive Maintenance Classification (Kaggle: shivamb/machine-predictive-maintenance-classification).
Phần A - Baseline & Data Leakage
●  Xây Dense Neural Network để dự đoán Machine failure. Kiểm tra feature availability và loại bỏ các biến định danh hoặc target-derived feature có nguy cơ leakage.
●  Báo cáo train/dev loss, accuracy, precision, recall và confusion matrix. Giải thích metric nào phản ánh tốt hơn hiệu quả phát hiện failure khi class bị mất cân bằng.
Phần B - Training và Testing trên Distribution khác nhau
●  Tạo hai dev set: (1) random/stratified dev có distribution gần train; (2) mismatch dev trong đó một nhóm Type hoặc một vùng điều kiện vận hành được giữ lại nhiều hơn so với train.
●  So sánh performance giữa train, matched-dev và mismatched-dev. Phân biệt trường hợp model bị high variance với trường hợp model gặp data distribution mismatch.
●  Đề xuất tối đa 2 cách xử lý mismatch, ví dụ thay đổi cách lấy dữ liệu train, reweight/resample các nhóm, bổ sung dữ liệu đại diện hoặc thiết kế feature bền vững hơn. Giải thích vì sao chọn chúng.
Phần C - Multi-task Learning
●  Thử một mô hình shared representation có hai đầu ra liên quan: Machine failure và Failure Type. So sánh với mô hình single-task chỉ dự đoán Machine failure.
●  Đánh giá xem auxiliary task có giúp task chính hay không. Nếu không giúp, phân tích khả năng hai task không đủ liên quan, loss weighting chưa hợp lý hoặc class imbalance làm một task chi phối task còn lại.
