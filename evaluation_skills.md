# Hướng dẫn đánh giá mô hình

## 1. Mục tiêu đánh giá

Bài toán phân loại nhị phân với biến mục tiêu `Revenue`:

- `False` (0): **Không mua** (`Non-purchase`)
- `True` (1): **Mua** (`Purchase`)

Dữ liệu mất cân bằng, trong đó lớp **Mua** là lớp thiểu số. Sử dụng **Average Precision (AP)** làm độ đo chính để so sánh mô hình. Không sử dụng ROC curve hoặc ROC-AUC.

## 2. Các hàm đánh giá của scikit-learn

| Hàm | Tên chỉ số | Vai trò |
|---|---|---|
| `average_precision_score` | **AP** | Độ đo chính để so sánh mô hình |
| `precision_recall_curve` | **PR curve** | Khảo sát sự đánh đổi giữa Precision và Recall theo ngưỡng |
| `precision_score` | **Precision (Mua)** | Tỷ lệ dự đoán Mua đúng trong số các phiên được dự đoán Mua |
| `recall_score` | **Recall (Mua)** | Tỷ lệ phiên Mua thực tế được mô hình phát hiện |
| `f1_score` | **F1 (Mua)** | Cân bằng giữa Precision và Recall của lớp Mua |
| `confusion_matrix` | **Confusion matrix** | Thống kê TP, FP, TN, FN tại ngưỡng đã chọn |
| `classification_report` | **Classification report** | Báo cáo Precision, Recall, F1-score và support theo lớp |

## 3. Quy tắc đặt tên và báo cáo

- Dùng tên **AP** cho giá trị tính bằng `average_precision_score`.
- Không gọi AP là `PR-AUC`, vì AP và diện tích hình học tính bằng quy tắc hình thang là hai đại lượng khác nhau.
- `precision_recall_curve` tạo dữ liệu để vẽ PR curve, không trả về một chỉ số tổng hợp duy nhất.
- Không sử dụng `auc(recall, precision)`.
- Khi đánh giá bằng cross-validation (CV), báo cáo **mean ± std** qua các fold.
- Quy ước tên chỉ số CV:
  - `CV AP`
  - `CV Precision (Mua)`
  - `CV Recall (Mua)`
  - `CV F1 (Mua)`
- Với Precision, Recall và F1, cần ghi rõ ngưỡng dự đoán được sử dụng hoặc quy trình chọn ngưỡng.
- Trên tập test, báo cáo AP và các chỉ số cần thiết. Theo quy ước của dự án, dùng `classification_report` và `confusion_matrix` để trình bày kết quả phân loại theo từng lớp.
- **Quy định về lưu trữ Confusion Matrix**: Thực hiện tính toán và trực quan hóa `confusion_matrix` trực tiếp trên tập Test trong notebook (nhằm quan sát phân bố các trường hợp phân loại nhầm tại ngưỡng đã chọn), nhưng **tuyệt đối không lưu kết quả ma trận nhầm lẫn (`confusion_matrix`)** vào file lưu trữ lịch sử thực nghiệm (`experiment_results.json`). File lịch sử chỉ lưu các chỉ số tóm tắt (AP, Accuracy, Precision, Recall, F1 theo từng nhãn cùng ngưỡng threshold).
- Trong bảng kết quả, thống nhất nhãn `0 = Không mua`, `1 = Mua`.

## 4. Quy ước định dạng các chỉ số đánh giá

- Các chỉ số dạng tỷ lệ như **AP, Precision (Mua), Recall (Mua), F1 (Mua)** được nhân với \(100\) và hiển thị dưới dạng phần trăm, với **2 chữ số thập phân**.
- Ví dụ: `0.1234` được trình bày thành `12.34%`.
- Khi báo cáo cross-validation (CV), tính **mean ± std** qua các fold. Cả mean và std đều được nhân với \(100\), làm tròn đến 2 chữ số thập phân và hiển thị dưới dạng phần trăm.
- Tên các chỉ số được thống nhất như sau:
  - `CV AP (%)`
  - `CV Precision (Mua) (%)`
  - `CV Recall (Mua) (%)`
  - `CV F1 (Mua) (%)`
- Trên tập test, áp dụng cùng quy ước định dạng phần trăm cho AP, Precision, Recall và F1.
- Với Precision, Recall và F1, cần ghi rõ ngưỡng dự đoán được sử dụng hoặc quy trình lựa chọn ngưỡng.
- **Lưu ý:** chỉ thay đổi cách hiển thị, không thay đổi giá trị thực tế được dùng trong tính toán hoặc so sánh mô hình.

## 5. Tóm tắt

**AP** là độ đo chính để so sánh mô hình; **PR curve** dùng để khảo sát ngưỡng; **Precision (Mua), Recall (Mua), F1 (Mua)** và **confusion matrix** dùng để đánh giá quyết định phân loại tại ngưỡng đã chọn. Không sử dụng ROC hoặc ROC-AUC.