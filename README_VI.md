# 🛒 Dự đoán ý định mua hàng trực tuyến (Online Shoppers' Purchasing Intention)
> **Ablation study & benchmarking**: Logistic Regression, XGBoost và PyTorch MLP trên dữ liệu bảng mất cân bằng (imbalanced tabular data).

---

## 📌 1. Tổng quan dự án

Dự án này nghiên cứu và xây dựng pipeline học máy / học sâu toàn diện để giải quyết bài toán phân loại nhị phân: **Dự đoán phiên truy cập của khách hàng có mang lại doanh thu mua hàng (`Revenue = True/False`) hay không** trên tập dữ liệu thương mại điện tử `online_shoppers_intention.csv` (12.330 phiên truy cập).

### Thách thức chính của bài toán:
- **Mất cân bằng lớp nghiêm trọng (severe class imbalance):** Chỉ ~15.47% phiên mang lại doanh thu (`Revenue = 1`).
- **Dữ liệu lệch phân phối cực mạnh (heavy skewness & outliers):** Các biến thời gian và số trang truy cập (`ProductRelated_Duration`, `PageValues`,...) lệch phải nặng.
- **Tương quan phức tạp giữa biến số & biến phân loại.**

### Phương pháp tiếp cận:
Thực hiện chuỗi thực nghiệm bóc tách (**ablation study**) từ **Exp 0 đến Exp 3**, kiểm soát chặt chẽ rò rỉ dữ liệu (**data leakage**), thẩm định chéo phân tầng lặp lại (**Repeated Stratified K-Fold: 4 folds × 3 repeats = 12 folds**) và đánh giá trên **Hold-out Test 20%** độc lập.

---

## 🚀 2. Tóm tắt quá trình thực nghiệm (Exp 0 ➔ Exp 3)

### 🔹 Exp 0: Baseline
- **Mục tiêu:** Thiết lập kết quả cơ sở ban đầu với bước tiền xử lý tối thiểu.
- **Tiền xử lý:** 
  - Biến số: Giữ nguyên thang đo thô (`passthrough`).
  - Biến phân loại: Mã hóa one-hot (`OneHotEncoder`).
- **Kết quả & nhận định:**
  - Mô hình tuyến tính và MLP gặp khó khăn do chênh lệch thang đo lớn (hàm mất mát dao động mạnh, hội tụ kém).
  - XGBoost vẫn thích ứng tốt nhờ bản chất dựa trên cây quyết định (không nhạy cảm với thang đo).

---

### 🔹 Exp 1: Feature transformation & scaling (Chuẩn hóa & xử lý độ lệch)
- **Mục tiêu:** Đánh giá độc lập tác động của việc chuẩn hóa thang đo và hiệu chỉnh phân phối lệch.
- **Thiết kế 2 nhánh thử nghiệm:**
  - **TN1 (Chỉ StandardScaler):** Đưa toàn bộ biến số về phân phối chuẩn chuẩn hóa Z-score ($\mu=0, \sigma=1$).
  - **TN2 (Log1p + StandardScaler):** Áp dụng phép biến đổi $\log(1 + x)$ trên 5 biến lệch mạnh (`Administrative`, `Administrative_Duration`, `ProductRelated`, `ProductRelated_Duration`, `PageValues`), sau đó chuẩn hóa `StandardScaler`.
- **Kết quả & nhận định:**
  - **TN2 vượt trội:** Giúp đường cong học tập (learning curves) của MLP giảm loss mượt mà, ổn định tuyệt đối và nâng Test AP lên đỉnh cao.
  - Logistic Regression cải thiện rõ rệt; XGBoost duy trì sự ổn định.

---

### 🔹 Exp 2: Feature importance ablation & LOFO analysis (Bóc tách tác động biên & hiện tượng che khuất)
- **Mục tiêu:** Đánh giá tác động biên (marginal impact) của từng đặc trưng và làm rõ hiện tượng che khuất đặc trưng (**feature shadowing**) gây ra bởi biến áp đảo `PageValues`.
- **Phương pháp thực hiện (Thẩm định trên 12-Fold Stratified CV của tập Train):**
  - **LOFO (Leave-One-Feature-Out) trên 17 biến:** Lần lượt loại bỏ từng biến và huấn luyện lại mô hình để đo mức độ suy giảm hiệu năng $\Delta\text{CV AP} = \text{CV AP}_{\text{bỏ}} - \text{CV AP}_{\text{chuẩn}}$ trên Logistic Regression và XGBoost.
  - **LOFO trên 16 biến (Không bao gồm `PageValues`):** Do có tới ~77.86% số phiên có `PageValues = 0`, việc loại bỏ `PageValues` giúp bóc tách và làm lộ diện vai trò thực sự của các biến hành vi còn lại đối với phần lớn người dùng.
- **Kết quả & nhận định:**
  - Khẳng định `PageValues` là biến quan trọng nhất nhưng gây ra hiện tượng che khuất mạnh đối với các biến khác.
  - Khi không có `PageValues`, các biến hành vi như `ExitRates`, `ProductRelated`, `Month` và `TrafficType` lập tức trở thành các nhân tố gánh vác chính trong việc dự đoán hành vi mua hàng.

---

### 🔹 Exp 3: Domain-driven feature generation (Tạo đặc trưng theo tri thức ngành)
- **Mục tiêu:** Đưa tri thức ngành (e-commerce domain knowledge) vào mô hình để tạo những cải thiện về hiệu năng.
- **16 đặc trưng hành vi được tạo mới:**
  - Nhóm tương tác trang: `Total_Pages`, `Total_Duration`, `Avg_Duration_Per_Page`,...
  - Nhóm tỷ lệ thoát & hành vi: `Bounce_to_Exit_Ratio`, `Exit_per_Page`,...
  - Nhóm giá trị kinh tế: `PageValue_per_ProductPage`, `High_Intent_Score`,...
- **Kết quả & nhận định:**
  - Cung cấp tín hiệu phân biệt mạnh mẽ giữa khách hàng "chỉ xem lướt" và khách hàng "có chủ đích mua hàng".
  - Cải thiện chỉ số đánh giá trên các mô hình.

---

## 📊 3. Bảng tổng hợp so sánh 3 mô hình

> **Độ đo chính:** **Average Precision (AP)** — Thước đo chuẩn mực cho bài toán dữ liệu mất cân bằng nghiêm trọng.

| Mô hình | Đặc điểm kiến trúc / Thuật toán | Test AP: Baseline (Exp 0) $\to$ Tốt nhất (Exp 3) | Nhận xét |
| :--- | :--- | :---: | :--- |
| **Logistic Regression** | Tuyến tính, L2 Regularization, Solver L-BFGS | $65.53\% \longrightarrow \mathbf{74.40\%}$ | Cải thiện mạnh nhất nhờ Log1p (Exp 1); Feature Engineering giúp tăng AP nhưng F1/Recall thực tế cải thiện không đáng kể. |
| **XGBoost** | Gradient Boosted Trees (100 cây, max depth 6) | $71.94\% \longrightarrow \mathbf{73.49\%}$ | Rất ổn định từ đầu, ít nhạy cảm với tiền xử lý, cân bằng tốc độ - hiệu năng. |
| **PyTorch MLP** | Mạng nơ-ron 2 lớp ẩn ($16 \to 8 \to 1$), ReLU, Dropout 0.2, Sigmoid, Adam | $66.88\% \longrightarrow \mathbf{75.49\%}$ | **Đạt AP cao nhất toàn bộ nghiên cứu**, học biểu diễn phi tuyến tốt nhất. |

---

## 🛠️ 4. Cấu trúc thư mục

```text
├── images/                  # Biểu đồ trực quan hóa
├── src/                     # Mã nguồn, dữ liệu và tài liệu
│   ├── eda.ipynb            # Phân tích khám phá dữ liệu
│   ├── exp0_baseline.ipynb   # Baseline
│   ├── exp1_transform_scale.ipynb  # Log1p & StandardScaler
│   ├── exp2_feature_ablation.ipynb # LOFO & Feature Shadowing
│   ├── exp3_feature_generation.ipynb # Feature Engineering
│   ├── logger.py            # MLPNet, huấn luyện, K-Fold CV & Logging
│   ├── visualization.py     # Các hàm trực quan hóa
│   ├── online_shoppers_intention.csv # Dữ liệu gốc
│   ├── dataset-description.md # Mô tả dữ liệu
│   ├── evaluation_skills.md # Quy chuẩn đánh giá
│   └── experiment_results.json # Kết quả thực nghiệm
├── report.pdf               # Báo cáo chi tiết
├── slide.pptx               # Slide thuyết trình
└── README.md                # Hướng dẫn dự án (hiển thị trang chủ GitHub)
```

---

## 💡 5. Hướng dẫn chạy nhanh (Quick Start)

### Cài đặt môi trường
```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
```

### Quy trình chạy thực nghiệm:
1. Mở và chạy `src/eda.ipynb` để nắm bức tranh tổng thể về phân phối dữ liệu và độ lệch.
2. Lần lượt chạy các notebook trong `src/`: `exp0` $\to$ `exp1` $\to$ `exp2` $\to$ `exp3`.

---

## 📚 6. Nguồn dữ liệu & Tham khảo (References)

- **Dataset Source:** [UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Bài báo gốc:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM neural networks*. Neural Comput & Applic 31, 6893–6908 (2019). [DOI: 10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0)
