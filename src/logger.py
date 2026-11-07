"""
Module: logger.py
Mục đích: Chuẩn hóa quy trình đánh giá, ghi nhận và so sánh kết quả thực nghiệm một cách có hệ thống.

Tuân thủ nghiêm ngặt theo tài liệu 'huong_dan_danh_gia_mo_hinh.md':
1. Bài toán: Phân loại nhị phân Revenue (0 = Không mua / Non-purchase, 1 = Mua / Purchase).
2. Độ đo chính: Average Precision (AP) tính bằng `average_precision_score`.
   - Tuyệt đối không sử dụng ROC curve hoặc ROC-AUC.
   - Dùng tên 'AP' cho giá trị tính bằng `average_precision_score`, không gọi là 'PR-AUC'.
   - Không sử dụng `auc(recall, precision)`.
3. Đánh giá Cross-Validation (CV):
   - Báo cáo định dạng 'mean ± std' qua các fold:
     - `CV AP (%)`
     - `CV Precision (Mua) (%)`
     - `CV Recall (Mua) (%)`
     - `CV F1 (Mua) (%)`
4. Ngưỡng quyết định (Threshold):
   - Ghi nhận rõ ngưỡng dự đoán được sử dụng cho Precision, Recall, F1 (mặc định: 0.5).
5. Đánh giá tập Test:
   - Báo cáo AP và bảng phân loại theo lớp (`classification_report` và `confusion_matrix`).
   - Thống nhất nhãn: 0 = Không mua, 1 = Mua.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

DEFAULT_JSON_PATH = 'experiment_results.json'


# ==============================================================================
# 1. BỘ ĐỘ ĐO CHUẨN CHO CROSS-VALIDATION (SCORERS)
# ==============================================================================

def get_cv_scorers(pos_label: int = 1) -> Dict[str, Any]:
    """
    Trả về dictionary các scorers chuẩn hóa cho sklearn.model_selection.cross_validate.

    Quy chuẩn:
    - 'ap': average_precision_score (Độ đo chính để so sánh mô hình)
    - 'prec_1', 'rec_1', 'f1_1': Precision, Recall, F1 cho lớp 1 (Mua)
    - 'prec_0', 'rec_0', 'f1_0': Precision, Recall, F1 cho lớp 0 (Không mua)
    - Tuyệt đối không chứa roc_auc.
    """
    return {
        'ap': make_scorer(average_precision_score, response_method='predict_proba'),
        'prec_1': make_scorer(precision_score, pos_label=pos_label, zero_division=0),
        'rec_1': make_scorer(recall_score, pos_label=pos_label, zero_division=0),
        'f1_1': make_scorer(f1_score, pos_label=pos_label, zero_division=0),
        'prec_0': make_scorer(precision_score, pos_label=0, zero_division=0),
        'rec_0': make_scorer(recall_score, pos_label=0, zero_division=0),
        'f1_0': make_scorer(f1_score, pos_label=0, zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    }


# ==============================================================================
# 2. XỬ LÝ VÀ TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (CV & TEST)
# ==============================================================================

def summarize_cv_results(cv_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trích xuất kết quả từ output của `cross_validate` thành cấu trúc chuẩn:
    Bao gồm mean và std của:
    - AP (Average Precision)
    - Precision, Recall, F1 cho Lớp 1 (Mua)
    - Precision, Recall, F1 cho Lớp 0 (Không mua)
    """
    summary = {}

    ap_key = None
    if 'test_ap' in cv_output:
        ap_key = 'test_ap'
    elif 'test_pr_auc' in cv_output:
        ap_key = 'test_pr_auc'

    if ap_key:
        summary['ap_mean'] = round(float(np.mean(cv_output[ap_key])), 4)
        summary['ap_std'] = round(float(np.std(cv_output[ap_key])), 4)
    else:
        summary['ap_mean'] = None
        summary['ap_std'] = None

    if 'test_accuracy' in cv_output:
        summary['accuracy_mean'] = round(float(np.mean(cv_output['test_accuracy'])), 4)
        summary['accuracy_std'] = round(float(np.std(cv_output['test_accuracy'])), 4)

    # Lớp 1 (Mua)
    summary['class_1'] = {
        'label': 'Mua',
        'precision_mean': round(float(np.mean(cv_output['test_prec_1'])), 4) if 'test_prec_1' in cv_output else None,
        'precision_std': round(float(np.std(cv_output['test_prec_1'])), 4) if 'test_prec_1' in cv_output else None,
        'recall_mean': round(float(np.mean(cv_output['test_rec_1'])), 4) if 'test_rec_1' in cv_output else None,
        'recall_std': round(float(np.std(cv_output['test_rec_1'])), 4) if 'test_rec_1' in cv_output else None,
        'f1_mean': round(float(np.mean(cv_output['test_f1_1'])), 4) if 'test_f1_1' in cv_output else None,
        'f1_std': round(float(np.std(cv_output['test_f1_1'])), 4) if 'test_f1_1' in cv_output else None,
    }

    # Lớp 0 (Không mua)
    summary['class_0'] = {
        'label': 'Không mua',
        'precision_mean': round(float(np.mean(cv_output['test_prec_0'])), 4) if 'test_prec_0' in cv_output else None,
        'precision_std': round(float(np.std(cv_output['test_prec_0'])), 4) if 'test_prec_0' in cv_output else None,
        'recall_mean': round(float(np.mean(cv_output['test_rec_0'])), 4) if 'test_rec_0' in cv_output else None,
        'recall_std': round(float(np.std(cv_output['test_rec_0'])), 4) if 'test_rec_0' in cv_output else None,
        'f1_mean': round(float(np.mean(cv_output['test_f1_0'])), 4) if 'test_f1_0' in cv_output else None,
        'f1_std': round(float(np.std(cv_output['test_f1_0'])), 4) if 'test_f1_0' in cv_output else None,
    }

    return summary


def evaluate_test(
    y_test: Union[np.ndarray, pd.Series, list],
    y_proba: Union[np.ndarray, pd.Series, list],
    threshold: float = 0.5,
    y_pred: Optional[Union[np.ndarray, pd.Series, list]] = None
) -> Dict[str, Any]:
    """
    Đánh giá mô hình trên tập Test theo đúng quy chuẩn:
    - AP: average_precision_score
    - Classification Report theo từng lớp (0 = Không mua, 1 = Mua)
    - Ngưỡng quyết định threshold
    - Lưu ý: Không lưu trữ confusion_matrix vào file JSON (chỉ tính và trực quan hóa trong notebook).
    """
    y_test_arr = np.asarray(y_test).astype(int)
    y_proba_arr = np.asarray(y_proba)
    if y_proba_arr.ndim == 2:
        y_proba_arr = y_proba_arr[:, 1]

    if y_pred is None:
        y_pred_arr = (y_proba_arr >= threshold).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred).astype(int)

    # 1. Chỉ số AP chính
    ap = average_precision_score(y_test_arr, y_proba_arr)

    # 2. Classification Report
    report = classification_report(
        y_test_arr, y_pred_arr,
        output_dict=True,
        zero_division=0
    )

    c1 = report.get('1', {})
    c0 = report.get('0', {})

    return {
        'ap': round(float(ap), 4),
        'threshold': round(float(threshold), 4),
        'accuracy': round(float(report.get('accuracy', 0.0)), 4),
        'class_1': {
            'label': 'Mua',
            'precision': round(float(c1.get('precision', 0.0)), 4),
            'recall': round(float(c1.get('recall', 0.0)), 4),
            'f1_score': round(float(c1.get('f1-score', 0.0)), 4),
            'support': int(c1.get('support', 0))
        },
        'class_0': {
            'label': 'Không mua',
            'precision': round(float(c0.get('precision', 0.0)), 4),
            'recall': round(float(c0.get('recall', 0.0)), 4),
            'f1_score': round(float(c0.get('f1-score', 0.0)), 4),
            'support': int(c0.get('support', 0))
        },
        'f1_macro': round(float(report.get('macro avg', {}).get('f1-score', 0.0)), 4)
    }


# ==============================================================================
# 3. GHI NHẬN THỰC NGHIỆM VÀO FILE JSON
# ==============================================================================

def log_experiment(
    exp_id: str,
    model_name: str,
    notebook: str,
    description: str,
    config: Dict[str, Any],
    cv_results: Optional[Dict[str, Any]] = None,
    y_test: Optional[Union[np.ndarray, pd.Series, list]] = None,
    y_proba: Optional[Union[np.ndarray, pd.Series, list]] = None,
    y_pred: Optional[Union[np.ndarray, pd.Series, list]] = None,
    threshold: float = 0.5,
    test_results: Optional[Dict[str, Any]] = None,
    json_path: str = DEFAULT_JSON_PATH
) -> Dict[str, Any]:
    """
    Ghi nhận hoặc cập nhật kết quả của một thực nghiệm vào file JSON.

    Tự động chuẩn hóa:
    - Xử lý cv_results qua summarize_cv_results (nếu truyền vào dict thô từ cross_validate)
    - Tự động tính toán test_results nếu truyền (y_test, y_proba)
    - Tuân thủ quy chuẩn AP, lược bỏ ROC-AUC, định danh rõ ngưỡng threshold.
    """
    # 1. Xử lý khối Test Results
    if test_results is None and y_test is not None and y_proba is not None:
        test_results = evaluate_test(y_test, y_proba, threshold=threshold, y_pred=y_pred)
    elif test_results is None:
        test_results = {}

    # 2. Xử lý khối CV Results
    cv_summary = {}
    if cv_results is not None:
        if 'test_ap' in cv_results or 'test_pr_auc' in cv_results or 'test_f1_1' in cv_results:
            cv_summary = summarize_cv_results(cv_results)
        else:
            cv_summary = cv_results

    # 3. Tạo entry thực nghiệm chuẩn mực
    entry = {
        'meta': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'notebook': notebook,
            'description': description
        },
        'config': config,
        'cv_results': cv_summary,
        'test_results': test_results
    }

    # 4. Đọc file JSON hiện có và cập nhật
    all_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    all_results[exp_id] = entry

    # 5. Lưu lại vào file JSON (UTF-8)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    try:
        print(f"-> Đã lưu thành công thực nghiệm '{exp_id}' vào {json_path}")
    except UnicodeEncodeError:
        print(f"-> [OK] Da luu thanh cong thuc nghiem '{exp_id}' vao {json_path}")

    return entry


# ==============================================================================
# 4. TRUY VẤN VÀ BÁO CÁO BẢNG SO SÁNH
# ==============================================================================

def load_experiments_json(json_path: str = DEFAULT_JSON_PATH) -> Dict[str, Any]:
    """Đọc toàn bộ file kết quả và trả về dictionary Python nguyên bản."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _fmt(val_mean: Optional[float], val_std: Optional[float] = None, as_percent: bool = True) -> str:
    """Format chỉ số dạng 'mean ± std' (%) hoặc '-' nếu rỗng."""
    if val_mean is None or np.isnan(val_mean):
        return '-'
    scale = 100.0 if as_percent else 1.0
    if val_std is not None and not np.isnan(val_std) and val_std > 0:
        return f"{val_mean * scale:.2f} ± {val_std * scale:.2f}"
    return f"{val_mean * scale:.2f}"


def load_experiments_df(
    json_path: str = DEFAULT_JSON_PATH,
    view: str = 'cv',
    as_percent: bool = True
) -> pd.DataFrame:
    """
    Đọc toàn bộ file JSON và xuất thành DataFrame chuẩn mực theo 'huong_dan_danh_gia_mo_hinh.md'.

    Parameters:
    -----------
    json_path : str
        Đường dẫn file kết quả JSON.
    view : str
        'cv': Bảng chi tiết toàn bộ chỉ số Cross-Validation (mean ± std) cho cả 2 nhãn.
        'test': Bảng chi tiết các chỉ số trên tập Test cho cả 2 nhãn.
    as_percent : bool
        Nếu True, nhân 100 hiển thị %. Mặc định True.
    """
    if not os.path.exists(json_path):
        try:
            print(f"Chưa tìm thấy file {json_path}")
        except UnicodeEncodeError:
            print(f"Chua tim thay file {json_path}")
        return pd.DataFrame()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pct = " (%)" if as_percent else ""
    scale = 100.0 if as_percent else 1.0
    rows = []

    for exp_id, item in data.items():
        meta = item.get('meta', {})
        cv = item.get('cv_results', {})
        test = item.get('test_results', {})

        if view == 'cv':
            cv_ap_mean = cv.get('ap_mean', cv.get('pr_auc_mean'))
            cv_ap_std = cv.get('ap_std', 0.0)
            cv_c1 = cv.get('class_1', {})
            cv_c0 = cv.get('class_0', {})

            rows.append({
                'Experiment ID': exp_id,
                'Model': meta.get('model_name', ''),
                'Notebook': meta.get('notebook', ''),
                f'CV AP{pct}': _fmt(cv_ap_mean, cv_ap_std, as_percent=as_percent),
                f'CV Precision (Mua){pct}': _fmt(cv_c1.get('precision_mean'), cv_c1.get('precision_std'), as_percent=as_percent),
                f'CV Recall (Mua){pct}': _fmt(cv_c1.get('recall_mean'), cv_c1.get('recall_std'), as_percent=as_percent),
                f'CV F1 (Mua){pct}': _fmt(cv_c1.get('f1_mean'), cv_c1.get('f1_std'), as_percent=as_percent),
                f'CV Precision (Không mua){pct}': _fmt(cv_c0.get('precision_mean'), cv_c0.get('precision_std'), as_percent=as_percent),
                f'CV Recall (Không mua){pct}': _fmt(cv_c0.get('recall_mean'), cv_c0.get('recall_std'), as_percent=as_percent),
                f'CV F1 (Không mua){pct}': _fmt(cv_c0.get('f1_mean'), cv_c0.get('f1_std'), as_percent=as_percent),
                f'CV F1 Macro{pct}': _fmt(
                    (cv_c1.get('f1_mean', 0) + cv_c0.get('f1_mean', 0)) / 2 if cv_c1.get('f1_mean') is not None and cv_c0.get('f1_mean') is not None else None,
                    np.sqrt(np.array(cv_c1.get('f1_std', 0))**2 + np.array(cv_c0.get('f1_std', 0))**2) / 2 if cv_c1.get('f1_std') is not None else 0.0,
                    as_percent=as_percent
                ),
            })
        elif view == 'test':
            test_ap = test.get('ap', test.get('pr_auc'))
            test_th = test.get('threshold', 0.5)
            test_c1 = test.get('class_1', {})
            test_c0 = test.get('class_0', {})
            f1_1_val = test_c1.get('f1_score')
            f1_0_val = test_c0.get('f1_score')
            macro_val = test.get('f1_macro')
            if macro_val is None and f1_1_val is not None and f1_0_val is not None:
                macro_val = (f1_1_val + f1_0_val) / 2

            rows.append({
                'Experiment ID': exp_id,
                'Model': meta.get('model_name', ''),
                'Notebook': meta.get('notebook', ''),
                'Ngưỡng (Threshold)': test_th if test else '-',
                f'Test AP{pct}': f"{test_ap * scale:.2f}" if test_ap is not None else '-',
                f'Accuracy{pct}': f"{test.get('accuracy') * scale:.2f}" if test.get('accuracy') is not None else '-',
                f'Test F1 (Không mua){pct}': f"{f1_0_val * scale:.2f}" if f1_0_val is not None else '-',
                f'Test F1 (Mua){pct}': f"{f1_1_val * scale:.2f}" if f1_1_val is not None else '-',
                f'Test F1 Macro{pct}': f"{macro_val * scale:.2f}" if macro_val is not None else '-',
                f'Test Precision (Mua){pct}': f"{test_c1.get('precision') * scale:.2f}" if test_c1.get('precision') is not None else '-',
                f'Test Recall (Mua){pct}': f"{test_c1.get('recall') * scale:.2f}" if test_c1.get('recall') is not None else '-',
                f'Test Precision (Không mua){pct}': f"{test_c0.get('precision') * scale:.2f}" if test_c0.get('precision') is not None else '-',
                f'Test Recall (Không mua){pct}': f"{test_c0.get('recall') * scale:.2f}" if test_c0.get('recall') is not None else '-',
            })

    return pd.DataFrame(rows)


# ==============================================================================
# 5. TIỆN ÍCH HỖ TRỢ DEEP LEARNING (PYTORCH MLP)
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import clone
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPNet(nn.Module):
    """
    Kiến trúc mạng nơ-ron đa tầng MLP chuẩn hóa cho dữ liệu dạng bảng:
    Layer 1: Linear(input_dim, 16) -> ReLU() -> Dropout(p=0.2)
    Layer 2: Linear(16, 8) -> ReLU() -> Dropout(p=0.2)
    Output:  Linear(8, 1) -> Sigmoid()
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_pytorch_mlp(
    X_tr_raw,
    y_tr_raw,
    preprocessor_base,
    lr: float = 0.002,
    weight_decay: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 50,
    patience: int = 8,
    random_state: int = 42,
    test_size: float = 0.15,
    return_history: bool = False
):
    """
    Huấn luyện mô hình PyTorch MLP kèm Early Stopping và khôi phục Best Checkpoint:
    - Tiền xử lý độc lập trên X_tr_raw chống Data Leakage
    - Tách test_size (15% hoặc 20%) làm validation nội bộ để Early Stopping
    - Khi return_history=True: Thu thập lịch sử Loss, AP, F1-1, F1-Macro qua từng epoch
    """
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)

    # 1. Fit bộ tiền xử lý độc lập
    prep = clone(preprocessor_base)
    X_proc = prep.fit_transform(X_tr_raw)
    X_arr = X_proc.toarray() if hasattr(X_proc, 'toarray') else np.asarray(X_proc, dtype=np.float32)
    y_arr = np.asarray(y_tr_raw, dtype=np.float32)

    tr_pct = int(round((1.0 - test_size) * 100))
    val_pct = int(round(test_size * 100))

    # 2. Tách validation nội bộ
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_arr, test_size=test_size, stratify=y_arr, random_state=random_state
    )

    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if return_history:
        tr_eval_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)

    model = MLPNet(X_arr.shape[1]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    best_weights = None
    best_epoch = 0
    patience_count = 0

    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_ap': [],   'val_ap': [],
        'train_f1_1': [], 'val_f1_1': [],
        'train_f1_macro': [], 'val_f1_macro': [],
        'best_epoch': 0,
        'tr_pct': tr_pct,
        'val_pct': val_pct
    }

    for epoch in range(1, epochs + 1):
        # A. Giai đoạn Train
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        # B. Giai đoạn Eval
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_probs_list = []
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
                val_probs_list.append(out.cpu().numpy())
            val_loss /= len(X_val)

            if return_history:
                val_probs = np.concatenate(val_probs_list)
                val_preds = (val_probs >= 0.5).astype(int)
                val_ap = average_precision_score(y_val, val_probs)
                val_f1_1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
                val_f1_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)

                tr_loss = 0.0
                tr_probs_list = []
                for bx, by in tr_eval_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    out = model(bx)
                    tr_loss += criterion(out, by).item() * len(by)
                    tr_probs_list.append(out.cpu().numpy())
                tr_loss /= len(X_tr)
                tr_probs = np.concatenate(tr_probs_list)
                tr_preds = (tr_probs >= 0.5).astype(int)
                tr_ap = average_precision_score(y_tr, tr_probs)
                tr_f1_1 = f1_score(y_tr, tr_preds, pos_label=1, zero_division=0)
                tr_f1_macro = f1_score(y_tr, tr_preds, average='macro', zero_division=0)

                history['epoch'].append(epoch)
                history['train_loss'].append(tr_loss)
                history['val_loss'].append(val_loss)
                history['train_ap'].append(tr_ap)
                history['val_ap'].append(val_ap)
                history['train_f1_1'].append(tr_f1_1)
                history['val_f1_1'].append(val_f1_1)
                history['train_f1_macro'].append(tr_f1_macro)
                history['val_f1_macro'].append(val_f1_macro)

        # C. Kiểm tra Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    history['best_epoch'] = best_epoch

    if return_history:
        return prep, model, history
    return prep, model


def predict_proba_mlp(prep, model, X_raw) -> np.ndarray:
    """
    Dự đoán xác suất P(Revenue = 1) trên dữ liệu thô cho mô hình MLP PyTorch:
    - Tự động gọi model.eval() tắt Dropout
    - prep.transform(X_raw) tiền xử lý dữ liệu
    - Trả về mảng numpy 1D xác suất lớp 1
    """
    model.eval()
    X_proc = prep.transform(X_raw)
    X_arr = X_proc.toarray() if hasattr(X_proc, 'toarray') else np.asarray(X_proc, dtype=np.float32)
    with torch.no_grad():
        tensor_x = torch.tensor(X_arr, dtype=torch.float32).to(DEVICE)
        proba = model(tensor_x).cpu().numpy()
    return proba


def cross_validate_mlp(
    X_data,
    y_data,
    preprocessor_base,
    cv_splitter,
    random_state: int = 42,
    test_size: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    Thực hiện Cross-Validation trực tiếp cho mô hình MLP PyTorch:
    - Lặp qua từng fold của cv_splitter (ví dụ RepeatedStratifiedKFold)
    - Trả về dictionary kết quả 12 folds chuẩn hóa tương thích log_experiment
    """
    scores = {
        'test_ap': [], 'test_prec_1': [], 'test_rec_1': [], 'test_f1_1': [],
        'test_prec_0': [], 'test_rec_0': [], 'test_f1_0': [], 'test_f1_macro': []
    }
    for fold_idx, (tr_idx, val_idx) in enumerate(cv_splitter.split(X_data, y_data), 1):
        X_fold_tr, y_fold_tr = X_data.iloc[tr_idx], y_data.iloc[tr_idx]
        X_fold_val, y_fold_val = X_data.iloc[val_idx], y_data.iloc[val_idx]

        prep_f, model_f = train_pytorch_mlp(
            X_fold_tr, y_fold_tr, preprocessor_base, random_state=random_state,
            test_size=test_size, return_history=False
        )
        prob_val = predict_proba_mlp(prep_f, model_f, X_fold_val)
        pred_val = (prob_val >= 0.5).astype(int)

        scores['test_ap'].append(average_precision_score(y_fold_val, prob_val))
        scores['test_prec_1'].append(precision_score(y_fold_val, pred_val, pos_label=1, zero_division=0))
        scores['test_rec_1'].append(recall_score(y_fold_val, pred_val, pos_label=1, zero_division=0))
        scores['test_f1_1'].append(f1_score(y_fold_val, pred_val, pos_label=1, zero_division=0))
        scores['test_prec_0'].append(precision_score(y_fold_val, pred_val, pos_label=0, zero_division=0))
        scores['test_rec_0'].append(recall_score(y_fold_val, pred_val, pos_label=0, zero_division=0))
        scores['test_f1_0'].append(f1_score(y_fold_val, pred_val, pos_label=0, zero_division=0))
        scores['test_f1_macro'].append(f1_score(y_fold_val, pred_val, average='macro', zero_division=0))

    return {k: np.array(v) for k, v in scores.items()}
