"""
Module: visualization.py
Mục đích: Tiện ích trực quan hóa (Visualization Helpers) theo chuẩn của dự án.

Tuân thủ nghiêm ngặt 'huong_dan_danh_gia_mo_hinh.md':
- Confusion Matrix: Hiển thị định dạng số nguyên 'd', gán nhãn rõ ràng: 0 = 'Không mua', 1 = 'Mua'.
- PR Curve: Sử dụng precision_recall_curve, ghi rõ điểm AP trong chú giải, không gọi là PR-AUC.
  Không sử dụng ROC curve hoặc ROC-AUC.
"""

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_confusion_matrix(
    cm: Union[np.ndarray, list],
    labels: Tuple[str, str] = ('Không mua (0)', 'Mua (1)'),
    title: str = 'Confusion Matrix',
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    model_name: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) chuẩn mực với số nguyên định dạng 'd'.

    Parameters:
    -----------
    cm : array-like
        Ma trận nhầm lẫn 2x2.
    labels : Tuple[str, str]
        Tên hiển thị cho các lớp: ('Không mua (0)', 'Mua (1)').
    title : str
        Tiêu đề đồ thị.
    threshold : Optional[float]
        Ngưỡng phân loại được dùng để tạo ma trận nhầm lẫn này (nếu có).
    ax : Optional[plt.Axes]
        Axes matplotlib vẽ lên (tùy chọn).
    cmap : str
        Bảng màu Seaborn. Mặc định 'Blues'.
    model_name : Optional[str]
        Tên mô hình (tùy chọn).
    """
    if model_name is not None and title == 'Confusion Matrix':
        title = f"Confusion Matrix - {model_name}"

    cm_arr = np.asarray(cm)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_arr, annot=True, fmt='d', cmap=cmap, cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax,
        annot_kws={'size': 11, 'weight': 'bold'}
    )

    full_title = title if threshold is None else f"{title} (Ngưỡng = {threshold})"
    ax.set_title(full_title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Nhãn dự đoán (Predicted)', fontsize=10)
    ax.set_ylabel('Nhãn thực tế (Actual)', fontsize=10)
    return ax


def plot_pr_curve(
    y_test: Union[np.ndarray, pd.Series, list],
    y_proba: Union[np.ndarray, pd.Series, list],
    model_name: str = 'Mô hình',
    ax: Optional[plt.Axes] = None,
    color: str = '#2980B9'
) -> Tuple[plt.Axes, float]:
    """
    Vẽ Precision-Recall Curve bằng `precision_recall_curve` và ghi rõ điểm AP trong chú giải.
    Tuyệt đối không gọi là PR-AUC.

    Parameters:
    -----------
    y_test : array-like
        Nhãn thực tế (0 hoặc 1).
    y_proba : array-like
        Xác suất dự đoán cho lớp 1 (hoặc ma trận xác suất 2 cột).
    model_name : str
        Tên mô hình để hiển thị trong chú giải legend.
    ax : Optional[plt.Axes]
        Axes matplotlib vẽ lên (tùy chọn).
    color : str
        Màu đường cong.
    """
    y_test_arr = np.asarray(y_test).astype(int)
    y_proba_arr = np.asarray(y_proba)
    if y_proba_arr.ndim == 2:
        y_proba_arr = y_proba_arr[:, 1]

    prec, rec, _ = precision_recall_curve(y_test_arr, y_proba_arr)
    ap = average_precision_score(y_test_arr, y_proba_arr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(rec, prec, label=f"{model_name} (AP = {ap:.4f})", color=color, lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Độ nhạy - Lớp Mua)', fontsize=10)
    ax.set_ylabel('Precision (Độ chính xác - Lớp Mua)', fontsize=10)
    ax.set_title(f'Precision-Recall (PR) Curve - {model_name}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3, linestyle=':')

    return ax, ap


def plot_learning_curves(
    history: dict,
    model_title: str = "MLP"
) -> plt.Figure:
    """
    Trực quan hóa tiến trình huấn luyện (Learning Curves) cho mô hình PyTorch MLP.
    Hiển thị lưới 2x2 gồm:
    1. Binary Cross-Entropy Loss (Train vs Val)
    2. Average Precision (AP - Độ đo chính) (%) (Train vs Val)
    3. F1-Score Lớp Mua (1) (%) (Train vs Val)
    4. F1-Macro (%) (Train vs Val)
    Đánh dấu Best Checkpoint bằng đường nét đứt màu đỏ.
    """
    epochs = history['epoch']
    best_ep = history.get('best_epoch', 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    metrics = [
        ('Loss (Binary Cross-Entropy)', 'train_loss', 'val_loss', axes[0, 0], False),
        ('Average Precision (AP - Độ đo chính)', 'train_ap', 'val_ap', axes[0, 1], True),
        ('F1-Score (Lớp Mua - 1)', 'train_f1_1', 'val_f1_1', axes[1, 0], True),
        ('F1-Macro (Cân bằng 2 lớp)', 'train_f1_macro', 'val_f1_macro', axes[1, 1], True)
    ]

    tr_pct = history.get('tr_pct', 80)
    val_pct = history.get('val_pct', 20)

    for title, tr_key, val_key, ax_i, is_percent in metrics:
        scale = 100.0 if is_percent else 1.0
        unit = " (%)" if is_percent else ""

        tr_vals = [v * scale for v in history[tr_key]]
        val_vals = [v * scale for v in history[val_key]]

        ax_i.plot(epochs, tr_vals, label=f'Train ({tr_pct}%)', color='#2980B9', lw=2)
        ax_i.plot(epochs, val_vals, label=f'Validation nội bộ ({val_pct}%)', color='#E67E22', lw=2)

        # Đánh dấu Best Checkpoint
        ax_i.axvline(best_ep, color='#C0392B', linestyle='--', lw=1.5,
                     label=f'Best Checkpoint (Epoch {best_ep})')

        ax_i.set_title(f"{title}", fontsize=11, fontweight='bold')
        ax_i.set_xlabel("Epoch", fontsize=10)
        ax_i.set_ylabel(f"Giá trị{unit}", fontsize=10)
        ax_i.grid(True, linestyle=':', alpha=0.6)
        ax_i.legend(loc='best', fontsize=9.5)

    plt.suptitle(f"TIẾN TRÌNH HUẤN LUYỆN (LEARNING CURVES) - {model_title.upper()}",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
