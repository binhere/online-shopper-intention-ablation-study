# 🛒 Online Shoppers' Purchasing Intention Prediction
> **Ablation study & benchmarking**: Logistic Regression, XGBoost and PyTorch MLP on imbalanced tabular data.

---

## 📌 1. Project Overview

This project builds a comprehensive machine learning / deep learning pipeline for a binary classification problem: **predicting whether a customer's browsing session results in a purchase (`Revenue = True/False`)** on the e-commerce dataset `online_shoppers_intention.csv` (12,330 sessions).

### Main challenges:
- **Severe class imbalance:** Only ~15.47% of sessions generate revenue (`Revenue = 1`).
- **Heavy skewness & outliers:** Time and page-count variables (`ProductRelated_Duration`, `PageValues`, ...) are strongly right-skewed.
- **Complex relationships between numerical & categorical variables.**

### Approach:
A series of ablation experiments from **Exp 0 to Exp 3**, with strict control of **data leakage**, **Repeated Stratified K-Fold cross-validation (4 folds × 3 repeats = 12 folds)**, and evaluation on an independent **20% hold-out test set**.

---

## 🚀 2. Experiment Summary (Exp 0 ➔ Exp 3)

### 🔹 Exp 0: Baseline
- **Goal:** Establish baseline results with minimal preprocessing.
- **Preprocessing:** 
  - Numerical features: kept on their raw scale (`passthrough`).
  - Categorical features: one-hot encoding (`OneHotEncoder`).
- **Results & observations:**
  - The linear model and MLP struggle with large scale differences (loss oscillates strongly, poor convergence).
  - XGBoost still performs well thanks to its tree-based nature (insensitive to feature scale).

---

### 🔹 Exp 1: Feature transformation & scaling
- **Goal:** Independently evaluate the effect of feature scaling and skewed-distribution correction.
- **Two experimental branches:**
  - **TN1 (StandardScaler only):** Standardize all numerical features with Z-score ($\mu=0, \sigma=1$).
  - **TN2 (Log1p + StandardScaler):** Apply $\log(1 + x)$ to 5 highly skewed features (`Administrative`, `Administrative_Duration`, `ProductRelated`, `ProductRelated_Duration`, `PageValues`), then apply `StandardScaler`.
- **Results & observations:**
  - **TN2 is superior:** It makes the MLP learning curves smoother and more stable, and raises Test AP to the highest level.
  - Logistic Regression improves clearly; XGBoost remains stable.

---

### 🔹 Exp 2: Feature importance ablation & LOFO analysis
- **Goal:** Evaluate the marginal impact of each feature and clarify the **feature shadowing** phenomenon caused by the dominant variable `PageValues`.
- **Method (validated with 12-fold Stratified CV on the training set):**
  - **LOFO (Leave-One-Feature-Out) on 17 features:** Remove each feature in turn and retrain to measure the performance change $\Delta\text{CV AP} = \text{CV AP}_{\text{dropped}} - \text{CV AP}_{\text{baseline}}$ for Logistic Regression and XGBoost.
  - **LOFO on 16 features (excluding `PageValues`):** Since ~77.86% of sessions have `PageValues = 0`, removing `PageValues` reveals the true role of the remaining behavioral features for the majority of users.
- **Results & observations:**
  - Confirms `PageValues` is the most important feature, but it strongly shadows the other features.
  - Without `PageValues`, behavioral features such as `ExitRates`, `ProductRelated`, `Month` and `TrafficType` immediately become the main drivers for predicting purchase behavior.

---

### 🔹 Exp 3: Domain-driven feature generation
- **Goal:** Inject e-commerce domain knowledge into the model to improve performance.
- **16 newly generated behavioral features:**
  - Page interaction group: `Total_Pages`, `Total_Duration`, `Avg_Duration_Per_Page`, ...
  - Exit & behavior ratio group: `Bounce_to_Exit_Ratio`, `Exit_per_Page`, ...
  - Economic value group: `PageValue_per_ProductPage`, `High_Intent_Score`, ...
- **Results & observations:**
  - Provides a strong signal to distinguish "window-shopping" visitors from visitors with "purchase intent".
  - Improves the evaluation metrics across the models.

---

## 📊 3. Comparison of the 3 Models

> **Primary metric:** **Average Precision (AP)** — the standard metric for severely imbalanced data.

| Model | Architecture / Algorithm | Test AP: Baseline (Exp 0) $\to$ Best (Exp 3) | Remarks |
| :--- | :--- | :---: | :--- |
| **Logistic Regression** | Linear, L2 Regularization, L-BFGS solver | $65.53\% \longrightarrow \mathbf{74.40\%}$ | Biggest gain comes from Log1p (Exp 1); Feature Engineering raises AP but F1/Recall barely improve in practice. |
| **XGBoost** | Gradient Boosted Trees (100 trees, max depth 6) | $71.94\% \longrightarrow \mathbf{73.49\%}$ | Very stable from the start, insensitive to preprocessing, good balance of speed and performance. |
| **PyTorch MLP** | 2-hidden-layer neural network ($16 \to 8 \to 1$), ReLU, Dropout 0.2, Sigmoid, Adam | $66.88\% \longrightarrow \mathbf{75.49\%}$ | **Highest AP of the entire study**, best at learning non-linear representations. |

---

## 🛠️ 4. Repository Structure

```text
├── images/                  # Visualization charts
├── src/                     # Source code, data and documents
│   ├── eda.ipynb            # Exploratory data analysis
│   ├── exp0_baseline.ipynb   # Baseline
│   ├── exp1_transform_scale.ipynb  # Log1p & StandardScaler
│   ├── exp2_feature_ablation.ipynb # LOFO & Feature Shadowing
│   ├── exp3_feature_generation.ipynb # Feature Engineering
│   ├── logger.py            # MLPNet, training, K-Fold CV & Logging
│   ├── visualization.py     # Visualization functions
│   ├── online_shoppers_intention.csv # Raw dataset
│   ├── dataset-description.md # Dataset description
│   ├── evaluation_skills.md # Evaluation guidelines
│   └── experiment_results.json # Experiment results
├── report.pdf               # Detailed report
├── slide.pptx               # Presentation slides
└── README.md                # Project guide (shown on the GitHub home page)
```

---

## 💡 5. Quick Start

### Environment setup
```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
```

### Running the experiments:
1. Open and run `src/eda.ipynb` to get an overview of the data distribution and skewness.
2. Run the notebooks in `src/` in order: `exp0` $\to$ `exp1` $\to$ `exp2` $\to$ `exp3`.

---

## 📚 6. Data Source & References

- **Dataset Source:** [UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Original paper:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM neural networks*. Neural Comput & Applic 31, 6893–6908 (2019). [DOI: 10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0)

