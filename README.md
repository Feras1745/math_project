# Project 52: Network Intrusion Detection Using Random Forest on NSL-KDD

## Overview
A complete end-to-end Machine Learning pipeline for Network Intrusion Detection using the NSL-KDD benchmark dataset. The system classifies network connections into five categories: **Normal, DoS, Probe, R2L, and U2R** using a tuned Random Forest classifier with SHAP-based explainability.

## Results Summary

| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.9985 | 0.9985 | 0.9985 | 0.9985 |
| Decision Tree (Baseline) | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| Random Forest (Default) | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| **Random Forest (Tuned) ★** | **0.9999** | **0.9999** | **0.9999** | **0.9999** |

**Mean AUC: 1.0000** (perfect ROC across all 5 attack classes)

---

## Project Structure

```
nsl_kdd_project/
├── data/
│   ├── KDDTrain.csv              # 125,973 training records
│   └── KDDTest.csv               # 22,544 test records
├── code/
│   ├── NSL_KDD_Intrusion_Detection.ipynb   # Main Jupyter notebook
│   └── nsl_kdd_pipeline.py                 # Standalone Python script
├── models/
│   ├── best_random_forest.joblib           # Trained RF model
│   ├── label_encoder.joblib                # LabelEncoder
│   ├── scaler.joblib                       # StandardScaler
│   ├── feature_names.joblib                # Feature column names
│   └── results_summary.json               # Performance metrics
├── figures/                               # 12 publication-quality figures
│   ├── 01_class_distribution.png
│   ├── 02_protocol_flag.png
│   ├── 03_feature_distributions.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_pca_visualization.png
│   ├── 06_confusion_matrix.png
│   ├── 07_classification_report.png
│   ├── 08_roc_curves.png
│   ├── 09_feature_importance.png
│   ├── 10_model_comparison.png
│   ├── 11_shap_importance.png
│   └── 12_learning_curve.png
├── Technical_Report.docx                  # 15-page IEEE-format report
├── Presentation_Slides.pptx               # 12-slide presentation
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

**Option A – Jupyter Notebook (recommended for interactive exploration):**
```bash
jupyter notebook code/NSL_KDD_Intrusion_Detection.ipynb
```

**Option B – Python script (for full automated run):**
```bash
python code/nsl_kdd_pipeline.py
```

### 3. Load the Trained Model for Inference

```python
import joblib
import numpy as np

# Load artifacts
model   = joblib.load('models/best_random_forest.joblib')
le      = joblib.load('models/label_encoder.joblib')
scaler  = joblib.load('models/scaler.joblib')

# Predict (replace X_new with your feature matrix)
X_scaled = scaler.transform(X_new)
y_pred   = model.predict(X_scaled)
labels   = le.inverse_transform(y_pred)
print(labels)  # e.g., ['normal', 'DoS', 'Probe', ...]
```

---

## Dataset

The **NSL-KDD** dataset is an improved version of KDD Cup 1999 that:
- Removes duplicate records (which biased previous evaluations)
- Stratifies difficulty levels across train/test splits
- Contains **41 raw features** + class label

| Split | Records | Normal | DoS | Probe | R2L | U2R |
|---|---|---|---|---|---|---|
| Train | 125,973 | 67,343 | 45,927 | 11,656 | 995 | 52 |
| Test  | 22,544  | 9,711  | 7,458  | 2,421  | 2,754 | 200 |

**Source:** [NSL-KDD at Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)

---

## Feature Engineering

Six domain-informed features were added to the 41 raw features:

| Feature | Formula | Rationale |
|---|---|---|
| `bytes_ratio` | `(src_bytes+1)/(dst_bytes+1)` | Asymmetric byte flow → DoS signature |
| `error_rate_combined` | `serror_rate + rerror_rate` | Combined error signal → flood detection |
| `srv_ratio` | `(count+1)/(srv_count+1)` | Service spread → Probe scanning |
| `host_srv_ratio` | `(dst_host_count+1)/(dst_host_srv_count+1)` | Host diversity → port scan |
| `log_src_bytes` | `log1p(src_bytes)` | Reduce skewness → improve LR |
| `log_dst_bytes` | `log1p(dst_bytes)` | Same as log_src_bytes |

---

## Hyperparameter Tuning

GridSearchCV (3-fold stratified CV) over:
```python
param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [None, 30],
    'min_samples_split': [2, 5],
    'max_features':      ['sqrt', 'log2']
}
```
**Best parameters:** `n_estimators=100, max_depth=None, max_features='sqrt', min_samples_split=2`

---

## SHAP Explainability (Innovation)

SHAP (SHapley Additive exPlanations) analysis reveals that error-rate features dominate predictions:

1. `error_rate_combined` (engineered) — most impactful
2. `dst_host_serror_rate`
3. `dst_host_srv_serror_rate`
4. `dst_host_same_src_port_rate`
5. `serror_rate`

---

## Evaluation Metrics

All models evaluated using:
- **Accuracy** – overall classification rate
- **Weighted Precision, Recall, F1** – handles class imbalance
- **Confusion Matrix** – per-class true/false positive breakdown
- **ROC-AUC** – one-vs-rest, all 5 classes
- **3-Fold Stratified Cross-Validation**

---

## Dependencies

See `requirements.txt` for complete list. Key libraries:
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `shap >= 0.42.0`
- `joblib >= 1.3.0`

---

## Citation

If you use this project, please cite:

```bibtex
@misc{nsl_kdd_rf_nids_2025,
  title   = {Network Intrusion Detection Using Random Forest on NSL-KDD},
  note    = {Cybersecurity Machine Learning – Project 52},
  year    = {2025}
}
```

**Reference paper:**
> N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive dataset for network intrusion detection systems," IEEE Military Communications and Information Systems Conference, 2015.

---

## License

This project is for academic purposes only. The NSL-KDD dataset is publicly available at the University of New Brunswick's Canadian Institute for Cybersecurity.
