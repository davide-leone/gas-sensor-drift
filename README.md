# Learning under Distribution Shift: Robust Classification of Gas Sensor Data under Temporal Drift

This repository contains the code and project report for evaluating machine learning model robustness under temporal distribution shift using the **Gas Sensor Array Drift Dataset**. 

The project investigates how temporal drift and prior probability shift impact predictive performance over extended deployment periods, comparing **Static Deployment** (trained once) with **Adaptive Learning** (periodically retrained).

---

## Project Overview

In real-world applications—such as chemical sensing arrays and High-Performance Computing (HPC) monitoring systems—data distributions evolve over time due to sensor aging, environmental changes, and hardware degradation. 

Using a chronological evaluation protocol (Batches 1–5 for training, Batches 6–10 as sequential test sets), this benchmark evaluates five representative classification algorithms across static and adaptive deployment strategies:
- **Logistic Regression (LogReg)**
- **Support Vector Machines (SVM)**
- **Random Forest (RF)**
- **Extreme Gradient Boosting (XGBoost)**
- **Multi-Layer Perceptron (MLP)**

### Key Research Findings
1. **Global Shift vs. Local Separability:** Global centroid displacement in PCA space does not reliably predict performance degradation. Predictive stability depends primarily on whether class distributions maintain local separability within historical feature bounds.
2. **Prior Probability Shift:** Performance loss is amplified by severe fluctuations in class frequencies across batches (e.g., Gas 6 accounts for 2.2% of historical training data but jumps to >20% in later batches).
3. **Adaptive Retraining:** Periodic retraining significantly mitigates temporal degradation across all architectures, narrowing the gap between simple linear models and complex non-linear classifiers.

---

## Repository Structure

```
.
├── AML - Project Report.pdf    # Full academic project report
├── gas_sensor_drift.ipynb      # Main Jupyter Notebook containing analysis & models
├── Dataset/                    # Raw gas sensor dataset batches
│   ├── batch1.dat
│   ├── batch2.dat
│   └── ... (batch3.dat to batch10.dat)
└── Images/                     # Visualizations and confusion matrices
    ├── cm_6_static.png
    ├── cm_10_static.png
    ├── cm_10_adaptive.png
    ├── pca_centroids_static.png
    ├── pca_centroids_adaptive.png
    ├── pca_batch_drift_analysis_static.png
    ├── pca_batch_drift_analysis_adaptive.png
    ├── pca_class_drift_analysis_static.png
    ├── pca_class_drift_analysis_adaptive.png
    ├── pca_class_drift_analysis_zoomed_8_9.png
    ├── temporal_drift_static.png
    └── temporat_drift_adaptive.png
```

For a detailed theoretical analysis, detailed methodology, confusion matrix diagnoses, and full mathematical setup, please refer to the complete report.
