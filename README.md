# Fingerprint Spoofing Detection using Machine Learning

## Overview
This repository contains the code and resources for the project on fingerprint spoofing detection using various machine learning techniques. The goal of the project is to classify fingerprint images as either *Authentic* or *Spoofed* based on their extracted feature representations. The project explores multiple classification models, evaluates their performance, and determines the most effective approach for spoof detection.

## Dataset
The dataset consists of fingerprint images represented as 10-dimensional embeddings. The key details are:
- **Authentic fingerprints** are labeled as `1`, while **Spoofed fingerprints** are labeled as `0`.
- Spoofed samples originate from six different spoofing techniques (though the specific technique is not provided).
- The training set contains **2325 samples**, while the test set consists of **7704 samples**.
- The dataset is highly imbalanced, with spoofed fingerprints being significantly overrepresented.

## Feature Analysis
Before applying classification models, an analysis of feature distributions was conducted:
- Some features exhibit **Gaussian-like distributions**, particularly for authentic fingerprints.
- Spoofed fingerprints show a **clustered distribution** due to multiple spoofing techniques.
- **Feature correlations** were evaluated using Pearson correlation matrices.
- **Dimensionality reduction** techniques, such as **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**, were explored to enhance model performance.

## Machine Learning Models Evaluated for Fingerprint Spoofing
The following models were applied and compared:

### Generative Models
- **Multivariate Gaussian Classifier (MVG)**
- **MVG with Diagonal, Tied, and Combined Covariance Constraints**

### Discriminative Models
- **Logistic Regression** (Prior-weighted and Quadratic)
- **Support Vector Machines (SVMs)**
  - Linear SVM
  - Polynomial Kernel SVM
  - Radial Basis Function (RBF) Kernel SVM
- **Gaussian Mixture Model (GMM)**
  - Various covariance structures (Diagonal, Full, Tied)
  - Different numbers of components for target/non-target classes

## Model Selection and Validation
- **K-Fold Cross-Validation (K=5)** was used for performance evaluation.
- **Detection Cost Function (DCF)** was the primary metric for model selection.
- The best-performing models were:
  1. **Gaussian Mixture Model (GMM)**: Tied covariance for non-targets, diagonal covariance for targets, and component setup (2,8) (DCF=0.205).
  2. **Quadratic Logistic Regression**: PCA=6, λ=10⁻², πT=0.1 (DCF=0.263).
  3. **SVM with RBF Kernel**: γ=10⁻³, C=10, PCA=6, πT=0.1 (DCF=0.285).
- Model **calibration** was applied where necessary to adjust thresholds.

## Results and Conclusions
The **Gaussian Mixture Model (GMM)** with **Tied covariance for the non-target class** was identified as the best model due to its ability to capture the structure of spoofed fingerprints more effectively. The **Quadratic Logistic Regression** and **SVM with RBF kernel** also performed well, confirming that **non-linear decision boundaries** provide superior classification for fingerprint spoof detection.

## Contributors
- **Federica Amato** (s310275)
- **Marco Colangelo** (s309798)

## License
This project is released under the MIT License.
