# Machine Learning for Tabular Data: Benchmarking Classification Models in High-Stakes Domains

## Overview

This repository contains the complete codebase and analysis for an MSc dissertation benchmarking machine learning approaches across three high-stakes classification problems: fraud detection, healthcare prediction, and public safety analytics.

The core objective is to compare traditional statistical models, gradient boosting, and neural network architectures including Bayesian uncertainty quantification to identify which approaches are most suitable for real-world deployment in imbalanced, mission-critical datasets.

## Datasets and Problem Domains

### 1. Bank Fraud Detection
- **Scale:** 1 million transactions
- **Target:** Binary classification of fraudulent vs legitimate transactions
- **Challenge:** Severe class imbalance, financial impact of false negatives
- **Key insight:** XGBoost achieves AUC 0.8888 with superior calibration for threshold-sensitive fraud operations

### 2. Diabetes Prediction
- **Domain:** Healthcare, high-stakes diagnosis support
- **Challenge:** Class imbalance, interpretability requirements for clinical settings
- **Analysis:** Compares model fairness across demographic groups, critical for equitable healthcare

### 3. UK Crime Prediction
- **Data source:** UK Police API (live integration)
- **Challenge:** Temporal patterns, geographic variation, prediction fairness across neighbourhoods
- **Innovation:** Real-time data pipeline demonstrating production-ready data engineering

## Methodology

### Model Architectures Evaluated

1. **Logistic Regression** - baseline statistical model with interpretability
2. **Random Forest** - ensemble baseline
3. **XGBoost** - gradient boosting with hyperparameter tuning via GridSearchCV
4. **TensorFlow Neural Network** - standard deep learning architecture with dropout regularization
5. **Bayesian Neural Network** - uncertainty quantification via Monte Carlo dropout

### Key Analytical Approaches

- **Threshold Optimization:** 99-point grid search per model to identify optimal operating points for precision-recall trade-offs
- **Calibration Analysis:** Brier score and reliability diagrams to ensure predicted probabilities match observed outcomes
- **Fairness Testing:** Chi-square tests and demographic parity analysis across protected attributes (age, gender, region)
- **Statistical Rigor:** McNemar's test for model comparison, Cohen's d effect sizes, Bonferroni correction for multiple testing
- **Uncertainty Quantification:** Bayesian approaches to quantify prediction confidence

## Key Results

### Model Performance Summary

| Model | Bank Fraud AUC | Diabetes AUC | Crime AUC | Best Use Case |
|------|---------------|-------------|-----------|--------------|
| Logistic Regression | 0.858 | - | - | Baseline, interpretability |
| XGBoost | 0.889 | - | - | Production fraud detection |
| TensorFlow NN | 0.863 | - | - | Non-linear patterns |
| Bayesian NN | 0.856 | - | - | Uncertainty-critical applications |


### Critical Findings

- **XGBoost dominates** in AUC and Brier score across fraud detection, offering the best calibration for downstream decision-making
- **Bayesian Neural Networks detect fairness issues** that standard models miss - chi-square p-value0.0000 for age-based bias, enabling bias mitigation
- **Threshold tuning matters more than model choice** - F1-scores vary by 200percent across threshold selections, critical for imbalanced data
- **Calibration is non-negotiable** - uncalibrated models produce unreliable confidence estimates in production systems

## Repository Structure

```
Machine-Learning-for-Tabular-Data/
├── Bank Fraud Dataset Analysis.ipynb       # 1M row fraud classification
├── Diabatese Dataset Analysis.ipynb        # Healthcare prediction & fairness
├── UKCrime Dataset Analysis.ipynb          # Real-time crime prediction pipeline
└── final_dissertation (2).pdf# Complete write-up with
