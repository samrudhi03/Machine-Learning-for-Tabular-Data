# ML for High-Stakes Decisions: Responsible AI Benchmarking Across Three Domains

**MSc Dissertation - University of Bristol, Department of Engineering Mathematics** | Ethics approved: Application 15208 

---

## Why This Project Exists

When AI models are used to make decisions in healthcare, finance, or public safety, getting the algorithm right is only half the problem. The other half is understanding when to trust it, who it might disadvantage, and what happens when it is wrong.

This dissertation systematically evaluates five machine learning models across three real-world high-stakes domains - not just for predictive accuracy, but for fairness, calibration, uncertainty, and responsible deployment. The findings are empirical, statistically validated, and directly relevant to organisations deploying AI in regulated or sensitive environments.

---

## The Core Finding

**No single algorithm eliminates bias when it exists in the training data. Responsible deployment requires fairness monitoring, calibration assessment, and human oversight - regardless of which model you choose.**

---

## Domains Studied

| Domain | Dataset | Scale | Primary Risk |
|---|---|---|---|
| Healthcare | Diabetes Prediction | 768 patients | Missed diagnoses harm patients |
| Finance | Bank Account Fraud | 1,000,000 transactions | False positives disrupt customers; bias harms minorities |
| Public Safety | UK Crime Classification | 17,713 incidents (live API) | Biased predictions can reinforce systemic inequity |

---

## Responsible AI Findings

### Bias & Fairness

Age-based bias was detected across all models in healthcare and finance - statistically significant (p < 0.0001) with real world consequences:

| Domain | Most Biased Model | Bias Level | Least Biased Model |
|---|---|---|---|
| Healthcare | TensorFlow NN | 39.55% demographic parity difference | Random Forest (21.65%) |
| Finance | Logistic Regression | 21.37% demographic parity difference | XGBoost (14.31%) |
| Public Safety | All similar | Low (8.2% avg) | All similar |

**Key implication:** In finance, senior customers faced 2-3x higher fraud detection rates across all models - a pattern that could breach anti-discrimination regulations and requires active bias monitoring in any production deployment.

**Key implication:** In healthcare, older patients were systematically under diagnosed for diabetes across every model tested - a data-driven finding with direct patient safety consequences.

Public safety models showed significantly lower geographic bias, suggesting that careful problem formulation can reduce discriminatory outcomes.

### Calibration

A model with high AUC can still give unreliable probability estimates. This matters enormously when a score is used to inform a decision rather than just rank outputs.

| Domain | Best Calibrated Model | Worst Calibrated | Why It Matters |
|---|---|---|---|
| Crime | Neural Network (ECE: 0.0924) | XGBoost (ECE: 0.1727) | XGBoost overconfident despite similar AUC |
| Fraud | XGBoost (Brier: 0.1077) | Logistic Regression (Brier: 0.1489) | Miscalibration compounds imbalance problem |

**Key finding:** XGBoost's overconfidence in crime prediction means it would systematically overstate certainty in public safety decisions - a governance risk that AUC scores alone would not reveal.

### Uncertainty Quantification

Bayesian Neural Networks (Monte Carlo Dropout, 100 forward passes) were implemented to quantify prediction uncertainty - flagging cases where the model does not know, not just cases where it predicts positive.

- Uncertainty error correlation: 0.15 - 0.35 across domains
- Prediction interval coverage: 94.1% for 95% intervals - well calibrated
- **Practical value:** High uncertainty predictions can be routed to human review, reducing the risk of automated decisions in edge cases

---

## Models Evaluated

Five architectures were systematically compared across all three domains using identical preprocessing pipelines, fixed random seeds, and 5 - fold stratified cross validation:

1. **Logistic Regression** - interpretable linear baseline
2. **Random Forest** - ensemble baseline
3. **XGBoost** - gradient boosting with hyperparameter tuning (972 parameter combinations)
4. **TensorFlow Neural Network** - standard deep learning with dropout regularisation
5. **Bayesian Neural Network** - Monte Carlo Dropout for uncertainty quantification

---

## Deployment Recommendations by Domain

### Healthcare
- Use **Logistic Regression** - competitive performance, best interpretability, clinicians can understand it
- Threshold optimisation matters more than model complexity - recall can be improved significantly without changing the model
- **Mandatory:** Age based fairness monitoring in any production deployment

### Finance
- Use **XGBoost** - best performance on imbalanced data, lowest bias, fastest retraining
- Do not use as a binary classifier in production - use as a **risk scorer** with human review for high risk cases
- **Mandatory:** Bias monitoring for age based disparities; consider fairness aware threshold calibration

### Public Safety
- All models perform similarly - choose based on **interpretability and transparency** requirements
- Logistic Regression preferred for legal defensibility and accountability
- Ongoing geographic fairness monitoring required even where current bias is low

---

## Technical Rigour

- **Statistical testing:** McNemar's test for model comparison, Cohen's d for effect sizes, Bonferroni correction for multiple testing
- **Fairness metrics:** Demographic parity, equal opportunity, chi-square significance testing
- **Calibration metrics:** Brier score, Expected Calibration Error, reliability diagrams
- **Uncertainty metrics:** Predictive entropy, prediction interval coverage
- **Data:** Live UK Police API integration for crime dataset (Bristol area, August 2024 - July 2025)
- **Ethics:** Faculty ethics committee approved, application 15208, 12/12

---

## Repository Structure
```
Machine-Learning-for-Tabular-Data/
├── Bank Fraud Dataset Analysis.ipynb       # 1M row fraud classification
├── Diabatese Dataset Analysis.ipynb        # Healthcare prediction & fairness
├── UKCrime Dataset Analysis.ipynb          # Live crime prediction pipeline
└── Report.pdf                  # Full write-up with methodology and results
```

---

## Related Work

This project is part of a broader portfolio demonstrating responsible AI in practice. See also: [Country Risk Analyser](https://github.com/samrudhi03/country-risk-analyser) - an LLM pipeline for financial risk assessment with governance framework embedded from the outset.

---

*MSc Data Science, University of Bristol, 2025*
