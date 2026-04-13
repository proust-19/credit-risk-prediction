# Week 3: Model Optimization + Evaluation

## Overview

- Goal of this week: tune the best baseline model (XGBoost), evaluate it with threshold-based business metrics, and use SHAP to understand what drives default predictions.
- Best model going in: XGBoost (from Week 2)

---

## 1. Hyperparameter Tuning

### Method Used

- Optuna
- Number of trials = 50
- Metric optimized: F1-score because the dataset is imbalanced and the project needed a better balance between catching bad-credit applicants and avoiding too many false alarms

### Parameters Tuned

| Parameter | Search Range | Best Value |
|-----------|-------------|------------|
| max_depth | 3 to 8 | 4 |
| learning_rate | 0.01 to 0.30 (log scale) | 0.0183 |
| n_estimators | 100 to 500 | 121 |
| subsample | 0.60 to 1.00 | 0.8407 |

### MLflow Tracking

- Baseline MLflow runs logged: 3
- Experiment name: `credit-risk-baseline`
- Baseline XGBoost run ID: `0605766306bb4877b32a8c6bbc6cc6c3`
- Final threshold-tuned run ID: `39b7ab97b2e7412aa1456288eed52c94`

---

## 2. Tuned Model Performance

### Test Set Results

| Metric | Baseline XGBoost | Tuned XGBoost |
|--------|-----------------|---------------|
| F1-score | 0.5946 | 0.6752 |
| Precision | 0.63 | 0.54 |
| Recall | 0.56 | 0.90 |
| ROC-AUC | 0.8077 | 0.8406 |

### Confusion Matrix (Tuned Model)

```
[[96, 45],
 [6, 53]]
```

### Improvement Summary

- F1-score improved from 0.5946 to 0.6752
- ROC-AUC improved from 0.8077 to 0.8406
- Recall on bad-credit applicants improved strongly from 0.56 to 0.90, reducing false negatives from 26 to 6
- Precision dropped from 0.63 to 0.54 because the lower threshold catches more risky applicants but also increases false positives

---

## 3. Threshold Tuning

### Why Threshold Matters Here

- FN (predicting good credit when it's bad) costs MORE than FP in banking
- Default threshold is 0.5 — not always optimal

### Threshold Analysis

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.3 | 0.3931 | 0.9661 | 0.5588 |
| 0.4 | 0.4825 | 0.9322 | 0.6358 |
| 0.5 | 0.5595 | 0.7966 | 0.6573 |
| 0.6 | 0.6038 | 0.5424 | 0.5714 |

### Chosen Threshold

- Best threshold from PR search: 0.4577 with F1 = 0.6753
- Final threshold used in the saved config: 0.45
- Reasoning: it kept recall very high (0.90), reduced false negatives to 6, and produced the best practical F1 among the tested operating points

---

## 4. Cost-Sensitive Evaluation

- Example cost matrix used for comparison: FN = 5, FP = 1
- Default threshold (0.50): cost = `12*5 + 37 = 97`
- Tuned threshold (0.45): cost = `6*5 + 45 = 75`
- Among the tested thresholds, `0.45` gives the lowest business cost under this assumption
- This supports choosing a lower threshold because approving a bad loan is more expensive than rejecting a good one

---

## 5. SHAP Analysis

### Global Feature Importance

- Top 5 features by mean absolute SHAP value:

  | Rank | Feature | Mean SHAP |
  |------|---------|-------------|
  | 1 | `overall_stability` | 0.3173 |
  | 2 | `financial_stability` | 0.3116 |
  | 3 | `duration` | 0.1736 |
  | 4 | `purpose_points` | 0.1729 |
  | 5 | `credit_history_A34` | 0.1451 |

- This shows that engineered stability features became more influential than many raw categorical variables, which is stronger evidence than simple built-in model importance alone

### Key Insights

- Engineered features that contributed meaningfully included `overall_stability`, `financial_stability`, `purpose_points`, and `monthly_payment`
- The SHAP ranking suggests repayment stability and applicant financial profile matter more than any single raw field by itself
- This is consistent with Week 1 EDA, where weaker financial condition and higher repayment burden were associated with bad-credit outcomes

### Individual Prediction Explanations

- Force-plot example saved in `models/shap_force_plot.png`
- Dependence plot for `financial_stability × duration` saved in `models/shap_dependence.png`
- Individual explanations were generated, but the main deployment decision for Week 4 was to keep SHAP as an analysis artifact rather than expose it in the API response

---

## 6. Final Model Decision

- Chosen model: Tuned XGBoost with threshold 0.45
- Saved to: `models/xgb_final.pkl`
- MLflow run ID for the final threshold-tuned model: `39b7ab97b2e7412aa1456288eed52c94`

---

## 7. Key Insights & Week 4 Goals

### What I Learned

- 2-3 concrete takeaways (not generic — specific to your results)

### Going into Week 4

- API input schema will need: `duration`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `num_dependents`, `credit_history`, `personal_status`, `other_parties`, `property_type`, `other_payment_plans`, `housing`, `job`, `telephone`, `foreign_worker`, `checking_status`, `savings_status`, `employment_status`, `purpose`
- Threshold to use in production: 0.45
- SHAP explanations to expose in API response: no
