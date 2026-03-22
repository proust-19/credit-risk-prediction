# Week 3: Model Optimization + Evaluation

## Overview
- Goal of this week (1-2 lines — tuning the best baseline, evaluating properly, SHAP)
- Best model going in: XGBoost (from Week 2)

---

## 1. Hyperparameter Tuning

### Method Used
- Optuna
- Number of trials = 50
- Metric optimized: F1-score (explain why — class imbalance context)

### Parameters Tuned
| Parameter | Search Range | Best Value |
|-----------|-------------|------------|
| max_depth | ... | ... |
| learning_rate | ... | ... |
| n_estimators | ... | ... |
| subsample | ... | ... |

### MLflow Tracking
- How many runs logged
- Best run ID / experiment name
- Screenshot or table of top 3 trials

---

## 2. Tuned Model Performance

### Test Set Results
| Metric | Baseline XGBoost | Tuned XGBoost |
|--------|-----------------|---------------|
| F1-score | ... | ... |
| Precision | ... | ... |
| Recall | ... | ... |
| ROC-AUC | ... | ... |

### Confusion Matrix (Tuned Model)
```
[[TN, FP],
 [FN, TP]]
```

### Improvement Summary
- What improved, what didn't, and why

---

## 3. Threshold Tuning

### Why Threshold Matters Here
- FN (predicting good credit when it's bad) costs MORE than FP in banking
- Default threshold is 0.5 — not always optimal

### Threshold Analysis
| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.3 | ... | ... | ... |
| 0.4 | ... | ... | ... |
| 0.5 | ... | ... | ... |
| 0.6 | ... | ... | ... |

### Chosen Threshold
- Final threshold: X
- Reasoning: business cost justification

---

## 4. Cost-Sensitive Evaluation

- Cost matrix used (FN penalty vs FP penalty)
- Total misclassification cost at default vs tuned threshold
- Which threshold minimizes business cost

---

## 5. SHAP Analysis

### Global Feature Importance
- Top 5 features by SHAP value (table or description)
- What this tells us vs Week 2's feature importance from the model

### Key Insights
- Which engineered features (monthly_payment, financial_stability, etc.) contributed
- Any surprises vs EDA findings from Week 1

### Individual Prediction Explanations
- 1-2 sample predictions explained with force plot findings
- "For this applicant, X pushed the prediction toward default because..."

---

## 6. Final Model Decision

- Chosen model: Tuned XGBoost with threshold X
- Saved to: `models/xgb_final.pkl`
- MLflow run ID for the final model

---

## 7. Key Insights & Week 4 Goals

### What I Learned
- 2-3 concrete takeaways (not generic — specific to your results)

### Going into Week 4
- API input schema will need: [list the features]
- Threshold to use in production: X
- SHAP explanations to expose in API response: yes/no