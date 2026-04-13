# Week 2: Feature Engineering + Baseline Models (Mar 16-22, 2026)

## Feature Engineering Summary

- Input: 1000 samples, 20 raw features
- Output: 33 model features after encoding (`34` columns including target)
- Train/Test split: 800/200 (80/20)
- No SMOTE used in the final baseline modeling notebook

## Engineered Features

### 1. monthly_payment = credit_amount / duration

- Bad borrowers actually have *lower* monthly payments due to longer loan durations
- Weak and counterintuitive predictor — useful as context, not a strong standalone signal

### 2. financial_stability = checking_points + savings_points

- checking_map: A11→0, A12→1, A13→2, A14→3
- savings_map: A61→0, A62→1, A63→2, A64→3, A65→2
- 44% default spread across levels (score 0: 52% default → score 6: 8% default)
- Second strongest engineered feature

### 3. employment_points (ordinal encoding)

- A71/A72 (unemployed / <1yr) → 0, A73 (1-4yr) → 1, A74/A75 (>4yr / >7yr) → 2
- 16% default spread — marginal but consistent improvement

### 4. overall_stability = financial_stability + employment_points

- Combined score ranging 0–8
- 50% default spread (score 0: 56% default → score 8: 7% default)
- Strongest engineered feature overall

### 5. purpose_points (risk-based grouping)

- High risk (cars/other) → 0, Medium-high → 1, Medium-low (furniture/appliances) → 2, Low risk (retraining) → 3
- 28% default spread across groups

## Model Comparison

| Model               | F1 Score | ROC-AUC | Accuracy | Precision (bad) | Recall (bad) |
|---------------------|----------|---------|----------|-----------------|--------------|
| Logistic Regression | 0.6111   | 0.8096  | 79%      | 0.67            | 0.58         |
| Random Forest       | 0.5600   | **0.8211** | 78%      | **0.68**        | 0.47         |
| XGBoost             | **0.5946** | 0.8077  | 78%      | 0.63            | 0.56         |

### Confusion Matrices

**Logistic Regression:**

- True Negatives: 125, False Positives: 16
- False Negatives: 26, True Positives: 33

**Random Forest:**

- True Negatives: 128, False Positives: 13
- False Negatives: 31, True Positives: 28

**XGBoost:**

- True Negatives: 122, False Positives: 19
- False Negatives: 26, True Positives: 33

## Why XGBoost Was Selected for Tuning

At baseline, all three models performed similarly (F1: 0.56-0.61).

- XGBoost was selected because:

1. Tunable — hyperparameter space is large, Optuna can extract significantly more performance
2. Handles non-linear feature interactions — financial risk patterns are rarely linear
3. Native scale_pos_weight — handles class imbalance without SMOTE

- Logistic Regression was competitive at baseline (best F1: 0.6111)
but has limited capacity for improvement without feature transformations. Random Forest had the best ROC-AUC but worst recall — not acceptable for credit risk.

- Decision validated: tuned XGBoost reached F1 0.6752 and recall 0.90, catching 53 of 59 defaulters.

## Key Insights

- **overall_stability is the most powerful feature**: 50% default spread confirms that combining financial + employment stability is highly predictive
- **The notebook did not use SMOTE**: these baseline results come from the original class distribution
- **Recall on bad credit is still low across all models**: even the best recall here is 0.56, so 44% of actual defaulters are still missed
- **Business cost perspective**: False negatives (approving bad loans) are more expensive than false positives (rejecting good applicants).   Threshold tuning in Week 3 should shift toward higher recall

## What to Optimize in Week 3

1. **Re-check whether XGBoost is worth tuning further**: current baseline does not beat Logistic Regression on F1
2. **Threshold tuning**: Lower decision threshold to improve recall on bad credit class (reduce false negatives)
3. **Cost-sensitive evaluation**: Weight FN errors higher than FP errors

4. **SHAP analysis**: Confirm which features are actually driving the tree-based models
