# Week 2: Feature Engineering + Baseline Models (Mar 16-22, 2026)

## Feature Engineering Summary
- Input: 1000 samples, 20 raw features
- Output: 33 engineered features after encoding + new features
- Train/Test split: 800/200 (80/20, stratified)
- SMOTE applied: Train set balanced to 559/559 (class 0 / class 1)

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
| Logistic Regression | 0.5763   | 0.7813  | 75%      | 0.58            | 0.58         |
| Random Forest       | 0.6126   | **0.8339**  | 79%      | 0.65            | 0.58         |
| **XGBoost**         | **0.6727** | 0.8275 | **82%** | **0.73**     | **0.63**     |

### Confusion Matrices

**Logistic Regression:**
- True Negatives: 116, False Positives: 25
- False Negatives: 25, True Positives: 34

**Random Forest:**
- True Negatives: 123, False Positives: 18
- False Negatives: 25, True Positives: 34

**XGBoost:**
- True Negatives: 127, False Positives: 14(rejecting good loans)
- False Negatives: 22, True Positives: 37

## Best Model: XGBoost

**Why XGBoost wins:**
- Highest F1 score (0.6727) — best balance of precision and recall
- Highest accuracy (82%)
- Fewest false negatives (22) — approving bad loans is costly. So, we use efficient model of approving 22 bad loans rather than 25 bad loans.
- Fewest false positives (14) — rejecting good loans  is also unaffordable.
- Highest precision on bad credit class (73%) — fewer false alarms

**Why not Random Forest:**
- Highest ROC-AUC (0.8339) but lower F1 and more false positives (18 vs 14)
- Less reliable at the default threshold

**Why not Logistic Regression:**
- Didn't converge fully (lbfgs hit 1000 iteration limit)
- Weakest on all metrics — linear boundary not sufficient for this dataset

## Key Insights

- **overall_stability is the most powerful feature**: 50% default spread confirms that combining financial + employment stability is highly predictive
- **XGBoost handles class imbalance better** than linear models even after SMOTE
- **Recall on bad credit (0.63) is still low** — we're missing 37% of actual defaulters. This is the main problem to fix in Week 3
- **Business cost perspective**: False negatives (approving bad loans) are more expensive than false positives (rejecting good applicants). Threshold tuning in Week 3 should shift toward higher recall

## What to Optimize in Week 3

- **Hyperparameter tuning on XGBoost**: max_depth, learning_rate, n_estimators, subsample, min_child_weight
- **Threshold tuning**: Lower decision threshold to improve recall on bad credit class (reduce false negatives)
- **Cost-sensitive evaluation**: Weight FN errors higher than FP errors
- **SHAP analysis**: Confirm which features are actually driving XGBoost predictions