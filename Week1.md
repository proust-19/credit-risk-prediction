# Week 1: EDA Summary (Mar 9-15, 2026)

## Dataset Overview

- 1000 samples, 20 features, 1 target
- Class distribution: 70% good credit (class 1), 30% bad credit (class 2)
- No missing values
- Mixed types: 13 categorical, 7 numerical

## Top 5 Most Predictive Features

### 1. Credit History

- **A30 (no credits taken / all paid duly)**: 62% default risk — people with no credit history are risky because banks can't verify their repayment behavior
- **A31-A33 (proven payment history)**: 32-57% default risk — moderate, depends on quality of history
- **A34 (critical account / other credits existing)**: 17% default risk — surprisingly low despite being flagged as "critical"
- **Clear pattern**: No history or critical existing credits = higher risk. Clean repayment history at this bank (A31) = safer.

### 2. Checking Account Status

- Checking account status is the strongest predictor because it reveals financial stability: A11 (no account, 49% default) indicates no banking relationship or financial exclusion, while A14 (high balance, 12% default) shows liquidity and responsible money management. 37% risk spread makes this critical for the model.

### 3. Savings Status

- Lower balance reflects higher default risk (36%-33%) likely with the amount less than 500DM and when amount greater than 1000DM it's default risk down to 12%.

### 4. Duration

- From Outliers (boxplots), The median of class 1 is 20 months but class 2 have 35 months.
- This is clear seperation, if duration time increases then the risk increases.

### 5. Employment Length

- Default risk is too high in range of 40% to 22% but still the employer working more than for 4 years can be considered lesser default risk near to 23%.
- And those who are Unemployed or working less than 1 year are considered to be high default risk of 37-40%

## Feature Engineering Opportunities

1. **Monthly payment burden** - if monthly burden is huge 8(**credit_amount / duration**) then risk would be higher.
2. **Financial Staility** - if there is high balance + high savings have lower risk than no account + no savings. this can be used as a feature by assigning points (**financial_stability_score = checking_points + savings_points**).
3. **Employment Stability (Binary)** - unemployed/short = 0 and stable = 1 points can be given for scaling.
4. **Purpose** can be categorize like **high risk purpose**(cars/repairs) and **lower risk purpose**(furniture/retraining) which decides the risk. It can be used as binary or ordinal feature.

## Class Imbalance Strategy

- we choose **threshold moving** method, which chooses best threshold value that balances the T.P.R and F.P.R. And using Better Evauation metrics(**F1-Score**) for selection of **threshold** value

## Key Insights for Modeling

- **Financial features**: Checking account, savings, and credit history have the strongest separation. These should be prioritized in feature selection.
- **Duration**: It is a strong predictor but also risky. In production,longer loans always predicted as risky regardless of other factors. So, we need to balance the prediction with financial stability features.
- **Demographics are weak**: Age, personal status, num_dependents barely separate classes. Don't over-engineer these.
- **Business cost matters**: Approving bad loans cost more than rejecting good applicants. Model should be tuned for precision on class 2, not just accuracy.
- **Small dataset means careful validation**: Only 300 bad credit samples. Need stratified CV, can't afford to overfit.
