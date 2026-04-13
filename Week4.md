# Week 4: Deployment (FastAPI)

## Overview

- Wrapped the final XGBoost credit-risk model in a FastAPI application for online inference
- Exposed health-check and prediction endpoints for simple deployment testing
- API accepts raw applicant fields, performs the same feature engineering used in training, and returns default-risk output

---

## 1. API Architecture

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health  | GET    | Returns API status and loaded model name (`XGB_Tuned`) |
| /predict | POST   | Accepts one loan application, preprocesses it, and returns predicted risk |

### Input Schema

- 20 raw input fields are collected from the user
- Numeric fields: `duration`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `num_dependents`
- Categorical fields: `credit_history`, `personal_status`, `other_parties`, `property_type`, `other_payment_plans`, `housing`, `job`, `telephone`, `foreign_worker`, `checking_status`, `savings_status`, `employment_status`, `purpose`
- Pydantic validation is used for the coded categorical fields before inference

### Output Schema

- `probabilty`: predicted probability of bad credit as a float rounded to 4 decimals
- `prediction`: binary class output (`0` or `1`)
- `risk`: human-readable label (`"low"` or `"high"`)

---

## 2. Preprocessing Pipeline

- All request data is passed into `preprocess_input()` in [`api/utils.py`](/home/purshotam_kumar/ml_projects/credit-risk-prediction/api/utils.py)
- Four raw coded categories are first converted into score-based features using mapping dictionaries:
  - `checking_status`
  - `savings_status`
  - `employment_status`
  - `purpose`
- Engineered features created at inference time:
  - `monthly_payment = credit_amount / duration`
  - `financial_stability = checking_points + savings_points`
  - `employment_points`
  - `overall_stability = financial_stability + employment_points`
  - `purpose_points`
- Categorical training features are reconstructed with manual one-hot encoding for:
  - `credit_history`
  - `personal_status`
  - `other_parties`
  - `property_type`
  - `other_payment_plans`
  - `housing`
  - `job`
  - `telephone`
  - `foreign_worker`
- Final model input is reordered to the exact `feature_order` list expected by the saved model
- Final feature vector size: 33 columns

---

## 3. Key Implementation Decisions

- The model is loaded once at application startup from `models/xgb_final.pkl` using `joblib`
- Validation is handled in [`api/schemas.py`](/home/purshotam_kumar/ml_projects/credit-risk-prediction/api/schemas.py) so invalid category codes are rejected before prediction
- Preprocessing is kept outside the endpoint in a utility function to match the training feature pipeline more reliably
- Prediction probability is converted to a native Python float before returning JSON
- Current API classification rule in [`api/main.py`](/home/purshotam_kumar/ml_projects/credit-risk-prediction/api/main.py) uses `0.5`
- Tuned threshold saved from Week 3 in [`models/model_config.json`](/home/purshotam_kumar/ml_projects/credit-risk-prediction/models/model_config.json) is `0.45`

---

## 4. Deployment

- Platform used: Railway
- Application entry point: FastAPI app served from `api/main.py`
- Deployment included the trained model file, schema validation, and preprocessing logic so prediction could run end-to-end in the hosted environment
- `/health` was used to verify that the service was running correctly after deployment
- `/predict` was tested with sample loan input to confirm the API returned probability, class prediction, and risk label
- Demo evidence is included in [`api/FastAPI - demo.pdf`](/home/purshotam_kumar/ml_projects/credit-risk-prediction/api/FastAPI%20-%20demo.pdf)

---

## 5. Challenges & Fixes

| Problem | Fix |
|---------|-----|
| Feature shape mismatch (got 5, expected 33) | Built full preprocessing pipeline in utils.py |
| numpy.float32 not JSON serializable | Wrapped prob with float() |
| ModuleNotFoundError on Railway | Changed import to api.schemas and api.utils |

---

## 6. Final Reflection

- What this project taught: [2-3 lines, be specific]
- What's missing for true production: model monitoring, drift detection, retraining pipeline
- Live API URL: Deleted after testing (see screenshots/api_demo.pdf)
