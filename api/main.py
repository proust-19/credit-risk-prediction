from fastapi import FastAPI
import joblib
from pathlib import Path
from schemas import LoanApplication
from utils import preprocess_input

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / 'models' / 'xgb_final.pkl')

@app.get("/health")
def health():
  return {"status": "ok", "model": "XGB_Tuned"} 

@app.post("/predict")
def predict(data: LoanApplication):
  features = preprocess_input(data.dict())
  prob = model.predict_proba(features)[0][1]
  prediction = int(prob >= 0.5)

  return {"probabilty": float(round(prob, 4)),
          "prediction": prediction,
          "risk": "high" if prediction == 1 else "low"
          }