import pickle
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.bin"

with MODEL_PATH.open("rb") as f_in:
    FEATURE_COLUMNS, pipeline = pickle.load(f_in)

app = FastAPI(title="stroke-prediction")


class Patient(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float | None = None
    smoking_status: str


class PredictionResponse(BaseModel):
    stroke_probability: float
    stroke: bool


def predict_stroke_probability(patient: dict) -> float:
    df = pd.DataFrame([patient])
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return float(pipeline.predict_proba(df[FEATURE_COLUMNS])[0, 1])


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: Patient) -> PredictionResponse:
    probability = predict_stroke_probability(patient.model_dump())
    return PredictionResponse(
        stroke_probability=probability,
        stroke=probability >= 0.5,
    )


if __name__ == "__main__":
    uvicorn.run("src.predict:app", host="0.0.0.0", port=9696, reload=True)
