# Stroke Risk Prediction

Midterm project for [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

Stroke is the second leading cause of death worldwide (WHO). Many strokes can be prevented or mitigated when high-risk patients are identified early, but clinicians and care teams often lack a simple, consistent way to score risk from routine patient attributes.

This project builds a **binary classification model** that predicts whether a patient is likely to have a stroke (`1`) or not (`0`) from demographics and clinical features such as age, hypertension, heart disease, average glucose level, BMI, and smoking status.

**How the model is used:** a healthcare analyst or application sends a patient's features to a local web service; the service returns a stroke probability (and optionally a yes/no decision at a chosen threshold). That score can support triage, outreach, or further clinical review — it is an assistive risk signal, not a medical diagnosis.

## Dataset

The project uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle / fedesoriano).

| Item | Detail |
|------|--------|
| File in repo | `data/healthcare-dataset-stroke-data.xls` (CSV content) |
| Rows | ~5,110 patients |
| Target | `stroke` (0 / 1) |
| Notable issue | ~201 missing `bmi` values (`N/A`) — imputed with median via `SimpleImputer` |

**Features:** `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

If you need to re-download the data:

1. Download from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
2. Place `healthcare-dataset-stroke-data.csv` (or `.xls`) under `data/`.

## Project structure

```text
.
├── data/
│   └── healthcare-dataset-stroke-data.xls
├── notebooks/
│   └── stroke.ipynb          # EDA, feature importance, model comparison & tuning
├── src/
│   ├── __init__.py
│   ├── train.py              # Train final model and save artifact
│   └── predict.py            # FastAPI service for predictions
├── model.bin                 # Trained model artifact (created by src/train.py)
├── pyproject.toml            # Dependencies (managed with uv)
├── Dockerfile
└── README.md
```

## Approach

1. **Data preparation** — normalize column names, handle missing BMI, encode categoricals.
2. **EDA** — distributions, missingness, target balance, feature relationships / importance.
3. **Model training** — compare logistic regression, random forest, and gradient boosting; tune hyperparameters; select by ROC-AUC.
4. **Export** — train the winning Gradient Boosting model in `src/train.py` and persist it to `model.bin`.
5. **Deploy locally** — serve predictions with FastAPI + Uvicorn (`src/predict.py`), including via Docker.

**Final model:** `GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=20)`

## Setup (dependencies & environment)

This project uses [uv](https://docs.astral.sh/uv/) and `pyproject.toml`.

### Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create the environment and install dependencies

From the project root:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Or run commands without activating via `uv run ...`.

## How to run

### 1. Exploratory notebook

Open `notebooks/stroke.ipynb` in your IDE or Jupyter. It covers data cleaning, EDA, feature importance, model selection, and parameter tuning.

### 2. Train the final model

```bash
uv run python src/train.py
```

This fits the selected Gradient Boosting model and writes `model.bin` for the prediction service. Run this before starting the API locally.

### 3. Run the prediction service locally

```bash
uv run python src/predict.py
# or:
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696
```

- Interactive API docs: [http://localhost:9696/docs](http://localhost:9696/docs)
- Predict endpoint: `POST /predict`

Example request:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

Example response:

```json
{
  "stroke_probability": 0.24,
  "stroke": false
}
```

## Docker

The image installs dependencies with uv, trains the model during the build, and starts the FastAPI app with Uvicorn.

Build:

```bash
docker build -t stroke-prediction .
```

Run (service on port 9696):

```bash
docker run -it --rm -p 9696:9696 stroke-prediction
```

Then:

- Docs: [http://localhost:9696/docs](http://localhost:9696/docs)
- Predict: `POST http://localhost:9696/predict` (same JSON body as above)

## Cloud deployment

*Coming soon* — instructions and a public URL or demo screenshot/video will be added here.

## Evaluation checklist (ML Zoomcamp)

| Criterion | Where it lives in this repo |
|-----------|-----------------------------|
| Problem description | This README |
| EDA / feature importance | `notebooks/stroke.ipynb` |
| Multiple models + tuning | `notebooks/stroke.ipynb` |
| Training script | `src/train.py` |
| Reproducibility (data + run steps) | `data/` + this README |
| Model deployment (web service) | `src/predict.py` (FastAPI) |
| Dependency / env management | `pyproject.toml` + uv instructions above |
| Containerization | `Dockerfile` + Docker section above |
| Cloud deployment | Cloud section (to be completed) |

## Acknowledgments

- Dataset: [fedesoriano — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle (educational use).
- Course: [DataTalksClub Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).
