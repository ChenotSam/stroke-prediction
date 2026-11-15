import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.xls"
MODEL_PATH = ROOT / "model.bin"

CATEGORICAL_COLUMNS = [
    "gender",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
]
NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# Best GradientBoostingClassifier from notebook validation tuning
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 5,
    "n_estimators": 20,
    "random_state": 1,
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
        ]
    )
    model = GradientBoostingClassifier(**MODEL_PARAMS)
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    df = load_data()

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_full_train["stroke"].values
    y_test = df_test["stroke"].values

    pipeline = build_pipeline()
    pipeline.fit(df_full_train[FEATURE_COLUMNS], y_train)

    y_pred = pipeline.predict_proba(df_test[FEATURE_COLUMNS])[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"test auc: {auc:.4f}")

    with MODEL_PATH.open("wb") as f_out:
        pickle.dump((FEATURE_COLUMNS, pipeline), f_out)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
