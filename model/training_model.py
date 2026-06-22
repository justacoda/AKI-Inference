import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from aki_model import Feature_Processor

# Configuration
MAX_ITER = 2000
CLASS_WEIGHT = {0: 1.0, 1: 10.0}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_THRESHOLD = 0.5

def find_best_threshold(x, y, thresholds=THRESHOLDS, random_state=42):
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = create_model()
    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_val)[:, 1]

    best_threshold = DEFAULT_THRESHOLD
    best_f3 = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f3 = fbeta_score(y_val, y_pred, beta=3)
        if f3 > best_f3:
            best_f3 = f3
            best_threshold = threshold

    return best_threshold, best_f3

def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)),
    ])

def load_dataframe(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    required_cols = ["age", "sex"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_ml_arrays(filepath):
    df = load_dataframe(filepath)

    age = df["age"].to_numpy()
    sex = df["sex"].map({"f": 0, "m": 1}).to_numpy()

    result_cols = [c for c in df.columns if c.startswith("creatinine_result_")]
    if not result_cols:
        raise ValueError("No creatinine result columns found")
    results_df = df[result_cols]

    feature_processor = Feature_Processor()
    features = feature_processor.extract_creatinine_features(results_df)

    x = np.column_stack([
        age,
        sex,
        features["Creatinine_CV"],
        features["Min_Creatinine"],
        features["Creatinine_Max_Change"],
        features["Creatinine_Std"],
        features["Creatinine_Max_Consec_Increase"],
        features["Creatinine_Last_gt130_Count"],
        features["Creatinine_Increase_Count"],
        features["Creatinine_Max_Consec_Decrease"],
        features["Creatinine_Count_Since_Peak"],
        features["Creatinine_Change"],
        features["Creatinine_gt1p5_Count"],
    ])

    y = None
    if "aki" in df.columns:
        aki_col = df["aki"]
        if aki_col.dtype == object:
            y = aki_col.map({"n": 0, "y": 1}).to_numpy()
        else:
            y = aki_col.to_numpy()

    return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="training.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model_output", default="model/aki_model.pkl")
    flags = parser.parse_args()

    x_train, y_train = prepare_ml_arrays(flags.train)

    if flags.threshold is None:
        threshold, f3 = find_best_threshold(x_train, y_train)
        print(f"Auto-tuned threshold: {threshold} (F3: {f3:.4f})")
    else:
        threshold = flags.threshold

    model = create_model()
    model.fit(x_train, y_train)

    try:
        with open(flags.model_output, 'wb') as f:
            pickle.dump({'model': model, 'threshold': threshold}, f)  
    except Exception as e:
        print(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()