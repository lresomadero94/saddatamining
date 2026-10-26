import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules.data_processing import clean_data

PREDICTORS = [
    "Facility_Type", "User_Category", "Participants",
    "Duration_Hours", "Lead_Days", "Weekday", "Start_Hour",
    "Facility_Capacity"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to facility_reservations.csv")
    args = parser.parse_args()

    df = clean_data(pd.read_csv(args.data))
    df = df.dropna(subset=["Reservation_Outcome"]).copy()

    # The midterm code uses Lead_Days as a predictor. If the source file
    # uses Lead_Time_Days, mirror that source field for compatibility.
    if "Lead_Days" not in df.columns and "Lead_Time_Days" in df.columns:
        df["Lead_Days"] = df["Lead_Time_Days"]

    features = [c for c in PREDICTORS if c in df.columns]
    if len(features) < 5:
        raise ValueError(f"Need at least five usable predictors; found {features}")

    X = df[features].copy()
    y = df["Reservation_Outcome"].copy()

    numeric = X.select_dtypes(include=["number","bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessor", pre),
        ("classifier", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3,
            random_state=42, class_weight="balanced"
        ))
    ])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision (Not Completed):", precision_score(y_test, pred, pos_label="Not Completed", zero_division=0))
    print("Recall (Not Completed):", recall_score(y_test, pred, pos_label="Not Completed", zero_division=0))
    print("F1 (Not Completed):", f1_score(y_test, pred, pos_label="Not Completed", zero_division=0))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/trained_model.joblib")
    print("Saved models/trained_model.joblib")

if __name__ == "__main__":
    main()
