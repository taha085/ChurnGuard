"""Train, compare, tune, evaluate and serialize ChurnGuard models."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from features import engineer_features

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn-by-IBM.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42

def load_dataset():
    path = ROOT / "data" / "Telco-Customer-Churn.csv"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(path, index=False)
    return pd.read_csv(path)

def make_pipeline(model, X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    return Pipeline([("prep", prep), ("model", model)])

def main():
    df = load_dataset()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    y = df["Churn"].astype(int)
    X = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=350, class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=350, max_depth=3, learning_rate=.04,
                                 subsample=.85, colsample_bytree=.85,
                                 eval_metric="logloss", random_state=SEED,
                                 n_jobs=-1)
    }
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall",
               "f1":"f1", "roc_auc":"roc_auc", "pr_auc":"average_precision"}
    rows = []
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator, X_train)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model":name, **{k:float(np.mean(scores["test_"+k])) for k in scoring}})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ART/"model_comparison_cv.csv", index=False)

    # Recall-oriented tuning; threshold selection must use CV/train data, never final test labels.
    xgb_pipe = make_pipeline(candidates["xgboost"], X_train)
    search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions={
            "model__n_estimators": randint(200, 650),
            "model__max_depth": randint(2, 6),
            "model__learning_rate": uniform(.015, .10),
            "model__subsample": uniform(.70, .30),
            "model__colsample_bytree": uniform(.70, .30),
            "model__min_child_weight": randint(1, 8)
        },
        n_iter=15, scoring="recall", cv=cv, random_state=SEED, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    probs = best.predict_proba(X_test)[:,1]
    threshold = 0.40  # business-oriented starting threshold; validate with cost/retention capacity.
    pred = (probs >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test,pred)),
        "precision": float(precision_score(y_test,pred,zero_division=0)),
        "recall": float(recall_score(y_test,pred,zero_division=0)),
        "f1": float(f1_score(y_test,pred,zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test,probs)),
        "pr_auc": float(average_precision_score(y_test,probs)),
        "brier_score": float(brier_score_loss(y_test,probs)),
        "classification_report": classification_report(y_test,pred,output_dict=True,zero_division=0),
        "confusion_matrix": confusion_matrix(y_test,pred).tolist(),
        "best_params": search.best_params_,
        "best_cv_recall": float(search.best_score_)
    }
    frac, mean = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    metrics["calibration"] = [{"mean_predicted":float(a),"observed_rate":float(b)} for a,b in zip(mean,frac)]
    (ART/"evaluation.json").write_text(json.dumps(metrics,indent=2))
    comparison.to_csv(ART/"model_comparison_cv.csv",index=False)
    tag = "churnguard_xgb_2026-11-03_v1"
    joblib.dump(best, ART/f"{tag}.joblib")
    try:
        import skops.io as sio
        sio.dump(best, ART/f"{tag}.skops")
    except Exception as e:
        warnings.warn(f"skops serialization skipped: {e}")
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    with mlflow.start_run(run_name=tag):
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(ART/"evaluation.json"))
        mlflow.log_artifact(str(ART/"model_comparison_cv.csv"))
    print(comparison.to_string(index=False))
    print(json.dumps(metrics,indent=2))
    print(f"Saved model: artifacts/{tag}.joblib")

if __name__ == "__main__":
    main()
