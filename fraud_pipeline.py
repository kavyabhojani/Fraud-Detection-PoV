import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from xgboost import XGBClassifier, plot_importance
import joblib

def load_data(path="data/creditcard.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put Kaggle creditcard.csv under data/."
        )
    df = pd.read_csv(path)
    if not {"Time", "Amount", "Class"}.issubset(df.columns):
        raise ValueError("Expected columns: Time, Amount, Class")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"])
    out["Hour"] = (out["Time"] // 3600) % 24
    return out

def plot_confusion_matrix(cm_list, filename):
    cm = np.array(cm_list)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc(y_true, y_proba, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_pr(y_true, y_proba, filename):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(4, 3))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importance_model(model, filename):
    plt.figure(figsize=(6, 4))
    plot_importance(model, max_num_features=10, importance_type="gain")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# train-time reference stats for PSI

def _hist_ref(series: pd.Series, bins=10):
    counts, edges = np.histogram(series.values, bins=bins)
    dist = (counts / counts.sum()).tolist() if counts.sum() > 0 else [0.0] * len(counts)
    return {"edges": edges.tolist(), "dist": dist}

def build_train_reference_stats(train_df: pd.DataFrame) -> dict:
    # Time & Amount as hist; Hour as categorical distribution (0..23)
    ref = {
        "Time": _hist_ref(train_df["Time"], bins=10),
        "Amount": _hist_ref(train_df["Amount"], bins=10),
        "Hour": train_df["Hour"].value_counts(normalize=True).reindex(range(24), fill_value=0).tolist(),
    }
    return ref

def train_model(df: pd.DataFrame):
    X = df[["Time", "Amount", "Amount_log", "Hour"]].copy()
    y = df["Class"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    # scale continuous only
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    cont_cols = ["Time", "Amount", "Amount_log"]
    X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])

    # handle imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    y_pred = (model.predict_proba(X_test_scaled)[:, 1] >= 0.5).astype(int)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "conf_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "average_precision": float(average_precision_score(y_test, y_proba)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgb.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # save train-time reference stats for drift checks
    train_refs = build_train_reference_stats(pd.concat([X_train.assign(Class=y_train)], axis=1))
    with open("models/train_stats.json", "w") as f:
        json.dump(train_refs, f)

    os.makedirs("images", exist_ok=True)
    plot_confusion_matrix(metrics["conf_matrix"], "images/conf_matrix.png")
    plot_roc(y_test, y_proba, "images/roc_curve.png")
    plot_pr(y_test, y_proba, "images/pr_curve.png")
    plot_feature_importance_model(model, "images/feature_importance.png")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    m = train_model(df)
    print(json.dumps({"status": "ok", "metrics": m}, indent=2))
