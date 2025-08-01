import os
import pandas as pd
import numpy as np
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
    average_precision_score
)
from xgboost import XGBClassifier, plot_importance
import joblib

def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)

def feature_engineering(df):
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] // 3600) % 24
    return df

def train_model(df):
    features = ["Time", "Amount", "Amount_log", "Hour"]
    X = df[features]
    y = df['Class']

    scaler = StandardScaler()
    X[['Time', 'Amount', 'Amount_log']] = scaler.fit_transform(X[['Time', 'Amount', 'Amount_log']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    #handling class imbalance
    scale = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=scale,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgb.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    os.makedirs("images", exist_ok=True)
    plot_confusion_matrix(metrics['conf_matrix'], "images/conf_matrix.png")
    plot_roc_curve(y_test, y_proba, "images/roc_curve.png")
    plot_pr_curve(y_test, y_proba, "images/pr_curve.png")
    plot_feature_importance(model, "images/feature_importance.png")

    return metrics

def plot_confusion_matrix(conf_matrix, filename):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks([0.5, 1.5], ["Legitimate (0)", "Fraud (1)"])
    plt.yticks([0.5, 1.5], ["Legitimate (0)", "Fraud (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def plot_roc_curve(y_true, y_proba, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"ROC curve saved to {filename}")

def plot_pr_curve(y_true, y_proba, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Precision-Recall curve saved to {filename}")

def plot_feature_importance(model, filename):
    plt.figure(figsize=(8, 6))
    plot_importance(model, max_num_features=10)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Feature importance saved to {filename}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    metrics = train_model(df)
    print("Training complete. Metrics:", metrics)
