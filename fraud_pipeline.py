# fraud_pipeline.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)

def feature_engineering(df):
    # Example additional features (these would normally need raw transaction data)
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] // 3600) % 24
    return df

def train_model(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X[['Time', 'Amount', 'Amount_log']] = scaler.fit_transform(X[['Time', 'Amount', 'Amount_log']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Handle class imbalance using scale_pos_weight
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

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgb.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return metrics

def plot_confusion_matrix(conf_matrix, filename="images/conf_matrix.png"):
    os.makedirs("images", exist_ok=True)
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

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    metrics = train_model(df)
    print("Training complete. Metrics:", metrics)

    # Save confusion matrix plot
    plot_confusion_matrix(metrics['conf_matrix'])
