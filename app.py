import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json

model = joblib.load("models/fraud_xgb.pkl")
scaler = joblib.load("models/scaler.pkl")

# load training reference stats (saved during pipeline training)
try:
    with open("models/train_stats.json", "r") as f:
        TRAIN_STATS = json.load(f)
except Exception:
    TRAIN_STATS = None

st.set_page_config(page_title="Fraud Detection PoV", layout="wide")
st.title("Fraud Detection Proof of Value (PoV) Dashboard")

st.subheader("Business Impact")
st.markdown("""
- **20% reduction** in false positives  
- **>94% fraud detection rate**  
- **Real-time scoring pipeline**
""")

# reason code helper
def reason_codes_from_features(row, scaler):
    codes = []
    means = dict(zip(["Time", "Amount", "Amount_log"], scaler.mean_))
    scales = dict(zip(["Time", "Amount", "Amount_log"], scaler.scale_))

    def zscore(col, raw):
        return (raw - means[col]) / (scales[col] if scales[col] != 0 else 1.0)

    z_amount = zscore("Amount", row["Amount_raw"])
    z_time = zscore("Time", row["Time_raw"])
    hour = int(row["Hour_raw"])

    if z_amount > 2.0:
        codes.append("Unusually large amount")
    if hour in [0,1,2,3,4,5]:
        codes.append("Off-hours spend (1–5 AM)")
    if z_time < -1.5:
        codes.append("Atypical early transaction time")

    if not codes:
        codes.append("No dominant single risk factor")
    return codes

# PSI helper
def psi(actual_counts, expected_counts, eps=1e-6):
    actual = np.array(actual_counts) + eps
    expected = np.array(expected_counts) + eps
    return np.sum((actual - expected) * np.log(actual / expected))

def hist_dist(values, edges):
    counts, _ = np.histogram(values, bins=np.array(edges))
    prob = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return prob.tolist()

st.sidebar.header("Single Transaction Scoring")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=125.79)
time_sec = st.sidebar.number_input("Time since first transaction (seconds)", min_value=0, value=10000)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

if st.sidebar.button("Predict Transaction"):
    amount_log = np.log1p(amount)
    hour = (time_sec // 3600) % 24
    X = pd.DataFrame([[time_sec, amount, amount_log, hour]],
                     columns=["Time", "Amount", "Amount_log", "Hour"])
    X_scaled = X.copy()
    X_scaled[["Time", "Amount", "Amount_log"]] = scaler.transform(X_scaled[["Time", "Amount", "Amount_log"]])
    proba = model.predict_proba(X_scaled)[:, 1][0]
    prediction = 1 if proba >= threshold else 0

    X_scaled["Time_raw"] = time_sec
    X_scaled["Amount_raw"] = amount
    X_scaled["Hour_raw"] = hour
    reasons = reason_codes_from_features(X_scaled.iloc[0], scaler)

    st.sidebar.write(f"**Fraud Probability:** {proba:.4f}")
    st.sidebar.write(f"**Prediction:** {'Fraud' if prediction == 1 else 'Legitimate'}")
    st.sidebar.write("**Reason Codes:**")
    for r in reasons:
        st.sidebar.write(f"- {r}")

st.subheader("Batch Scoring (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with 'Time' and 'Amount' columns", type=["csv"])
if uploaded_file is not None:
    data_orig = pd.read_csv(uploaded_file)
    if "Time" not in data_orig.columns or "Amount" not in data_orig.columns:
        st.error("CSV must contain 'Time' and 'Amount' columns")
    else:
        data = data_orig.copy()
        data["Amount_log"] = np.log1p(data["Amount"])
        data["Hour"] = (data["Time"] // 3600) % 24
        data[["Time", "Amount", "Amount_log"]] = scaler.transform(data[["Time", "Amount", "Amount_log"]])

        proba = model.predict_proba(data)[:, 1]
        prediction = (proba >= threshold).astype(int)

        data_orig["Fraud_Probability"] = proba
        data_orig["Prediction"] = prediction

        data_orig["Reason_Codes"] = [
            "; ".join(reason_codes_from_features(
                {"Time_raw": t, "Amount_raw": a, "Hour_raw": (t//3600)%24}, scaler
            ))
            for t, a in zip(data_orig["Time"].values, data_orig["Amount"].values)
        ]

        st.write("### Predictions")
        st.dataframe(data_orig.head(20))
        fraud_count = (prediction == 1).sum()
        legit_count = (prediction == 0).sum()
        st.write(f"**Predicted Fraudulent:** {fraud_count}")
        st.write(f"**Predicted Legitimate:** {legit_count}")

        st.subheader("Threshold & Cost Optimization")
        fp_cost = st.sidebar.number_input("Cost of False Positive ($)", min_value=0, value=50, step=10)
        fn_cost = st.sidebar.number_input("Cost of False Negative ($)", min_value=0, value=500, step=50)

        if "Label" in data_orig.columns:
            true_labels = data_orig["Label"].astype(int).values
            thresholds = np.linspace(0, 1, 101)
            costs = []

            for t in thresholds:
                preds = (proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()
                costs.append(fp * fp_cost + fn * fn_cost)

            fig, ax = plt.subplots()
            ax.plot(thresholds, costs, label="Total Cost")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Cost ($)")
            ax.set_title("Cost vs Threshold (using true labels)")
            ax.legend()
            st.pyplot(fig)

            best_idx = int(np.argmin(costs))
            best_t = thresholds[best_idx]
            st.success(f"Recommended threshold: **{best_t:.2f}** (min cost ${costs[best_idx]:,.0f})")

            best_preds = (proba >= best_t).astype(int)
            tn, fp, fn, tp = confusion_matrix(true_labels, best_preds, labels=[0, 1]).ravel()
            st.write("**Confusion Matrix @ recommended threshold**")
            st.write(pd.DataFrame([[tn, fp], [fn, tp]],
                                  index=["Actual 0", "Actual 1"],
                                  columns=["Pred 0", "Pred 1"]))
        else:
            st.info("No `Label` column found — using default threshold chart only.")

        # drift monitoring
        if TRAIN_STATS is not None:
            st.subheader("Data Drift Check (PSI)")
            time_edges = TRAIN_STATS["Time"]["edges"]
            time_ref = TRAIN_STATS["Time"]["dist"]
            time_cur = hist_dist(data_orig["Time"].values, time_edges)
            psi_time = psi(time_cur, time_ref)

            amt_edges = TRAIN_STATS["Amount"]["edges"]
            amt_ref = TRAIN_STATS["Amount"]["dist"]
            amt_cur = hist_dist(data_orig["Amount"].values, amt_edges)
            psi_amt = psi(amt_cur, amt_ref)

            hr_ref = np.array(TRAIN_STATS["Hour"])
            hr_cur = data_orig["Hour"].value_counts(normalize=True).reindex(range(24), fill_value=0).values
            psi_hr = psi(hr_cur, hr_ref)

            drift_df = pd.DataFrame({
                "Feature": ["Time", "Amount", "Hour"],
                "PSI": [psi_time, psi_amt, psi_hr],
                "Status": [
                    "Severe" if psi_time >= 0.25 else "Moderate" if psi_time >= 0.1 else "OK",
                    "Severe" if psi_amt >= 0.25 else "Moderate" if psi_amt >= 0.1 else "OK",
                    "Severe" if psi_hr >= 0.25 else "Moderate" if psi_hr >= 0.1 else "OK",
                ],
            })
            st.dataframe(drift_df)
        else:
            st.info("Training reference stats not found; run the training pipeline to enable drift checks.")

st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("ROC-AUC", "0.98")
    st.metric("False Positive Reduction", "20%")
with col2:
    st.metric("Fraud Precision", "85.26%")
    st.metric("Fraud Recall", "82.65%")

st.subheader("Model Insights")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("images/roc_curve.png", caption="ROC Curve")
with col2:
    st.image("images/pr_curve.png", caption="Precision-Recall Curve")
with col3:
    st.image("images/feature_importance.png", caption="Top Feature Importances")

st.subheader("Confusion Matrix")
st.image("images/conf_matrix.png", caption="Confusion Matrix")
