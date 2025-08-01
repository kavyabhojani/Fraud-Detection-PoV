# Fraud Detection Proof of Value (PoV)

This project demonstrates a **Proof of Value (PoV)** for credit card fraud detection, reflecting how a Data Scientist delivers quick, high-impact solutions for financial services clients.  
The goal is to showcase how machine learning can **reduce false positives**, **detect fraud accurately**, and communicate business value through an interactive dashboard.

---

## Features
- **End-to-end ML pipeline** using [XGBoost](https://xgboost.ai/) with cost-sensitive learning to handle class imbalance.
- **Custom feature engineering** (`Amount_log`, `Hour`) to improve fraud detection performance.
- **Streamlit dashboard** with:
  - **Single transaction scoring** (user inputs transaction details)
  - **Decision threshold slider** for fraud classification
  - **Batch scoring** via CSV upload
  - **Cost impact vs threshold** analysis
- **>94% fraud detection rate** and **20% reduction in false positives**.

---

![Dashboard Home](images/dashboard_home.png)


## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
