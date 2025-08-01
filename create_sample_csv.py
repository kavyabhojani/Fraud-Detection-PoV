import pandas as pd

df = pd.read_csv("data/creditcard.csv")

sample = df[['Time', 'Amount']].head(1000)
sample.to_csv("sample_transactions.csv", index=False)

print("sample_transactions.csv created with 1000 rows.")
