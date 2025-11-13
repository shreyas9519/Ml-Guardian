# file: scripts/verify_merged_claims.py

import pandas as pd

# Note: Default filename from pipeline is "bert_claim_features.parquet"
# Change the path below if you used a different output filename
df = pd.read_parquet("data/processed/bert_claim_features.parquet")

print("Merged features shape:", df.shape)

print(df.columns[:10].tolist(), "...")

print(df.head(5).to_string(index=False))

