# scripts/run_only_rav.py
"""
Run lightweight Retrieval + Verification (RAV) over classifier predictions.

- Uses the features + trained XGBoost model to generate base labels
- Then runs retrieval + NLI verification for ambiguous claims
- Applies consensus rule:
      → At least 2 retrieved chunks must have entailment ≥ 0.6
        and aggregate entailment ≥ 0.7 before flipping to SUPPORTS.
"""

import os
import json
import pandas as pd
from pathlib import Path
from core.predict_with_rav import run_rav_on_batch, evaluate_results

# ==========================
# CONFIG — adjust as needed
# ==========================
FEATURES_PATH = "data/processed/bert_features.parquet"
XGB_MODEL = "models/xgb_bert_pipeline.joblib"
INDEX_DIR = "data/index"

RAV_SAMPLE = 2000          # number of examples to process (reduce for faster test)
RAV_TOPK = 7               # ↑ increased from 3 → 7 for deeper evidence retrieval
RAV_NLI_MODEL = None       # or e.g. "roberta-large-mnli"
LOW_THRESH = 0.35          # below this = confident → skip RAV
HIGH_THRESH = 0.75         # above this = confident → skip RAV
KEEP_EXAMPLES = 2000       # safety cap to prevent memory blow-up

# ==========================================================
# MAIN EXECUTION — runs classifier + RAV + evaluation/report
# ==========================================================
def main():
    # ---------------------
    # Sanity checks
    # ---------------------
    if not os.path.exists(FEATURES_PATH):
        raise SystemExit(f"Features file missing: {FEATURES_PATH}")

    if not os.path.exists(XGB_MODEL):
        raise SystemExit(f"XGBoost model missing: {XGB_MODEL}")

    if not os.path.exists(INDEX_DIR) or not (Path(INDEX_DIR) / "faiss.index").exists():
        raise SystemExit(f"FAISS index not found in {INDEX_DIR}. Build index first.")

    # ---------------------
    # Load features
    # ---------------------
    feats_df = pd.read_parquet(FEATURES_PATH)

    # extract claims
    if "claim" in feats_df.columns:
        claims = feats_df["claim"].astype(str).tolist()
    else:
        claims = [""] * len(feats_df)

    # numeric feature matrix
    if "features" in feats_df.columns:
        import numpy as np
        X = feats_df["features"].values
        Xmat = np.vstack([np.array(x) for x in X])
    else:
        numeric_df = feats_df.select_dtypes(include=[float, int])
        Xmat = numeric_df.values

    # ---------------------
    # Sampling subset
    # ---------------------
    sample_n = min(len(Xmat), RAV_SAMPLE)
    Xs = Xmat[:sample_n]
    cs = claims[:sample_n]

    # ---------------------
    # Run RAV
    # ---------------------
    print(f"Running RAV on {sample_n} examples (top_k={RAV_TOPK}) ...")

    results = run_rav_on_batch(
        features_np=Xs,
        claims=cs,
        model_path=XGB_MODEL,
        index_dir=INDEX_DIR,
        nli_model=RAV_NLI_MODEL,
        top_k=RAV_TOPK,          # now using higher retrieval depth
        low_thresh=LOW_THRESH,
        high_thresh=HIGH_THRESH,
        keep_examples=KEEP_EXAMPLES,
    )

    # ---------------------
    # Save outputs
    # ---------------------
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    results_path = reports_dir / "rav_results.jsonl"
    with open(results_path, "w", encoding="utf8") as wf:
        for r in results:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"RAV results saved to {results_path}")

    # ---------------------
    # Optional evaluation if gold labels exist
    # ---------------------
    gold_labels = None
    if "label" in feats_df.columns:
        label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
        if pd.api.types.is_numeric_dtype(feats_df["label"].dtype):
            gold_labels = feats_df["label"].map(label_map).tolist()[:sample_n]
        else:
            gold_labels = feats_df["label"].astype(str).tolist()[:sample_n]

    if gold_labels:
        metrics = evaluate_results(results, gold_labels=gold_labels)
        metrics_path = reports_dir / "rav_metrics.json"
        with open(metrics_path, "w", encoding="utf8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        print(f"RAV metrics saved to {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
