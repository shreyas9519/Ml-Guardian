# scripts/merge_claims_to_bert_features.py
import json
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_claim_features(claims_result):
    """
    claims_result: parsed JSON object for one example (format produced by your claim pipeline)
    returns: dict of engineered scalar features
    """
    per = claims_result.get("result", {})
    claims = per.get("claims", [])
    severity = per.get("severity", {}).get("severity", None)

    num_claims = len(claims)
    if num_claims == 0:
        return {
            "num_claims": 0,
            "mean_claim_score": np.nan,
            "max_contradiction_score": 0.0,
            "num_contradictions": 0,
            "severity": severity if severity is not None else 0.0
        }

    scores = []
    contra_scores = []
    num_contra = 0
    for c in claims:
        # raw_score is per-claim aggregated score (the pipeline outputs 'score' or 'raw_score')
        raw = c.get("score", None) or c.get("raw_score", None) or 0.0
        scores.append(float(raw))

        # find strongest contradiction evidence for this claim (if available)
        evid = c.get("evidence", [])
        max_contra = 0.0
        for e in evid:
            nli = e.get("nli", {})
            contra_prob = nli.get("contra_prob", None)
            if contra_prob is not None:
                max_contra = max(max_contra, float(contra_prob))
            # fallback: if label exists and is "contradiction"
            if nli.get("label") == "contradiction":
                num_contra += 1
        contra_scores.append(max_contra)

    mean_score = float(np.nanmean(scores)) if scores else 0.0
    max_contra_score = float(np.max(contra_scores)) if contra_scores else 0.0

    return {
        "num_claims": num_claims,
        "mean_claim_score": mean_score,
        "max_contradiction_score": max_contra_score,
        "num_contradictions": num_contra,
        "severity": severity if severity is not None else 0.0
    }

def main(claims_jsonl, bert_features_parquet, out_parquet):
    claims_jsonl = Path(claims_jsonl)
    bert_features_parquet = Path(bert_features_parquet)
    out_parquet = Path(out_parquet)
    assert claims_jsonl.exists(), f"Missing {claims_jsonl}"
    assert bert_features_parquet.exists(), f"Missing {bert_features_parquet}"

    # 1) read claims JSONL into DataFrame keyed by id
    rows = []
    with open(claims_jsonl, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            example_id = j.get("id")  # expectation: your claim pipeline used "id"
            feats = compute_claim_features(j)
            feats["id"] = example_id
            rows.append(feats)

    claims_df = pd.DataFrame(rows).set_index("id")
    print(f"[merge] Loaded {len(claims_df)} claim rows.")

    # 2) load bert features parquet
    bert_df = pd.read_parquet(str(bert_features_parquet))
    print(f"[merge] BERT features loaded: shape={bert_df.shape}, has 'id' column: {'id' in bert_df.columns}")
    
    # If bert_df has an 'id' column use it, else we assume index order aligns with claims ids.
    if "id" in bert_df.columns:
        bert_df = bert_df.set_index("id")
        # Filter to only rows that match claim IDs
        common_ids = bert_df.index.intersection(claims_df.index)
        if len(common_ids) == 0:
            raise RuntimeError(
                f"No matching IDs found between claims ({len(claims_df)} rows with IDs: {list(claims_df.index[:5])}) "
                f"and bert_features ({len(bert_df)} rows). "
                f"Make sure both files are from the same dataset."
            )
        if len(common_ids) < len(claims_df):
            print(f"[merge] Warning: Only {len(common_ids)}/{len(claims_df)} claim IDs found in bert_features. Filtering...")
        bert_df = bert_df.loc[common_ids]
        claims_df = claims_df.loc[common_ids]
    else:
        # try to align by index if number of rows matches:
        if len(bert_df) != len(claims_df):
            raise RuntimeError(
                f"bert_features.parquet has no 'id' column and row counts don't match:\n"
                f"  - Claims JSONL: {len(claims_df)} rows (IDs: {list(claims_df.index[:5])})\n"
                f"  - BERT features: {len(bert_df)} rows\n"
                f"Either add an 'id' column to bert_features or ensure both files have the same number of rows."
            )
        # otherwise assign claims_df.index to bert_df index in same order
        bert_df.index = claims_df.index

    print(f"[merge] BERT features shape: {bert_df.shape}. Claim features shape: {claims_df.shape}")

    # 3) join (claims -> bert features)
    merged = bert_df.join(claims_df, how="left")
    # fill NaNs for claim features with safe defaults
    merged["num_claims"] = merged["num_claims"].fillna(0).astype(int)
    merged["mean_claim_score"] = merged["mean_claim_score"].fillna(0.0).astype(float)
    merged["max_contradiction_score"] = merged["max_contradiction_score"].fillna(0.0).astype(float)
    merged["num_contradictions"] = merged["num_contradictions"].fillna(0).astype(int)
    merged["severity"] = merged["severity"].fillna(0.0).astype(float)

    # 4) save
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(out_parquet), index=False)
    print(f"[merge] Saved merged parquet to {out_parquet} (shape={merged.shape})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--claims", default="outputs/claims_out.jsonl", help="claims pipeline JSONL")
    p.add_argument("--bert", default="data/processed/bert_features.parquet", help="bert features parquet")
    p.add_argument("--out", default="data/processed/bert_claim_features.parquet", help="merged output parquet")
    args = p.parse_args()
    main(args.claims, args.bert, args.out)
