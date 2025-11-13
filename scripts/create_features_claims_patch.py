# scripts/create_features_claims_patch.py
"""
Compute claim-level features from outputs/claims_out.jsonl and merge into features parquet.
Run: python scripts/create_features_claims_patch.py

Outputs:
 - data/processed/bert_claim_features.parquet  (new parquet with claim features merged)
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

CLAIMS_JSONL = Path("outputs/claims_out.jsonl")  # from your last step
FEATURES_PARQUET = Path("data/processed/bert_features.parquet")
OUT_PARQUET = Path("data/processed/bert_claim_features.parquet")

def load_claims_map(claims_jsonl_path: Path):
    claims_map = {}
    if not claims_jsonl_path.exists():
        print(f"[create_features_claims_patch] WARNING: claims file not found: {claims_jsonl_path}")
        return claims_map
    with claims_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("id")
            if "result" in rec:
                claims_map[cid] = rec["result"]
            else:
                claims_map[cid] = rec.get("result", rec)
    return claims_map

def compute_claim_features_for_row(result):
    if not result:
        return {
            "num_claims": 0,
            "mean_claim_score": np.nan,
            "max_contradiction_score": 0.0,
            "severity": 0.0,
            "num_claims_contradiction": 0,
            "top_evidence_rel_score": 0.0
        }
    claims = result.get("claims", [])
    num_claims = len(claims)
    if num_claims == 0:
        return {
            "num_claims": 0,
            "mean_claim_score": np.nan,
            "max_contradiction_score": 0.0,
            "severity": result.get("severity", {}).get("severity", 0.0),
            "num_claims_contradiction": 0,
            "top_evidence_rel_score": 0.0
        }
    mean_claim_score = np.mean([c.get("score", 0.5) for c in claims])
    contra_scores = [1.0 - c.get("score", 0.5) for c in claims]
    max_contra = float(max(contra_scores)) if contra_scores else 0.0
    num_contra = int(sum(1 for c in claims if c.get("verdict") == "contradiction"))
    rels = []
    for c in claims:
        for e in c.get("evidence", []):
            rels.append(e.get("rel_score", 0.0))
    top_rel = float(max(rels)) if rels else 0.0
    severity_val = result.get("severity", {}).get("severity", 0.0)
    return {
        "num_claims": int(num_claims),
        "mean_claim_score": float(mean_claim_score),
        "max_contradiction_score": float(max_contra),
        "severity": float(severity_val),
        "num_claims_contradiction": int(num_contra),
        "top_evidence_rel_score": float(top_rel)
    }

def main():
    # load existing feature parquet if available
    if FEATURES_PARQUET.exists():
        print(f"[create_features_claims_patch] Loading {FEATURES_PARQUET}")
        df = pd.read_parquet(FEATURES_PARQUET)
    else:
        print("[create_features_claims_patch] No base parquet found, creating dummy DataFrame.")
        df = pd.DataFrame([{"id": "ex1"}, {"id": "ex2"}, {"id": "ex3"}])
    if 'id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'id'})
    claims_map = load_claims_map(CLAIMS_JSONL)
    feat_rows = []
    for uid in df['id'].astype(str).tolist():
        res = claims_map.get(uid)
        feats = compute_claim_features_for_row(res)
        feats['id'] = uid
        feat_rows.append(feats)
    feat_df = pd.DataFrame(feat_rows).set_index('id')
    df = df.set_index(df['id'].astype(str))
    merged = df.join(feat_df, how='left')
    merged.reset_index(drop=True).to_parquet(OUT_PARQUET, index=False)
    print(f"[create_features_claims_patch] Wrote {OUT_PARQUET}")
    print(merged.head().to_string())

if __name__ == "__main__":
    main()
