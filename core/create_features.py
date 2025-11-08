# core/create_features.py
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Allow imports from parent dir
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.config import cfg

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(raw_path=None, out_path=None, batch_size=32):
    raw_path = raw_path or cfg["paths"]["raw_train"]
    out_path = out_path or cfg["paths"]["processed_features"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logging.info(f"[create_features] Loading raw: {raw_path}")
    df = pd.read_json(raw_path, lines=True)
    logging.info(f"[create_features] Loaded {len(df)} rows")

    # Expected columns: claim, evidence, label
    if not {"claim", "evidence", "label"} <= set(df.columns):
        raise ValueError("Input JSONL must contain 'claim', 'evidence', 'label' columns")

    # Initialize transformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[create_features] Using device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Combine claim and evidence for joint embedding
    sentences = [
        f"Claim: {c} Evidence: {e}"
        for c, e in zip(df["claim"], df["evidence"])
    ]

    logging.info(f"[create_features] Encoding {len(sentences)} pairs...")
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, device=device, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # Convert embeddings to DataFrame
    feat_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    feats_df = pd.DataFrame(embeddings, columns=feat_cols)

    # Attach labels
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    feats_df["label"] = df["label"].map(label_map).fillna(-1).astype(int)

    logging.info(f"[create_features] Saving {feats_df.shape} to {out_path}")
    feats_df.to_parquet(out_path, index=False)
    logging.info("[create_features] ✅ Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None, help="Path to raw JSONL (train.jsonl)")
    parser.add_argument("--out", default=None, help="Path to output parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.raw, args.out, args.batch_size)
