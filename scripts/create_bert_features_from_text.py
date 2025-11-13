# scripts/create_bert_features_from_text.py
"""
Create BERT features from JSONL with 'id' and 'text' fields.
Outputs parquet with 'id' column for merging with claim pipeline results.
"""
import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser(description="Create BERT embeddings from text JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL with 'id' and 'text' fields")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for BERT inference")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="BERT model name")
    args = parser.parse_args()

    # Load data
    print(f"[bert] Loading data from: {args.input}")
    data = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"[bert] Loaded {len(data)} samples")

    # Extract IDs and texts
    ids = [d.get('id', f'row_{i}') for i, d in enumerate(data)]
    texts = [d.get('text', '') for d in data]

    # Load BERT model
    print(f"[bert] Loading BERT model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[bert] Using device: {device}")

    # Extract features
    print(f"[bert] Generating embeddings for {len(texts)} samples...")
    all_features = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i + args.batch_size]
        encodings = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            # Mean pooling of token embeddings
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.extend(features)

    # Create DataFrame
    feature_cols = [f"bert_{i}" for i in range(len(all_features[0]))]
    df = pd.DataFrame(all_features, columns=feature_cols)
    df["id"] = ids  # Add ID column for merging

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[bert] ✅ Saved {df.shape} features to: {args.output}")

if __name__ == "__main__":
    main()

