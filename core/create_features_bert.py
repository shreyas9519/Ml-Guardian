import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_data(raw_path):
    logging.info(f"📂 Loading data from: {raw_path}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    logging.info(f"✅ Loaded {len(data)} samples.")
    return data

def extract_evidence_text(ev):
    """Extract evidence text from nested FEVER dataset format.
    
    FEVER evidence format: [[[page_id, line_id, 'entity_name', sentence_idx], ...], ...]
    We extract entity names as evidence since actual text requires Wikipedia database.
    """
    if isinstance(ev, str):
        return ev
    elif isinstance(ev, list):
        if not ev:
            return ""
        # Check if it's the nested FEVER format
        if isinstance(ev[0], list) and len(ev[0]) > 0:
            if isinstance(ev[0][0], list):
                # Deeply nested: [[[page_id, line_id, 'entity', idx], ...], ...]
                ev_texts = []
                for part in ev:
                    if isinstance(part, list):
                        for p in part:
                            if isinstance(p, list) and len(p) > 2:
                                # Extract entity name (index 2)
                                entity = p[2] if isinstance(p[2], str) else ""
                                if entity:
                                    # Replace underscores with spaces for better readability
                                    ev_texts.append(entity.replace('_', ' '))
                return " ".join(ev_texts) if ev_texts else "NOT ENOUGH INFO"
            else:
                # Single level list
                return " ".join([str(x) for x in ev if x])
        else:
            # Simple list of strings
            return " ".join([str(x) for x in ev if x])
    else:
        return str(ev) if ev else ""

def extract_features(data, model_name="bert-base-uncased", batch_size=16):
    logging.info(f"🚀 Loading BERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")
        logging.info("✅ Using GPU")
    else:
        logging.info("⚠️ Using CPU (will be slower)")

    all_features = []
    # Extract evidence text properly from nested structure
    texts = []
    for d in data:
        claim = d.get('claim', '')
        evidence = extract_evidence_text(d.get('evidence', ''))
        texts.append(f"{claim} [SEP] {evidence}")

    logging.info(f"⚙️ Generating embeddings for {len(texts)} samples...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)

        if torch.cuda.is_available():
            encodings = {k: v.to("cuda") for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            # Mean pooling of token embeddings
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_features.extend(features)

    logging.info("✅ Finished feature extraction.")
    return all_features

def main():
    parser = argparse.ArgumentParser(description="Create BERT embeddings from FEVER dataset")
    parser.add_argument("--raw", required=True, help="Path to input JSONL file")
    parser.add_argument("--out", required=True, help="Path to output parquet file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for BERT inference")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="BERT model name from HuggingFace")
    args = parser.parse_args()

    data = load_data(args.raw)
    features = extract_features(data, model_name=args.model_name, batch_size=args.batch_size)

    # Create DataFrame with feature columns
    feature_cols = [f"bert_{i}" for i in range(len(features[0]))]
    df = pd.DataFrame(features, columns=feature_cols)
    df["label"] = [d.get("label", "NOT ENOUGH INFO") for d in data]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    logging.info(f"💾 Saved embeddings to: {args.out}")

if __name__ == "__main__":
    print("🚀 Starting create_features_bert.py ...")
    main()
