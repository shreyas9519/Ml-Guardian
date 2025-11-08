# neural/train_transformer.py
import os, argparse, logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from core.config import cfg
from core.labels import LABEL_MAP
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

SEED = cfg["seed"]

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def prepare_hf_dataset(df, tokenizer, max_length=256):
    # Combine claim and evidence with special token
    texts = (df["claim"].astype(str) + " [SEP] " + df["evidence"].astype(str)).tolist()
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    dataset = Dataset.from_dict({**enc, "label": df["label"].astype(int).tolist()})
    return dataset

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}

def main(raw_features=None, out_dir=None):
    set_seed()
    # Load original data for transformer (not processed features)
    raw_data_path = cfg["paths"]["raw_data"]
    out_dir = out_dir or cfg["paths"]["transformer_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"[train_transformer] Loading raw data from {raw_data_path}")
    
    # Load original JSONL data
    import json
    rows = []
    with open(raw_data_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    
    # Convert to DataFrame with proper format
    records = []
    for r in rows:
        claim = r.get("claim") or r.get("sentence") or ""
        label = r.get("label", None)
        evidence_text = ""
        ev = r.get("evidence") or r.get("evidence_text") or r.get("evidence_sentences")
        if isinstance(ev, list):
            if ev and isinstance(ev[0], list):
                ev_texts = []
                for part in ev:
                    try:
                        for p in part:
                            if isinstance(p, str):
                                ev_texts.append(p)
                    except Exception:
                        pass
                evidence_text = " ".join(ev_texts)
            else:
                evidence_text = " ".join([str(x) for x in ev if x])
        else:
            evidence_text = ev or ""

        records.append({
            "claim": claim,
            "evidence": evidence_text,
            "label": label
        })
    
    df = pd.DataFrame(records)
    # Convert labels to integers
    from core.labels import REVERSE_LABEL_MAP
    df["label"] = df["label"].map(REVERSE_LABEL_MAP).fillna(2).astype(int)
    
    # create train/val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer"]["model_name"])
    train_ds = prepare_hf_dataset(train_df, tokenizer, max_length=cfg["transformer"]["max_length"])
    val_ds = prepare_hf_dataset(val_df, tokenizer, max_length=cfg["transformer"]["max_length"])
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    model = AutoModelForSequenceClassification.from_pretrained(cfg["transformer"]["model_name"], num_labels=len(LABEL_MAP))
    # args = TrainingArguments(
    #     output_dir=out_dir,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=cfg["transformer"]["lr"],
    #     per_device_train_batch_size=cfg["transformer"]["batch_size"],
    #     per_device_eval_batch_size=cfg["transformer"]["batch_size"],
    #     num_train_epochs=cfg["transformer"]["epochs"],
    #     weight_decay=0.01,
    #     seed=SEED,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1_macro"
    # )
    args = TrainingArguments(
    output_dir=out_dir,
    logging_dir=f"{out_dir}/logs",
    evaluation_strategy="epoch",         
    save_strategy="epoch",               
    save_total_limit=2,                 
    learning_rate=cfg["transformer"]["lr"],
    per_device_train_batch_size=cfg["transformer"]["batch_size"],
    per_device_eval_batch_size=cfg["transformer"]["batch_size"],
    num_train_epochs=cfg["transformer"]["epochs"],
    weight_decay=0.01,
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",  
    logging_steps=50,
    report_to="none"                        
)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Save model & tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logging.info(f"[train_transformer] Saved to {out_dir}")

    # Save validation predictions (probabilities) for stacking
    logging.info("[train_transformer] Generating validation probabilities for stacking...")
    preds_output = trainer.predict(dataset["validation"])
    logits = preds_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    np.save(cfg["paths"]["transformer_preds"], probs)
    logging.info(f"[train_transformer] Saved probs to {cfg['paths']['transformer_preds']}")

def _main_entry():
    import logging, argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main(args.features, args.out)

if __name__ == "__main__" or __name__ == "neural.train_transformer":
    _main_entry()