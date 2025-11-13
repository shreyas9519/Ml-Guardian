# core/entailment.py
"""
Light NLI verifier wrapper using HuggingFace transformers.
Default: roberta-large-mnli (works well off-the-shelf).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

DEFAULT_MODEL = "roberta-large-mnli"

class Verifier:
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # labels for MNLI-like models: contradiction, neutral, entailment
        self.labels = ["contradiction", "neutral", "entailment"]

    def verify_pair(self, premise, hypothesis, max_length=512):
        """
        Return dict of probabilities for contradiction/neutral/entailment.
        """
        inputs = self.tokenizer(premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return dict(zip(self.labels, probs))

    def verify_topk(self, evidence_texts, claim, aggregate="max"):
        """
        Run verify_pair over each evidence chunk.
        aggregate: 'max' or 'mean' (per-class)
        Returns aggregated dict and per-chunk dicts.
        """
        per_chunk = [self.verify_pair(ev, claim) for ev in evidence_texts]
        # aggregate
        agg = {}
        for key in self.labels:
            vals = [pc[key] for pc in per_chunk]
            agg[key] = float(max(vals) if aggregate == "max" else (sum(vals)/len(vals)))
        return agg, per_chunk
