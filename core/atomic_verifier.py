# core/atomic_verifier.py

"""
Atomic verifier using a transformers NLI model (e.g., roberta-large-mnli or deBERTa).
We expose `verify_claim` which runs retrieval + NLI and returns structured verdicts.

Requirements:
    pip install transformers torch sentencepiece
Tune:
    - nli_model_name default uses 'roberta-large-mnli' (change as needed)
    - If you have a fine-tuned NLI checkpoint (FEVER-like), set via config
"""
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import math

# default model
_DEFAULT_NLI = "roberta-large-mnli"


class NLIModel:
    def __init__(self, model_name: str = _DEFAULT_NLI, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        # label mapping for MNLI-style: 0 - contradiction, 1 - neutral, 2 - entailment
        self.labels = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def predict(self, premise: str, hypothesis: str) -> Dict:
        """
        Returns {'label': str, 'probs': [p0, p1, p2], 'score': float}
        'score' is probability of entailment by default (tunable).
        """
        encoded = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.model(**encoded)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = self.labels[pred_idx]
            # interpret score: use entailment prob; if contradiction highest, set negative score
            entail_prob = float(probs[2])
            contra_prob = float(probs[0])
            neutral_prob = float(probs[1])
        return {"label": label, "probs": probs.tolist(), "entail_prob": entail_prob,
                "contra_prob": contra_prob, "neutral_prob": neutral_prob}


def verify_claim(claim: str, retriever_fn, nli_model: Optional[NLIModel] = None, topk: int = 5) -> Dict:
    """
    Runs retrieval (retriever_fn) and NLI verification.
    retriever_fn(query, k) -> List[{"doc_id","text","score"}]
    Returns:
    {
      "claim": claim,
      "verdict": "entailment"/"contradiction"/"neutral",
      "score": float (0..1, entailment prob),
      "evidence": [{"doc_id", "snippet", "rel_score", "nli": {...}}],
      "metadata": {...}
    }
    """
    if nli_model is None:
        nli_model = NLIModel()  # lazy init

    evidence = retriever_fn(claim, k=topk)
    best = None
    evidence_results = []
    # Evaluate NLI for each evidence doc (premise=doc_text, hyp=claim)
    for ev in evidence:
        premise = ev["text"]
        nli = nli_model.predict(premise, claim)
        evidence_results.append({
            "doc_id": ev["doc_id"],
            "snippet": premise[:1000],  # save preview; you can do smarter snippet extraction
            "rel_score": ev.get("score", 0.0),
            "nli": nli
        })
        # choose best by max (contra or entail) significance:
        # prioritize high entail_prob; keep best entail or highest contradiction for severity scoring
        if best is None:
            best = evidence_results[-1]
        else:
            # pick candidate with larger max(entail_prob, contra_prob) weighted by rel_score
            prev_max = max(best["nli"]["entail_prob"], best["nli"]["contra_prob"]) * (1.0 + best["rel_score"])
            cur_max = max(nli["entail_prob"], nli["contra_prob"]) * (1.0 + ev.get("score", 0.0))
            if cur_max > prev_max:
                best = evidence_results[-1]

    # If no evidence docs, neutral with low confidence
    if not evidence_results:
        return {"claim": claim, "verdict": "neutral", "score": 0.0, "evidence": [], "metadata": {"note":"no_evidence"}}

    # Determine verdict by best NLI label
    best_nli = best["nli"]
    verdict = best_nli["label"]
    # use a signed score: entail_prob - contra_prob normalized into 0..1
    signed_score = best_nli["entail_prob"] - best_nli["contra_prob"]
    # map signed_score (-1..1) to 0..1
    score = (signed_score + 1.0) / 2.0
    return {"claim": claim, "verdict": verdict, "score": float(score), "evidence": evidence_results, "metadata": {"best_doc": best["doc_id"]}}

