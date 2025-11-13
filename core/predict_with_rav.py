# core/predict_with_rav.py
"""
Orchestration wrapper to run classifier + light RAV verification.
Works with XGBoost sklearn API (joblib) or a PyTorch MLP (if provided).
"""

import joblib
import numpy as np
import json
from pathlib import Path
from core.retriever import Retriever
from core.entailment import Verifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABEL_MAP_INV = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

def load_xgb(path):
    return joblib.load(path)

def predict_proba_xgb(model, X):
    return model.predict_proba(X)

def run_rav_on_batch(features_np, claims, model_path,
                     index_dir="data/index",
                     nli_model=None,
                     top_k=3,
                     low_thresh=0.35,
                     high_thresh=0.75,
                     keep_examples=1000):
    """
    features_np: numpy array (N x D) of features used by classifier
    claims: list[str] claim texts in same order as features
    Returns: results list of dicts with base + rav decisions
    """
    model = load_xgb(model_path)
    retriever = Retriever(index_dir=index_dir)
    verifier = Verifier(model_name=nli_model) if nli_model else Verifier()
    probs = predict_proba_xgb(model, features_np)
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)

    results = []
    for i, (feat_row, claim, pred, prob) in enumerate(zip(features_np, claims, preds, max_probs)):
        base_label = LABEL_MAP_INV[int(pred)]
        out = {
            "claim": claim,
            "base_label": base_label,
            "base_prob": float(prob),
            "evidence": [],
            "final_label": base_label,
            "rav_used": False,
            "rav_meta": {}
        }

        # thresholding: only run RAV if model confidence is ambiguous
        if low_thresh < prob < high_thresh:
            try:
                hits = retriever.query(claim, top_k=top_k)
                ev_texts = [h["text"] for h in hits]
                out["evidence"] = [{"text": t, "score": h["score"]} for t,h in zip(ev_texts, hits)]
                out["rav_used"] = True
                agg, per_chunk = verifier.verify_topk(ev_texts, claim, aggregate="max")
                out["rav_meta"]["nli_agg"] = agg
                out["rav_meta"]["per_chunk"] = per_chunk

                # replace existing decision rules with this
                # compute mean evidence score (retriever score in [0,1], may be small)
                mean_ev_score = float(np.mean([h.get("score", 0.0) for h in hits])) if hits else 0.0
                ent = agg["entailment"]
                contr = agg["contradiction"]

                # New conservative rules:
                MIN_NLI_FOR_FLIP = 0.75    # require stronger NLI signal
                MIN_EVIDENCE_SCORE = 0.28  # require minimum retrieval score (tuneable)

                if ent >= MIN_NLI_FOR_FLIP and mean_ev_score >= MIN_EVIDENCE_SCORE:
                    out["final_label"] = "SUPPORTS"
                elif contr >= MIN_NLI_FOR_FLIP and mean_ev_score >= MIN_EVIDENCE_SCORE:
                    out["final_label"] = "REFUTES"
                else:
                    # keep base label but mark as verified/unverified
                    out["final_label"] = out["base_label"]

                out["rav_meta"]["mean_evidence_score"] = mean_ev_score
                out["rav_meta"]["nli_thresh_used"] = MIN_NLI_FOR_FLIP
                out["rav_meta"]["evidence_score_thresh"] = MIN_EVIDENCE_SCORE

            except Exception as e:
                out["rav_meta"]["error"] = str(e)

        results.append(out)
        # small safety: avoid huge memory if running on large eval set and user only wants sample
        if keep_examples and i + 1 >= keep_examples:
            break
    return results

def evaluate_results(results, gold_labels=None):
    """
    results: list of dicts (final_label)
    gold_labels: list[str] or None
    Returns simple metrics dict
    """
    preds = [r["final_label"] for r in results]
    if gold_labels:
        acc = accuracy_score(gold_labels, preds)
        cr = classification_report(gold_labels, preds, digits=4, zero_division=0, output_dict=True)
        cm = confusion_matrix(gold_labels, preds, labels=["SUPPORTS","REFUTES","NOT ENOUGH INFO"]).tolist()
        return {"accuracy": acc, "classification_report": cr, "confusion_matrix": cm}
    else:
        return {"n": len(preds)}
