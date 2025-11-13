# core/severity.py

"""
Aggregate claim-level verifications into continuous severity score and band.

Severity formula (configurable):
    severity = 1 - mean(entailment_prob) weighted by claim importance and retrieval relevance,
    but amplify contradictions.

Output band:
    severity in [0, 0.33) -> 'safe'
    [0.33, 0.66) -> 'review'
    [0.66, 1.0] -> 'unsafe'
"""

from typing import List, Dict
import math
import numpy as np


def compute_severity(claim_verifications: List[Dict], weights: Dict = None) -> Dict:
    """
    claim_verifications: list of dicts returned by verify_claim
    returns dict: {"severity": float, "band": str, "per_claim": [...], "metadata": {...}}
    """
    if not claim_verifications:
        return {"severity": 0.0, "band": "safe", "per_claim": [], "metadata": {"note": "no_claims"}}

    per_claim_scores = []
    # default weighting: longer claims and those with higher retriever score get more weight
    for cv in claim_verifications:
        claim_text = cv["claim"]
        score = cv.get("score", 0.5)  # 0..1 where 1 -> entailment-leaning, 0 -> contradiction-leaning
        # invert: we want severity high when contradiction strong -> use 1 - score
        contra_strength = 1.0 - score
        # importance heuristics
        length_weight = min(1.0, len(claim_text.split()) / 20.0)  # longer claims up to weight 1
        # retrieval relevance: take max rel_score from evidence
        rel_scores = [e.get("rel_score", 0.0) for e in cv.get("evidence", [])]
        rel_weight = max(rel_scores) if rel_scores else 0.0
        # combined weight
        w = 0.5 * length_weight + 0.5 * rel_weight
        # per-claim severity contribution
        per_score = contra_strength * (0.5 + w)  # base amplification
        per_claim_scores.append({"claim": claim_text, "per_score": per_score, "weight": w, "raw_score": score, "verdict": cv.get("verdict")})

    # normalize into 0..1
    scores = [c["per_score"] for c in per_claim_scores]
    if not scores:
        severity = 0.0
    else:
        # robust aggregator: 1 - exp(-mean*alpha) to amplify severe cases (alpha tunable)
        alpha = 1.2
        mean_raw = float(np.mean(scores))
        severity = min(1.0, 1 - math.exp(-alpha * mean_raw))

    if severity < 0.33:
        band = "safe"
    elif severity < 0.66:
        band = "review"
    else:
        band = "unsafe"

    return {"severity": float(severity), "band": band, "per_claim": per_claim_scores}

