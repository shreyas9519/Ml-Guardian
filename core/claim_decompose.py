# core/claim_decompose.py

"""
Heuristic + optional LLM-prompt claim decomposer.

Primary function:
    decompose_to_claims(text: str, max_claims=6) -> List[str]

Notes:
- Start conservative to avoid over-splitting.
- You can later swap heuristics for an LLM prompt-based splitter if you have an LLM available.
"""

import re
from typing import List

_CONJUNCTIONS = r'\b(and|or|but|however|while|although|whereas|,)\b'


def _split_on_punctuation(text: str) -> List[str]:
    # Split on sentence punctuation first
    # Keep abbreviations naive approach
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_on_conjunctions(clause: str) -> List[str]:
    # further split on conjunctions if clause is long
    pieces = re.split(_CONJUNCTIONS, clause)
    # remove pure conjunction tokens and very short pieces
    out = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(p.split()) < 3:
            # skip trivial pieces (tune 3 -> 4 tokens)
            continue
        out.append(p)
    return out


def decompose_to_claims(text: str, max_claims: int = 6) -> List[str]:
    """
    Return list of atomic claims extracted from text.
    Conservative defaults to avoid over-splitting.
    """
    if not text or not text.strip():
        return []

    sentences = _split_on_punctuation(text)
    claims = []
    for s in sentences:
        if len(s.split()) <= 12:
            # short sentence -> keep
            claims.append(s)
            continue
        # long sentence -> try to split by conjunctions/cause phrases
        pieces = _split_on_conjunctions(s)
        if pieces:
            for p in pieces:
                claims.append(p)
        else:
            # fallback: split by commas but keep only substantial pieces
            comma_parts = [c.strip() for c in s.split(',') if len(c.strip().split()) >= 3]
            if comma_parts:
                claims.extend(comma_parts)
            else:
                claims.append(s)

        if len(claims) >= max_claims:
            break

    # final clean-up: strip and deduplicate preserving order
    seen = set()
    out = []
    for c in claims:
        c = re.sub(r'\s+', ' ', c).strip()
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    return out[:max_claims]

