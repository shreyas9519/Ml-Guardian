# tests/test_claim_pipeline.py

"""
Quick local sanity test. Requires you to have data/index/corpus.jsonl and built TF-IDF index.
If not available, create a tiny corpus for test.
"""

import os
import json
from core.claim_decompose import decompose_to_claims
from core.rerank_retriever import build_tfidf_index, retrieve_evidence
from core.atomic_verifier import NLIModel, verify_claim
from core.severity import compute_severity


def make_dummy_corpus():
    os.makedirs("data/index", exist_ok=True)
    docs = [
        {"id": "d1", "text": "Paris is the capital city of France. It has the Eiffel Tower."},
        {"id": "d2", "text": "The Nile is the longest river in Africa."},
        {"id": "d3", "text": "Python is a programming language created by Guido van Rossum."}
    ]
    with open("data/index/corpus.jsonl", "w", encoding="utf8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    build_tfidf_index()


def test_pipeline():
    make_dummy_corpus()
    text = "Paris is the capital of France, and the Nile flows through Germany."
    claims = decompose_to_claims(text)
    nli = NLIModel()
    claim_results = []
    for c in claims:
        rv = verify_claim(c, retrieve_evidence, nli_model=nli, topk=3)
        claim_results.append(rv)
    agg = compute_severity(claim_results)
    print("Claims:", claims)
    print("Claim results:", claim_results)
    print("Aggregate:", agg)


if __name__ == "__main__":
    test_pipeline()

