# core/rerank_retriever.py

"""
Lightweight retriever: TF-IDF + optional FAISS (if installed).
Provides retrieve_evidence(query, k=5).

This is intentionally simple: it gives you retrieval features without heavy RAG setup.
If you already have a FAISS index from RAV, swap the retrieve implementation.

Requires:
    - sklearn
    - joblib (for caching)
Optional:
    - faiss (if you build an index using embeddings)
"""

from typing import List, Dict
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

_INDEX_DIR = "data/index"
_TFIDF_MODEL = os.path.join(_INDEX_DIR, "tfidf.joblib")
_DOCS_FILE = os.path.join(_INDEX_DIR, "corpus.jsonl")  # expected: [{"id":..., "text": ...}, ...]


def build_tfidf_index(corpus_jsonl_path: str = _DOCS_FILE, out_model_path: str = _TFIDF_MODEL):
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    docs = []
    ids = []
    # tolerate UTF-8 BOM if present
    with open(corpus_jsonl_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            item = json.loads(line)
            ids.append(item.get('id'))
            docs.append(item.get('text', ''))
    vec = TfidfVectorizer(max_features=200000, ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform(docs)
    joblib.dump({"vectorizer": vec, "matrix": X, "ids": ids}, out_model_path)
    print(f"[retriever] TF-IDF index built and saved to {out_model_path}")


def load_tfidf_index(model_path: str = _TFIDF_MODEL):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TF-IDF model not found at {model_path}. Run build_tfidf_index()")
    data = joblib.load(model_path)
    return data["vectorizer"], data["matrix"], data["ids"]


def retrieve_evidence(query: str, k: int = 5, model_path: str = _TFIDF_MODEL) -> List[Dict]:
    """
    Returns list of dicts: {"doc_id": str, "text": str, "score": float}
    """
    vec, matrix, ids = load_tfidf_index(model_path)
    qv = vec.transform([query])
    cosine_similarities = linear_kernel(qv, matrix).flatten()
    top_idx = np.argsort(cosine_similarities)[::-1][:k]
    results = []
    # we must have access to the corpus file to fetch full text - read lazily
    # Assume corpus_jsonl path is same as used while building: _DOCS_FILE
    corpus = {}
    with open(_DOCS_FILE, 'r', encoding='utf-8-sig') as f:
        for line in f:
            item = json.loads(line)
            corpus[item['id']] = item.get('text','')
    for idx in top_idx:
        doc_id = ids[idx]
        results.append({
            "doc_id": doc_id,
            "text": corpus.get(doc_id, ""),
            "score": float(cosine_similarities[idx])
        })
    return results

