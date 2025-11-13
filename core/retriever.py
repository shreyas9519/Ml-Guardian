# core/retriever.py
"""
FAISS + SBERT Retriever with simple on-disk cache.
Designed to be fast and plug into the RAV pipeline.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import hashlib
import os
import pickle

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_dir="data/index", model_name=DEFAULT_MODEL, use_bm25=False):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "meta.json"
        self.cache_path = self.index_dir / "query_cache.pkl"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []
        self.use_bm25 = use_bm25
        self._load_index_if_exists()
        self._load_cache()

    def _load_index_if_exists(self):
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.index = None
                self.meta = []

    def _load_cache(self):
        if Path(self.cache_path).exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass

    def build(self, texts, save=True):
        """
        Build FAISS index for a list of texts.
        texts: list[str]
        """
        if not texts:
            raise ValueError("Empty corpus passed to Retriever.build()")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine sim
        index.add(embs)
        self.index = index
        self.meta = [{"id": i, "text": t} for i, t in enumerate(texts)]
        if save:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "w", encoding="utf8") as f:
                json.dump(self.meta, f, ensure_ascii=False)

    def _hash(self, text, top_k):
        return hashlib.sha1((text + str(top_k)).encode("utf8")).hexdigest()

    def query(self, text, top_k=5, use_cache=True):
        """
        Return list of dicts: {id, text, score}
        """
        if use_cache:
            key = self._hash(text, top_k)
            if key in self._cache:
                return self._cache[key]

        if self.index is None:
            raise RuntimeError("Retriever index not built. Call build() first or provide index files.")

        q_emb = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            md = self.meta[idx]
            results.append({"id": md["id"], "text": md["text"], "score": float(score)})
        if use_cache:
            self._cache[key] = results
            self._save_cache()
        return results

    def load_from_dir(self, index_dir):
        """Reload index and meta from a directory."""
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "meta.json"
        self._load_index_if_exists()
        self._load_cache()
