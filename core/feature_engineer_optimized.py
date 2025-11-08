"""
OptimizedFeatureEngineer (GPU-enabled version for Windows/macOS/Linux)
💡 Supports RTX GPU acceleration (via CUDA in embedding subprocess).
Embeddings handled via subprocess runner (embedding_runner.py).
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# --- Thread controls to avoid CPU oversubscription ---
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re, json, subprocess, warnings, time
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple

# --- GPU check for subprocess ---
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"
    print(f"[INIT] PyTorch GPU available: {GPU_AVAILABLE} ({GPU_DEVICE})")
except Exception:
    GPU_AVAILABLE = False
    GPU_DEVICE = "CPU"
    print("[INFO] torch not available — embeddings will use CPU")

# --- spaCy setup ---
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
    print("[INFO] spaCy not available - advanced NLP features disabled")


class OptimizedFeatureEngineer:
    def __init__(self, spacy_model_name: str = "en_core_web_sm"):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model_name, disable=["parser", "textcat"])
                self.nlp.max_length = 2_000_000
                print(f"[SUCCESS] Loaded spaCy model: {spacy_model_name}")
            except OSError:
                print(f"[INFO] spaCy model '{spacy_model_name}' not found. Install it using:")
                print(f"    python -m spacy download {spacy_model_name}")
            except Exception as e:
                print(f"[WARNING] spaCy load failed: {e}")
        self.word_pattern = re.compile(r"\b\w+\b")
        self.number_pattern = re.compile(r"\d+")
        self.whitespace_pattern = re.compile(r"\s+")
        print(f"[READY] FeatureEngineer ready — embeddings via subprocess (GPU={GPU_AVAILABLE})")

    # --------------------------------------------------
    def _safe_text(self, x: Any) -> str:
        return str(x).strip() if x is not None else ""

    # ---------- basic + structural + advanced ----------
    def _basic_lexical_features(self, c, e):
        cw, ew = set(self.word_pattern.findall(c.lower())), set(self.word_pattern.findall(e.lower()))
        overlap, union = len(cw & ew), len(cw | ew) or 1
        ct, et = c.split(), e.split()
        return {
            "word_overlap": overlap / (len(cw) or 1),
            "jaccard_similarity": overlap / union,
            "claim_length": len(ct),
            "evidence_length": len(et),
            "length_ratio": len(ct) / (len(et) + 1e-8),
            "token_overlap_count": overlap,
        }

    def _structural_features(self, c, e):
        chn, ehn = bool(self.number_pattern.search(c)), bool(self.number_pattern.search(e))
        ca = ''.join(filter(str.isalpha, c))
        ea = ''.join(filter(str.isalpha, e))
        cc = sum(1 for x in ca if x.isupper()) / (len(ca) or 1)
        ec = sum(1 for x in ea if x.isupper()) / (len(ea) or 1)
        cs = [s for s in self.whitespace_pattern.sub(' ', c).split('.') if s.strip()]
        es = [s for s in self.whitespace_pattern.sub(' ', e).split('.') if s.strip()]
        return {
            "claim_has_numbers": float(chn),
            "evidence_has_numbers": float(ehn),
            "claim_caps_ratio": cc,
            "evidence_caps_ratio": ec,
            "claim_sentence_count": len(cs),
            "evidence_sentence_count": len(es)
        }

    def _advanced_similarity_features(self, c, e):
        feats = {}
        if not c.strip() or not e.strip():
            return {k: 0.0 for k in [
                "entity_overlap_ratio", "entity_jaccard", "noun_similarity",
                "verb_similarity", "adj_similarity"
            ]}

        if self.nlp:
            try:
                cd, ed = self.nlp(c), self.nlp(e)
                ce, ee = {ent.text.lower() for ent in cd.ents}, {ent.text.lower() for ent in ed.ents}
                if ce:
                    o = len(ce & ee)
                    feats["entity_overlap_ratio"] = o / len(ce)
                    feats["entity_jaccard"] = o / len(ce | ee) if (ce | ee) else 0.0
                cp, ep = [t.pos_ for t in cd], [t.pos_ for t in ed]
                for pos in ["NOUN", "VERB", "ADJ"]:
                    cr = cp.count(pos) / len(cp)
                    er = ep.count(pos) / len(ep)
                    feats[f"{pos.lower()}_similarity"] = 1.0 - abs(cr - er)
            except Exception as ex:
                warnings.warn(f"spaCy failed: {ex}")
        for k in ["entity_overlap_ratio", "entity_jaccard", "noun_similarity", "verb_similarity", "adj_similarity"]:
            feats.setdefault(k, 0.0)
        return feats

    def _composite_features(self, base):
        sc, eo, wo, js = (
            base.get("semantic_cosine", 0.0),
            base.get("entity_overlap_ratio", 0.0),
            base.get("word_overlap", 0.0),
            base.get("jaccard_similarity", 0.0),
        )
        return {
            "semantic_entity_interaction": sc * eo,
            "semantic_lexical_balance": sc * wo,
            "composite_confidence": sc * 0.4 + eo * 0.3 + wo * 0.2 + js * 0.1,
        }

    def _clean(self, d):
        out = {}
        for k, v in d.items():
            try:
                v = float(v)
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
                if k in ["semantic_cosine", "jaccard_similarity", "word_overlap"]:
                    v = max(0.0, min(1.0, v))
                out[k] = v
            except Exception:
                out[k] = 0.0
        return out

    # ---------- subprocess embedding with GPU ----------
    def _run_embedding_batch(self, texts):
        try:
            inp = json.dumps({"texts": texts}).encode("utf8")
            runner_path = os.path.join(os.path.dirname(__file__), "..", "neural", "embedding_runner.py")

            # Pass GPU flag to subprocess
            env_vars = {**os.environ}
            if GPU_AVAILABLE:
                env_vars["USE_CUDA"] = "1"

            proc = subprocess.run(
                [sys.executable, runner_path],
                input=inp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                env=env_vars
            )

            if proc.returncode != 0:
                warnings.warn(f"embedding subprocess failed (code {proc.returncode})\nStderr: {proc.stderr.decode('utf8')[:300]}")
                return [[0.0] * 384 for _ in texts]

            out = json.loads(proc.stdout.decode("utf8") or "{}")
            if "embeddings" in out:
                return out["embeddings"]
            elif "error" in out:
                warnings.warn(f"embedding subprocess error: {out['error']}")
        except Exception as e:
            warnings.warn(f"embedding subprocess failed: {e}")
        return [[0.0] * 384 for _ in texts]

    # ---------- unified feature extractor ----------
    def extract_all_features(self, claim: str, evidence: str) -> Dict[str, float]:
        claim = self._safe_text(claim)
        evidence = self._safe_text(evidence)

        try:
            base_feats = {}
            base_feats.update(self._basic_lexical_features(claim, evidence))
            base_feats.update(self._structural_features(claim, evidence))
            base_feats.update(self._advanced_similarity_features(claim, evidence))
            base_feats.update(self._composite_features(base_feats))
            return self._clean(base_feats)
        except Exception as e:
            warnings.warn(f"extract_all_features failed: {e}")
            return {}

    # ---------- batch extraction ----------
    def batch_extract_with_embeddings(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        total = len(df)
        feats, claim_embs, evid_embs = [], [], []
        claims = df["claim"].astype(str).tolist()
        evids = df["evidence"].astype(str).tolist()
        pbar = tqdm(total=total, desc="Extracting claim–evidence pairs", ncols=100)
        start = time.time()

        for i in range(0, total, batch_size):
            c_batch = claims[i:i + batch_size]
            e_batch = evids[i:i + batch_size]

            for c, e in zip(c_batch, e_batch):
                feats.append(self.extract_all_features(c, e))

            claim_embs.extend(self._run_embedding_batch(c_batch))
            evid_embs.extend(self._run_embedding_batch(e_batch))

            pbar.update(len(c_batch))
            done = i + len(c_batch)
            if done % 500 == 0 or done == total:
                elapsed = time.time() - start
                speed = done / max(elapsed, 1e-6)
                eta = (total - done) / max(speed, 1e-6)
                pbar.set_postfix({"done": f"{done/total:.1%}", "ETA": f"{eta/60:.1f}m"})
        pbar.close()

        print(f"[SUCCESS] Finished {total} pairs in {round((time.time() - start) / 60, 2)} min")
        return pd.DataFrame(feats).fillna(0.0), np.array(claim_embs, dtype="float32"), np.array(evid_embs, dtype="float32")
