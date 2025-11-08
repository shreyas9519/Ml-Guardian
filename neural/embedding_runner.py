#!/usr/bin/env python3
"""
embedding_runner.py
Reads a JSON array of texts from stdin and writes JSON array of embeddings to stdout.
✅ Fully GPU-optimized for RTX 4050 (PyTorch + SentenceTransformers)
✅ Safe subprocess behavior (no TF interference, no thread explosion)
"""

import sys
import json
import os
import warnings
import torch

# ------------------------ Safe threading and environment setup ------------------------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TF_CPP_MIN_LOG_LEVEL": "3",  # suppress TensorFlow logs if installed
})
warnings.filterwarnings("ignore")

# ------------------------ Device setup ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[embedding_runner] Using device: {DEVICE}", file=sys.stderr)

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ------------------------ Model setup ------------------------
from sentence_transformers import SentenceTransformer
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def load_model():
    """Load SentenceTransformer model on GPU (cached globally)."""
    global _model
    if _model is None:
        try:
            print(f"[embedding_runner] Loading model '{MODEL_NAME}' on {DEVICE}", file=sys.stderr)
            _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Model load failed: {e}\n")
            sys.stdout.write(json.dumps({"error": f"model_load_failed: {str(e)}"}))
            sys.exit(1)
    return _model


# ------------------------ Main embedding routine ------------------------
def main():
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw.decode("utf8"))
        texts = payload.get("texts") if isinstance(payload, dict) else payload
        if not isinstance(texts, list):
            raise ValueError("Invalid input: expected list of texts")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Bad input: {e}\n")
        sys.stdout.write(json.dumps({"error": f"bad_input: {e}"}))
        sys.exit(1)

    try:
        model = load_model()
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                device=DEVICE,
                show_progress_bar=False,
            )
        sys.stdout.write(json.dumps({"embeddings": embeddings.tolist()}))
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"[ERROR] Encode failed: {e}\n")
        sys.stdout.write(json.dumps({"error": f"encode_failed: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()