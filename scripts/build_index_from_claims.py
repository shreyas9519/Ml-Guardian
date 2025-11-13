# scripts/build_index_from_claims.py
from core.retriever import Retriever
import json
from pathlib import Path

RAW = "data/raw/fever_data/train.jsonl"
INDEX_DIR = "data/index"

texts = []
p = Path(RAW)
if not p.exists():
    raise SystemExit(f"Raw file not found: {RAW}")

with open(RAW, "r", encoding="utf8") as f:
    for i, line in enumerate(f):
        try:
            d = json.loads(line)
            claim = d.get("claim", "")
            texts.append(claim if claim is not None else "")
        except Exception:
            texts.append("")

print(f"Building index with {len(texts)} entries -> {INDEX_DIR} (this may take a minute)...")
r = Retriever(index_dir=INDEX_DIR)
r.build(texts, save=True)
print("Index build finished. Files written to:", INDEX_DIR)
