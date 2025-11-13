# inference/run_claim_pipeline.py

"""
CLI wrapper to run claim pipeline for a single text or a batch file (newline JSONL).
Produces per-example JSONL outputs with claim-level verifications and overall severity.

Usage:
    python inference/run_claim_pipeline.py --text "Some generated sentence..."
    python inference/run_claim_pipeline.py --input data/to_verify.jsonl --output outputs/claims_out.jsonl

Expected input JSONL format (if using --input):
{"id": "<unique_id>", "text": "<generated_text>"}
"""
# inference/run_claim_pipeline.py (top of file)
import os
import sys

# add repo root to PYTHONPATH so "import core.*" works when running the script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
from core.claim_decompose import decompose_to_claims
from core.rerank_retriever import retrieve_evidence, build_tfidf_index
from core.atomic_verifier import verify_claim, NLIModel
from core.severity import compute_severity
from typing import List


def process_text(text: str, retriever_fn, nli_model, topk: int = 5):
    claims = decompose_to_claims(text)
    claim_results = []
    for c in claims:
        vr = verify_claim(c, retriever_fn, nli_model=nli_model, topk=topk)
        claim_results.append(vr)
    agg = compute_severity(claim_results)
    return {"text": text, "claims": claim_results, "severity": agg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Single text to process")
    parser.add_argument("--input", type=str, default=None, help="JSONL input file with id+text")
    parser.add_argument("--output", type=str, default="outputs/claim_pipeline_out.jsonl")
    parser.add_argument("--build_index", action="store_true", help="(Re)build TF-IDF index from data/index/corpus.jsonl")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--nli_model", type=str, default=None)
    args = parser.parse_args()

    if args.build_index:
        print("[cli] Building TF-IDF index from data/index/corpus.jsonl ...")
        build_tfidf_index()

    retriever_fn = lambda q, k: retrieve_evidence(q, k=k)
    nli = NLIModel(model_name=args.nli_model) if args.nli_model else NLIModel()

    outputs = []
    if args.text:
        out = process_text(args.text, retriever_fn, nli, topk=args.topk)
        print(json.dumps(out, indent=2))
        return

    if args.input:
        # ensure output dir exists
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.input, 'r', encoding='utf-8-sig') as fin, open(args.output, 'w', encoding='utf8') as fout:
            for line in fin:
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                item = json.loads(line)
                uid = item.get("id")
                text = item.get("text","")
                out = process_text(text, retriever_fn, nli, topk=args.topk)
                out_record = {"id": uid, "result": out}
                fout.write(json.dumps(out_record) + "\n")
        print(f"[cli] Wrote outputs to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

