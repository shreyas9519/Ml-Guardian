"""
BERT Pipeline Runner
Complete pipeline for training models with BERT embeddings:
1. Create BERT features from raw data
2. Train XGBoost model with BERT features
3. Train MLP model with BERT features
4. (Optional) Run Retrieval-Augmented Verification (RAV) evaluation
"""

import argparse
import subprocess
import sys
import os
import logging
import joblib
import pandas as pd
import json
from pathlib import Path
import numpy as np

# RAV helper functions (already implemented in core/)
from core.predict_with_rav import run_rav_on_batch, evaluate_results

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_cmd(label, cmd):
    """Helper to run subprocess with clean logs and error handling."""
    print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[SUCCESS] {label} completed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {label} failed with exit code {e.returncode}")
        return False


def _extract_feature_matrix(feats_df):
    """
    Try to extract a numeric feature matrix (numpy array) from the features dataframe.
    Handles common shapes:
    - A column named 'features' containing ndarray-like objects
    - Per-dimension numeric columns (all numeric columns except text/label columns)
    - If neither, raises ValueError
    """
    # Common textual columns to exclude
    exclude_cols = {"claim", "evidence", "label", "labels", "id", "text"}
    # 1) 'features' column (object / list / ndarray per row)
    if "features" in feats_df.columns:
        try:
            # If features entries are lists/ndarrays, stack into 2D array
            arrs = feats_df["features"].values
            # If already a 2D numpy array stored in a single cell, handle that
            if isinstance(arrs[0], (list, tuple, np.ndarray)):
                features_np = np.vstack([np.array(x) for x in arrs])
                return features_np
        except Exception:
            pass

    # 2) Numeric columns approach: select numeric dtypes and drop excluded
    numeric_df = feats_df.select_dtypes(include=[np.number]).copy()
    # drop label if numeric (we don't want label as feature)
    for c in list(numeric_df.columns):
        if c.lower() in exclude_cols:
            numeric_df.drop(columns=[c], inplace=True, errors=True)
    if numeric_df.shape[1] >= 1:
        return numeric_df.values

    # 3) Maybe features stored as multiple columns named feat_0, feat_1...
    feat_cols = [c for c in feats_df.columns if c.startswith("feat_") or c.startswith("embedding_")]
    if feat_cols:
        return feats_df[feat_cols].values

    raise ValueError("Could not extract feature matrix from features dataframe. "
                     "Ensure 'features' column or numeric feature columns exist.")


def main():
    parser = argparse.ArgumentParser(description="BERT Pipeline Runner for ML Guardian")
    parser.add_argument("--raw", default="data/raw/fever_data/train_tiny.jsonl",
                       help="Path to input JSONL file (default: train_tiny.jsonl)")
    parser.add_argument("--features", default="data/processed/bert_features.parquet",
                       help="Path to output BERT features (default: data/processed/bert_features.parquet)")
    parser.add_argument("--xgb_model", default="models/xgb_bert_pipeline.joblib",
                       help="Path to XGBoost model output (default: models/xgb_bert_pipeline.joblib)")
    parser.add_argument("--mlp_model", default="models/mlp_bert_pipeline.pt",
                       help="Path to MLP model output (default: models/mlp_bert_pipeline.pt)")
    parser.add_argument("--bert_model", default="bert-base-uncased",
                       help="BERT model name (default: bert-base-uncased)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for BERT inference (default: 16)")
    parser.add_argument("--skip_features", action="store_true",
                       help="Skip feature extraction if features already exist")
    parser.add_argument("--skip_xgb", action="store_true",
                       help="Skip XGBoost training")
    parser.add_argument("--skip_mlp", action="store_true",
                       help="Skip MLP training")
    
    # Claim verification options (Step 2.5)
    parser.add_argument("--use_claims", action="store_true",
                       help="Run claim verification pipeline and merge with BERT features")
    parser.add_argument("--claim_input", default="tmp/generated.jsonl",
                       help="Input JSONL for claim pipeline (default: tmp/generated.jsonl)")
    parser.add_argument("--claim_output", default="outputs/claims_out.jsonl",
                       help="Output JSONL from claim pipeline (default: outputs/claims_out.jsonl)")
    parser.add_argument("--merged_features", default="data/processed/bert_claim_features.parquet",
                       help="Output path for merged BERT+claim features (default: data/processed/bert_claim_features.parquet)")

    # RAV / verification options (Step 4)
    parser.add_argument("--use_rav", action="store_true",
                        help="Run Retrieval-Augmented Verification (RAV) after training")
    parser.add_argument("--rav_index", default="data/index",
                        help="Directory where FAISS index and meta.json live (default: data/index)")
    parser.add_argument("--rav_topk", type=int, default=3,
                        help="Top-k passages to retrieve for RAV (default: 3)")
    parser.add_argument("--rav_nli_model", default=None,
                        help="HuggingFace model name for NLI verifier (e.g., roberta-large-mnli). If None uses default.")
    parser.add_argument("--rav_low_thresh", type=float, default=0.35,
                        help="Lower threshold for base model confidence to trigger RAV (default: 0.35)")
    parser.add_argument("--rav_high_thresh", type=float, default=0.75,
                        help="Upper threshold for base model confidence to trigger RAV (default: 0.75)")
    parser.add_argument("--rav_sample", type=int, default=2000,
                        help="Number of evaluation examples to run RAV on (default: 2000). Use 0 or omit to run on full set.")
    parser.add_argument("--rav_keep_examples", type=int, default=2000,
                        help="Max examples to return in results array (safety) (default: 2000)")

    args = parser.parse_args()

    # Step 1: Create BERT features
    if not args.skip_features:
        if os.path.exists(args.features):
            logger.info(f"⚠️ Features file already exists: {args.features}")
            response = input("Do you want to regenerate features? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Skipping feature extraction...")
                args.skip_features = True

    if not args.skip_features:
        success = run_cmd(
            "STEP 1: Create BERT Features",
            [
                sys.executable, "-m", "core.create_features_bert",
                "--raw", args.raw,
                "--out", args.features,
                "--model_name", args.bert_model,
                "--batch_size", str(args.batch_size)
            ]
        )
        if not success:
            logger.error("Feature extraction failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("✅ Skipping feature extraction (using existing features)")

    # Step 2.5: Optional Claim Verification + Merge
    features_path = args.features  # Default to original BERT features
    if args.use_claims:
        logger.info("\n" + "="*70)
        logger.info(">>> STEP 2.5: Running Claim Verification + Merge")
        logger.info("="*70)
        
        # Check if claim input file exists
        if not os.path.exists(args.claim_input):
            logger.warning(f"Claim input file not found: {args.claim_input}")
            logger.warning("Skipping claim verification step. Using original BERT features.")
        else:
            # Run claim pipeline
            claim_success = run_cmd(
                "Claim Verification Pipeline",
                [
                    sys.executable, "inference/run_claim_pipeline.py",
                    "--input", args.claim_input,
                    "--output", args.claim_output
                ]
            )
            
            if claim_success and os.path.exists(args.claim_output):
                # Merge claim features with BERT features
                merge_success = run_cmd(
                    "Merge Claim Features with BERT Features",
                    [
                        sys.executable, "scripts/merge_claims_to_bert_features.py",
                        "--claims", args.claim_output,
                        "--bert", args.features,
                        "--out", args.merged_features
                    ]
                )
                
                if merge_success and os.path.exists(args.merged_features):
                    features_path = args.merged_features
                    logger.info(f"✅ Using merged features: {features_path}")
                else:
                    logger.warning("Merge failed. Falling back to original BERT features.")
            else:
                logger.warning("Claim verification failed. Falling back to original BERT features.")
        
        logger.info("="*70 + "\n")
    else:
        logger.info("ℹ️  Claim verification step skipped (use --use_claims to enable)")

    # Step 2: Train XGBoost model
    if not args.skip_xgb:
        success = run_cmd(
            "STEP 2: Train XGBoost Model with BERT Features",
            [
                sys.executable, "-m", "core.train_xgb_bert",
                "--features", features_path,
                "--out", args.xgb_model
            ]
        )
        if not success:
            logger.error("XGBoost training failed.")
    else:
        logger.info("✅ Skipping XGBoost training")

    # Step 3: Train MLP model
    if not args.skip_mlp:
        success = run_cmd(
            "STEP 3: Train MLP Model with BERT Features",
            [
                sys.executable, "-m", "core.train_mlp_bert",
                "--features", features_path,
                "--out", args.mlp_model
            ]
        )
        if not success:
            logger.error("MLP training failed.")
    else:
        logger.info("✅ Skipping MLP training")

    # Step 4: Optional RAV evaluation
    if args.use_rav:
        logger.info("\n" + "="*70)
        logger.info(">>> STEP 4: Running Retrieval-Augmented Verification (RAV) evaluation")
        logger.info("="*70)
        # Load features parquet (use merged if available)
        features_file = features_path if args.use_claims and os.path.exists(features_path) else args.features
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}. Cannot run RAV.")
        else:
            try:
                feats_df = pd.read_parquet(features_file)
            except Exception as e:
                logger.error(f"Failed to read features parquet {features_file}: {e}")
                feats_df = None

            if feats_df is None:
                logger.error("No features dataframe loaded; skipping RAV.")
            else:
                # Extract claims list
                claims = None
                if "claim" in feats_df.columns:
                    claims = feats_df["claim"].astype(str).tolist()
                else:
                    # fallback: try to load raw file and align by index
                    if os.path.exists(args.raw):
                        try:
                            raw_claims = []
                            with open(args.raw, "r", encoding="utf8") as rf:
                                for line in rf:
                                    try:
                                        d = json.loads(line)
                                        raw_claims.append(d.get("claim", ""))
                                    except Exception:
                                        raw_claims.append("")
                            if len(raw_claims) >= len(feats_df):
                                claims = raw_claims[:len(feats_df)]
                                logger.info("Aligned claims from raw JSONL to features.")
                            else:
                                # can't align perfectly, use empty claims
                                claims = [""] * len(feats_df)
                                logger.warning("Raw file shorter than features; using empty claims for missing rows.")
                        except Exception as e:
                            logger.warning(f"Could not load raw file to extract claims: {e}")
                            claims = [""] * len(feats_df)
                    else:
                        claims = [""] * len(feats_df)
                        logger.warning("No 'claim' column found and raw file missing; claims will be empty strings.")

                # Build feature matrix
                try:
                    features_np = _extract_feature_matrix(feats_df)
                except Exception as e:
                    logger.error(f"Failed to extract feature matrix: {e}")
                    features_np = None

                if features_np is None:
                    logger.error("No features matrix available; skipping RAV.")
                else:
                    # sample subset if specified
                    n_total = features_np.shape[0]
                    sample_n = n_total if (args.rav_sample is None or args.rav_sample <= 0) else min(n_total, args.rav_sample)
                    if sample_n < n_total:
                        logger.info(f"[RAV] Sampling first {sample_n} of {n_total} examples for quick evaluation.")
                    Xs = features_np[:sample_n]
                    cs = claims[:sample_n]

                    logger.info(f"[RAV] Running RAV on {sample_n} examples. Index dir: {args.rav_index}, top_k={args.rav_topk}")
                    try:
                        results = run_rav_on_batch(
                            features_np=Xs,
                            claims=cs,
                            model_path=args.xgb_model,
                            index_dir=args.rav_index,
                            nli_model=args.rav_nli_model,
                            top_k=args.rav_topk,
                            low_thresh=args.rav_low_thresh,
                            high_thresh=args.rav_high_thresh,
                            keep_examples=args.rav_keep_examples
                        )
                    except Exception as e:
                        logger.error(f"RAV run failed: {e}")
                        results = None

                    reports_dir = Path("reports")
                    reports_dir.mkdir(exist_ok=True)

                    if results is not None:
                        # Save results JSONL
                        results_path = reports_dir / "rav_results.jsonl"
                        try:
                            with open(results_path, "w", encoding="utf8") as wf:
                                for r in results:
                                    wf.write(json.dumps(r, ensure_ascii=False) + "\n")
                            logger.info(f"[RAV] Results saved to {results_path}")
                        except Exception as e:
                            logger.error(f"Failed to save RAV results: {e}")

                        # If we have gold labels in feats_df, evaluate and save metrics
                        gold_labels = None
                        if "label" in feats_df.columns:
                            # Map numeric labels (0/1/2) to text labels consistent with predict_with_rav
                            try:
                                # If numeric
                                if pd.api.types.is_integer_dtype(feats_df["label"].dtype) or pd.api.types.is_float_dtype(feats_df["label"].dtype):
                                    label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
                                    gold_labels = feats_df["label"].map(label_map).tolist()[:sample_n]
                                else:
                                    # assume already textual (maybe 'SUPPORTS' etc.)
                                    gold_labels = feats_df["label"].astype(str).tolist()[:sample_n]
                            except Exception:
                                gold_labels = None

                        if gold_labels is not None:
                            try:
                                metrics = evaluate_results(results, gold_labels=gold_labels)
                                metrics_path = reports_dir / "rav_metrics.json"
                                with open(metrics_path, "w", encoding="utf8") as mf:
                                    json.dump(metrics, mf, ensure_ascii=False, indent=2)
                                logger.info(f"[RAV] Metrics written to {metrics_path}")
                            except Exception as e:
                                logger.error(f"Failed to compute/save metrics: {e}")
                        else:
                            logger.info("[RAV] No gold labels found in features; metrics not computed.")
                    else:
                        logger.error("[RAV] No results produced.")
        logger.info(">>> STEP 4: RAV evaluation finished.")
        logger.info("="*70)

    logger.info("\n" + "="*70)
    logger.info("✅ BERT Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"Features Used: {features_path}")
    if args.use_claims and features_path != args.features:
        logger.info(f"  (Original BERT: {args.features})")
        logger.info(f"  (Merged with claims: {features_path})")
    if not args.skip_xgb:
        logger.info(f"XGBoost Model: {args.xgb_model}")
    if not args.skip_mlp:
        logger.info(f"MLP Model: {args.mlp_model}")
    logger.info("\nYou can now use these models for prediction!")


if __name__ == "__main__":
    main()
