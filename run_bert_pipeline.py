"""
BERT Pipeline Runner
Complete pipeline for training models with BERT embeddings:
1. Create BERT features from raw data
2. Train XGBoost model with BERT features
3. Train MLP model with BERT features
"""

import argparse
import subprocess
import sys
import os
import logging

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
    
    # Step 2: Train XGBoost model
    if not args.skip_xgb:
        success = run_cmd(
            "STEP 2: Train XGBoost Model with BERT Features",
            [
                sys.executable, "-m", "core.train_xgb_bert",
                "--features", args.features,
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
                "--features", args.features,
                "--out", args.mlp_model
            ]
        )
        if not success:
            logger.error("MLP training failed.")
    else:
        logger.info("✅ Skipping MLP training")
    
    logger.info("\n" + "="*70)
    logger.info("✅ BERT Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"BERT Features: {args.features}")
    if not args.skip_xgb:
        logger.info(f"XGBoost Model: {args.xgb_model}")
    if not args.skip_mlp:
        logger.info(f"MLP Model: {args.mlp_model}")
    logger.info("\nYou can now use these models for prediction!")

if __name__ == "__main__":
    main()

