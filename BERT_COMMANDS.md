# BERT Pipeline - Quick Reference Commands

## Setup (One-time)

```powershell
# Activate virtual environment
.\mlvenv\Scripts\Activate.ps1

# Or if using mlvenv310
.\mlvenv310\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt
```

## Quick Start - Complete Pipeline

```powershell
# Run complete BERT pipeline (recommended)
python run_bert_pipeline.py
```

## Step-by-Step Commands

### Step 1: Create BERT Features
```powershell
python -m core.create_features_bert --raw data/raw/fever_data/train_tiny.jsonl --out data/processed/bert_features.parquet --model_name bert-base-uncased --batch_size 16
```

### Step 2: Train XGBoost
```powershell
python -m core.train_xgb_bert --features data/processed/bert_features.parquet --out models/xgb_bert_pipeline.joblib
```

### Step 3: Train MLP
```powershell
python -m core.train_mlp_bert --features data/processed/bert_features.parquet --out models/mlp_bert_pipeline.pt --epochs 10 --batch_size 64
```

## Alternative: Using Different Datasets

```powershell
# Small dataset (fast testing)
python run_bert_pipeline.py --raw data/raw/fever_data/train_tiny.jsonl

# Medium dataset
python run_bert_pipeline.py --raw data/raw/fever_data/train_sample.jsonl

# Full dataset
python run_bert_pipeline.py --raw data/raw/fever_data/train.jsonl --batch_size 32
```

## Using Different BERT Models

```powershell
# Faster model
python run_bert_pipeline.py --bert_model distilbert-base-uncased

# Better accuracy
python run_bert_pipeline.py --bert_model roberta-base

# Best accuracy (slower)
python run_bert_pipeline.py --bert_model bert-large-uncased --batch_size 8
```

## Skip Steps (if already done)

```powershell
# Skip feature extraction
python run_bert_pipeline.py --skip_features

# Skip XGBoost training
python run_bert_pipeline.py --skip_xgb

# Skip MLP training
python run_bert_pipeline.py --skip_mlp
```

## All-in-One Command (Copy-Paste Ready)

```powershell
# Activate environment and run complete pipeline
.\mlvenv\Scripts\Activate.ps1
python run_bert_pipeline.py --raw data/raw/fever_data/train_tiny.jsonl --batch_size 16
```



