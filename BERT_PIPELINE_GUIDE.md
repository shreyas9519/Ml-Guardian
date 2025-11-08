# BERT Pipeline Guide for ML Guardian

This guide explains how to run the ML Guardian hallucination detection system using BERT embeddings.

## Overview

The BERT pipeline consists of three main steps:
1. **Feature Extraction**: Generate BERT embeddings from raw FEVER dataset
2. **XGBoost Training**: Train XGBoost classifier on BERT features
3. **MLP Training**: Train Multi-Layer Perceptron on BERT features

## Prerequisites

1. **Activate Virtual Environment**
   ```powershell
   # Windows PowerShell
   .\mlvenv\Scripts\Activate.ps1
   
   # Or if using mlvenv310
   .\mlvenv310\Scripts\Activate.ps1
   ```

2. **Install Dependencies** (if not already installed)
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```powershell
   python -c "import torch; import transformers; print('✅ All dependencies installed')"
   ```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Use the automated pipeline runner:

```powershell
# Activate virtual environment first
.\mlvenv\Scripts\Activate.ps1

# Run complete BERT pipeline with default settings
python run_bert_pipeline.py

# Or with custom settings
python run_bert_pipeline.py --raw data/raw/fever_data/train_tiny.jsonl --bert_model bert-base-uncased --batch_size 16
```

### Option 2: Run Steps Individually

#### Step 1: Create BERT Features

```powershell
python -m core.create_features_bert --raw data/raw/fever_data/train_tiny.jsonl --out data/processed/bert_features.parquet --model_name bert-base-uncased --batch_size 16
```

**Parameters:**
- `--raw`: Path to input JSONL file (e.g., `train_tiny.jsonl`, `train_sample.jsonl`, or `train.jsonl`)
- `--out`: Path to output parquet file with BERT embeddings
- `--model_name`: BERT model name from HuggingFace (default: `bert-base-uncased`)
- `--batch_size`: Batch size for inference (default: 16, increase if you have GPU memory)

**Available BERT Models:**
- `bert-base-uncased` (default, 768 dimensions)
- `bert-large-uncased` (1024 dimensions, slower but more accurate)
- `distilbert-base-uncased` (768 dimensions, faster)
- `roberta-base` (768 dimensions)

#### Step 2: Train XGBoost Model

```powershell
python -m core.train_xgb_bert --features data/processed/bert_features.parquet --out models/xgb_bert_pipeline.joblib
```

**Parameters:**
- `--features`: Path to BERT features parquet file
- `--out`: Path to save trained XGBoost model

#### Step 3: Train MLP Model

```powershell
python -m core.train_mlp_bert --features data/processed/bert_features.parquet --out models/mlp_bert_pipeline.pt --epochs 10 --batch_size 64
```

**Parameters:**
- `--features`: Path to BERT features parquet file
- `--out`: Path to save trained MLP model
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 64)

## Advanced Usage

### Skip Specific Steps

If you've already generated features or trained a model, you can skip steps:

```powershell
# Skip feature extraction (use existing features)
python run_bert_pipeline.py --skip_features

# Skip XGBoost training
python run_bert_pipeline.py --skip_xgb

# Skip MLP training
python run_bert_pipeline.py --skip_mlp

# Only train MLP (skip features and XGBoost)
python run_bert_pipeline.py --skip_features --skip_xgb
```

### Use Different Dataset Sizes

```powershell
# Small test dataset (fast)
python run_bert_pipeline.py --raw data/raw/fever_data/train_tiny.jsonl

# Medium dataset
python run_bert_pipeline.py --raw data/raw/fever_data/train_sample.jsonl

# Full dataset (slow, requires more memory)
python run_bert_pipeline.py --raw data/raw/fever_data/train.jsonl --batch_size 32
```

### Use Different BERT Models

```powershell
# Use DistilBERT (faster)
python run_bert_pipeline.py --bert_model distilbert-base-uncased

# Use RoBERTa (better accuracy)
python run_bert_pipeline.py --bert_model roberta-base

# Use BERT Large (best accuracy, slower)
python run_bert_pipeline.py --bert_model bert-large-uncased --batch_size 8
```

## Expected Output

### Step 1: Feature Extraction
- Creates a parquet file with BERT embeddings (768 or 1024 dimensions depending on model)
- File: `data/processed/bert_features.parquet`
- Contains columns: `bert_0`, `bert_1`, ..., `bert_767`, `label`

### Step 2: XGBoost Training
- Performs 5-fold cross-validation
- Outputs mean CV accuracy and confusion matrix
- Saves model: `models/xgb_bert_pipeline.joblib`

### Step 3: MLP Training
- Performs 5-fold cross-validation
- Outputs mean CV accuracy, confusion matrix, and classification report
- Saves model: `models/mlp_bert_pipeline.pt`

## File Structure

```
├── core/
│   ├── create_features_bert.py    # BERT feature extraction
│   ├── train_xgb_bert.py          # XGBoost training with BERT
│   └── train_mlp_bert.py          # MLP training with BERT
├── data/
│   ├── raw/fever_data/
│   │   ├── train_tiny.jsonl       # Small test dataset
│   │   ├── train_sample.jsonl     # Medium dataset
│   │   └── train.jsonl            # Full dataset
│   └── processed/
│       └── bert_features.parquet  # Generated BERT features
├── models/
│   ├── xgb_bert_pipeline.joblib   # Trained XGBoost model
│   └── mlp_bert_pipeline.pt       # Trained MLP model
└── run_bert_pipeline.py           # Pipeline runner script
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size
```powershell
python run_bert_pipeline.py --batch_size 8
```

### Issue: Module not found errors
**Solution:** Activate virtual environment and install dependencies
```powershell
.\mlvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: Slow feature extraction
**Solution:** 
- Use GPU if available (CUDA will be automatically detected)
- Use smaller BERT model: `--bert_model distilbert-base-uncased`
- Reduce batch size if using CPU
- Use smaller dataset for testing: `--raw data/raw/fever_data/train_tiny.jsonl`

### Issue: Evidence extraction warnings
**Note:** The FEVER dataset contains evidence as Wikipedia references, not actual text. The pipeline extracts entity names from these references. This is expected behavior.

## Performance Tips

1. **Use GPU**: The pipeline automatically detects and uses CUDA if available
2. **Batch Size**: 
   - GPU: 16-32 (depending on GPU memory)
   - CPU: 8-16
3. **Model Selection**:
   - Fast: `distilbert-base-uncased`
   - Balanced: `bert-base-uncased` (default)
   - Best Accuracy: `bert-large-uncased` or `roberta-base`
4. **Dataset Size**: Start with `train_tiny.jsonl` for testing, then scale up

## Next Steps

After training, you can:
1. Use the trained models for prediction (see `predict.py`)
2. Evaluate model performance
3. Fine-tune hyperparameters
4. Train ensemble models combining XGBoost and MLP

## Command Summary

```powershell
# Complete pipeline
python run_bert_pipeline.py

# Individual steps
python -m core.create_features_bert --raw data/raw/fever_data/train_tiny.jsonl --out data/processed/bert_features.parquet
python -m core.train_xgb_bert --features data/processed/bert_features.parquet --out models/xgb_bert_pipeline.joblib
python -m core.train_mlp_bert --features data/processed/bert_features.parquet --out models/mlp_bert_pipeline.pt
```



