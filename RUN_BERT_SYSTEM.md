# How to Run the BERT System

## 🚀 Quick Start

### Step 1: Activate Virtual Environment
```powershell
# Navigate to project directory
cd "d:\Comdur\ML Project\-ML-Guardian-Hallucination-Detector"

# Activate virtual environment
.\mlvenv\Scripts\Activate.ps1

# Or if using mlvenv310
.\mlvenv310\Scripts\Activate.ps1
```

### Step 2: Run Complete BERT Pipeline
```powershell
# Run with default settings (uses train_tiny.jsonl)
python run_bert_pipeline.py
```

## 📋 Detailed Commands

### Option A: Automated Pipeline (Recommended)

```powershell
# Complete pipeline with default settings
python run_bert_pipeline.py

# With custom dataset
python run_bert_pipeline.py --raw data/raw/fever_data/train_sample.jsonl

# With custom BERT model
python run_bert_pipeline.py --bert_model distilbert-base-uncased

# With larger batch size (if you have GPU)
python run_bert_pipeline.py --batch_size 32
```

### Option B: Manual Step-by-Step

#### Step 1: Create BERT Features
```powershell
python -m core.create_features_bert --raw data/raw/fever_data/train_tiny.jsonl --out data/processed/bert_features.parquet --model_name bert-base-uncased --batch_size 16
```

#### Step 2: Train XGBoost Model
```powershell
python -m core.train_xgb_bert --features data/processed/bert_features.parquet --out models/xgb_bert_pipeline.joblib
```

#### Step 3: Train MLP Model
```powershell
python -m core.train_mlp_bert --features data/processed/bert_features.parquet --out models/mlp_bert_pipeline.pt --epochs 10 --batch_size 64
```

## 🎯 All Commands in One Place

Copy and paste these commands in order:

```powershell
# 1. Activate environment
.\mlvenv\Scripts\Activate.ps1

# 2. Create BERT features
python -m core.create_features_bert --raw data/raw/fever_data/train_tiny.jsonl --out data/processed/bert_features.parquet --model_name bert-base-uncased --batch_size 16

# 3. Train XGBoost
python -m core.train_xgb_bert --features data/processed/bert_features.parquet --out models/xgb_bert_pipeline.joblib

# 4. Train MLP
python -m core.train_mlp_bert --features data/processed/bert_features.parquet --out models/mlp_bert_pipeline.pt --epochs 10 --batch_size 64
```

## 🔧 Configuration Options

### Available Datasets
- `data/raw/fever_data/train_tiny.jsonl` - Small test dataset (fast)
- `data/raw/fever_data/train_sample.jsonl` - Medium dataset
- `data/raw/fever_data/train.jsonl` - Full dataset (slow)

### Available BERT Models
- `bert-base-uncased` - Default, balanced (768 dims)
- `distilbert-base-uncased` - Faster (768 dims)
- `roberta-base` - Better accuracy (768 dims)
- `bert-large-uncased` - Best accuracy, slower (1024 dims)

### Batch Size Recommendations
- **CPU**: 8-16
- **GPU (4GB)**: 16-32
- **GPU (8GB+)**: 32-64

## 📁 Output Files

After running the pipeline, you'll have:
- `data/processed/bert_features.parquet` - BERT embeddings
- `models/xgb_bert_pipeline.joblib` - Trained XGBoost model
- `models/mlp_bert_pipeline.pt` - Trained MLP model

## ⚠️ Troubleshooting

### If you get "Module not found" errors:
```powershell
# Make sure virtual environment is activated
.\mlvenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### If you get CUDA out of memory:
```powershell
# Reduce batch size
python run_bert_pipeline.py --batch_size 8
```

### If feature extraction is slow:
- Use smaller dataset: `--raw data/raw/fever_data/train_tiny.jsonl`
- Use faster model: `--bert_model distilbert-base-uncased`
- Check if GPU is being used (should see "✅ Using GPU" in logs)

## 📊 Expected Runtime

- **Feature Extraction**: 2-10 minutes (depending on dataset size and GPU)
- **XGBoost Training**: 1-5 minutes
- **MLP Training**: 5-15 minutes

## ✅ Verification

After training, verify the models were created:
```powershell
# Check if files exist
Test-Path data/processed/bert_features.parquet
Test-Path models/xgb_bert_pipeline.joblib
Test-Path models/mlp_bert_pipeline.pt
```

## 🎓 Next Steps

After training:
1. Use models for prediction (see `predict.py`)
2. Evaluate model performance
3. Fine-tune hyperparameters
4. Train on larger datasets

## 📚 Additional Resources

- See `BERT_PIPELINE_GUIDE.md` for detailed documentation
- See `BERT_COMMANDS.md` for quick reference
- See `README.md` for project overview

