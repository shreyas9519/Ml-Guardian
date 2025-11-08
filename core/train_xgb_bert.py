import pandas as pd
import numpy as np
import logging
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def detect_tree_method():
    """
    Automatically select the best XGBoost tree_method.
    If GPU XGBoost is available, use 'gpu_hist', else 'hist'.
    """
    try:
        if torch.cuda.is_available():
            # Try to verify GPU support
            test_model = xgb.XGBClassifier(tree_method="gpu_hist")
            _ = test_model.get_params()  # test parameter validity
            logger.info("✅ CUDA GPU detected and supported by XGBoost.")
            return "gpu_hist"
        else:
            logger.warning("⚠️ CUDA not available. Using CPU mode ('hist').")
            return "hist"
    except Exception as e:
        logger.warning(f"⚠️ GPU mode unavailable ({e}). Falling back to CPU.")
        return "hist"

def main(features_path="data/processed/bert_features.parquet", out_path="models/xgb_bert_pipeline.joblib"):
    logger.info(f"📂 Loading BERT features from {features_path}")
    df = pd.read_parquet(features_path)

    if "label" not in df.columns:
        raise ValueError("❌ The features file must include a 'label' column.")

    # Prepare features and labels
    X = df.drop("label", axis=1).values
    
    # Convert textual labels to numeric using LABEL_MAP
    from core.labels import LABEL_MAP
    if df["label"].dtype == object:
        y = df["label"].map(LABEL_MAP).values
        if pd.isna(y).any():
            logger.warning("⚠️ Some labels could not be mapped. Using default mapping.")
            label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2, "NEI": 2}
            y = df["label"].map(label_map).fillna(2).astype(int).values
    else:
        y = df["label"].astype(int).values

    # Detect GPU or CPU
    tree_method = detect_tree_method()
    logger.info(f"🚀 Using XGBoost tree_method = '{tree_method}'")

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    conf_total = np.zeros((3, 3), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"\n===== Fold {fold} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost model
        model = xgb.XGBClassifier(
            learning_rate=0.03,
            max_depth=8,
            n_estimators=600,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method=tree_method,
            random_state=42,
            use_label_encoder=False
        )

        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Evaluate
        acc = accuracy_score(y_val, preds)
        conf = confusion_matrix(y_val, preds)
        scores.append(acc)
        conf_total += conf

        logger.info(f"✅ Fold {fold} Accuracy: {acc:.4f}")

    # Summary
    mean_acc = np.mean(scores)
    logger.info(f"\n🎯 Mean CV Accuracy: {mean_acc:.4f}")
    logger.info(f"📊 Confusion Matrix (rows=true, cols=pred):\n{conf_total}")

    # Train final model on full dataset
    logger.info("\n🧠 Training final model on full dataset...")
    final_model = xgb.XGBClassifier(
        learning_rate=0.03,
        max_depth=8,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method=tree_method,
        random_state=42,
        use_label_encoder=False
    )
    final_model.fit(X, y)

    # Save model
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_path)
    logger.info(f"💾 Saved final model to: {out_path}")
    logger.info("✅ Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost on BERT features")
    parser.add_argument("--features", type=str, default="data/processed/bert_features.parquet")
    parser.add_argument("--out", type=str, default="models/xgb_bert_pipeline.joblib")
    args = parser.parse_args()

    main(args.features, args.out)
