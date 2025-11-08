# core/train_xgb.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, logging
import numpy as np
import pandas as pd
from core.config import cfg
from core.labels import LABEL_MAP
from core.persistence import save_pipeline
from core.metrics import classification_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

def main(features_path=None, out_path=None):
    features_path = features_path or cfg["paths"]["processed_features"]
    out_path = out_path or cfg["paths"]["xgb_model"]

    logging.info(f"[train_xgb] Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    if "label" not in df:
        raise ValueError("Need 'label' column in features parquet")

    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values

    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    logging.info(f"[train_xgb] Class weights: {class_weight_dict}")

    set_seed(cfg["seed"])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["seed"])

    # Try small grid of parameter combinations
    param_grid = [
        {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 800},
        {"learning_rate": 0.03, "max_depth": 8, "n_estimators": 1000},
        {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 600}
    ]

    best_acc = 0
    best_params = None

    for grid in param_grid:
        logging.info(f"[train_xgb] Testing params: {grid}")
        fold_acc = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            sample_weights = np.array([class_weight_dict[label] for label in y_tr])

            model = XGBClassifier(
                learning_rate=grid["learning_rate"],
                max_depth=grid["max_depth"],
                n_estimators=grid["n_estimators"],
                subsample=cfg["xgb"]["subsample"],
                colsample_bytree=cfg["xgb"]["colsample_bytree"],
                reg_lambda=cfg["xgb"].get("reg_lambda", 1.0),
                use_label_encoder=False,
                eval_metric=cfg["xgb"]["eval_metric"],
                n_jobs=-1,
                random_state=cfg["seed"]
            )

            model.fit(X_tr, y_tr, sample_weight=sample_weights, verbose=False)
            preds = model.predict(X_va)
            acc = np.mean(preds == y_va)
            fold_acc.append(acc)

        avg_acc = np.mean(fold_acc)
        logging.info(f"→ Mean CV Accuracy: {avg_acc:.4f}")
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = grid

    logging.info(f"[train_xgb] ✅ Best Params: {best_params} with CV Acc={best_acc:.4f}")

    # Train final model
    logging.info("[train_xgb] Training final model on full data...")
    final_weights = np.array([class_weight_dict[label] for label in y])

    final_model = XGBClassifier(
        **best_params,
        subsample=cfg["xgb"]["subsample"],
        colsample_bytree=cfg["xgb"]["colsample_bytree"],
        reg_lambda=cfg["xgb"].get("reg_lambda", 1.0),
        use_label_encoder=False,
        eval_metric=cfg["xgb"]["eval_metric"],
        n_jobs=-1,
        random_state=cfg["seed"]
    )

    final_model.fit(X, y, sample_weight=final_weights, verbose=False)

    pipeline = {
        "model": final_model,
        "feature_names": list(df.drop(columns=["label"]).columns),
        "label_map": LABEL_MAP,
        "config": cfg
    }
    save_pipeline(pipeline, out_path)
    logging.info(f"[train_xgb] Saved pipeline to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    main(args.features, args.out)
