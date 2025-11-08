# Add parent dir to sys.path for module-safe imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# neural/train_mlp.py
import os, argparse, logging
import numpy as np
import pandas as pd
from core.config import cfg
from core.labels import LABEL_MAP, REVERSE_LABEL_MAP
from core.metrics import classification_metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = cfg["seed"]

def set_seed(seed=SEED):
    import random, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_mlp(input_dim, emb_dim=0, hidden_sizes=(256,128), dropout=0.2, num_classes=3):
    inputs = Input(shape=(input_dim,), name="features")
    x = inputs
    if emb_dim>0:
        emb_in = Input(shape=(emb_dim,), name="emb")
        x = Concatenate()([inputs, emb_in])
    for h in hidden_sizes:
        x = Dense(h, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    out = Dense(num_classes, activation="softmax")(x)
    if emb_dim>0:
        model = Model(inputs=[inputs, emb_in], outputs=out)
    else:
        model = Model(inputs=inputs, outputs=out)
    return model

def main(features_path=None, claim_emb_path=None, evid_emb_path=None, out_model=None, use_embeddings=True):
    set_seed()
    features_path = features_path or cfg["paths"]["processed_features"]
    claim_emb_path = claim_emb_path or cfg["paths"]["claim_emb"]
    evid_emb_path = evid_emb_path or cfg["paths"]["evid_emb"]
    out_model = out_model or cfg["paths"]["mlp_model"]
    logging.info(f"[train_mlp] Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].astype(int).values
    # load embeddings if requested
    if use_embeddings:
        if not os.path.exists(claim_emb_path) or not os.path.exists(evid_emb_path):
            logging.warning("Embeddings not found. Training MLP without embeddings.")
            use_embeddings = False
            emb = None
            emb_dim = 0
        else:
            try:
                claim_emb = np.load(claim_emb_path)
                evid_emb = np.load(evid_emb_path)
                emb = np.concatenate([claim_emb, evid_emb], axis=1)
                emb_dim = emb.shape[1]
                logging.info(f"[train_mlp] Loaded embeddings with dimension {emb_dim}")
            except Exception as e:
                logging.warning(f"Failed to load embeddings: {e}. Training without embeddings.")
                use_embeddings = False
                emb = None
                emb_dim = 0
    else:
        emb = None
        emb_dim = 0

    # train/val split with index alignment to keep embeddings in sync
    idx = np.arange(len(X))
    idx_tr, idx_va, y_tr, y_va = train_test_split(idx, y, test_size=0.15, random_state=SEED, stratify=y)
    X_tr, X_va = X[idx_tr], X[idx_va]
    if use_embeddings:
        emb_tr = emb[idx_tr]
        emb_va = emb[idx_va]
    else:
        emb_tr = emb_va = None

    # scaling
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    # class weights
    counts = Counter(y_tr)
    total = sum(counts.values())
    class_weight = {k: total/(len(counts)*v) for k,v in counts.items()}
    logging.info(f"[train_mlp] class weight: {class_weight}")

    model = build_mlp(input_dim=X_tr.shape[1], emb_dim=emb_dim,
                      hidden_sizes=tuple(cfg["mlp"]["hidden_sizes"]),
                      dropout=cfg["mlp"]["dropout"], num_classes=len(LABEL_MAP))
    model.compile(optimizer=Adam(cfg["mlp"]["lr"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(out_model+".h5", save_best_only=True, monitor="val_loss")
    ]
    if use_embeddings:
        history = model.fit([X_tr, emb_tr], y_tr, validation_data=([X_va, emb_va], y_va),
                          epochs=cfg["mlp"]["epochs"], batch_size=cfg["mlp"]["batch_size"],
                          class_weight=class_weight, callbacks=callbacks)
    else:
        history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                          epochs=cfg["mlp"]["epochs"], batch_size=cfg["mlp"]["batch_size"],
                          class_weight=class_weight, callbacks=callbacks)
    # Evaluate on validation and print accuracy + confusion matrix
    try:
        if use_embeddings:
            prob_va = model.predict([X_va, emb_va], verbose=0)
        else:
            prob_va = model.predict(X_va, verbose=0)
        y_pred = prob_va.argmax(axis=1)
        labels_order = list(LABEL_MAP.keys())
        metrics = classification_metrics(y_va, y_pred, labels=labels_order)
        logging.info(f"[train_mlp] Val accuracy: {metrics['accuracy']:.4f}")
        label_names = [LABEL_MAP[i] for i in labels_order]
        cm = metrics["confusion_matrix"]
        header = "\t" + "\t".join(label_names)
        logging.info("[train_mlp] Confusion Matrix (rows=true, cols=pred):")
        logging.info(header)
        for i, row in enumerate(cm):
            row_str = label_names[i] + "\t" + "\t".join(str(v) for v in row)
            logging.info(row_str)
    except Exception as e:
        logging.warning(f"[train_mlp] Failed to compute/print validation metrics: {e}")
    # Save scaler + model wrapper
    try:
        import joblib
        joblib.dump({"scaler": scaler, "model_path": out_model+".h5", "use_embeddings": use_embeddings}, out_model+".joblib")
    except Exception as e:
        logging.warning(f"Failed to save joblib wrapper: {e}")
    logging.info(f"[train_mlp] Saved model to {out_model}.h5 and wrapper {out_model}.joblib")

if __name__ == "__main__":
    import logging, argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=None)
    p.add_argument("--claim_emb", default=None)
    p.add_argument("--evid_emb", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no_embeddings", action="store_true")
    args = p.parse_args()
    main(args.features, args.claim_emb, args.evid_emb, args.out, use_embeddings=(not args.no_embeddings))