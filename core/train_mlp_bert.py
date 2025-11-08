import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.config import cfg
from core.labels import LABEL_MAP
from core.persistence import save_pipeline


# -------------------------------
# MLP Model Definition
# -------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Training Function (One Fold)
# -------------------------------
def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                true.extend(yb.cpu().numpy())

        acc = accuracy_score(true, preds)
        logging.info(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.4f} | Loss: {running_loss/len(train_loader):.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_acc


# -------------------------------
# Main Training Loop (Cross-Validation)
# -------------------------------
def main(features_path=None, out_path=None, epochs=10, batch_size=64):
    logging.info("[train_mlp_bert] Starting MLP training using BERT embeddings...")

    features_path = features_path or cfg["paths"]["processed_features"]
    out_path = out_path or "models/mlp_bert_pipeline.pt"

    df = pd.read_parquet(features_path)
    if "label" not in df:
        raise ValueError("Expected 'label' column in features parquet")

    # Convert textual labels to numeric using LABEL_MAP
    if df["label"].dtype == object:
        df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isnull().any():
        raise ValueError("Some labels could not be mapped to numeric values. Check LABEL_MAP.")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].astype(int).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    all_preds, all_true = [], []
    fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"[train_mlp_bert] Fold {fold}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        model, best_acc = train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs)
        fold_acc.append(best_acc)

        # Collect validation predictions
        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, _ in val_loader:
                xb = xb.to(device)
                outputs = model(xb)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_preds.extend(val_preds)
            all_true.extend(y_val)

    acc = accuracy_score(all_true, all_preds)
    logging.info(f"[train_mlp_bert] ✅ Mean CV Accuracy: {acc:.4f}")
    logging.info(f"[train_mlp_bert] Fold Accuracies: {fold_acc}")

    cm = confusion_matrix(all_true, all_preds)

    # ✅ Fixed Section
    present_labels = unique_labels(all_true, all_preds)

    # Handle both mapping directions
    if all(isinstance(k, int) for k in LABEL_MAP.keys()):
        target_names = [LABEL_MAP[i] for i in present_labels if i in LABEL_MAP]
    else:
        inv_label_map = {v: k for k, v in LABEL_MAP.items()}
        target_names = [inv_label_map.get(i, str(i)) for i in present_labels]

    if not target_names:  # Fallback
        target_names = [f"class_{i}" for i in present_labels]

    report = classification_report(all_true, all_preds, labels=present_labels, target_names=target_names)
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(f"Classification Report:\n{report}")

    # -------------------------------
    # Final Training on Full Dataset
    # -------------------------------
    logging.info("[train_mlp_bert] Training final model on full dataset...")
    full_ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

    final_model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        final_model.train()
        running_loss = 0.0
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = final_model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Final Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(full_loader):.4f}")

    pipeline = {
        "model_state_dict": final_model.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "label_map": LABEL_MAP,
        "config": cfg
    }
    
    # Save as PyTorch file if .pt extension, otherwise use joblib
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith('.pt') or out_path.endswith('.pth'):
        torch.save(pipeline, out_path)
    else:
        save_pipeline(pipeline, out_path)
    logging.info(f"[train_mlp_bert] ✅ Saved MLP model to {out_path}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args.features, args.out, args.epochs, args.batch_size)
