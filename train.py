"""
NeuroAlert — Phase 4: Model Training
Two model options:
  A) Random Forest on extracted features  (fast, interpretable)
  B) 1D-CNN on raw EEG epochs            (more impressive, takes longer)

Run with:  python train.py --model rf      (Random Forest, default)
           python train.py --model cnn     (1D-CNN with PyTorch)
"""

import os
import sys
import argparse
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Option A — Random Forest (recommended for 24-hr hackathon)
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X: np.ndarray, y: np.ndarray) -> dict:
    """Train a Random Forest with 5-fold cross-validation."""
    print("\n── Training Random Forest ──────────────────────────────")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validation
    print("Running 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(clf, X_sc, y, cv=skf, method='predict_proba')
    y_pred       = (y_pred_proba[:, 1] > 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba[:, 1])
    f1  = f1_score(y, y_pred, zero_division=0)

    print(f"\n  AUC-ROC : {auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print("\n" + classification_report(y, y_pred,
          target_names=['Inter-ictal', 'Pre-ictal']))

    # Final model on all data
    clf.fit(X_sc, y)

    # Save
    joblib.dump(clf,    MODELS_DIR / "rf_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    metrics = {"model": "RandomForest", "auc": round(auc, 4), "f1": round(f1, 4)}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved → models/rf_model.pkl + models/scaler.pkl")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Option B — 1D-CNN with PyTorch
# ══════════════════════════════════════════════════════════════════════════════

def train_cnn(X_raw: np.ndarray, y: np.ndarray) -> dict:
    """Train a 1D-CNN on raw EEG epochs."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    print("\n── Training 1D-CNN ─────────────────────────────────────")

    class SeizureCNN(nn.Module):
        def __init__(self, n_channels: int = 23, n_samples: int = 1280):
            super().__init__()
            self.conv_block = nn.Sequential(
                # Temporal convolution across all EEG channels
                nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Dropout(0.4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8, 128),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.classifier(self.conv_block(x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Train/val split stratified
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    def to_tensor(arr, lbl):
        return (torch.FloatTensor(arr).to(device),
                torch.LongTensor(lbl).to(device))

    X_tr_t,  y_tr_t  = to_tensor(X_tr,  y_tr)
    X_val_t, y_val_t = to_tensor(X_val, y_val)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True
    )

    model     = SeizureCNN().to(device)
    weights   = torch.FloatTensor([1.0, float(np.sum(y == 0) / (np.sum(y == 1) + 1e-8))]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_auc, best_state = 0.0, None
    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_val, probs)

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val-AUC={auc:.4f}")

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs  = torch.softmax(model(X_val_t), dim=1)[:, 1].cpu().numpy()
    preds = (probs > 0.5).astype(int)
    f1    = f1_score(y_val, preds, zero_division=0)

    print(f"\n  Best Val AUC : {best_auc:.4f}")
    print(f"  Final F1     : {f1:.4f}")
    print("\n" + classification_report(y_val, preds,
          target_names=['Inter-ictal', 'Pre-ictal']))

    torch.save(model.state_dict(), MODELS_DIR / "cnn_model.pt")
    torch.save({"n_channels": X_raw.shape[1], "n_samples": X_raw.shape[2]},
               MODELS_DIR / "cnn_config.pt")

    metrics = {"model": "1D-CNN", "auc": round(best_auc, 4), "f1": round(f1, 4)}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved → models/cnn_model.pt + models/cnn_config.pt")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "cnn"], default="rf",
                        help="rf = Random Forest (fast), cnn = 1D-CNN (powerful)")
    parser.add_argument("--data",  default=None,
                        help="Path to CHB-MIT data directory (optional; uses synthetic if not given)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.data:
        # Real CHB-MIT data
        sys.path.insert(0, ".")
        from data_loader import load_patient_data
        print(f"Loading real data from {args.data}...")
        X_raw, y = load_patient_data(args.data, patient="chb01")
        np.save(DATA_DIR / "X_raw.npy", X_raw)
        np.save(DATA_DIR / "y_labels.npy", y)
    else:
        # Use synthetic data (for hackathon demo without dataset download)
        raw_path = DATA_DIR / "X_raw.npy"
        lbl_path = DATA_DIR / "y_labels.npy"

        if not raw_path.exists():
            print("No data found — generating synthetic EEG data...")
            sys.path.insert(0, ".")
            from data_loader import generate_synthetic_data
            X_raw, y = generate_synthetic_data(n_samples=2000)
            np.save(raw_path, X_raw)
            np.save(lbl_path, y)
        else:
            X_raw = np.load(raw_path)
            y     = np.load(lbl_path)

    print(f"Dataset: {X_raw.shape[0]} epochs, class distribution: {np.bincount(y)}")

    if args.model == "rf":
        # Feature extraction needed for RF
        feat_path = DATA_DIR / "X_features.npy"
        if feat_path.exists():
            X_feat = np.load(feat_path)
            print(f"Loaded cached features: {X_feat.shape}")
        else:
            print("Extracting features...")
            sys.path.insert(0, ".")
            from features import extract_features
            X_feat = extract_features(X_raw)
            np.save(feat_path, X_feat)
        train_random_forest(X_feat, y)

    elif args.model == "cnn":
        train_cnn(X_raw, y)


if __name__ == "__main__":
    main()
