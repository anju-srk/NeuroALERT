"""
NeuroAlert — Inference Engine
Loads the trained model and provides a predict() function for the dashboard.
Supports both RF (sklearn) and CNN (PyTorch) backends.
"""

import numpy as np
import joblib
import json
from pathlib import Path
from typing import Tuple

MODELS_DIR = Path("models")
SFREQ      = 256
N_CHANNELS = 23
EPOCH_SAMPLES = SFREQ * 5   # 5-second window


def _load_backend():
    """Auto-detect which model was trained and load it."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None, None

    with open(metrics_path) as f:
        meta = json.load(f)
    model_type = meta.get("model", "RandomForest")

    if "CNN" in model_type:
        return _load_cnn()
    else:
        return _load_rf()


def _load_rf():
    try:
        clf    = joblib.load(MODELS_DIR / "rf_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        return ("rf", {"clf": clf, "scaler": scaler})
    except FileNotFoundError:
        return None, None


def _load_cnn():
    try:
        import torch
        import torch.nn as nn

        cfg = torch.load(MODELS_DIR / "cnn_config.pt", map_location="cpu")

        class SeizureCNN(nn.Module):
            def __init__(self, n_channels, n_samples):
                super().__init__()
                self.conv_block = nn.Sequential(
                    nn.Conv1d(n_channels, 64,  kernel_size=7, padding=3),
                    nn.BatchNorm1d(64), nn.ELU(),
                    nn.MaxPool1d(2), nn.Dropout(0.3),
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128), nn.ELU(),
                    nn.MaxPool1d(2), nn.Dropout(0.3),
                    nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256), nn.ELU(),
                    nn.AdaptiveAvgPool1d(8), nn.Dropout(0.4),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 8, 128), nn.ELU(), nn.Dropout(0.5),
                    nn.Linear(128, 2),
                )
            def forward(self, x):
                return self.classifier(self.conv_block(x))

        model = SeizureCNN(cfg["n_channels"], cfg["n_samples"])
        model.load_state_dict(torch.load(MODELS_DIR / "cnn_model.pt", map_location="cpu"))
        model.eval()
        return ("cnn", {"model": model})
    except (FileNotFoundError, Exception) as e:
        print(f"CNN load error: {e}")
        return None, None


class NeuroAlertPredictor:
    """
    Wraps the trained model for inference.
    Usage:
        predictor = NeuroAlertPredictor()
        risk, proba = predictor.predict(epoch)   # epoch: (23, 1280) ndarray
    """

    def __init__(self):
        self.backend_type, self.backend = _load_backend()
        if self.backend is None:
            print("⚠ No trained model found — using demo mode (random scores).")

    @property
    def is_loaded(self) -> bool:
        return self.backend is not None

    def predict(self, epoch: np.ndarray) -> Tuple[float, float]:
        """
        Predict seizure risk for a single 5-second EEG epoch.
        epoch: ndarray of shape (n_channels, n_samples)
        Returns: (risk_score 0-100, raw_probability 0-1)
        """
        if not self.is_loaded:
            return self._demo_predict(epoch)

        if self.backend_type == "rf":
            return self._rf_predict(epoch)
        else:
            return self._cnn_predict(epoch)

    def _rf_predict(self, epoch: np.ndarray) -> Tuple[float, float]:
        import sys
        sys.path.insert(0, ".")
        from features import extract_features

        epoch_norm = self._normalize(epoch)
        feats = extract_features(epoch_norm[np.newaxis])[0]
        feats_sc = self.backend["scaler"].transform(feats.reshape(1, -1))
        proba = float(self.backend["clf"].predict_proba(feats_sc)[0, 1])
        return round(proba * 100, 1), proba

    def _cnn_predict(self, epoch: np.ndarray) -> Tuple[float, float]:
        import torch
        epoch_norm = self._normalize(epoch)
        x = torch.FloatTensor(epoch_norm).unsqueeze(0)
        with torch.no_grad():
            logits = self.backend["model"](x)
            proba  = float(torch.softmax(logits, dim=1)[0, 1].item())
        return round(proba * 100, 1), proba

    def _demo_predict(self, epoch: np.ndarray) -> Tuple[float, float]:
        """Heuristic demo predictor using raw signal energy (no trained model needed)."""
        # High-frequency energy as a proxy for seizure activity
        from scipy.signal import welch
        hf_power = 0.0
        for ch in range(min(epoch.shape[0], N_CHANNELS)):
            freqs, psd = welch(epoch[ch], fs=SFREQ, nperseg=128)
            mask = (freqs >= 20) & (freqs <= 40)
            hf_power += float(np.mean(psd[mask]))
        hf_power /= N_CHANNELS

        # Normalize to 0-1 range empirically
        proba = float(np.clip(hf_power / 5.0, 0.0, 1.0))
        return round(proba * 100, 1), proba

    @staticmethod
    def _normalize(epoch: np.ndarray) -> np.ndarray:
        mean = epoch.mean(axis=1, keepdims=True)
        std  = epoch.std(axis=1, keepdims=True) + 1e-8
        return ((epoch - mean) / std).astype(np.float32)


# Singleton — import this in the dashboard
predictor = NeuroAlertPredictor()
