"""
NeuroAlert — Phase 3: Feature Extraction
Extracts clinically-meaningful features from EEG epochs.
Used by the classical ML pipeline (Random Forest / XGBoost).
The deep learning pipeline uses raw epochs directly.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from typing import List


# ─── Frequency bands used in epilepsy research ────────────────────────────────
BANDS = {
    'delta': (0.5, 4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 40),
}

SFREQ = 256  # Hz


def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    """Compute absolute power in a frequency band using the trapezoidal rule."""
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def extract_channel_features(ch_signal: np.ndarray) -> np.ndarray:
    """
    Extract 12 features from a single EEG channel:
      - 5 band powers (delta, theta, alpha, beta, gamma)
      - 5 band power ratios (theta/alpha, beta/alpha, gamma/delta, delta/alpha, beta/theta)
      - Mean, variance, skewness, kurtosis
      - Zero-crossing rate
      - Line length (proxy for seizure energy)
      - Hjorth parameters: activity, mobility, complexity
    """
    feats = []

    # Welch PSD
    freqs, psd = welch(ch_signal, fs=SFREQ, nperseg=min(256, len(ch_signal) // 2))

    # Band powers
    powers = {}
    for name, (lo, hi) in BANDS.items():
        bp = band_power(psd, freqs, lo, hi)
        powers[name] = bp
        feats.append(bp)

    # Clinically meaningful ratios
    eps = 1e-10
    feats.append(powers['theta'] / (powers['alpha'] + eps))   # slowing ratio
    feats.append(powers['beta']  / (powers['alpha'] + eps))   # activation ratio
    feats.append(powers['gamma'] / (powers['delta'] + eps))   # fast/slow ratio
    feats.append(powers['delta'] / (powers['alpha'] + eps))   # delta dominance
    feats.append(powers['beta']  / (powers['theta'] + eps))   # beta/theta

    # Statistical features
    feats.append(float(np.mean(ch_signal)))
    feats.append(float(np.var(ch_signal)))
    feats.append(float(skew(ch_signal)))
    feats.append(float(kurtosis(ch_signal)))

    # Zero-crossing rate
    zcr = np.sum(np.diff(np.sign(ch_signal)) != 0) / len(ch_signal)
    feats.append(float(zcr))

    # Line length (captures high-frequency, high-amplitude activity)
    ll = np.sum(np.abs(np.diff(ch_signal))) / len(ch_signal)
    feats.append(float(ll))

    # Hjorth parameters
    activity   = float(np.var(ch_signal))
    diff1      = np.diff(ch_signal)
    diff2      = np.diff(diff1)
    mob        = float(np.sqrt(np.var(diff1) / (activity + eps)))
    complexity = float(np.sqrt(np.var(diff2) / (np.var(diff1) + eps)) / (mob + eps))
    feats.extend([activity, mob, complexity])

    return np.array(feats, dtype=np.float32)


def extract_features(epochs: np.ndarray) -> np.ndarray:
    """
    Extract features from all epochs.
    epochs: shape (n_epochs, n_channels, n_samples)
    Returns: shape (n_epochs, n_channels * features_per_channel)
    """
    n_epochs, n_channels, _ = epochs.shape
    sample_feats = extract_channel_features(epochs[0, 0])
    n_feats = len(sample_feats)

    X = np.zeros((n_epochs, n_channels * n_feats), dtype=np.float32)

    for i, epoch in enumerate(epochs):
        ch_feats = []
        for ch in range(n_channels):
            ch_feats.append(extract_channel_features(epoch[ch]))
        X[i] = np.concatenate(ch_feats)

        if (i + 1) % 100 == 0:
            print(f"  Features: {i+1}/{n_epochs} epochs done", end='\r')

    print(f"  Feature extraction complete: {X.shape}")
    return X


def get_feature_names() -> List[str]:
    """Return feature names in the same order as extraction (for importance plots)."""
    band_names    = list(BANDS.keys())
    ratio_names   = ['theta_alpha','beta_alpha','gamma_delta','delta_alpha','beta_theta']
    stat_names    = ['mean','var','skew','kurt','zcr','linelen','hjorth_act','hjorth_mob','hjorth_comp']
    per_ch = band_names + ratio_names + stat_names

    names = []
    for ch in range(23):
        for f in per_ch:
            names.append(f"ch{ch:02d}_{f}")
    return names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from data_loader import generate_synthetic_data
    import os

    os.makedirs("data", exist_ok=True)
    print("Generating synthetic data...")
    X_raw, y = generate_synthetic_data(500)

    print("Extracting features...")
    X_feat = extract_features(X_raw)

    np.save("data/X_features.npy", X_feat)
    np.save("data/y_labels.npy",   y)
    np.save("data/X_raw.npy",      X_raw)
    print(f"Saved: features {X_feat.shape}, labels {y.shape}")
    print(f"Feature names count: {len(get_feature_names())}")
