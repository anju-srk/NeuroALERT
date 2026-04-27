"""
NeuroAlert — Phase 1 & 2: Data Loading and Preprocessing
Downloads and preprocesses the CHB-MIT EEG dataset.
"""

import os
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

# ─── Constants ────────────────────────────────────────────────────────────────
SFREQ = 256          # CHB-MIT sampling frequency (Hz)
EPOCH_SEC = 5        # window length in seconds
PREICTAL_SEC = 30    # label this many seconds before seizure as pre-ictal
OVERLAP = 0.5        # 50% overlap between windows
N_CHANNELS = 23      # CHB-MIT standard channel count
EPOCH_SAMPLES = SFREQ * EPOCH_SEC   # 1280 samples per window

# CHB-MIT seizure annotations (patient chb01 — add more patients as needed)
# Format: { filename: [(onset_sec, offset_sec), ...] }
SEIZURE_TIMES = {
    "chb01_03.edf": [(2996, 3036)],
    "chb01_04.edf": [(1467, 1494)],
    "chb01_15.edf": [(1732, 1772)],
    "chb01_16.edf": [(1015, 1066)],
    "chb01_18.edf": [(1720, 1810)],
    "chb01_21.edf": [(327,  420)],
    "chb01_26.edf": [(1862, 1963)],
}

CHANNEL_NAMES = [
    'FP1-F7','F7-T7','T7-P7','P7-O1',
    'FP1-F3','F3-C3','C3-P3','P3-O1',
    'FZ-CZ','CZ-PZ',
    'FP2-F4','F4-C4','C4-P4','P4-O2',
    'FP2-F8','F8-T8','T8-P8','P8-O2',
    'P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8'
]


def load_edf(filepath: str) -> Tuple[np.ndarray, float]:
    """Load an EDF file and return (data, sfreq). Handles channel count mismatches."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)

    data = raw.get_data()  # shape: (n_channels, n_samples)

    # Standardise to N_CHANNELS channels
    if data.shape[0] > N_CHANNELS:
        data = data[:N_CHANNELS]
    elif data.shape[0] < N_CHANNELS:
        pad = np.zeros((N_CHANNELS - data.shape[0], data.shape[1]))
        data = np.vstack([data, pad])

    return data, raw.info['sfreq']


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently."""
    mean = epoch.mean(axis=1, keepdims=True)
    std  = epoch.std(axis=1, keepdims=True) + 1e-8
    return (epoch - mean) / std


def extract_epochs(
    data: np.ndarray,
    sfreq: float,
    seizure_intervals: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across the recording and label each window.
    Returns (epochs, labels) where label=1 means pre-ictal.
    """
    step = int(EPOCH_SAMPLES * (1 - OVERLAP))
    n_samples = data.shape[1]

    # Build a per-sample label array: 1 = pre-ictal, 0 = inter-ictal
    sample_labels = np.zeros(n_samples, dtype=np.int8)
    for onset, offset in seizure_intervals:
        onset_s  = int(onset  * sfreq)
        offset_s = int(offset * sfreq)
        preictal_start = max(0, onset_s - int(PREICTAL_SEC * sfreq))
        sample_labels[preictal_start:onset_s] = 1   # pre-ictal window
        # mark actual seizure as -1 so we can exclude it
        sample_labels[onset_s:offset_s] = -1

    epochs, labels = [], []
    for start in range(0, n_samples - EPOCH_SAMPLES, step):
        end = start + EPOCH_SAMPLES
        window_label = sample_labels[start:end]

        # Skip windows that contain actual seizure samples
        if np.any(window_label == -1):
            continue

        label = 1 if np.mean(window_label) > 0.5 else 0
        epoch = data[:, start:end].astype(np.float32)
        epochs.append(normalize_epoch(epoch))
        labels.append(label)

    return np.array(epochs), np.array(labels)


def load_patient_data(data_dir: str, patient: str = "chb01") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all EDF files for a given patient and return combined epochs + labels.
    data_dir: path to the folder containing chb01/*.edf files
    """
    patient_dir = Path(data_dir) / patient
    all_epochs, all_labels = [], []

    edf_files = sorted(patient_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {patient_dir}")

    print(f"Found {len(edf_files)} EDF files for {patient}")

    for edf_path in edf_files:
        fname = edf_path.name
        seizures = SEIZURE_TIMES.get(fname, [])
        print(f"  Loading {fname} — {len(seizures)} seizure(s)")

        try:
            data, sfreq = load_edf(str(edf_path))
            epochs, labels = extract_epochs(data, sfreq, seizures)
            all_epochs.append(epochs)
            all_labels.append(labels)
            n_pre = labels.sum()
            print(f"    → {len(labels)} windows ({n_pre} pre-ictal, {len(labels)-n_pre} inter-ictal)")
        except Exception as e:
            print(f"    ✗ Skipped {fname}: {e}")

    if not all_epochs:
        raise RuntimeError("No data loaded. Check your data directory.")

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {X.shape[0]} epochs, {y.sum()} pre-ictal ({y.mean()*100:.1f}%)")
    return X, y


def generate_synthetic_data(n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic EEG data for demo/testing when
    the real dataset is not available. Pre-ictal windows have elevated
    high-frequency power and reduced alpha power.
    """
    np.random.seed(42)
    t = np.linspace(0, EPOCH_SEC, EPOCH_SAMPLES)
    X, y = [], []

    n_preictal   = n_samples // 5   # 20% pre-ictal (realistic ratio)
    n_interictal = n_samples - n_preictal

    def make_epoch(preictal: bool) -> np.ndarray:
        epoch = np.zeros((N_CHANNELS, EPOCH_SAMPLES), dtype=np.float32)
        for ch in range(N_CHANNELS):
            # Base brain rhythm components
            delta = 2.0  * np.sin(2 * np.pi * 2   * t + np.random.rand() * 2 * np.pi)
            theta = 1.0  * np.sin(2 * np.pi * 6   * t + np.random.rand() * 2 * np.pi)
            alpha = (0.3 if preictal else 1.2) * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            beta  = (1.8 if preictal else 0.4) * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            noise = 0.3  * np.random.randn(EPOCH_SAMPLES)

            if preictal:
                # Spike discharges characteristic of pre-ictal state
                n_spikes = np.random.randint(3, 8)
                spike_times = np.random.choice(EPOCH_SAMPLES, n_spikes, replace=False)
                spikes = np.zeros(EPOCH_SAMPLES)
                for st in spike_times:
                    width = np.random.randint(5, 15)
                    s = max(0, st - width // 2)
                    e = min(EPOCH_SAMPLES, st + width // 2)
                    spikes[s:e] += np.random.uniform(2, 5) * np.hanning(e - s)
                epoch[ch] = delta + theta + alpha + beta + noise + spikes
            else:
                epoch[ch] = delta + theta + alpha + beta + noise

        return normalize_epoch(epoch)

    for _ in range(n_interictal):
        X.append(make_epoch(preictal=False))
        y.append(0)

    for _ in range(n_preictal):
        X.append(make_epoch(preictal=True))
        y.append(1)

    idx = np.random.permutation(len(X))
    return np.array(X)[idx], np.array(y)[idx]


if __name__ == "__main__":
    print("Generating synthetic data for testing...")
    X, y = generate_synthetic_data(1000)
    print(f"X shape: {X.shape}, y distribution: {np.bincount(y)}")
    np.save("data/X_synthetic.npy", X)
    np.save("data/y_synthetic.npy", y)
    print("Saved to data/")
