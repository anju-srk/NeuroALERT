"""
NeuroAlert — Streamlit Dashboard (Phase 5)
Run: streamlit run dashboard.py
"""

import sys
import os
import time
import json
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import welch
from alerter import alerter

sys.path.insert(0, str(Path(__file__).parent))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAlert",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0a0a0f; }
  .block-container { padding-top: 1rem; }

  .metric-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-val   { font-size: 2.4rem; font-weight: 700; margin: 0; }
  .metric-label { font-size: 0.75rem; color: #666; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-top: 4px; }

  .alert-box {
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 8px 0;
  }
  .alert-safe    { background:#0d2e1a; color:#2ecc71; border:1.5px solid #2ecc71; }
  .alert-warning { background:#2e1f0d; color:#f39c12; border:1.5px solid #f39c12; }
  .alert-danger  { background:#2e0d0d; color:#e74c3c; border:1.5px solid #e74c3c; }

  .stMetric { background: #12121a; border-radius:10px; padding: 14px; }
  h1, h2, h3 { color: #e0e0f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── EEG Signal Simulator ─────────────────────────────────────────────────────

class EEGSimulator:
    """
    Streams synthetic EEG with realistic inter-ictal and pre-ictal episodes.
    Used when no real EDF file is uploaded.
    """
    SFREQ    = 256
    N_CH     = 23
    SEQ_LEN  = 90   # seconds per full scenario cycle

    def __init__(self):
        self.t = 0
        np.random.seed(0)

    def get_window(self, n_seconds: int = 5) -> np.ndarray:
        n = self.SFREQ * n_seconds
        t = np.linspace(self.t, self.t + n_seconds, n)
        # Scenario phase (0–90 s cycle)
        phase = (self.t % self.SEQ_LEN) / self.SEQ_LEN

        preictal = (0.45 < phase < 0.75)  # 40 s pre-ictal window
        ictal    = (0.75 < phase < 0.85)  # brief actual seizure

        data = np.zeros((self.N_CH, n), dtype=np.float32)
        for ch in range(self.N_CH):
            base  = 2.0 * np.sin(2 * np.pi * 2  * t + ch * 0.3)
            alpha = (0.3 if preictal else 1.1) * np.sin(2 * np.pi * 10 * t + ch * 0.5)
            beta  = (1.6 if preictal else 0.3) * np.sin(2 * np.pi * 22 * t + ch * 0.2)
            noise = 0.25 * np.random.randn(n)

            if ictal:
                spikes = 3.5 * np.sin(2 * np.pi * 4 * t) * np.abs(np.random.randn(n))
                data[ch] = base + spikes + noise
            else:
                data[ch] = base + alpha + beta + noise

        self.t += n_seconds
        return data


def load_edf_signal(file_obj, n_channels: int = 23) -> np.ndarray:
    """Load an uploaded EDF file and return raw data array."""
    import tempfile, mne
    mne.set_log_level('WARNING')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    raw.filter(0.5, 40., verbose=False)
    data = raw.get_data()

    if data.shape[0] > n_channels:
        data = data[:n_channels]
    elif data.shape[0] < n_channels:
        data = np.vstack([data, np.zeros((n_channels - data.shape[0], data.shape[1]))])

    os.unlink(tmp_path)
    return data


# ─── Plotting helpers ─────────────────────────────────────────────────────────

DISPLAY_CHANNELS = ['FP1-F7','F7-T7','T7-P7','FP1-F3','F3-C3','CZ-PZ','FP2-F4','T8-P8']
DARK_BG = '#0e0e16'

def plot_eeg_waveforms(data: np.ndarray, sfreq: int = 256,
                       channels: list = None, risk: float = 0) -> plt.Figure:
    """Multi-channel EEG waterfall plot."""
    channels = channels or DISPLAY_CHANNELS
    n_ch     = min(len(channels), data.shape[0])
    n_s      = data.shape[1]
    t        = np.arange(n_s) / sfreq

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    offset  = 0
    spacing = 4.5
    color   = '#e74c3c' if risk > 70 else '#f39c12' if risk > 40 else '#2ecc71'

    for i in range(n_ch):
        sig = data[i] if i < data.shape[0] else np.zeros(n_s)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        ax.plot(t, sig + offset, linewidth=0.6, color=color, alpha=0.85)
        ax.text(-0.04, offset, channels[i], color='#999', fontsize=7.5,
                ha='right', va='center', transform=ax.get_yaxis_transform())
        offset += spacing

    ax.set_xlim(0, t[-1])
    ax.set_ylim(-spacing, offset)
    ax.set_xlabel("Time (s)", color='#888', fontsize=9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#1e1e2e')
    ax.tick_params(colors='#888')
    ax.grid(axis='x', color='#1e1e2e', linewidth=0.4)

    return fig


def plot_risk_gauge(risk: float) -> plt.Figure:
    """Semicircular risk gauge."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1] * 200, color='#1e1e2e', linewidth=16, solid_capstyle='round')

    # Colored risk arc
    end_angle = np.pi - (risk / 100) * np.pi
    theta2    = np.linspace(np.pi, end_angle, max(3, int(risk * 2)))
    arc_color = '#e74c3c' if risk > 70 else '#f39c12' if risk > 40 else '#2ecc71'
    ax.plot(theta2, [1] * len(theta2), color=arc_color, linewidth=16, solid_capstyle='round')

    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xlim(0, np.pi)
    ax.axis('off')

    ax.text(np.pi / 2, 0.1, f"{risk:.0f}%", ha='center', va='center',
            color=arc_color, fontsize=26, fontweight='bold')
    ax.text(np.pi / 2, -0.35, "SEIZURE RISK", ha='center', va='center',
            color='#666', fontsize=8)

    return fig


def plot_spectral_heatmap(data: np.ndarray, sfreq: int = 256) -> plt.Figure:
    """Per-channel frequency heatmap (brain fingerprint view)."""
    n_ch = min(data.shape[0], 16)
    freq_matrix = np.zeros((n_ch, 40))

    for ch in range(n_ch):
        freqs, psd = welch(data[ch], fs=sfreq, nperseg=min(256, data.shape[1] // 2))
        for fi, f in enumerate(range(1, 41)):
            idx = np.argmin(np.abs(freqs - f))
            freq_matrix[ch, fi] = np.log1p(psd[idx])

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    im = ax.imshow(freq_matrix, aspect='auto', cmap='plasma',
                   origin='lower', interpolation='bilinear')
    ax.set_xlabel("Frequency (Hz)", color='#888', fontsize=9)
    ax.set_ylabel("Channel", color='#888', fontsize=9)
    ax.set_xticks(range(0, 40, 5))
    ax.set_xticklabels(range(1, 41, 5), color='#888', fontsize=7)
    ax.set_yticks(range(0, n_ch, 4))
    ax.set_yticklabels(range(0, n_ch, 4), color='#888', fontsize=7)
    for sp in ax.spines.values():
        sp.set_color('#1e1e2e')
    plt.colorbar(im, ax=ax).ax.tick_params(colors='#888')

    return fig


def plot_risk_timeline(history: list) -> plt.Figure:
    """Rolling risk score over time."""
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    if not history:
        return fig

    xs = list(range(len(history)))
    ax.fill_between(xs, history, alpha=0.25, color='#e74c3c')

    # Colour line by risk level
    for i in range(1, len(history)):
        c = '#e74c3c' if history[i] > 70 else '#f39c12' if history[i] > 40 else '#2ecc71'
        ax.plot([xs[i-1], xs[i]], [history[i-1], history[i]], color=c, linewidth=1.5)

    ax.axhline(70, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(40, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(len(history) - 1, 1))
    ax.set_ylabel("Risk %", color='#888', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#1e1e2e')

    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/ios-filled/100/e74c3c/brain.png", width=60)
        st.title("NeuroAlert")
        st.caption("Early Seizure Detection System")
        st.divider()

        mode = st.radio("Signal source", ["Live simulation", "Upload EDF file"])

        uploaded_file = None
        if mode == "Upload EDF file":
            uploaded_file = st.file_uploader("Upload .edf file", type=["edf"])

        st.divider()
        alert_threshold = st.slider("Alert threshold (%)", 50, 90, 70)
        warn_threshold  = st.slider("Warning threshold (%)", 20, 60, 40)
        window_size     = st.select_slider("Window (s)", [3, 5, 10], value=5)

        st.divider()
        auto_run = st.toggle("Auto-run demo", value=True)
        st.divider()
        if st.button("📱 Send test alert"):
            alerter.last_alert_time = 0
            alerter.send_all(risk=85.0)
            st.success("Test alert sent! Check your phone.")

        # Model metrics
        metrics_path = Path("models/metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            st.divider()
            st.caption("Model performance")
            col_a, col_b = st.columns(2)
            col_a.metric("AUC-ROC", f"{m.get('auc', 'N/A')}")
            col_b.metric("F1 Score", f"{m.get('f1',  'N/A')}")

    # ── Session state ─────────────────────────────────────────────────────────
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []
    if "simulator"  not in st.session_state:
        st.session_state.simulator = EEGSimulator()
    if "edf_data"   not in st.session_state:
        st.session_state.edf_data = None
    if "edf_cursor" not in st.session_state:
        st.session_state.edf_cursor = 0
    if "alert_count" not in st.session_state:
        st.session_state.alert_count = 0
    if "max_risk"   not in st.session_state:
        st.session_state.max_risk = 0.0

    # Load EDF once
    if uploaded_file is not None and st.session_state.edf_data is None:
        with st.spinner("Loading EDF file..."):
            try:
                st.session_state.edf_data   = load_edf_signal(uploaded_file)
                st.session_state.edf_cursor = 0
                st.success(f"Loaded: {st.session_state.edf_data.shape[1]/256:.1f}s recording")
            except Exception as e:
                st.error(f"EDF load failed: {e}")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='margin-bottom:0'>🧠 NeuroAlert</h1>
    <p style='color:#666;margin-top:4px'>Real-time EEG seizure prediction · CHB-MIT dataset · LSTM/1D-CNN</p>
    """, unsafe_allow_html=True)

    # ── Get current window ────────────────────────────────────────────────────
    sfreq       = 256
    n_window    = sfreq * window_size

    if mode == "Upload EDF file" and st.session_state.edf_data is not None:
        cursor = st.session_state.edf_cursor
        data   = st.session_state.edf_data
        if cursor + n_window > data.shape[1]:
            cursor = 0
        window = data[:, cursor:cursor + n_window]
        st.session_state.edf_cursor = cursor + n_window // 2  # 50% step
    else:
        window = st.session_state.simulator.get_window(window_size)

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        from predictor import predictor
        risk, _ = predictor.predict(window)
    except Exception:
        # Fallback heuristic if model not trained
        from scipy.signal import welch as _welch
        hf = 0.0
        for ch in range(window.shape[0]):
            fr, ps = _welch(window[ch], fs=sfreq, nperseg=128)
            mask   = (fr >= 20) & (fr <= 40)
            hf    += float(np.mean(ps[mask]))
        hf  /= window.shape[0]
        risk = float(np.clip(hf * 18, 0, 100))

    # Smooth risk with exponential moving average
    if st.session_state.risk_history:
        risk = 0.65 * risk + 0.35 * st.session_state.risk_history[-1]

    risk = round(risk, 1)
    st.session_state.risk_history.append(risk)
    if len(st.session_state.risk_history) > 60:
        st.session_state.risk_history = st.session_state.risk_history[-60:]

    st.session_state.max_risk = max(st.session_state.max_risk, risk)

    alert_level = "danger"  if risk >= alert_threshold else \
                  "warning" if risk >= warn_threshold  else "safe"

    if alert_level == "danger":
        st.session_state.alert_count += 1
        alerter.send_all(risk=risk) 

    # ── Top row: alert + gauge + metrics ──────────────────────────────────────
    col_alert, col_gauge, col_metrics = st.columns([2, 1.5, 1.5])

    with col_alert:
        labels = {"safe": "● MONITORING", "warning": "⚠ WARNING", "danger": "🚨 SEIZURE ALERT"}
        box_cls = f"alert-{alert_level}"
        st.markdown(f'<div class="alert-box {box_cls}">{labels[alert_level]}</div>',
                    unsafe_allow_html=True)

        status_text = {
            "safe":    "Brain activity is within normal inter-ictal range.",
            "warning": "Early pre-ictal activity detected. Monitoring closely.",
            "danger":  "High-confidence pre-ictal pattern. Seizure imminent."
        }
        st.caption(status_text[alert_level])

        # Risk timeline
        st.markdown("**Risk timeline (last 60 windows)**")
        st.pyplot(plot_risk_timeline(st.session_state.risk_history), use_container_width=True)

    with col_gauge:
        st.pyplot(plot_risk_gauge(risk), use_container_width=True)

    with col_metrics:
        st.metric("Current risk",  f"{risk:.1f}%",
                  delta=f"{risk - (st.session_state.risk_history[-2] if len(st.session_state.risk_history)>1 else risk):.1f}%")
        st.metric("Session peak",  f"{st.session_state.max_risk:.1f}%")
        st.metric("Alert triggers", st.session_state.alert_count)
        st.metric("Windows analysed", len(st.session_state.risk_history))

    st.divider()

    # ── EEG waveforms ─────────────────────────────────────────────────────────
    st.markdown("**Live EEG — 8-channel view**")
    st.pyplot(plot_eeg_waveforms(window, sfreq, risk=risk), use_container_width=True)

    # ── Spectral heatmap ──────────────────────────────────────────────────────
    st.markdown("**Spectral power heatmap (log PSD)**")
    st.pyplot(plot_spectral_heatmap(window, sfreq), use_container_width=True)

    # ── Band power bar ────────────────────────────────────────────────────────
    st.markdown("**Mean band power across all channels**")
    bands = {'Delta\n(0.5–4)': (0.5,4), 'Theta\n(4–8)': (4,8),
             'Alpha\n(8–13)': (8,13), 'Beta\n(13–30)': (13,30), 'Gamma\n(30–40)': (30,40)}
    band_vals = []
    for (lo, hi) in bands.values():
        bp = 0.0
        for ch in range(window.shape[0]):
            fr, ps = welch(window[ch], fs=sfreq, nperseg=min(256, n_window // 2))
            mask   = (fr >= lo) & (fr <= hi)
            bp    += float(np.trapezoid(ps[mask], fr[mask]))
        band_vals.append(bp / window.shape[0])

    fig_bp, ax_bp = plt.subplots(figsize=(10, 2))
    fig_bp.patch.set_facecolor(DARK_BG)
    ax_bp.set_facecolor(DARK_BG)
    colors_bp = ['#3498db','#9b59b6','#2ecc71','#f39c12','#e74c3c']
    bars = ax_bp.bar(list(bands.keys()), band_vals, color=colors_bp, edgecolor='none', width=0.5)
    ax_bp.set_ylabel("Power (µV²/Hz)", color='#888', fontsize=8)
    ax_bp.tick_params(colors='#888', labelsize=7)
    for sp in ax_bp.spines.values():
        sp.set_color('#1e1e2e')
    ax_bp.set_facecolor(DARK_BG)
    st.pyplot(fig_bp, use_container_width=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_run:
        time.sleep(1.5)
        st.rerun()
    else:
        if st.button("▶ Next window"):
            st.rerun()


if __name__ == "__main__":
    main()
