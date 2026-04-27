#!/usr/bin/env bash
# NeuroAlert — one-shot setup and launch script
# Usage:
#   bash run.sh demo         → train on synthetic data + launch dashboard
#   bash run.sh train-rf     → train Random Forest on synthetic data
#   bash run.sh train-cnn    → train 1D-CNN on synthetic data
#   bash run.sh real <path>  → train on real CHB-MIT data at <path>

set -e
cd "$(dirname "$0")"

echo ""
echo "  🧠  NeuroAlert — Early Seizure Prediction"
echo "  ──────────────────────────────────────────"
echo ""

CMD=${1:-demo}

install_deps() {
  echo "▸ Installing dependencies..."
  pip install -q -r requirements.txt
  echo "  ✓ Dependencies ready"
}

make_dirs() {
  mkdir -p data models
}

case "$CMD" in
  demo)
    install_deps
    make_dirs
    echo ""
    echo "▸ Training Random Forest on synthetic data..."
    python train.py --model rf
    echo ""
    echo "▸ Launching dashboard..."
    echo "  Open http://localhost:8501 in your browser"
    echo ""
    streamlit run dashboard.py
    ;;

  train-rf)
    install_deps
    make_dirs
    echo "▸ Training Random Forest..."
    python train.py --model rf
    ;;

  train-cnn)
    install_deps
    make_dirs
    echo "▸ Training 1D-CNN (this takes ~5–10 min on CPU)..."
    python train.py --model cnn
    ;;

  real)
    DATA_PATH=${2:-""}
    if [ -z "$DATA_PATH" ]; then
      echo "  ✗ Provide path to CHB-MIT data: bash run.sh real /path/to/chb-mit"
      exit 1
    fi
    install_deps
    make_dirs
    echo "▸ Training on real CHB-MIT data at $DATA_PATH..."
    python train.py --model rf --data "$DATA_PATH"
    echo "▸ Launching dashboard..."
    streamlit run dashboard.py
    ;;

  dashboard)
    echo "▸ Launching dashboard (no training)..."
    streamlit run dashboard.py
    ;;

  *)
    echo "Unknown command: $CMD"
    echo "Usage: bash run.sh [demo|train-rf|train-cnn|real <path>|dashboard]"
    exit 1
    ;;
esac
