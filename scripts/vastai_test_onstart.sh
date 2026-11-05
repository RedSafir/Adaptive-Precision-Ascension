#!/bin/bash
set -e

echo "================================================================="
echo "  VAST.AI ONSTART: SINGLE TEST RUN (YOLOS APA GAMMA 0.9 & THETA 0.9)"
echo "  Validation Test Separated from Cluster 1 Suite                 "
echo "================================================================="

# 1. Setup GitHub Credentials & Pull Repository
REPO_DIR="/workspace/Adaptive-Precision-Ascension"
if [ ! -d "$REPO_DIR" ]; then
    echo "[*] Cloning Adaptive-Precision-Ascension repository..."
    if [ -n "$GITHUB_PAT" ]; then
        git clone https://RedSafir:${GITHUB_PAT}@github.com/RedSafir/Adaptive-Precision-Ascension.git "$REPO_DIR"
    else
        git clone https://github.com/RedSafir/Adaptive-Precision-Ascension.git "$REPO_DIR"
    fi
fi

cd "$REPO_DIR"
git pull origin main

# 2. Setup Kaggle Credentials & Download Dataset
if [ -n "$KAGGLE_API_TOKEN" ]; then
    export KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN"
    mkdir -p ~/.kaggle
    echo "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
fi

DATASET_DIR="$REPO_DIR/datasets/detection-person-and-ball/merged_yolo_person_ball"
if [ ! -d "$DATASET_DIR" ]; then
    echo "[*] Downloading dataset from Kaggle (rifqilukmansyah381/detection-person-and-ball)..."
    mkdir -p "$REPO_DIR/datasets/detection-person-and-ball"
    pip install kaggle -q
    kaggle datasets download -d rifqilukmansyah381/detection-person-and-ball -p "$REPO_DIR/datasets/detection-person-and-ball" --unzip
    echo "[✓] Dataset downloaded and unzipped successfully!"
else
    echo "[*] Dataset already present at $DATASET_DIR."
fi

# 3. Setup Python dependencies & native APA CUDA kernels
echo "[*] Installing dependencies & building APA kernels..."
pip install -e ".[examples]"
pip install vastai -q

# 4. Launch Single Test Runner (Default 50 epochs, or overridden by TEST_EPOCHS)
EPOCHS_TO_RUN="${TEST_EPOCHS:-50}"
echo "[*] Launching Single Test (Epochs: $EPOCHS_TO_RUN, Gamma: 0.9, Theta Underflow: 0.9, Hybrid CUDA Graph)..."
python scripts/run_vastai_single_test.py \
    --gamma 0.9 \
    --theta_underflow 0.9 \
    --epochs "$EPOCHS_TO_RUN" \
    --batch_size 16 \
    --check_interval 8 \
    --cuda_graph \
    --on_finish "${ON_FINISH:-stop}" \
    --github_pat "${GITHUB_PAT}"
