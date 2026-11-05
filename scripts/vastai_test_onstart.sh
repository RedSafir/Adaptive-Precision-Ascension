#!/bin/bash
set -e

echo "================================================================="
echo "  VAST.AI ONSTART: SINGLE TEST RUN (YOLOS APA GAMMA 0.9 & THETA 0.9)"
echo "  Dataset: Hugging Face (miftahN/soccers-player-ball)            "
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

# 2. Download Clean Dataset from Hugging Face CDN (Fast ~2 mins, 26 GB)
DATASET_DIR="$REPO_DIR/datasets/detection-person-and-ball/merged_yolo_person_ball"
if [ ! -d "$DATASET_DIR" ]; then
    echo "[*] Downloading 26 GB clean dataset from Hugging Face (miftahN/soccers-player-ball)..."
    mkdir -p "$REPO_DIR/datasets/detection-person-and-ball"
    TAR_PATH="$REPO_DIR/datasets/detection-person-and-ball/merged_yolo_person_ball.tar"
    curl -fL --retry 3 --retry-delay 2 "https://huggingface.co/datasets/miftahN/soccers-player-ball/resolve/main/merged_yolo_person_ball.tar" -o "$TAR_PATH"
    echo "[*] Extracting dataset archive..."
    tar -xf "$TAR_PATH" -C "$REPO_DIR/datasets/detection-person-and-ball"
    rm -f "$TAR_PATH"
    echo "[✓] Dataset extracted successfully and ready at $DATASET_DIR!"
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
