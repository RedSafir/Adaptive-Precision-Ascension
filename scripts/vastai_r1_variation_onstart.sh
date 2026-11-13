#!/bin/bash
set -e

shutdown_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ] && [ "${ON_FINISH:-stop}" = "stop" ]; then
        echo "[!] Error occurred during onstart setup (exit code $exit_code). Stopping Vast.ai instance to prevent idle billing..."
        RAW_ID="${VAST_CONTAINERLABEL:-$CONTAINER_ID}"
        CLEAN_ID="${RAW_ID#C.}"
        API_KEY="${CONTAINER_API_KEY:-$VAST_API_KEY}"
        if [ -n "$CLEAN_ID" ] && [ -n "$API_KEY" ]; then
            curl -s -X PUT "https://console.vast.ai/api/v1/instances/${CLEAN_ID}/" \
                -H "Authorization: Bearer ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d '{"state": "stopped"}' || true
        fi
    fi
}
trap shutdown_on_error EXIT

echo "================================================================="
echo "  VAST.AI ONSTART: R1 VARIATION (YOLOS APA GAMMA 0.1, THETA 0.1)"
echo "  Check Intervals: ci in [8, 985] | 50 Epochs per Run           "
echo "  Dataset: Hugging Face (miftahN/soccers-player-ball)            "
echo "================================================================="

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

DATASET_DIR="$REPO_DIR/datasets/detection-person-and-ball/merged_yolo_person_ball"
if [ ! -d "$DATASET_DIR" ]; then
    echo "[*] Streaming 26 GB clean dataset directly to disk (zero duplicate .tar overhead)..."
    mkdir -p "$REPO_DIR/datasets/detection-person-and-ball"
    AUTH_HEADER=()
    if [ -n "$HF_TOKEN" ]; then
        echo "[*] Using authenticated Hugging Face token for priority CDN speed."
        AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
    fi
    curl -fL "${AUTH_HEADER[@]}" --retry 3 --retry-delay 2 \
        "https://huggingface.co/datasets/miftahN/soccers-player-ball/resolve/main/merged_yolo_person_ball.tar" \
        | tar -x -C "$REPO_DIR/datasets/detection-person-and-ball"
    echo "[✓] Dataset extracted successfully and ready at $DATASET_DIR!"
else
    echo "[*] Dataset already present at $DATASET_DIR."
fi

echo "[*] Installing dependencies & building APA kernels..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
which python || ln -sf /usr/bin/python3 /usr/bin/python
python3 -m pip install --no-cache-dir opencv-python-headless scipy transformers accelerate tqdm pillow datasets huggingface_hub vastai -q
python3 -m pip install --no-cache-dir -e ".[examples]"

echo "[*] Launching R1 Variation Suite (Gamma=0.1, Theta=0.1, ci=8 & ci=985)..."
python3 scripts/run_vastai_r1_variation.py \
    --gamma 0.1 \
    --theta_underflow 0.1 \
    --ci_list 8 985 \
    --epochs "${R1_VAR_EPOCHS:-50}" \
    --batch_size 16 \
    --enable_frobenius \
    --on_finish "${ON_FINISH:-stop}" \
    --github_pat "${GITHUB_PAT}"
