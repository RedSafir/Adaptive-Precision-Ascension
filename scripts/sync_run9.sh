#!/usr/bin/env bash
set -e

REMOTE_HOST="142.252.100.103"
REMOTE_PORT="55018"
REMOTE_DIR="/workspace/Adaptive-Precision-Ascension/runs/taguchi_run9_gamma_0.9_theta_0.9_ci_8"
LOCAL_DIR="runs/taguchi_run9_gamma_0.9_theta_0.9_ci_8"
GDRIVE_DIR="gdrive:result/9_Percobaan/taguchi_run9_gamma_0.9_theta_0.9_ci_8"

echo "[*] Step 1: Downloading taguchi_run9_gamma_0.9_theta_0.9_ci_8 from Vast.ai..."
mkdir -p "$LOCAL_DIR"
rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p $REMOTE_PORT" root@$REMOTE_HOST:$REMOTE_DIR/ "$LOCAL_DIR/"

echo "[✓] Step 1 Complete: taguchi_run9_gamma_0.9_theta_0.9_ci_8 downloaded locally."
echo "[*] Step 2: Uploading taguchi_run9_gamma_0.9_theta_0.9_ci_8 to Google Drive ($GDRIVE_DIR)..."
rclone copy "$LOCAL_DIR/" "$GDRIVE_DIR/" --progress

echo "[✓] Step 2 Complete: taguchi_run9_gamma_0.9_theta_0.9_ci_8 successfully uploaded to Google Drive!"
