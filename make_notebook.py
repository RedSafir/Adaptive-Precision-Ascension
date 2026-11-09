import json

notebook = {
    "cells": [],
    "metadata": {
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(code_str):
    lines = [line + "\n" for line in code_str.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# Section 1
add_md("# 1. Environment Setup & Google Drive Mount")
add_code("""
import os
import sys
import json
import gzip
import glob
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ipywidgets as widgets
from IPython.display import display, Markdown

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

try:
    from google.colab import drive
    drive.mount("/content/drive")
    possible_paths = [
        "/content/drive/MyDrive/result/9_Percobaan",
        "/content/drive/My Drive/result/9_Percobaan",
        "./runs",
        "../runs"
    ]
    DEFAULT_BASE = next((p for p in possible_paths if os.path.exists(p)), "/content/drive/MyDrive/result/9_Percobaan")
except Exception:
    DEFAULT_BASE = "./runs"
""")

# Section 2
add_md("# 2. Select Run to Analyze")
add_code("""
base_dir = DEFAULT_BASE if os.path.exists(DEFAULT_BASE) else "./runs"
available_runs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]) if os.path.exists(base_dir) else []

run_selector = widgets.Dropdown(
    options=available_runs if available_runs else ["No runs found in directory"],
    value=available_runs[0] if available_runs else None,
    description="Run ID:",
    layout=widgets.Layout(width="60%")
)
display(run_selector)
""")

# Section 3
add_md("# 3. Load & Parse Telemetry Data")
add_code("""
selected_run = run_selector.value
run_path = os.path.join(base_dir, selected_run)

def read_jsonl(filepath):
    if not os.path.exists(filepath):
        gz_path = filepath + ".gz"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_records = read_jsonl(os.path.join(run_path, "yolos_training.jsonl"))
events_records = read_jsonl(os.path.join(run_path, "yolos_training_apa_events.jsonl"))
frob_records = read_jsonl(os.path.join(run_path, "yolos_training_frobenius.jsonl"))

header_meta = {}
step_list = []
epoch_list = []

for r in train_records:
    rec_type = r.get("type")
    if rec_type == "header":
        header_meta = r.get("meta", {})
    elif rec_type == "step":
        step_list.append(r)
    elif rec_type == "epoch":
        epoch_list.append(r)

df_steps = pd.DataFrame(step_list)
df_epochs = pd.DataFrame(epoch_list)
df_events = pd.DataFrame(events_records)

display(Markdown(f"### Target Run: `{selected_run}`"))
display(pd.DataFrame([header_meta]).T.rename(columns={0: "Hyperparameter"}))
""")

# Section 4
add_md("# 4. Training Loss & Convergence Analysis")
add_code("""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

if not df_epochs.empty:
    axes[0, 0].plot(df_epochs["epoch"], df_epochs["avg_loss"], marker="o", color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Total Average Loss per Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(df_epochs["epoch"], df_epochs["avg_ce"], marker="s", color="#ff7f0e", linewidth=2)
    axes[0, 1].set_title("Classification Loss (CE)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Cross Entropy")

    axes[1, 0].plot(df_epochs["epoch"], df_epochs["avg_bbox"], marker="^", color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Bounding Box L1 Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L1 Loss")

    axes[1, 1].plot(df_epochs["epoch"], df_epochs["avg_giou"], marker="d", color="#d62728", linewidth=2)
    axes[1, 1].set_title("GIoU Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("GIoU")

plt.tight_layout()
plt.show()
""")

# Section 5
add_md("# 5. Throughput (FPS) & Latency Analysis")
add_code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

if not df_epochs.empty:
    ax1.plot(df_epochs["epoch"], df_epochs["avg_fps"], color="#2ca02c", marker="o", linewidth=2)
    ax1.axhline(df_epochs["avg_fps"].mean(), color="black", linestyle="--", alpha=0.7)
    ax1.set_title(f"Throughput per Epoch (Mean: {df_epochs['avg_fps'].mean():.1f} FPS)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("FPS")

    ax2.plot(df_epochs["epoch"], df_epochs["avg_latency_ms"], color="#9467bd", marker="s", linewidth=2)
    ax2.set_title(f"Average Step Latency (Mean: {df_epochs['avg_latency_ms'].mean():.1f} ms)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Latency (ms)")

plt.tight_layout()
plt.show()
""")

# Section 6
add_md("# 6. Memory & Hardware Utilization (Peak VRAM)")
add_code("""
fig, ax = plt.subplots(figsize=(10, 5))

if not df_epochs.empty:
    ax.fill_between(df_epochs["epoch"], df_epochs["peak_vram_mb"], color="#1f77b4", alpha=0.3)
    ax.plot(df_epochs["epoch"], df_epochs["peak_vram_mb"], color="#1f77b4", marker="o", linewidth=2)
    ax.axhline(16311, color="red", linestyle="--", label="Blackwell 16GB Limit")
    ax.set_title("Peak VRAM Usage per Epoch (MB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VRAM (MB)")
    ax.set_ylim(0, 17000)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
""")

# Section 7
add_md("# 7. Adaptive Precision Ascension (APA) Layer Dynamics")
add_code("""
total_managed_layers = 72
layer_final_level = {}

if not df_events.empty:
    for _, row in df_events.iterrows():
        m = row.get("module")
        lvl = row.get("new_level")
        if m and lvl is not None and any(k in str(m) for k in ["query", "key", "value", "dense"]):
            layer_final_level[str(m)] = int(lvl)

fp16_count = sum(1 for lvl in layer_final_level.values() if lvl == 1)
tf32_count = sum(1 for lvl in layer_final_level.values() if lvl == 2)
fp8_count = max(0, total_managed_layers - fp16_count - tf32_count)

precision_counts = {"FP8": fp8_count, "FP16": fp16_count, "TF32": tf32_count}
pie_slices = {k: v for k, v in precision_counts.items() if v > 0}
colors_map = {"FP8": "#2ca02c", "FP16": "#ff7f0e", "TF32": "#1f77b4"}
colors = [colors_map[k] for k in pie_slices.keys()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

if pie_slices:
    ax1.pie(pie_slices.values(), labels=[f"{k} ({v})" for k, v in pie_slices.items()], autopct="%1.1f%%", colors=colors, startangle=140)
else:
    ax1.text(0.5, 0.5, "No precision data", ha="center", va="center")
ax1.set_title(f"Final Layer Precision Distribution ({sum(precision_counts.values())} Linear Layers)")

if not df_events.empty and "reason" in df_events.columns:
    reasons = df_events["reason"].value_counts()
    ax2.barh(reasons.index, reasons.values, color="#1f77b4")
    ax2.set_title(f"Escalation Triggers (Total Events: {len(df_events)})")
    ax2.set_xlabel("Occurrences")

plt.tight_layout()
plt.show()
""")

# Section 8
add_md("# 8. Frobenius Norm & Disparity Analysis")
add_code("""
frob_steps = []
weight_disp = []
grad_disp = []

for r in frob_records:
    if "weight_disparity" in r and "step" in r:
        frob_steps.append(r["step"])
        weight_disp.append(r["weight_disparity"])
        grad_disp.append(r.get("grad_disparity", 0.0))

if frob_steps:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(frob_steps, weight_disp, color="#d62728", linewidth=1.5)
    ax1.set_title("Weight Frobenius Disparity Across Layers")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Disparity Ratio")

    ax2.plot(frob_steps, grad_disp, color="#8c564b", linewidth=1.5)
    ax2.set_title("Gradient Frobenius Disparity Across Layers")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Gradient Disparity")

    plt.tight_layout()
    plt.show()
else:
    print("No Frobenius disparity records available for this run.")
""")

# Section 9
add_md("# 9. Model Checkpoint Inspection (`.pt`)")
add_code("""
ckpt_path = os.path.join(run_path, "yolos_apa_final.pt")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    info = {
        "File Size (MB)": os.path.getsize(ckpt_path) / (1024 * 1024),
        "Epoch Saved": ckpt.get("epoch"),
        "Keys in Checkpoint": list(ckpt.keys()),
        "State Dict Modules": len(ckpt.get("model_state_dict", {}))
    }
    display(pd.DataFrame([info]).T.rename(columns={0: "Checkpoint Metadata"}))
else:
    print(f"Checkpoint not found at: {ckpt_path}")
""")

# Section 10
add_md("# 10. Summary Metrics & Taguchi Overview")
add_code("""
report_path = os.path.join(run_path, "taguchi_l9_yolos_report.md")
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        display(Markdown(f.read()))
else:
    summary_data = {
        "Run": selected_run,
        "Total Epochs": len(df_epochs),
        "Final Loss": df_epochs["avg_loss"].iloc[-1] if not df_epochs.empty else None,
        "Final CE": df_epochs["avg_ce"].iloc[-1] if not df_epochs.empty else None,
        "Final BBox": df_epochs["avg_bbox"].iloc[-1] if not df_epochs.empty else None,
        "Final GIoU": df_epochs["avg_giou"].iloc[-1] if not df_epochs.empty else None,
        "Average FPS": df_epochs["avg_fps"].mean() if not df_epochs.empty else None,
        "Peak VRAM (MB)": df_epochs["peak_vram_mb"].max() if not df_epochs.empty else None,
        "FP8 Layers": precision_counts["FP8"],
        "FP16 Layers": precision_counts["FP16"],
        "TF32 Layers": precision_counts["TF32"],
        "Total Escalations": len(df_events)
    }
    display(pd.DataFrame([summary_data]).T.rename(columns={0: "Run Metric"}))
""")

with open("yolos_apa_taguchi_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("[✓] Successfully generated yolos_apa_taguchi_analysis.ipynb")
