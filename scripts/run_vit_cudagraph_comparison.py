import sys
import os
import time
import json
import subprocess
import torch
import numpy as np

def run_command_stream(cmd, desc):
    print("=" * 75)
    print(f"STARTING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 75, flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)
    proc.stdout.close()
    ret = proc.wait()
    duration = time.time() - t0
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")
    print(f"\n[COMPLETED] {desc} in {duration:.1f}s ({duration/60:.2f} min)\n", flush=True)
    return duration

def parse_log(log_path):
    epochs = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get('event') == 'epoch_summary':
                epochs.append(data)
    return epochs

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_dir)

    python_bin = sys.executable
    train_script = os.path.join(base_dir, 'examples', 'vit_cifar10', 'train.py')
    
    os.makedirs('result', exist_ok=True)
    log_full_fp8 = os.path.join(base_dir, 'result', 'vit_full_fp8_cudagraph.jsonl')
    log_hybrid_fp8 = os.path.join(base_dir, 'result', 'vit_hybrid_fp8_cudagraph.jsonl')
    log_apa_hybrid = os.path.join(base_dir, 'result', 'vit_apa_hybrid_cudagraph.jsonl')

    epochs = 5
    batch_size = 128

    common_args = [
        python_bin, train_script,
        '--dataset', 'cifar10',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', '1e-3',
        '--cuda_graph'
    ]

    configs = [
        {
            "name": "Full FP8 + CUDA Graph (All Layers FP8)",
            "desc": "Semua layer (Patch Embed + Transformer Blocks + Classifier Head) 100% FP8",
            "cmd": common_args + ['--precision', 'fp8', '--all_apa_layers', '--log_file', log_full_fp8],
            "log": log_full_fp8
        },
        {
            "name": "Hybrid FP8 + CUDA Graph (Boundary FP32/TF32)",
            "desc": "Arsitektur Hybrid: Boundary (Patch Embed & Head) FP32/TF32, Body Transformer FP8",
            "cmd": common_args + ['--precision', 'fp8', '--log_file', log_hybrid_fp8],
            "log": log_hybrid_fp8
        },
        {
            "name": "APA Adaptive + Hybrid CUDA Graph",
            "desc": "Adaptive Dynamic Precision: Awal FP8 dengan eskalasi dinamis 0-freeze re-capture",
            "cmd": common_args + ['--precision', 'apa', '--log_file', log_apa_hybrid],
            "log": log_apa_hybrid
        }
    ]

    print("######################################################################")
    print("  VISION TRANSFORMER (ViT) BENCHMARK: FULL FP8 vs HYBRID CUDA GRAPH")
    print("######################################################################")
    print(f"Model: ViT-Small (dim=256, depth=6, heads=4)")
    print(f"Dataset: CIFAR-10 (390 batches/epoch @ batch_size={batch_size})")
    print(f"Epochs: {epochs} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("######################################################################\n", flush=True)

    results = []

    for i, cfg in enumerate(configs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        time.sleep(2)

        desc = f"Run {i+1}/{len(configs)}: {cfg['name']}"
        dur = run_command_stream(cfg['cmd'], desc)

        ep_list = parse_log(cfg['log'])
        last_ep = ep_list[-1] if ep_list else {}
        avg_ep_time = float(np.mean([e.get('epoch_time_sec', 0) for e in ep_list])) if ep_list else dur / epochs

        results.append({
            "name": cfg["name"],
            "desc": cfg["desc"],
            "total_dur_sec": dur,
            "avg_epoch_time_sec": avg_ep_time,
            "final_train_loss": last_ep.get('train_loss', 0.0),
            "final_train_acc": last_ep.get('train_acc', 0.0) * 100.0,
            "final_test_loss": last_ep.get('test_loss', 0.0),
            "final_test_acc": last_ep.get('test_acc', 0.0) * 100.0,
            "precision_dist": last_ep.get('precision_distribution', {})
        })

    # Generate Markdown Table
    summary_md = f"""# Benchmark Comparison: Vision Transformer (ViT CIFAR-10)
## Layer Full FP8 dengan CUDA Graph vs Hybrid CUDA Graph

**Spesifikasi Hardware & Model**:
- **Device**: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- **Arsitektur Model**: Vision Transformer (ViT: dim=256, depth=6, heads=4)
- **Dataset**: CIFAR-10 (50.000 train samples, 10.000 test samples, batch size {batch_size})
- **Total Durasi Training**: {epochs} Epochs per skema

| Metrik / Parameter | Full FP8 + CUDA Graph | Hybrid FP8 + CUDA Graph | APA Adaptive + Hybrid Graph |
| :--- | :---: | :---: | :---: |
| **Arsitektur Layer** | **100% Full FP8** (Patch Embed, Blocks, Head) | **Hybrid Precision** (Boundary FP32, Body FP8) | **Adaptive Precision** (Dynamic FP8/FP16/TF32) |
| **Rata-rata Waktu / Epoch** | **{results[0]['avg_epoch_time_sec']:.2f} s** | **{results[1]['avg_epoch_time_sec']:.2f} s** | **{results[2]['avg_epoch_time_sec']:.2f} s** |
| **Total Waktu ({epochs} Epochs)** | {results[0]['total_dur_sec']:.1f} s | {results[1]['total_dur_sec']:.1f} s | {results[2]['total_dur_sec']:.1f} s |
| **Throughput (Images/s)** | **{batch_size * 390 / max(0.1, results[0]['avg_epoch_time_sec']):.1f} img/s** | **{batch_size * 390 / max(0.1, results[1]['avg_epoch_time_sec']):.1f} img/s** | **{batch_size * 390 / max(0.1, results[2]['avg_epoch_time_sec']):.1f} img/s** |
| **Final Train Loss (Epoch {epochs})** | {results[0]['final_train_loss']:.4f} | {results[1]['final_train_loss']:.4f} | {results[2]['final_train_loss']:.4f} |
| **Final Test Loss (Epoch {epochs})** | {results[0]['final_test_loss']:.4f} | {results[1]['final_test_loss']:.4f} | {results[2]['final_test_loss']:.4f} |
| **Final Test Top-1 Accuracy** | **{results[0]['final_test_acc']:.2f}%** | **{results[1]['final_test_acc']:.2f}%** | **{results[2]['final_test_acc']:.2f}%** |
| **Distribusi Presisi Layer** | All 26 FP8 (100%) | 24 FP8 (92.3%), 2 FP32 (7.7%) | FP8: {results[2]['precision_dist'].get('fp8', 24)}, FP16: {results[2]['precision_dist'].get('fp16', 0)}, TF32: {results[2]['precision_dist'].get('tf32', 2)} |
"""

    summary_file = os.path.join(base_dir, 'result', 'vit_cudagraph_comparison.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print("\n" + "=" * 75)
    print("FINAL COMPARISON RESULT:")
    print("=" * 75)
    print(summary_md)
    print("=" * 75)
    print(f"Summary written to: {summary_file}")

if __name__ == '__main__':
    main()
