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
    
    epochs = 3
    batch_size = 128

    common_args = [
        python_bin, train_script,
        '--dataset', 'cifar10',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', '1e-3',
    ]

    configs = [
        {
            "name": "FP32 (Standard IEEE-754)",
            "desc": "Single precision FP32, TF32 Tensor Cores dinonaktifkan",
            "cmd": common_args + ['--precision', 'fp32', '--log_file', os.path.join(base_dir, 'result', 'vit_bench_fp32.jsonl')],
            "log": os.path.join(base_dir, 'result', 'vit_bench_fp32.jsonl')
        },
        {
            "name": "TF32 (TensorFloat-32)",
            "desc": "FP32 dengan akselerasi hardware TF32 Tensor Cores (Ampere/Ada/Blackwell)",
            "cmd": common_args + ['--precision', 'tf32', '--log_file', os.path.join(base_dir, 'result', 'vit_bench_tf32.jsonl')],
            "log": os.path.join(base_dir, 'result', 'vit_bench_tf32.jsonl')
        },
        {
            "name": "FP16 (Half Precision AMP)",
            "desc": "PyTorch native mixed precision AMP FP16",
            "cmd": common_args + ['--precision', 'fp16', '--log_file', os.path.join(base_dir, 'result', 'vit_bench_fp16.jsonl')],
            "log": os.path.join(base_dir, 'result', 'vit_bench_fp16.jsonl')
        },
        {
            "name": "FP8 APA (Eager)",
            "desc": "FP8 Fixed Level 0 via APA Eager Mode (tanpa CUDA Graph)",
            "cmd": common_args + ['--precision', 'fp8', '--log_file', os.path.join(base_dir, 'result', 'vit_bench_fp8_eager.jsonl')],
            "log": os.path.join(base_dir, 'result', 'vit_bench_fp8_eager.jsonl')
        },
        {
            "name": "FP8 APA + CUDA Graph",
            "desc": "FP8 Fixed Level 0 dengan akselerasi APA CUDA Graph Runner",
            "cmd": common_args + ['--precision', 'fp8', '--cuda_graph', '--log_file', os.path.join(base_dir, 'result', 'vit_bench_fp8_graph.jsonl')],
            "log": os.path.join(base_dir, 'result', 'vit_bench_fp8_graph.jsonl')
        }
    ]

    print("######################################################################")
    print("  VISION TRANSFORMER (ViT) BENCHMARK: 5 PRECISION MODES")
    print("  FP32 vs TF32 vs FP16 vs FP8 (Eager) vs FP8 + CUDA Graph")
    print("######################################################################")
    print(f"Model: ViT-Small (dim=256, depth=6, heads=4)")
    print(f"Dataset: CIFAR-10 (390 batches/epoch @ batch_size={batch_size})")
    print(f"Epochs per mode: {epochs} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("######################################################################\n", flush=True)

    results = []

    for i, cfg in enumerate(configs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        time.sleep(1)

        desc = f"Run {i+1}/{len(configs)}: {cfg['name']}"
        dur = run_command_stream(cfg['cmd'], desc)

        epochs_data = parse_log(cfg['log'])
        if epochs_data:
            # Skip first epoch if warmup or average over all
            epoch_times = [ep['epoch_time_sec'] for ep in epochs_data]
            mean_epoch_s = float(np.mean(epoch_times))
            mean_step_ms = (mean_epoch_s / len(epochs_data[0].get('step_time_ms', [1]*390))) * 1000.0 if 'step_time_ms' in epochs_data[0] else (mean_epoch_s / 390.0) * 1000.0
            throughput = (batch_size * 390.0) / mean_epoch_s
            last_ep = epochs_data[-1]
            train_loss = last_ep.get('train_loss', 0.0)
            test_acc = last_ep.get('test_top1_acc', 0.0)
            max_vram = last_ep.get('max_memory_allocated_mb', 0)
        else:
            mean_epoch_s = dur / epochs
            mean_step_ms = (mean_epoch_s / 390.0) * 1000.0
            throughput = (batch_size * 390.0) / mean_epoch_s
            train_loss = 0.0
            test_acc = 0.0
            max_vram = 0.0

        results.append({
            "name": cfg["name"],
            "desc": cfg["desc"],
            "duration_sec": dur,
            "mean_epoch_sec": mean_epoch_s,
            "mean_step_ms": mean_step_ms,
            "throughput_fps": throughput,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "max_vram_mb": max_vram
        })

    # Summary Markdown
    ref_step = results[0]["mean_step_ms"] # FP32 reference
    summary_md = f"""# Benchmark Comparison: Vision Transformer (ViT)
## 5 Precision Modes: FP32 vs TF32 vs FP16 vs FP8 (Eager) vs FP8 + CUDA Graph

**Hardware & Dataset Environment**:
- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- Model: `Vision Transformer (ViT-Small)`: dim=256, depth=6, heads=4 (~3.2M params, >98% `nn.Linear`)
- Dataset: `CIFAR-10` (50,000 images, 390 batches/epoch @ batch_size={batch_size})
- Epochs: `{epochs}` per mode

| Precision Mode | Waktu / Epoch | Latensi per Step | Throughput (FPS) | Peak VRAM | Final Test Acc | Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        sp = ref_step / max(1e-6, r["mean_step_ms"])
        summary_md += f"| **{r['name']}** | **{r['mean_epoch_sec']:.2f} s** | {r['mean_step_ms']:.2f} ms | {r['throughput_fps']:.1f} fps | {r['max_vram_mb']} MB | {r['test_acc']:.2f}% | **{sp:.2f}x** |\n"

    summary_file = os.path.join(base_dir, 'result', 'vit_5_precisions_comparison.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print("\n" + "=" * 85)
    print("FINAL COMPARISON RESULT:")
    print("=" * 85)
    print(summary_md)
    print("=" * 85)
    print(f"Summary written to: {summary_file}")

if __name__ == '__main__':
    main()
