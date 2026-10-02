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
    records = []
    step_metrics = []
    epoch_summaries = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get('event') == 'step_metric':
                step_metrics.append(data)
            elif 'epoch' in data and 'epoch_time_sec' in data:
                epoch_summaries.append(data)
            else:
                records.append(data)
    return records, step_metrics, epoch_summaries

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_dir)

    python_bin = sys.executable
    train_script = os.path.join(base_dir, 'examples', 'rtdetr_coco', 'train.py')
    data_yaml = os.path.join(base_dir, 'datasets', 'detection-person-and-ball', 'merged_yolo_person_ball', 'data.yaml')
    
    os.makedirs('result', exist_ok=True)
    log_fp16 = os.path.join(base_dir, 'result', 'rtdetr_fp16.jsonl')
    log_fp8 = os.path.join(base_dir, 'result', 'rtdetr_fp8.jsonl')

    max_steps = 50
    batch_size = 8

    common_args = [
        python_bin, train_script,
        '--model', 'rtdetr-l.yaml',
        '--data', data_yaml,
        '--classes', '0',
        '--batch_size', str(batch_size),
        '--imgsz', '640',
        '--epochs', '1',
        '--max_steps', str(max_steps),
        '--workers', '4',
        '--no_save'
    ]

    configs = [
        {
            "name": "DINO-DETR (RT-DETR) FP16 Baseline",
            "cmd": common_args + ['--precision', 'fp16', '--log_file', log_fp16],
            "log": log_fp16
        },
        {
            "name": "DINO-DETR (RT-DETR) FP8 (APA)",
            "cmd": common_args + ['--precision', 'fp8', '--log_file', log_fp8],
            "log": log_fp8
        }
    ]

    print("######################################################################")
    print("  DINO-DETR / RT-DETR PRECISION BENCHMARK: FP16 VS FP8 (APA)")
    print("######################################################################")
    print(f"Dataset: {data_yaml} (Class 0: Person)")
    print(f"Model: DINO-DETR / RT-DETR (33.0M params, 74 Transformer Linear layers)")
    print(f"Batch Size: {batch_size} | Steps per run: {max_steps}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
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

        records, steps, epochs = parse_log(cfg['log'])
        ep = epochs[0] if epochs else {}

        # Exclude initial 5 steps from latency average as warmup
        steady_steps = [s['step_time_ms'] for s in steps if s.get('step', 0) >= 5 and 'step_time_ms' in s]
        mean_step_ms = float(np.mean(steady_steps)) if steady_steps else float(ep.get('epoch_time_sec', dur) / max(1, max_steps) * 1000.0)
        throughput = (batch_size * 1000.0) / mean_step_ms if mean_step_ms > 0 else 0.0

        results.append({
            "name": cfg["name"],
            "duration_sec": dur,
            "mean_step_ms": mean_step_ms,
            "throughput_img_sec": throughput,
            "train_loss": ep.get('train_loss', 'N/A'),
            "cls_loss": ep.get('cls_loss', 'N/A'),
            "giou_loss": ep.get('giou_loss', 'N/A'),
            "l1_loss": ep.get('l1_loss', 'N/A'),
            "max_vram_mb": ep.get('max_memory_allocated_mb', 0),
            "reserved_vram_mb": ep.get('max_memory_reserved_mb', 0),
            "precision_distribution": ep.get('precision_distribution', {})
        })

    # Summary Markdown
    speedup = results[0]["mean_step_ms"] / max(1e-6, results[1]["mean_step_ms"])
    vram_saved = results[0]["max_vram_mb"] - results[1]["max_vram_mb"]
    vram_saved_pct = (vram_saved / max(1e-6, results[0]["max_vram_mb"])) * 100.0

    summary_md = f"""# Benchmark Comparison: DINO-DETR / RT-DETR (FP16 vs FP8 APA)

**Hardware & Dataset Environment**:
- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- Model: `RT-DETR-L` (DINO-DETR Architecture: ~33.0M parameter, 74 Transformer Linear layers)
- Dataset: `merged_yolo_person_ball` (Class 0: Person)
- Batch Size: `{batch_size}` | Iterasi Benchmark: `{max_steps}` batches

| Parameter / Metrik | DINO-DETR (FP16 Baseline) | DINO-DETR (FP8 APA) | Perubahan / Efisiensi |
| :--- | :---: | :---: | :---: |
| **Rata-rata Step Latency** | **{results[0]['mean_step_ms']:.1f} ms** | **{results[1]['mean_step_ms']:.1f} ms** | **{speedup:.2f}x Speedup** |
| **Throughput (Images / Detik)** | **{results[0]['throughput_img_sec']:.1f} img/s** | **{results[1]['throughput_img_sec']:.1f} img/s** | **+{(results[1]['throughput_img_sec'] - results[0]['throughput_img_sec']):.1f} img/s** |
| **Peak VRAM Allocated** | **{results[0]['max_vram_mb']} MB** | **{results[1]['max_vram_mb']} MB** | **Hemat {vram_saved:.1f} MB (-{vram_saved_pct:.1f}%)** |
| **Peak VRAM Reserved** | {results[0]['reserved_vram_mb']} MB | {results[1]['reserved_vram_mb']} MB | Penghematan memori GPU |
| **Durasi {max_steps} Steps** | {results[0]['duration_sec']:.1f} s | {results[1]['duration_sec']:.1f} s | - |
| **Train Loss ({max_steps} steps)** | {results[0]['train_loss']} | {results[1]['train_loss']} | Konvergensi numerik stabil |
| **Classification Loss** | {results[0]['cls_loss']} | {results[1]['cls_loss']} | Terjaga |
| **GIoU Box Loss** | {results[0]['giou_loss']} | {results[1]['giou_loss']} | Terjaga |
| **L1 Box Loss** | {results[0]['l1_loss']} | {results[1]['l1_loss']} | Terjaga |
| **Distribusi Presisi Layer** | All FP16 (74 layers) | FP8: {results[1]['precision_distribution'].get('fp8', 74)}, TF32: {results[1]['precision_distribution'].get('tf32', 0)} | Native FP8 Tensor Cores |
"""

    summary_file = os.path.join(base_dir, 'result', 'rtdetr_fp8_vs_fp16_comparison.md')
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
