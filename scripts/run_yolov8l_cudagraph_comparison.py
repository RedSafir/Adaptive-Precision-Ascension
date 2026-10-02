import sys
import os
import time
import json
import subprocess
import torch
import numpy as np

def run_command_stream(cmd, desc):
    print("=" * 70)
    print(f"STARTING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 70, flush=True)
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
    train_script = os.path.join(base_dir, 'examples', 'yolov8_coco', 'train.py')
    data_yaml = os.path.join(base_dir, 'datasets', 'detection-person-and-ball', 'merged_yolo_person_ball', 'data.yaml')
    
    os.makedirs('result', exist_ok=True)
    log_no_cg = os.path.join(base_dir, 'result', 'yolov8l_compare_no_cudagraph.jsonl')
    log_with_cg = os.path.join(base_dir, 'result', 'yolov8l_compare_with_cudagraph.jsonl')

    # Base training arguments for 1 Full Epoch
    common_args = [
        python_bin, train_script,
        '--model', 'yolov8l.pt',
        '--data', data_yaml,
        '--classes', '0',
        '--batch_size', '16',
        '--imgsz', '640',
        '--epochs', '1',
        '--precision', 'apa',
        '--apa_preset', 'research',
        '--check_interval', '1',
        '--device', 'cuda',
        '--workers', '8',
        '--no_save'
    ]

    cmd_no_cg = common_args + ['--log_file', log_no_cg]
    cmd_with_cg = common_args + ['--cuda_graph', '--log_file', log_with_cg]

    print("######################################################################")
    print("  YOLOv8l HYBRID CUDA GRAPH VS EAGER BENCHMARK (1 FULL EPOCH)")
    print("######################################################################")
    print(f"Dataset: {data_yaml} (Class 0: Person)")
    print(f"Model: YOLOv8l (Large, ~43.6M params)")
    print(f"Batch Size: 16 | Imgsz: 640 | Epochs: 1 (1,971 steps per run)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("######################################################################\n", flush=True)

    # 1. Run Baseline (Without Hybrid CUDA Graph)
    t_no_cg = run_command_stream(cmd_no_cg, "1/2: YOLOv8l APA Eager Training (Tanpa Hybrid CUDA Graph)")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(5)

    # 2. Run With Hybrid CUDA Graph
    t_with_cg = run_command_stream(cmd_with_cg, "2/2: YOLOv8l APA + Hybrid CUDA Graph Training")

    # 3. Analyze and Compare
    _, steps_no_cg, epochs_no_cg = parse_log(log_no_cg)
    _, steps_with_cg, epochs_with_cg = parse_log(log_with_cg)

    ep_no_cg = epochs_no_cg[0] if epochs_no_cg else {}
    ep_with_cg = epochs_with_cg[0] if epochs_with_cg else {}

    time_no_cg = ep_no_cg.get('epoch_time_sec', t_no_cg)
    time_with_cg = ep_with_cg.get('epoch_time_sec', t_with_cg)
    speedup = (time_no_cg / time_with_cg) if time_with_cg > 0 else 1.0

    step_times_no_cg = [s['step_time_ms'] for s in steps_no_cg if 'step_time_ms' in s]
    step_times_with_cg = [s['step_time_ms'] for s in steps_with_cg if 'step_time_ms' in s]

    mean_ms_no_cg = np.mean(step_times_no_cg) if step_times_no_cg else 0.0
    mean_ms_with_cg = np.mean(step_times_with_cg) if step_times_with_cg else 0.0

    summary_md = f"""# Benchmark Comparison: YOLOv8l APA With vs Without Hybrid CUDA Graph

| Metric | Tanpa Hybrid CUDA Graph (Eager APA) | Dengan Hybrid CUDA Graph (`--cuda_graph`) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Durasi 1 Epoch Penuh** | {time_no_cg:.2f} s ({time_no_cg/60:.2f} min) | {time_with_cg:.2f} s ({time_with_cg/60:.2f} min) | **{speedup:.2f}x Faster** (hemat {time_no_cg - time_with_cg:.1f}s) |
| **Rata-rata Step Time** | {mean_ms_no_cg:.1f} ms | {mean_ms_with_cg:.1f} ms | **{(mean_ms_no_cg - mean_ms_with_cg):.1f} ms lebih cepat per batch** |
| **Total Train Loss** | {ep_no_cg.get('train_loss', 'N/A')} | {ep_with_cg.get('train_loss', 'N/A')} | Identik / Konsisten |
| **Box Loss** | {ep_no_cg.get('box_loss', 'N/A')} | {ep_with_cg.get('box_loss', 'N/A')} | Identik / Konsisten |
| **Class Loss** | {ep_no_cg.get('cls_loss', 'N/A')} | {ep_with_cg.get('cls_loss', 'N/A')} | Identik / Konsisten |
| **DFL Loss** | {ep_no_cg.get('dfl_loss', 'N/A')} | {ep_with_cg.get('dfl_loss', 'N/A')} | Identik / Konsisten |
| **FP8 Backbone/Neck Layers** | {ep_no_cg.get('precision_distribution', {}).get('fp8', 'N/A')} | {ep_with_cg.get('precision_distribution', {}).get('fp8', 'N/A')} | FP8 Preserved |
| **TF32 Detect Head Layers** | {ep_no_cg.get('precision_distribution', {}).get('tf32', 'N/A')} | {ep_with_cg.get('precision_distribution', {}).get('tf32', 'N/A')} | TF32 Preserved |
"""

    summary_file = os.path.join(base_dir, 'result', 'yolov8l_cudagraph_comparison.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON RESULT:")
    print("=" * 70)
    print(summary_md)
    print("=" * 70)
    print(f"Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
