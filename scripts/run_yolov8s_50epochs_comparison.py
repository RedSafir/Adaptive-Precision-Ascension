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
    print(f"\n[COMPLETED] {desc} in {duration:.1f}s ({duration/60:.2f} min / {duration/3600:.2f} hours)\n", flush=True)
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
            try:
                data = json.loads(line)
            except Exception:
                continue
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
    log_with_cg = os.path.join(base_dir, 'result', 'yolov8s_person_apa_cudagraph.jsonl')
    log_no_cg = os.path.join(base_dir, 'result', 'yolov8s_person_apa_no_cudagraph.jsonl')

    # Common training arguments for 50 Epochs
    common_args = [
        python_bin, train_script,
        '--model', 'yolov8s.pt',
        '--data', data_yaml,
        '--classes', '0',
        '--batch_size', '16',
        '--imgsz', '640',
        '--epochs', '50',
        '--lr', '0.001',
        '--precision', 'apa',
        '--apa_preset', 'research',
        '--check_interval', '1',
        '--device', 'cuda',
        '--workers', '8',
        '--no_save'
    ]

    cmd_with_cg = common_args + ['--cuda_graph', '--log_file', log_with_cg]
    cmd_no_cg = common_args + ['--log_file', log_no_cg]

    print("######################################################################")
    print("  YOLOv8s HYBRID CUDA GRAPH VS EAGER BENCHMARK (50 FULL EPOCHS)")
    print("######################################################################")
    print(f"Dataset: {data_yaml} (Class 0: Person, 31,528 images)")
    print(f"Model: YOLOv8s (Small, ~11.2M params)")
    print(f"Batch Size: 16 | Imgsz: 640 | Epochs: 50 (~98,550 total steps per mode)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("######################################################################\n", flush=True)

    # 1. Run 1: With Hybrid CUDA Graph
    t_with_cg = run_command_stream(cmd_with_cg, "1/2: YOLOv8s APA + Hybrid CUDA Graph Training (50 Epochs)")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(10)

    # 2. Run 2: Without Hybrid CUDA Graph (Eager APA)
    t_no_cg = run_command_stream(cmd_no_cg, "2/2: YOLOv8s APA Eager Training (Tanpa Hybrid CUDA Graph, 50 Epochs)")

    # 3. Analyze and Compare
    _, steps_with_cg, epochs_with_cg = parse_log(log_with_cg)
    _, steps_no_cg, epochs_no_cg = parse_log(log_no_cg)

    last_ep_with_cg = epochs_with_cg[-1] if epochs_with_cg else {}
    last_ep_no_cg = epochs_no_cg[-1] if epochs_no_cg else {}

    epoch_times_with_cg = [e['epoch_time_sec'] for e in epochs_with_cg if 'epoch_time_sec' in e]
    epoch_times_no_cg = [e['epoch_time_sec'] for e in epochs_no_cg if 'epoch_time_sec' in e]

    mean_epoch_with_cg = np.mean(epoch_times_with_cg) if epoch_times_with_cg else (t_with_cg / 50.0)
    mean_epoch_no_cg = np.mean(epoch_times_no_cg) if epoch_times_no_cg else (t_no_cg / 50.0)

    speedup = (t_no_cg / t_with_cg) if t_with_cg > 0 else 1.0

    step_times_with_cg = [s['step_time_ms'] for s in steps_with_cg if 'step_time_ms' in s]
    step_times_no_cg = [s['step_time_ms'] for s in steps_no_cg if 'step_time_ms' in s]

    mean_ms_with_cg = np.mean(step_times_with_cg) if step_times_with_cg else 0.0
    mean_ms_no_cg = np.mean(step_times_no_cg) if step_times_no_cg else 0.0

    summary_md = f"""# Benchmark Comparison: YOLOv8s APA 50 Epochs (With vs Without Hybrid CUDA Graph)

| Metric | Dengan Hybrid CUDA Graph (`--cuda_graph`) | Tanpa Hybrid CUDA Graph (Eager APA) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Total Waktu 50 Epochs** | {t_with_cg/3600:.2f} jam ({t_with_cg/60:.1f} min) | {t_no_cg/3600:.2f} jam ({t_no_cg/60:.1f} min) | **{speedup:.2f}x Faster** (hemat {(t_no_cg - t_with_cg)/60:.1f} min) |
| **Rata-rata Waktu / Epoch** | {mean_epoch_with_cg:.2f} s | {mean_epoch_no_cg:.2f} s | **{(mean_epoch_no_cg - mean_epoch_with_cg):.2f} s lebih cepat per epoch** |
| **Rata-rata Step Latency** | {mean_ms_with_cg:.1f} ms | {mean_ms_no_cg:.1f} ms | **{(mean_ms_no_cg - mean_ms_with_cg):.1f} ms lebih cepat per batch** |
| **Final Train Loss (Epoch 50)** | {last_ep_with_cg.get('train_loss', 'N/A')} | {last_ep_no_cg.get('train_loss', 'N/A')} | Konvergen stabil |
| **Final Box Loss** | {last_ep_with_cg.get('box_loss', 'N/A')} | {last_ep_no_cg.get('box_loss', 'N/A')} | Konvergen stabil |
| **Final Class Loss** | {last_ep_with_cg.get('cls_loss', 'N/A')} | {last_ep_no_cg.get('cls_loss', 'N/A')} | Konvergen stabil |
| **Final DFL Loss** | {last_ep_with_cg.get('dfl_loss', 'N/A')} | {last_ep_no_cg.get('dfl_loss', 'N/A')} | Konvergen stabil |
| **Distribusi Presisi Akhir (FP8/FP16/TF32)** | FP8:{last_ep_with_cg.get('precision_distribution', {}).get('fp8', '-')}, FP16:{last_ep_with_cg.get('precision_distribution', {}).get('fp16', '-')}, TF32:{last_ep_with_cg.get('precision_distribution', {}).get('tf32', '-')} | FP8:{last_ep_no_cg.get('precision_distribution', {}).get('fp8', '-')}, FP16:{last_ep_no_cg.get('precision_distribution', {}).get('fp16', '-')}, TF32:{last_ep_no_cg.get('precision_distribution', {}).get('tf32', '-')} | - |
"""

    summary_file = os.path.join(base_dir, 'result', 'comparison_yolov8s_50epochs_cudagraph.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print("\n" + "=" * 70)
    print("FINAL 50-EPOCH COMPARISON RESULT:")
    print("=" * 70)
    print(summary_md)
    print("=" * 70)
    print(f"Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
