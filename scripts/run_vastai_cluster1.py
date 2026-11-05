#!/usr/bin/env python3
"""
Vast.ai Automated Orchestration Runner: Klaster 1 (YOLOS APA Threshold Spectrum)
Runs 3 experiments across the overflow spectrum: gamma in [0.1, 0.5, 0.9] with theta_underflow fixed at 0.9.
Automatically generates comparative reports, pushes logs to GitHub, and shuts down instance.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Dict, Any, List

DEFAULT_GAMMA_LIST = [0.1, 0.5, 0.9]
DEFAULT_THETA_UNDERFLOW = 0.9

def parse_args():
    parser = argparse.ArgumentParser(description="Run Klaster 1 YOLOS APA Threshold Spectrum Suite on Vast.ai")
    parser.add_argument('--gamma_list', type=float, nargs='+', default=DEFAULT_GAMMA_LIST,
                        help="List of gamma overflow thresholds to test (default: 0.1 0.5 0.9)")
    parser.add_argument('--theta_underflow', type=float, default=DEFAULT_THETA_UNDERFLOW,
                        help="Fixed underflow ratio threshold (default: 0.9)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per run (default: 50)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--imgsz', type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument('--check_interval', type=int, default=8, help="Soft-check interval in steps (default: 8)")
    parser.add_argument('--enable_frobenius', action='store_true', default=True, help="Enable Frobenius norm diagnostic telemetry (default: True)")
    parser.add_argument('--no_frobenius', dest='enable_frobenius', action='store_false', help="Disable Frobenius norm telemetry")
    parser.add_argument('--frobenius_check_interval', type=int, default=8, help="Step interval for Frobenius norm logging (default: 8)")
    parser.add_argument('--cuda_graph', action='store_true', default=True, help="Enable Hybrid CUDA Graph acceleration")
    parser.add_argument('--no_cuda_graph', dest='cuda_graph', action='store_false', help="Disable CUDA Graph")
    parser.add_argument('--data_dir', type=str, default='datasets/detection-person-and-ball/merged_yolo_person_ball',
                        help="Path to dataset directory")
    parser.add_argument('--github_pat', type=str, default=os.environ.get('GITHUB_PAT', None),
                        help="GitHub Personal Access Token for auto-pushing results")
    parser.add_argument('--on_finish', type=str, choices=['none', 'stop', 'destroy'], default='stop',
                        help="Action to perform on Vast.ai instance after completion: 'stop', 'destroy', or 'none'")
    parser.add_argument('--smoke_test', action='store_true', help="Run 3 steps per epoch for quick validation")
    parser.add_argument('--max_steps', type=int, default=0, help="Max steps per epoch (0 = full)")
    return parser.parse_args()

def parse_jsonl(log_path: str):
    header_meta = {}
    epoch_summaries = []
    step_records = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    r_type = record.get('type')
                    if r_type == 'header':
                        header_meta = record.get('meta', {})
                    elif r_type == 'epoch':
                        epoch_summaries.append(record)
                    elif r_type == 'step':
                        step_records.append(record)
                except Exception:
                    pass
    return header_meta, epoch_summaries, step_records

def count_escalation_events(events_path: str) -> Dict[str, int]:
    counts = {'OVERFLOW': 0, 'SILENT_UNDERFLOW': 0, 'TOTAL': 0}
    if os.path.exists(events_path):
        with open(events_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    trigger = ev.get('trigger_reason', '')
                    if 'OVERFLOW' in trigger:
                        counts['OVERFLOW'] += 1
                    elif 'UNDERFLOW' in trigger:
                        counts['SILENT_UNDERFLOW'] += 1
                    counts['TOTAL'] += 1
                except Exception:
                    pass
    return counts

def is_run_completed(log_path: str, target_epochs: int) -> bool:
    if not os.path.exists(log_path):
        return False
    _, epochs, _ = parse_jsonl(log_path)
    return len(epochs) >= target_epochs

def run_single_experiment(val_gamma: float, val_theta: float, args: argparse.Namespace, base_dir: str):
    tag = f"gamma_{val_gamma}_theta_{val_theta}"
    save_dir = os.path.join(base_dir, 'runs', f"cluster1_yolos_{tag}")
    log_file = os.path.join(save_dir, "yolos_training.jsonl")
    events_file = os.path.join(save_dir, "yolos_training_apa_events.jsonl")

    print("\n" + "=" * 90)
    print(f"  [KLASTER 1] RUNNING EXPERIMENT: GAMMA={val_gamma} | THETA_UNDERFLOW={val_theta}")
    print(f"  Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | Check Interval: {args.check_interval}")
    print(f"  CUDA Graph: {args.cuda_graph} | Save Dir: {save_dir}")
    print("=" * 90, flush=True)

    if is_run_completed(log_file, args.epochs):
        print(f"[*] Found completed run for gamma={val_gamma}, theta={val_theta} ({args.epochs} epochs). Skipping.")
        return log_file, events_file

    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(base_dir, "examples", "yolos_person", "train.py"),
        "--model", "hustvl/yolos-small",
        "--data_dir", args.data_dir,
        "--precision", "apa",
        "--gamma", str(val_gamma),
        "--theta_underflow", str(val_theta),
        "--check_interval", str(args.check_interval),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--imgsz", str(args.imgsz),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--classes", "0",
        "--save_dir", save_dir,
        "--log_file", log_file,
        "--no_save"
    ]
    if args.cuda_graph:
        cmd.append("--cuda_graph")
    if args.enable_frobenius:
        cmd.extend([
            "--enable_frobenius_telemetry",
            "--frobenius_check_interval", str(args.frobenius_check_interval)
        ])
    if args.smoke_test:
        cmd.extend(["--max_steps", "3"])
    elif args.max_steps > 0:
        cmd.extend(["--max_steps", str(args.max_steps)])

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=base_dir)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Experiment gamma={val_gamma}, theta={val_theta} failed with exit code {proc.returncode}")

    print(f"[✓] Experiment gamma={val_gamma}, theta={val_theta} completed in {elapsed/60:.2f} minutes.")
    return log_file, events_file

def generate_cluster1_report(results: List[Dict[str, Any]], theta_val: float, report_path: str):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    headers = ["Metrik Evaluasi"] + [f"$\\gamma={r['gamma']}$ ($\\theta={theta_val}$)" for r in results]
    aligns = [":---"] + [":---:" for _ in results]

    lines = [
        "# Laporan Evaluasi Komparatif: Klaster 1 YOLOS APA (Spektrum Overflow)",
        f"## Evaluasi Sensitivitas Gamma $\\gamma \\in [0.1, 0.5, 0.9]$ dengan Underflow Tetap $\\theta = {theta_val}$ (50 Epochs)",
        "",
        "Laporan ini mendokumentasikan hasil pengujian komprehensif pengaruh ambang batas overflow $\\gamma$ pada model **YOLOS-Small** (27.4M parameter) dengan akselerasi **Hybrid CUDA Graph**.",
        "",
        "---",
        "",
        "### 1. Ringkasan Kinerja Keseluruhan (50 Epochs)",
        "",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(aligns)} |"
    ]

    def fmt(val, template="{:.4f}"):
        return template.format(val) if val is not None else "N/A"

    def get_stat(r, key, default=None):
        return r.get(key, default)

    durations = [f"{get_stat(r, 'total_time_min', 0):.1f} min" for r in results]
    throughputs = [f"{get_stat(r, 'avg_fps', 0):.1f} fps" for r in results]
    latencies = [f"{get_stat(r, 'avg_latency_ms', 0):.1f} ms" for r in results]
    vrams = [f"{get_stat(r, 'peak_vram_mb', 0):.0f} MB" for r in results]
    total_loss = [fmt(get_stat(r, 'final_total_loss')) for r in results]
    bbox_loss = [fmt(get_stat(r, 'final_bbox_l1')) for r in results]
    giou_loss = [fmt(get_stat(r, 'final_giou')) for r in results]
    ce_loss = [fmt(get_stat(r, 'final_ce')) for r in results]
    escalations = [str(get_stat(r, 'escalation_total', 0)) for r in results]
    underflows = [str(get_stat(r, 'underflow_events', 0)) for r in results]
    overflows = [str(get_stat(r, 'overflow_events', 0)) for r in results]

    lines.append(f"| **Total Durasi Training** | {' | '.join(durations)} |")
    lines.append(f"| **Training Throughput** | {' | '.join(throughputs)} |")
    lines.append(f"| **Latensi per Step** | {' | '.join(latencies)} |")
    lines.append(f"| **Peak VRAM** | {' | '.join(vrams)} |")
    lines.append(f"| **Final Total Loss** | {' | '.join(total_loss)} |")
    lines.append(f"| **BBox L1 Loss (Regresi)** | {' | '.join(bbox_loss)} |")
    lines.append(f"| **GIoU Loss (Overlap)** | {' | '.join(giou_loss)} |")
    lines.append(f"| **Class CE Loss** | {' | '.join(ce_loss)} |")
    lines.append(f"| **Total Event Eskalasi** | {' | '.join(escalations)} |")
    lines.append(f"| - *Silent Underflow* | {' | '.join(underflows)} |")
    lines.append(f"| - *Hard Overflow* | {' | '.join(overflows)} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 2. Analisis Ilmiah & Kesimpulan Sensitivitas Parameter")
    lines.append("1. **Isolasi Variabel Overflow (Gamma)**: Dengan mengunci $\\theta_{\\text{underflow}}=0.9$, toleransi underflow dibuka lebar sehingga eskalasi layer murni dipicu oleh lonjakan magnitudo tensor yang melampaui $T_{\\text{max}} = \\gamma \\times \\text{dtype\\_max}$.")
    lines.append("2. **Perilaku $\\gamma=0.1$ vs $\\gamma=0.9$**: Mengamati apakah batasan overflow ketat ($\\gamma=0.1$, $T_{\\text{max}}=44.8$) memicu promosi layer lebih dini ke FP16/TF32 dibanding batasan longgar ($\\gamma=0.9$, $T_{\\text{max}}=403.2$).")
    lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[✓] Consolidated report written to: {report_path}")

def auto_push_results_to_github(github_pat: str, base_dir: str):
    if not github_pat:
        print("[!] GITHUB_PAT not provided. Skipping remote git push.")
        return False

    print("\n" + "=" * 80)
    print("  [GIT AUTO-PUSH] Pushing Cluster 1 Results & Logs to GitHub...")
    print("=" * 80)
    try:
        subprocess.run(["git", "config", "user.name", "RedSafir"], cwd=base_dir, check=True)
        subprocess.run(["git", "config", "user.email", "mfthnugraha@gmail.com"], cwd=base_dir, check=True)
        subprocess.run(["git", "add", "result/", "runs/"], cwd=base_dir, check=True)

        status = subprocess.run(["git", "status", "--porcelain"], cwd=base_dir, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", "chore(vastai): auto-commit cluster 1 50-epoch results [skip ci]"], cwd=base_dir, check=True)
        else:
            print("[*] No new file changes to commit.")

        remote_url = f"https://RedSafir:{github_pat}@github.com/RedSafir/Adaptive-Precision-Ascension.git"
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'
        env['GIT_ASKPASS'] = ''
        subprocess.run(["git", "push", remote_url, "main"], cwd=base_dir, check=True, env=env)
        print("[✓] All results successfully pushed to GitHub origin/main!")
        return True
    except Exception as e:
        print(f"[!] Git push failed: {e}")
        return False

def shutdown_vastai_instance(action: str):
    if action not in ('stop', 'destroy'):
        print(f"[*] on_finish is set to '{action}'. Instance will remain active.")
        return

    container_id = os.environ.get('VAST_CONTAINERLABEL', os.environ.get('CONTAINER_ID', None))
    api_key = os.environ.get('CONTAINER_API_KEY', None)

    print("\n" + "=" * 80)
    print(f"  [AUTO-SHUTDOWN] Triggering Vast.ai instance {action.upper()}...")
    print(f"  Container Label: {container_id} | API Key present: {bool(api_key)}")
    print("=" * 80)

    # 1. Try vastai CLI
    if container_id:
        cmd = ["vastai", action, "instance", str(container_id)]
        if api_key:
            cmd.extend(["--api-key", api_key])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[✓] Vast.ai instance {container_id} {action} succeeded via CLI: {res.stdout.strip()}")
                return
            else:
                print(f"[!] Vast CLI returned code {res.returncode}: {res.stderr.strip()}")
        except FileNotFoundError:
            print("[*] vastai CLI not installed, trying curl API fallback...")

    # 2. Try Vast.ai REST API via curl fallback
    if container_id and api_key:
        try:
            import urllib.request
            method = "DELETE" if action == "destroy" else "PUT"
            url = f"https://console.vast.ai/api/v0/instances/{container_id}/"
            data = None if action == "destroy" else json.dumps({"state": "stopped"}).encode('utf-8')
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"[✓] Vast.ai API {action} call responded with HTTP {resp.status}")
                return
        except Exception as e:
            print(f"[!] Vast REST API fallback failed: {e}")

    # 3. Last resort local OS shutdown
    print("[*] Executing system poweroff as fallback to stop billing...")
    os.system("poweroff || shutdown -h now")

def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(base_dir)

    print("=" * 90)
    print("  VAST.AI ORCHESTRATION SUITE: KLASTER 1 (YOLOS APA THRESHOLD SPECTRUM)")
    print(f"  Gamma List: {args.gamma_list} | Fixed Theta Underflow: {args.theta_underflow}")
    print(f"  Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | CUDA Graph: {args.cuda_graph}")
    print(f"  On Finish: {args.on_finish.upper()}")
    print("=" * 90)

    results = []
    for val_gamma in args.gamma_list:
        log_file, events_file = run_single_experiment(val_gamma, args.theta_underflow, args, base_dir)
        header, epochs, steps = parse_jsonl(log_file)
        ev_counts = count_escalation_events(events_file)

        last_epoch = epochs[-1] if epochs else {}
        total_time_min = sum(e.get('duration_sec', 0) for e in epochs) / 60.0
        avg_fps = sum(e.get('avg_fps', 0) for e in epochs) / max(len(epochs), 1)
        avg_lat = sum(e.get('avg_latency_ms', 0) for e in epochs) / max(len(epochs), 1)
        peak_vram = max([e.get('peak_vram_mb', 0) for e in epochs] + [0])

        results.append({
            'gamma': val_gamma,
            'theta_underflow': args.theta_underflow,
            'total_time_min': total_time_min,
            'avg_fps': avg_fps,
            'avg_latency_ms': avg_lat,
            'peak_vram_mb': peak_vram,
            'final_total_loss': last_epoch.get('avg_loss', None),
            'final_bbox_l1': last_epoch.get('avg_bbox', None),
            'final_giou': last_epoch.get('avg_giou', None),
            'final_ce': last_epoch.get('avg_ce', None),
            'escalation_total': ev_counts['TOTAL'],
            'underflow_events': ev_counts['SILENT_UNDERFLOW'],
            'overflow_events': ev_counts['OVERFLOW']
        })

    # Generate Markdown Report
    report_file = os.path.join(base_dir, 'result', 'cluster1_yolos_threshold_spectrum_report.md')
    generate_cluster1_report(results, args.theta_underflow, report_file)

    # Push to GitHub
    push_ok = auto_push_results_to_github(args.github_pat, base_dir)

    # Auto Shutdown Vast.ai Instance
    shutdown_vastai_instance(args.on_finish)

if __name__ == '__main__':
    main()
