#!/usr/bin/env python3
"""
Vast.ai Automated Orchestration Runner: Taguchi L9 Design of Experiments Suite
Runs 9 experiments across:
  - Factor A (Gamma): [0.1, 0.5, 0.9]
  - Factor B (Theta Underflow): [0.1, 0.5, 0.9]
  - Factor C (Check Interval): [1, 8, 985]

After every single run completes:
  1. Generates/updates comparative Taguchi report in result/taguchi_l9_yolos_report.md
  2. Compresses .jsonl -> .jsonl.gz
  3. Uploads raw telemetry to Hugging Face Private Dataset (miftahN/apa-research-telemetry)
  4. Auto-pushes updated report markdown to GitHub (if GITHUB_PAT provided)
  5. Resumes cleanly if interrupted
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import time
import json
import gzip
import shutil
import argparse
import subprocess
from typing import Dict, Any, List, Optional

# Standard Taguchi L9(3^3) Orthogonal Array Design Matrix
TAGUCHI_L9_RUNS = [
    {"run_id": 1, "gamma": 0.1, "theta_underflow": 0.1, "check_interval": 1},
    {"run_id": 2, "gamma": 0.1, "theta_underflow": 0.5, "check_interval": 8},
    {"run_id": 3, "gamma": 0.1, "theta_underflow": 0.9, "check_interval": 985},
    {"run_id": 4, "gamma": 0.5, "theta_underflow": 0.1, "check_interval": 8},
    {"run_id": 5, "gamma": 0.5, "theta_underflow": 0.5, "check_interval": 985},
    {"run_id": 6, "gamma": 0.5, "theta_underflow": 0.9, "check_interval": 1},
    {"run_id": 7, "gamma": 0.9, "theta_underflow": 0.1, "check_interval": 985},
    {"run_id": 8, "gamma": 0.9, "theta_underflow": 0.5, "check_interval": 1},
    {"run_id": 9, "gamma": 0.9, "theta_underflow": 0.9, "check_interval": 8},
]

DEFAULT_HF_REPO_ID = "miftahN/apa-research-telemetry"

def _get_default_hf_token() -> Optional[str]:
    token = os.environ.get('HF_TOKEN', None)
    if token:
        return token.strip()
    token_path = os.path.expanduser('~/.cache/huggingface/token')
    if os.path.exists(token_path):
        try:
            with open(token_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run Taguchi L9(3^3) YOLOS APA Suite on Vast.ai")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per run (default: 50)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument('--imgsz', type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument('--enable_frobenius', action='store_true', default=True, help="Enable Frobenius norm diagnostic telemetry (default: True)")
    parser.add_argument('--no_frobenius', dest='enable_frobenius', action='store_false', help="Disable Frobenius norm telemetry")
    parser.add_argument('--frobenius_check_interval', type=int, default=8, help="Fixed step interval for Frobenius norm logging (default: 8)")
    parser.add_argument('--cuda_graph', action='store_true', default=False, help="Enable Hybrid CUDA Graph acceleration (default: False)")
    parser.add_argument('--no_cuda_graph', dest='cuda_graph', action='store_false', help="Disable CUDA Graph")
    parser.add_argument('--data_dir', type=str, default='datasets/detection-person-and-ball/merged_yolo_person_ball',
                        help="Path to dataset directory")
    parser.add_argument('--github_pat', type=str, default=os.environ.get('GITHUB_PAT', None),
                        help="GitHub Personal Access Token for auto-pushing markdown reports")
    parser.add_argument('--hf_token', type=str, default=_get_default_hf_token(),
                        help="Hugging Face token for auto-uploading raw telemetry after each run")
    parser.add_argument('--hf_repo_id', type=str, default=DEFAULT_HF_REPO_ID,
                        help=f"Hugging Face dataset repo ID (default: {DEFAULT_HF_REPO_ID})")
    parser.add_argument('--on_finish', type=str, choices=['none', 'stop', 'destroy'], default='stop',
                        help="Action to perform on Vast.ai instance after completion: 'stop', 'destroy', or 'none'")
    parser.add_argument('--smoke_test', action='store_true', help="Run 3 steps per epoch for quick validation")
    parser.add_argument('--max_steps', type=int, default=0, help="Max steps per epoch (0 = full)")
    parser.add_argument('--start_run', type=int, default=1, help="Start from run_id (1-9)")
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
                    if ev.get('event') not in (None, 'escalation'):
                        continue
                    reason = ev.get('reason')
                    if reason in counts:
                        counts[reason] += 1
                    counts['TOTAL'] += 1
                except Exception:
                    pass
    return counts

def run_single_taguchi_experiment(cfg: Dict[str, Any], args, base_dir: str):
    run_id = cfg["run_id"]
    val_gamma = cfg["gamma"]
    val_theta = cfg["theta_underflow"]
    val_ci = cfg["check_interval"]

    run_name = f"taguchi_run{run_id}_gamma_{val_gamma}_theta_{val_theta}_ci_{val_ci}"
    save_dir = os.path.join(base_dir, 'runs', run_name)
    log_file = os.path.join(save_dir, 'yolos_training.jsonl')
    events_file = os.path.join(save_dir, 'yolos_training_apa_events.jsonl')

    # Check if run already completed (Auto-Resume)
    if os.path.exists(log_file):
        _, epochs, _ = parse_jsonl(log_file)
        if len(epochs) >= args.epochs:
            print(f"[SKIP] Run {run_id} ({run_name}) already completed with {len(epochs)} epochs.", flush=True)
            return log_file, events_file

    print("\n" + "=" * 90, flush=True)
    print(f"  [TAGUCHI L9 - RUN {run_id}/9] Gamma={val_gamma} | Theta_underflow={val_theta} | Check_interval={val_ci}", flush=True)
    print(f"  Target Epochs: {args.epochs} | Batch Size: {args.batch_size} | CUDA Graph: {args.cuda_graph}", flush=True)
    print(f"  Save Dir: {save_dir}", flush=True)
    print("=" * 90, flush=True)

    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, 'examples/yolos_person/train.py',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--imgsz', str(args.imgsz),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--precision', 'apa',
        '--gamma', str(val_gamma),
        '--theta_underflow', str(val_theta),
        '--check_interval', str(val_ci),
        '--data_dir', args.data_dir,
        '--save_dir', save_dir,
        '--log_file', log_file,
    ]

    if args.cuda_graph:
        cmd.append('--cuda_graph')
    if args.enable_frobenius:
        cmd.extend(['--enable_frobenius_telemetry', '--frobenius_check_interval', str(args.frobenius_check_interval)])
    if args.smoke_test:
        cmd.extend(['--max_steps', '3'])
    elif args.max_steps > 0:
        cmd.extend(['--max_steps', str(args.max_steps)])

    start_t = time.time()
    proc = subprocess.Popen(cmd, cwd=base_dir)
    ret = proc.wait()
    dur_min = (time.time() - start_t) / 60.0

    if ret != 0:
        raise RuntimeError(f"Run {run_id} ({run_name}) failed with exit code {ret} after {dur_min:.2f} mins.")

    print(f"[✓] Run {run_id} completed successfully in {dur_min:.2f} minutes.", flush=True)
    return log_file, events_file

def upload_run_to_huggingface(hf_token: Optional[str], hf_repo_id: str, save_dir: str, report_file: str, run_tag: str):
    token = hf_token or _get_default_hf_token()
    if not token:
        print("[!] HF_TOKEN not provided or found. Skipping Hugging Face upload.", flush=True)
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[!] huggingface_hub package not installed. Skipping Hugging Face upload.", flush=True)
        return

    print(f"[*] Preparing Hugging Face upload for {run_tag} to dataset '{hf_repo_id}'...", flush=True)
    try:
        # Compress jsonl files
        for fname in os.listdir(save_dir):
            if fname.endswith(".jsonl") and not fname.endswith(".jsonl.gz"):
                raw_path = os.path.join(save_dir, fname)
                gz_path = raw_path + ".gz"
                if not os.path.exists(gz_path) or os.path.getmtime(raw_path) > os.path.getmtime(gz_path):
                    with open(raw_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)

        if os.path.exists(report_file):
            shutil.copy2(report_file, os.path.join(save_dir, os.path.basename(report_file)))

        api.upload_folder(
            folder_path=save_dir,
            path_in_repo=run_tag,
            repo_id=hf_repo_id,
            repo_type="dataset",
            allow_patterns=["*.jsonl.gz", "*.md", "*_apa_events.jsonl", "yolos_training.jsonl"],
            commit_message=f"Upload Taguchi run {run_tag} telemetry"
        )
        print(f"[✓] Successfully uploaded {run_tag} to Hugging Face: {hf_repo_id}/{run_tag}", flush=True)
    except Exception as e:
        print(f"[!] Warning: Failed to upload {run_tag} to Hugging Face: {e}", flush=True)

def generate_taguchi_report(results: List[Dict[str, Any]], report_path: str):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = [
        "# Laporan Rekapitulasi Desain Eksperimen: Taguchi Orthogonal Array $L_9(3^3)$",
        "## Evaluasi Hyperparameter Adaptive Precision Ascension (APA) pada Model YOLOS",
        "",
        "Laporan ini mendokumentasikan hasil eksekusi 9 kombinasi ortogonal $L_9(3^3)$ secara berurutan:",
        "- **Faktor A (Gamma / Overflow Threshold)**: $\\gamma \\in [0.1, 0.5, 0.9]$",
        "- **Faktor B (Theta / Underflow Tolerance)**: $\\theta_{\\text{underflow}} \\in [0.1, 0.5, 0.9]$",
        "- **Faktor C (Check Interval / Evaluasi Soft-Check)**: $k \\in [1, 8, 985]$",
        "",
        "---",
        "",
        "### 1. Tabel Hasil Matriks Ortogonal $L_9$",
        "",
        "| Run ID | $\\gamma$ (Ovf) | $\\theta_{\\text{uf}}$ (Uf) | $k$ (CI) | Durasi (m) | Throughput (FPS) | Latensi (ms) | Peak VRAM | Final Loss | BBox L1 | GIoU | Eskalasi |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        run_id = r.get('run_id', '-')
        gamma = r.get('gamma', 0.0)
        theta = r.get('theta_underflow', 0.0)
        ci = r.get('check_interval', 0)
        dur = f"{r.get('total_time_min', 0.0):.1f}"
        fps = f"{r.get('avg_fps', 0.0):.1f}"
        lat = f"{r.get('avg_latency_ms', 0.0):.1f}"
        vram = f"{r.get('peak_vram_mb', 0.0):.1f} MB"
        loss = f"{r.get('final_total_loss', 0.0):.4f}" if r.get('final_total_loss') is not None else "-"
        bbox = f"{r.get('final_bbox_l1', 0.0):.4f}" if r.get('final_bbox_l1') is not None else "-"
        giou = f"{r.get('final_giou', 0.0):.4f}" if r.get('final_giou') is not None else "-"
        esk = f"{r.get('escalation_total', 0)} ({r.get('overflow_events', 0)}O/{r.get('underflow_events', 0)}U)"

        lines.append(f"| **R{run_id}** | `{gamma}` | `{theta}` | `{ci}` | {dur} | {fps} | {lat} | {vram} | {loss} | {bbox} | {giou} | {esk} |")

    lines.extend([
        "",
        "---",
        "",
        f"*Laporan diperbarui otomatis pada: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*"
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[✓] Taguchi L9 report updated at: {report_path}", flush=True)

def auto_push_results_to_github(github_pat: Optional[str], base_dir: str):
    pat = github_pat or os.environ.get('GITHUB_PAT', None)
    if not pat:
        return
    try:
        subprocess.run(['git', 'config', 'user.name', 'Miftah Nugraha'], cwd=base_dir, check=False)
        subprocess.run(['git', 'config', 'user.email', 'miftah.nugraha@mhs.itenas.ac.id'], cwd=base_dir, check=False)
        subprocess.run(['git', 'add', 'result/taguchi_l9_yolos_report.md'], cwd=base_dir, check=False)
        status_proc = subprocess.run(['git', 'status', '--porcelain', 'result/taguchi_l9_yolos_report.md'],
                                     cwd=base_dir, capture_output=True, text=True, check=False)
        if not status_proc.stdout.strip():
            return
        subprocess.run(['git', 'commit', '-m', "docs: update Taguchi L9 comparative report [skip ci]"],
                       cwd=base_dir, check=True)
        remote_url = f"https://RedSafir:{pat}@github.com/RedSafir/Adaptive-Precision-Ascension.git"
        subprocess.run(['git', 'push', remote_url, 'HEAD:main'], cwd=base_dir, check=True)
        print("[✓] Automatically pushed Taguchi report markdown to GitHub.", flush=True)
    except Exception as e:
        print(f"[!] Warning: git auto-push to GitHub failed: {e}", flush=True)

def shutdown_vastai_instance(action: str):
    if action == 'none':
        return
    raw_id = os.environ.get('VAST_CONTAINERLABEL', os.environ.get('CONTAINER_ID', ''))
    clean_id = raw_id.replace('C.', '').strip() if raw_id else ''
    api_key = os.environ.get('CONTAINER_API_KEY', os.environ.get('VAST_API_KEY', '')).strip()

    if clean_id and api_key:
        try:
            target_state = 'stopped' if action == 'stop' else 'deleted'
            cmd = ['curl', '-s', '-X', 'PUT', f'https://console.vast.ai/api/v1/instances/{clean_id}/',
                   '-H', f'Authorization: Bearer {api_key}', '-H', 'Content-Type: application/json',
                   '-d', json.dumps({'state': target_state})]
            subprocess.run(cmd, check=False, timeout=10)
        except Exception:
            pass

    if action in ('stop', 'destroy'):
        os.system("poweroff || shutdown -h now || true")

def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    report_file = os.path.join(base_dir, 'result', 'taguchi_l9_yolos_report.md')

    print("=" * 90, flush=True)
    print("  VAST.AI ORCHESTRATION RUNNER: TAGUCHI L9(3^3) YOLOS SUITE (9 RUNS)", flush=True)
    print(f"  Target Epochs per Run: {args.epochs} | Batch Size: {args.batch_size} | CUDA Graph: {args.cuda_graph}")
    print(f"  Frobenius Telemetry: {args.enable_frobenius} (Interval: {args.frobenius_check_interval})")
    print(f"  Auto-Upload HF Repo: {args.hf_repo_id}")
    print(f"  On Finish: {args.on_finish.upper()}")
    print("=" * 90, flush=True)

    results = []
    try:
        for run_cfg in TAGUCHI_L9_RUNS:
            if run_cfg["run_id"] < args.start_run:
                continue

            run_id = run_cfg["run_id"]
            val_gamma = run_cfg["gamma"]
            val_theta = run_cfg["theta_underflow"]
            val_ci = run_cfg["check_interval"]

            log_file, events_file = run_single_taguchi_experiment(run_cfg, args, base_dir)
            save_dir = os.path.dirname(log_file)
            run_tag = f"yolos_apa_taguchi_r{run_id}_gamma{val_gamma}_theta{val_theta}_ci{val_ci}_epochs{args.epochs}"

            header, epochs, steps = parse_jsonl(log_file)
            ev_counts = count_escalation_events(events_file)

            last_epoch = epochs[-1] if epochs else {}
            total_time_min = sum(e.get('duration_sec', 0) for e in epochs) / 60.0
            avg_fps = sum(e.get('avg_fps', 0) for e in epochs) / max(len(epochs), 1)
            avg_lat = sum(e.get('avg_latency_ms', 0) for e in epochs) / max(len(epochs), 1)
            peak_vram = max([e.get('peak_vram_mb', 0) for e in epochs] + [0])

            results.append({
                'run_id': run_id,
                'gamma': val_gamma,
                'theta_underflow': val_theta,
                'check_interval': val_ci,
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

            try:
                generate_taguchi_report(results, report_file)
            except Exception as rep_err:
                print(f"[!] Warning: report generation failed: {rep_err}", flush=True)

            try:
                upload_run_to_huggingface(args.hf_token, args.hf_repo_id, save_dir, report_file, run_tag)
            except Exception as hf_err:
                print(f"[!] Warning: HF upload failed: {hf_err}", flush=True)

            try:
                auto_push_results_to_github(args.github_pat, base_dir)
            except Exception:
                pass

    except Exception as e:
        print(f"[ERROR] Exception during Taguchi L9 execution: {e}", flush=True)
    finally:
        try:
            auto_push_results_to_github(args.github_pat, base_dir)
        except Exception:
            pass
        shutdown_vastai_instance(args.on_finish)

if __name__ == '__main__':
    main()
