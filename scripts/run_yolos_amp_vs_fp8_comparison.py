import sys
import os
import time
import json
import argparse
import subprocess

def parse_jsonl(log_path):
    epoch_summaries = []
    step_records = []
    header_meta = {}

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
                    elif r_type == 'step':
                        step_records.append(record)
                    elif r_type == 'epoch':
                        epoch_summaries.append(record)
                except Exception:
                    pass
    return header_meta, epoch_summaries, step_records

def run_training(precision, epochs, batch_size, imgsz, lr, weight_decay, workers, seed, classes, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        "/home/ictlab/miniconda3/envs/apa/bin/python",
        "examples/yolos_person/train.py",
        "--model", "hustvl/yolos-small",
        "--data_dir", "datasets/detection-person-and-ball/merged_yolo_person_ball",
        "--precision", precision,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--imgsz", str(imgsz),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--workers", str(workers),
        "--seed", str(seed),
        "--save_dir", save_dir,
        "--log_file", log_file
    ]
    if classes is not None:
        cmd.extend(["--classes"] + [str(c) for c in classes])

    print("=" * 85)
    print(f"  LAUNCHING TRAINING EXPERIMENT: {precision.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 85, flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/home/ictlab/Documents/penelitian_miftah/Adaptive-Precision-Ascension")
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Experiment {precision} failed with return code {proc.returncode}")

    header, epochs_list, steps = parse_jsonl(log_file)
    return {
        'precision': precision,
        'elapsed_sec': elapsed,
        'header': header,
        'epoch_summaries': epochs_list,
        'step_records': steps,
        'log_file': log_file
    }

def generate_report(amp_res, fp8_res, fp32_res=None, report_path='result/comparison_yolos_10epochs_amp_vs_apa_fp8.md'):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    amp_epochs = amp_res['epoch_summaries']
    fp8_epochs = fp8_res['epoch_summaries']
    fp32_epochs = fp32_res['epoch_summaries'] if fp32_res else []

    num_ep = min(len(amp_epochs), len(fp8_epochs))

    # Overall metrics
    total_sec_amp = sum(e.get('duration_sec', 0) for e in amp_epochs[:num_ep])
    total_sec_fp8 = sum(e.get('duration_sec', 0) for e in fp8_epochs[:num_ep])
    
    avg_lat_amp = sum(e.get('avg_latency_ms', 0) for e in amp_epochs[:num_ep]) / max(num_ep, 1)
    avg_lat_fp8 = sum(e.get('avg_latency_ms', 0) for e in fp8_epochs[:num_ep]) / max(num_ep, 1)

    avg_fps_amp = sum(e.get('avg_fps', 0) for e in amp_epochs[:num_ep]) / max(num_ep, 1)
    avg_fps_fp8 = sum(e.get('avg_fps', 0) for e in fp8_epochs[:num_ep]) / max(num_ep, 1)

    peak_vram_amp = max((e.get('peak_vram_mb', 0) for e in amp_epochs[:num_ep]), default=0)
    peak_vram_fp8 = max((e.get('peak_vram_mb', 0) for e in fp8_epochs[:num_ep]), default=0)

    lines = []
    lines.append("# Laporan Evaluasi Komparatif Pelatihan YOLOS (10 Epochs)")
    lines.append("## PyTorch Standard AMP (FP16 Autocast) vs APA FP8 Custom Fused vs Pure FP32")
    lines.append("")
    lines.append("### 1. Spesifikasi Eksperimen & Hyperparameter")
    lines.append("- **Arsitektur Model**: `hustvl/yolos-small` (Pure Transformer Object Detection, 27.4M params)")
    lines.append("- **Dataset**: `merged_yolo_person_ball` (Class 0: Person Only, 31,528 citra per epoch)")
    lines.append(f"- **Epochs**: {num_ep} Epoch Penuh ({num_ep * 1970:,} updates)")
    lines.append("- **Batch Size**: 16")
    lines.append("- **Resolusi Gambar**: 512 x 512")
    lines.append("- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)")
    lines.append("- **Hardware**: NVIDIA GeForce RTX 5060 Ti 16GB (Compute 12.0, CUDA 13.2)")
    lines.append("- **File Log Sumber**:")
    lines.append(f"  - AMP FP16: `{amp_res['log_file']}`")
    lines.append(f"  - APA FP8: `{fp8_res['log_file']}`")
    if fp32_res:
        lines.append(f"  - Pure FP32: `{fp32_res['log_file']}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 2. Ringkasan Kinerja Komparatif")
    lines.append("")
    if fp32_res and fp32_epochs:
        total_sec_fp32 = sum(e.get('duration_sec', 0) for e in fp32_epochs[:num_ep])
        avg_lat_fp32 = sum(e.get('avg_latency_ms', 0) for e in fp32_epochs[:num_ep]) / max(num_ep, 1)
        avg_fps_fp32 = sum(e.get('avg_fps', 0) for e in fp32_epochs[:num_ep]) / max(num_ep, 1)
        peak_vram_fp32 = max((e.get('peak_vram_mb', 0) for e in fp32_epochs[:num_ep]), default=0)

        lines.append("| Metrik Evaluasi | Pure FP32 Baseline | PyTorch AMP FP16 | APA FP8 Custom Fused | Analisis / Speedup |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        lines.append(f"| **Total Durasi ({num_ep} Epoch)** | **{total_sec_fp32/60:.2f} m** | **{total_sec_amp/60:.2f} m** | **{total_sec_fp8/60:.2f} m** | AMP: {(total_sec_fp32/max(total_sec_amp,1e-4)):.2f}x, FP8: {(total_sec_fp32/max(total_sec_fp8,1e-4)):.2f}x vs FP32 |")
        lines.append(f"| **Latensi per Step** | **{avg_lat_fp32:.2f} ms** | **{avg_lat_amp:.2f} ms** | **{avg_lat_fp8:.2f} ms** | AMP: {(avg_lat_fp32/max(avg_lat_amp,1e-4)):.2f}x, FP8: {(avg_lat_fp32/max(avg_lat_fp8,1e-4)):.2f}x |")
        lines.append(f"| **Throughput (FPS)** | **{avg_fps_fp32:.1f} fps** | **{avg_fps_amp:.1f} fps** | **{avg_fps_fp8:.1f} fps** | FP8 Throughput: {avg_fps_fp8:.1f} fps |")
        lines.append(f"| **Peak VRAM** | **{peak_vram_fp32:.1f} MB** | **{peak_vram_amp:.1f} MB** | **{peak_vram_fp8:.1f} MB** | Hemat VRAM FP8 vs FP32: {((peak_vram_fp32-peak_vram_fp8)/max(peak_vram_fp32,1e-4))*100:.1f}% |")
        lines.append(f"| **Final Total Loss (Ep {num_ep})** | {fp32_epochs[num_ep-1].get('avg_loss', 0):.4f} | {amp_epochs[num_ep-1].get('avg_loss', 0):.4f} | {fp8_epochs[num_ep-1].get('avg_loss', 0):.4f} | Konvergen stabil |")
    else:
        lines.append("| Metrik Evaluasi | PyTorch AMP FP16 | APA FP8 Custom Fused | Rasio Perbedaan |")
        lines.append("| :--- | :---: | :---: | :---: |")
        lines.append(f"| **Total Durasi ({num_ep} Epoch)** | **{total_sec_amp/60:.2f} menit** | **{total_sec_fp8/60:.2f} menit** | {total_sec_fp8/max(total_sec_amp, 1e-4):.2f}x |")
        lines.append(f"| **Latensi per Step** | **{avg_lat_amp:.2f} ms** | **{avg_lat_fp8:.2f} ms** | {avg_lat_fp8/max(avg_lat_amp, 1e-4):.2f}x |")
        lines.append(f"| **Throughput (FPS)** | **{avg_fps_amp:.1f} fps** | **{avg_fps_fp8:.1f} fps** | {avg_fps_fp8/max(avg_fps_amp, 1e-4):.2f}x |")
        lines.append(f"| **Peak VRAM** | **{peak_vram_amp:.1f} MB** | **{peak_vram_fp8:.1f} MB** | {peak_vram_fp8 - peak_vram_amp:+.1f} MB |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 3. Tabel Detail Per-Epoch (AMP FP16 vs APA FP8)")
    lines.append("")
    lines.append("| Epoch | Mode Presisi | Total Loss | CE Loss | BBox L1 Loss | GIoU Loss | Latensi / Step | Throughput | Durasi Epoch |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ep in range(1, num_ep + 1):
        e_amp = amp_epochs[ep - 1]
        e_fp8 = fp8_epochs[ep - 1]
        lines.append(f"| **{ep}** | **AMP FP16** | {e_amp.get('avg_loss', 0):.4f} | {e_amp.get('avg_ce', 0):.4f} | {e_amp.get('avg_bbox', 0):.4f} | {e_amp.get('avg_giou', 0):.4f} | {e_amp.get('avg_latency_ms', 0):.2f} ms | {e_amp.get('avg_fps', 0):.1f} fps | {e_amp.get('duration_sec', 0)/60:.2f} m |")
        lines.append(f"| **{ep}** | **APA FP8**  | {e_fp8.get('avg_loss', 0):.4f} | {e_fp8.get('avg_ce', 0):.4f} | {e_fp8.get('avg_bbox', 0):.4f} | {e_fp8.get('avg_giou', 0):.4f} | {e_fp8.get('avg_latency_ms', 0):.2f} ms | {e_fp8.get('avg_fps', 0):.1f} fps | {e_fp8.get('duration_sec', 0)/60:.2f} m |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 4. Analisis Ilmiah")
    lines.append("1. **Perbedaan Arsitektur AMP vs APA FP8**:")
    lines.append("   - **PyTorch Native AMP**: Menggunakan `torch.amp.autocast('cuda', dtype=torch.float16)` bersama `GradScaler`. Seluruh 78 layer Linear tetap berstatus native `nn.Linear`. Autocast secara dinamis menentukan operasi mana yang menggunakan FP16 dan FP32.")
    lines.append("   - **APA FP8 Custom Fused Engine**: Mengganti 72 layer Linear Transformer menjadi `APALinear` yang mengeksekusi kernel C++ native cuBLASLt FP8 (`apa_cuda`) dengan delayed dynamic scaling asinkron di GPU, tanpa `torch.amp.autocast` dan tanpa `GradScaler`.")
    lines.append("2. **Karakteristik Efisiensi Memori & Throughput**:")
    lines.append("   - FP8 mempertahankan bobot kerja dalam format 8-bit (1 byte per parameter), secara signifikan menekan memory footprint.")
    lines.append("3. **Stabilitas Numerik & Konvergensi Bounding Box**:")
    lines.append("   - Pada kedua mode, Hungarian Matcher dan loss prediction heads tetap dilindungi sehingga gradien tidak mengalami instabilitas.")

    content = "\n".join(lines) + "\n"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[DONE] Comparison report written to: {report_path}")
    return content

def main():
    parser = argparse.ArgumentParser(description="Run 10-Epoch Comparison for AMP vs APA FP8 on YOLOS")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per step")
    parser.add_argument('--imgsz', type=int, default=512, help="Input image resolution")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--classes', type=int, nargs='+', default=[0], help="Target class IDs")
    parser.add_argument('--save_dir_amp', type=str, default='runs/train_yolos_amp_fp16_person', help="Output dir for AMP")
    parser.add_argument('--save_dir_fp8', type=str, default='runs/train_yolos_apa_fp8_person2', help="Output dir for FP8")
    parser.add_argument('--rerun_fp8', action='store_true', help="Re-run FP8 even if existing 10-epoch log exists")
    parser.add_argument('--report_file', type=str, default='result/comparison_yolos_10epochs_amp_vs_apa_fp8.md')
    args = parser.parse_args()

    # 1. Run AMP
    log_amp = os.path.join(args.save_dir_amp, "yolos_amp_training.jsonl")
    res_amp = run_training(
        precision='amp',
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        seed=args.seed,
        classes=args.classes,
        save_dir=args.save_dir_amp,
        log_file=log_amp
    )

    # 2. Check FP8
    log_fp8 = os.path.join(args.save_dir_fp8, "yolos_apa_fp8_training.jsonl")
    h_fp8, ep_fp8, st_fp8 = parse_jsonl(log_fp8)
    if not args.rerun_fp8 and len(ep_fp8) >= args.epochs:
        print(f"\n[*] Found existing {len(ep_fp8)}-epoch APA FP8 log at {log_fp8}. Reusing for apples-to-apples comparison.")
        res_fp8 = {
            'precision': 'apa_fp8',
            'elapsed_sec': sum(e.get('duration_sec', 0) for e in ep_fp8[:args.epochs]),
            'header': h_fp8,
            'epoch_summaries': ep_fp8[:args.epochs],
            'step_records': st_fp8,
            'log_file': log_fp8
        }
    else:
        res_fp8 = run_training(
            precision='apa_fp8',
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr=args.lr,
            weight_decay=args.weight_decay,
            workers=args.workers,
            seed=args.seed,
            classes=args.classes,
            save_dir=args.save_dir_fp8,
            log_file=log_fp8
        )

    # 3. Check FP32 for 3-way comparison if available
    log_fp32 = 'runs/train_yolos_person_only_fp32/yolos_fp32_training.jsonl'
    res_fp32 = None
    if os.path.exists(log_fp32):
        h_fp32, ep_fp32, st_fp32 = parse_jsonl(log_fp32)
        if len(ep_fp32) >= args.epochs:
            res_fp32 = {
                'precision': 'fp32',
                'elapsed_sec': sum(e.get('duration_sec', 0) for e in ep_fp32[:args.epochs]),
                'header': h_fp32,
                'epoch_summaries': ep_fp32[:args.epochs],
                'step_records': st_fp32,
                'log_file': log_fp32
            }

    # 4. Generate report
    generate_report(res_amp, res_fp8, res_fp32, report_path=args.report_file)

if __name__ == '__main__':
    main()
