import sys
import os
import argparse
import time
import json
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Dict, Any

warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
warnings.filterwarnings("ignore", message=".*You passed `num_labels=2`.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from examples.yolos_person.model import create_yolos_model, convert_yolos_to_apa
from examples.yolos_person.dataset import build_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOS on Person-Ball Dataset with Pure FP32 vs APA FP8 Custom Fused.")
    parser.add_argument('--model', type=str, default='hustvl/yolos-small', help="YOLOS model name or checkpoint")
    parser.add_argument('--data_dir', type=str, default='datasets/detection-person-and-ball/merged_yolo_person_ball',
                        help="Path to YOLO dataset directory containing images/ and labels/")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per training step")
    parser.add_argument('--imgsz', type=int, default=512, help="Input image resolution (e.g. 512)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--precision', type=str, default='apa_fp8',
                        choices=['fp32', 'apa_fp8', 'apa', 'fp16', 'tf32'],
                        help="Precision engine: 'fp32' (Pure FP32 baseline, no TF32), 'apa_fp8' (APA FP8 Custom Fused Engine), 'apa' (Adaptive FP8->FP16->TF32)")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['research', 'conservative', 'aggressive'])
    parser.add_argument('--check_interval', type=int, default=1, help="Evaluation interval for soft checks in APA")
    parser.add_argument('--workers', type=int, default=4, help="Number of DataLoader worker processes")
    parser.add_argument('--max_steps', type=int, default=0, help="Max steps per epoch (0 = full epoch)")
    parser.add_argument('--log_file', type=str, default=None, help="Path to save JSONL training log")
    parser.add_argument('--save_dir', type=str, default='runs/train_yolos_person', help="Directory to save checkpoints")
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help="Filter dataset and model to specific class IDs (e.g. --classes 0 for person only)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--no_save', action='store_true', help="Disable saving checkpoint weights")
    return parser.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)

    print("=" * 80)
    print("  YOLOS (PURE VISION TRANSFORMER OBJECT DETECTION) TRAINING PIPELINE")
    print(f"  Model: {args.model} | Resolution: {args.imgsz}x{args.imgsz}")
    print(f"  Precision Mode: {args.precision.upper()} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA device is required for native FP8 / FP32 comparison.")

    # 1. Precision & Hardware Setup
    use_apa = False
    freeze_level = None

    if args.precision == 'fp32':
        # Pure IEEE-754 FP32 baseline (disable TF32 Tensor Cores)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')
        use_apa = False
        mode_desc = "Pure IEEE-754 FP32 Baseline (TF32 Disabled)"
    elif args.precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        use_apa = False
        mode_desc = "TF32 Tensor Cores Baseline"
    elif args.precision == 'apa_fp8':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = True
        freeze_level = LEVEL_FP8
        mode_desc = "APA FP8 Custom Fused Kernel Engine (Level 0 FP8, Native C++ Fused GEMM)"
    elif args.precision == 'apa':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = True
        freeze_level = None # Adaptive escalation
        mode_desc = "APA Full Adaptive Precision (FP8 -> FP16 -> TF32 Dynamic Escalation)"
    elif args.precision == 'fp16':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = True
        freeze_level = LEVEL_FP16
        mode_desc = "APA Fixed FP16 (Custom APALinear, NO AMP)"
    else:
        raise ValueError(f"Unknown precision: {args.precision}")

    print(f"[*] Hardware Precision Config: {mode_desc}")

    # 2. Build Model
    num_labels = len(args.classes) if args.classes is not None else 2
    model = create_yolos_model(
        model_name=args.model,
        num_labels=num_labels,
        img_size=args.imgsz,
        pretrained=True
    ).to(device)

    # 3. Apply APA Conversion if required
    apa_manager = None
    if use_apa:
        if args.apa_preset == 'research':
            apa_cfg = APAConfig.research_default()
        elif args.apa_preset == 'conservative':
            apa_cfg = APAConfig.conservative_default()
        else:
            apa_cfg = APAConfig.aggressive_default()

        apa_cfg.device = 'cuda'
        apa_cfg.check_interval = args.check_interval
        apa_cfg.freeze_level = freeze_level
        apa_cfg.fp8_output_dtype = 'float16'
        if args.log_file:
            apa_cfg.log_file = args.log_file.replace('.jsonl', '_apa_events.jsonl')

        model = convert_yolos_to_apa(model, config=apa_cfg, preserve_critical_heads=True)
        apa_manager = APAManager(model, config=apa_cfg)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 4. Build DataLoaders
    train_img_dir = os.path.join(args.data_dir, 'images', 'train')
    train_lbl_dir = os.path.join(args.data_dir, 'labels', 'train')
    val_img_dir = os.path.join(args.data_dir, 'images', 'val')
    val_lbl_dir = os.path.join(args.data_dir, 'labels', 'val')

    train_loader = build_dataloader(
        img_dir=train_img_dir,
        label_dir=train_lbl_dir,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        shuffle=True,
        num_workers=args.workers,
        class_filter=args.classes,
        max_samples=None
    )

    total_train_samples = len(train_loader.dataset)
    steps_per_epoch = len(train_loader)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    print(f"[*] Dataset: {total_train_samples:,} images in train set ({len(train_loader)} batches/epoch @ batch_size={args.batch_size})")
    print(f"[*] Training will run for {args.epochs} epoch(s), {steps_per_epoch} steps per epoch.")

    # 5. Setup Logging
    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_file is None:
        args.log_file = os.path.join(args.save_dir, f"yolos_{args.precision}_training.jsonl")

    log_fd = open(args.log_file, 'w', encoding='utf-8')
    header_meta = {
        'mode': args.precision,
        'mode_desc': mode_desc,
        'model': args.model,
        'batch_size': args.batch_size,
        'img_size': args.imgsz,
        'lr': args.lr,
        'epochs': args.epochs,
        'steps_per_epoch': steps_per_epoch,
        'gpu': torch.cuda.get_device_name(0),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    log_fd.write(json.dumps({'type': 'header', 'meta': header_meta}) + '\n')
    log_fd.flush()

    # 6. Training Loop
    total_training_start = time.time()
    epoch_summaries = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_bbox = 0.0
        running_giou = 0.0
        step_times = []

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch [{epoch}/{args.epochs}] [{args.precision.upper()}]")

        for step_idx, (pixel_values, labels) in pbar:
            if args.max_steps > 0 and step_idx >= args.max_steps:
                break

            step_t0 = time.time()

            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in labels]

            if apa_manager is not None:
                apa_manager.pre_step()

            optimizer.zero_grad(set_to_none=True)

            # Pure forward pass - No torch.amp.autocast!
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss_dict = outputs.loss_dict

            # Backward pass
            loss.backward()

            # Post-backward evaluation & sync
            if apa_manager is not None:
                step_ok = apa_manager.post_backward_sync_and_eval()
                if step_ok:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()

            torch.cuda.synchronize()
            step_latency_ms = (time.time() - step_t0) * 1000.0
            step_times.append(step_latency_ms)

            # Metrics
            loss_val = loss.item()
            ce_val = loss_dict.get('loss_ce', torch.tensor(0.0)).item()
            bbox_val = loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
            giou_val = loss_dict.get('loss_giou', torch.tensor(0.0)).item()

            running_loss += loss_val
            running_ce += ce_val
            running_bbox += bbox_val
            running_giou += giou_val

            curr_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            curr_fps = (args.batch_size * 1000.0) / max(step_latency_ms, 1e-4)

            pbar.set_postfix({
                'loss': f"{loss_val:.3f}",
                'ce': f"{ce_val:.3f}",
                'bbox': f"{bbox_val:.3f}",
                'ms': f"{step_latency_ms:.1f}",
                'fps': f"{curr_fps:.1f}",
                'vram': f"{curr_vram_mb:.0f}MB"
            })

            # Log step telemetry
            step_record = {
                'type': 'step',
                'epoch': epoch,
                'step': step_idx + 1,
                'loss': loss_val,
                'loss_ce': ce_val,
                'loss_bbox': bbox_val,
                'loss_giou': giou_val,
                'latency_ms': step_latency_ms,
                'fps': curr_fps,
                'vram_mb': curr_vram_mb
            }
            log_fd.write(json.dumps(step_record) + '\n')
            if (step_idx + 1) % 50 == 0:
                log_fd.flush()

        epoch_duration_sec = time.time() - epoch_start_time
        num_steps_completed = len(step_times)
        avg_loss = running_loss / max(num_steps_completed, 1)
        avg_ce = running_ce / max(num_steps_completed, 1)
        avg_bbox = running_bbox / max(num_steps_completed, 1)
        avg_giou = running_giou / max(num_steps_completed, 1)
        # Skip warm-up steps for clean latency calculation
        clean_latencies = step_times[5:] if len(step_times) > 10 else step_times
        avg_latency_ms = sum(clean_latencies) / max(len(clean_latencies), 1)
        avg_fps = (args.batch_size * 1000.0) / max(avg_latency_ms, 1e-4)
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        epoch_summary = {
            'type': 'epoch',
            'epoch': epoch,
            'duration_sec': epoch_duration_sec,
            'avg_loss': avg_loss,
            'avg_ce': avg_ce,
            'avg_bbox': avg_bbox,
            'avg_giou': avg_giou,
            'avg_latency_ms': avg_latency_ms,
            'avg_fps': avg_fps,
            'peak_vram_mb': peak_vram_mb,
            'steps_completed': num_steps_completed
        }
        epoch_summaries.append(epoch_summary)
        log_fd.write(json.dumps(epoch_summary) + '\n')
        log_fd.flush()

        print("\n" + "-" * 75)
        print(f"  Epoch {epoch} Complete [{args.precision.upper()}]:")
        print(f"  Duration: {epoch_duration_sec:.1f}s ({epoch_duration_sec/60.0:.2f}m) | Steps: {num_steps_completed}")
        print(f"  Avg Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, BBox: {avg_bbox:.4f}, GIoU: {avg_giou:.4f})")
        print(f"  Avg Latency: {avg_latency_ms:.2f} ms/step | Throughput: {avg_fps:.1f} fps")
        print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")
        print("-" * 75 + "\n")

    total_training_sec = time.time() - total_training_start
    log_fd.write(json.dumps({
        'type': 'final',
        'total_duration_sec': total_training_sec,
        'epochs': args.epochs,
        'summary': epoch_summaries
    }) + '\n')
    log_fd.close()

    # Save final model
    if not args.no_save:
        save_path = os.path.join(args.save_dir, f"yolos_{args.precision}_final.pt")
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'precision': args.precision,
            'epoch_summary': epoch_summaries[-1] if epoch_summaries else {}
        }, save_path)
        print(f"[*] Checkpoint saved to: {save_path}")

    print(f"\n[DONE] Training finished successfully in {total_training_sec:.1f}s ({total_training_sec/60.0:.2f}m).")
    print(f"[*] Log saved to: {args.log_file}")

if __name__ == '__main__':
    main()
