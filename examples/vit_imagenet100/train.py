#!/usr/bin/env python3
"""
Training Vision Transformer on ImageNet-100 via Hugging Face Hub (clane9/imagenet-100)

Supports:
- Multi-precision: APA (Adaptive), FP8 (Native Triton), FP16 (AMP), TF32, and FP32 (Strict IEEE 754)
- Model Presets: ViT-Tiny (~5.7M), ViT-Small (~22M), and ViT-Base (~86M)
- Hugging Face authentication token (--hf_token or HF_TOKEN env var)
- Top-1 and Top-5 accuracy evaluation
- JSONL structured event logging for comparative analysis

Usage:
    # Train ViT-Small with APA on ImageNet-100:
    python examples/vit_imagenet100/train.py --precision apa --model_size small --batch_size 128

    # Train with strict FP32 baseline:
    python examples/vit_imagenet100/train.py --precision fp32 --model_size small --batch_size 128
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import build_vit_imagenet100, MODEL_PRESETS
from dataset import get_imagenet100_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on ImageNet-100 with APA or Baselines.")
    parser.add_argument('--precision', type=lambda s: s.lower(), default='apa',
                        choices=['apa', 'fp8', 'fp16', 'tf32', 'fp32'],
                        help="Precision mode: 'apa' (Adaptive), 'fp8' (Fixed FP8), 'fp16' (AMP), 'tf32' (Tensor Cores FP32), or 'fp32' (Strict IEEE 754)")
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'],
                        help="ViT model scale: 'tiny' (~5.7M), 'small' (~22M, default), or 'base' (~86M)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size per step (default: 128)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate for cosine decay")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Linear warmup epochs (default: 5)")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for AdamW (default: 0.05)")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability (default: 0.1)")
    
    # Hugging Face & Data Options
    parser.add_argument('--hf_token', type=str, default=None,
                        help="Hugging Face API token (can also be provided via HF_TOKEN environment variable)")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Optional custom directory to cache the downloaded dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader worker processes (default: 4)")
    
    # APA Configuration
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'],
                        help="APA config preset (default: research)")
    parser.add_argument('--fp8_sim', action='store_true', help="Use FP8 simulation mode")
    parser.add_argument('--strict_fp32', action='store_true',
                        help="Force strict IEEE 754 FP32 math (disables TF32 Tensor Cores)")
    parser.add_argument('--all_apa_layers', action='store_true',
                        help="Apply APA to boundary patch_embed and head layers as well")
    parser.add_argument('--forensic', action='store_true', help="Enable detailed forensic logging on escalation")
    parser.add_argument('--forensic_argmax', action='store_true', help="Capture flat argmax index in forensic snapshots")
    parser.add_argument('--interval_telemetry', action='store_true',
                        help="Sample telemetry only on check_interval steps instead of every step")
    parser.add_argument('--telemetry_interval', type=int, default=0,
                        help="Periodic step interval to log per-layer telemetry time series")
    parser.add_argument('--log_file', type=str, default='apa_vit_imagenet100_log.jsonl',
                        help="Output JSONL log file (default: apa_vit_imagenet100_log.jsonl)")
    
    return parser.parse_args()

def compute_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Precision Mode Setup
    freeze_level = None
    use_amp = False
    if args.precision == 'fp8':
        use_apa = True
        freeze_level = LEVEL_FP8
        mode_str = "Pure FP8 (Fixed Level 0 with Native Triton Kernel)"
        math_mode = "FP8 E4M3/E5M2 Tensor Cores"
    elif args.precision == 'fp16':
        use_apa = False
        use_amp = True
        mode_str = "Pure FP16 (Automatic Mixed Precision AMP)"
        math_mode = "FP16 Half Precision Tensor Cores"
    elif args.precision == 'tf32':
        use_apa = False
        args.strict_fp32 = False
        mode_str = "Pure TF32 (Standard FP32 with TF32 Tensor Cores)"
        math_mode = "Standard FP32 (TF32 Tensor Cores Enabled)"
    elif args.precision == 'fp32':
        use_apa = False
        args.strict_fp32 = True
        mode_str = "Pure FP32 Baseline (Strict IEEE 754, TF32 Disabled)"
        math_mode = "Strict IEEE 754 Single Precision (TF32 Disabled)"
    else:  # apa
        use_apa = True
        mode_str = "APA (Adaptive Precision Architecture)"
        math_mode = "Adaptive Escalation (FP8 -> FP16 -> TF32)"
        
    # Configure Hardware Matrix Multiply Precision
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    preset_info = MODEL_PRESETS[args.model_size]
    print("=" * 75)
    print("🌟 Vision Transformer ImageNet-100 Training")
    print(f"Model Scale : {preset_info['desc']}")
    print(f"Mode        : {mode_str}")
    print(f"Math Dtype  : {math_mode}")
    print(f"Device      : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print(f"Epochs      : {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Log File    : {args.log_file}")
    print("=" * 75)

    # 1. Dataset & DataLoaders
    train_loader, val_loader = get_imagenet100_loaders(
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        image_size=224,
        pin_memory=True
    )

    # 2. APA Config & Model Instantiation
    if use_apa:
        preset_map = {
            'conservative': APAConfig.conservative,
            'aggressive': APAConfig.aggressive,
            'research': APAConfig.research_default,
        }
        telemetry_int = args.telemetry_interval if args.telemetry_interval > 0 else (50 if args.forensic else 0)
        config = preset_map[args.apa_preset](
            device=str(device),
            log_file=args.log_file,
            enable_forensic_logging=args.forensic,
            forensic_capture_argmax_index=args.forensic_argmax,
            telemetry_log_interval=telemetry_int,
            interval_telemetry=args.interval_telemetry,
            freeze_level=freeze_level,
            fp8_output_dtype='float16'
        )
        if args.fp8_sim:
            config.fp8_simulation_mode = True

        preserve_crit = not args.all_apa_layers
        model = build_vit_imagenet100(
            model_size=args.model_size,
            config=config,
            use_apa=True,
            preserve_critical_layers=preserve_crit,
            dropout=args.dropout
        ).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = build_vit_imagenet100(
            model_size=args.model_size,
            config=None,
            use_apa=False,
            dropout=args.dropout
        ).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} ({total_params / 1e6:.2f}M params)\n")

    # 3. Optimizer & Scaler & LR Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Cosine scheduler with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Training Loop
    best_top1_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        total_train_samples = 0
        train_top1_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", dynamic_ncols=True)
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if apa_manager is not None:
                apa_manager.pre_step()
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_accepted = True
            else:
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)
                loss.backward()
                if apa_manager is not None:
                    step_accepted = apa_manager.post_backward_sync_and_eval()
                    if step_accepted:
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    step_accepted = True

            loss_val = loss.item()
            total_train_loss += loss_val
            train_batches += 1
            total_train_samples += targets.size(0)

            top1 = compute_accuracy(outputs, targets, topk=(1,))[0]
            train_top1_sum += top1

            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'top1': f"{top1:.1f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                'status': 'OK' if step_accepted else 'SKIP'
            })

        scheduler.step()
        train_loss_avg = total_train_loss / train_batches if train_batches > 0 else 0.0
        train_acc_avg = train_top1_sum / train_batches if train_batches > 0 else 0.0

        # 5. Validation Loop
        model.eval()
        total_val_loss = 0.0
        val_top1_sum = 0.0
        val_top5_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False, dynamic_ncols=True):
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(images)
                        loss = F.cross_entropy(outputs, targets)
                else:
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, targets)

                total_val_loss += loss.item()
                top1, top5 = compute_accuracy(outputs, targets, topk=(1, 5))
                val_top1_sum += top1
                val_top5_sum += top5
                val_batches += 1

        val_loss_avg = total_val_loss / val_batches if val_batches > 0 else 0.0
        val_top1_avg = val_top1_sum / val_batches if val_batches > 0 else 0.0
        val_top5_avg = val_top5_sum / val_batches if val_batches > 0 else 0.0
        epoch_time = time.perf_counter() - epoch_start

        if val_top1_avg > best_top1_acc:
            best_top1_acc = val_top1_avg
            star = " 🏆 (New Best!)"
        else:
            star = ""

        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss_avg:.4f} (Top-1: {train_acc_avg:.1f}%) | "
            f"Val Loss: {val_loss_avg:.4f} (Top-1: {val_top1_avg:.2f}%, Top-5: {val_top5_avg:.2f}%){star} | "
            f"Peak VRAM: {vram_mb:.1f} MB"
        )

        # 6. Log epoch metrics to JSONL
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss_avg, 4),
            "train_acc": round(train_acc_avg / 100.0, 4),
            "test_loss": round(val_loss_avg, 4),
            "test_acc": round(val_top1_avg / 100.0, 4),
            "test_acc_top5": round(val_top5_avg / 100.0, 4),
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time_sec": round(epoch_time, 2),
            "peak_vram_mb": round(vram_mb, 1),
            "precision": args.precision,
            "model_size": args.model_size
        }
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(epoch_record) + '\n')

    print("=" * 75)
    print(f"🎉 Training Completed. Best Val Top-1 Accuracy: {best_top1_acc:.2f}%")
    print(f"Metrics saved to: {args.log_file}")
    print("=" * 75)

if __name__ == '__main__':
    main()
