import sys
import os
import argparse
import time
import json
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import convert_rtdetr_to_apa

try:
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
except ImportError as e:
    raise ImportError("ultralytics package is required. Install via: pip install ultralytics") from e

def get_args():
    parser = argparse.ArgumentParser(description="Train DINO-DETR / RT-DETR with Adaptive Precision Architecture (APA).")
    parser.add_argument('--model', type=str, default='rtdetr-l.yaml', help="RT-DETR config or checkpoint (e.g. rtdetr-l.yaml, rtdetr-l.pt)")
    parser.add_argument('--data', type=str, default='coco8.yaml', help="Dataset config (e.g. data.yaml)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Target device (cuda or cpu)")
    parser.add_argument('--workers', type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument('--precision', type=str, default='apa', choices=['apa', 'fp8', 'fp16', 'tf32', 'fp32_baseline'],
                        help="Precision mode: 'apa' (Adaptive), 'fp8' (Fixed FP8), 'fp16' (Fixed FP16), 'tf32' (Fixed TF32), 'fp32_baseline' (Pure PyTorch FP32)")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['research', 'conservative', 'aggressive'], help="APA config preset")
    parser.add_argument('--check_interval', type=int, default=1, help="Evaluation interval for soft checks")
    parser.add_argument('--all_apa_layers', action='store_true', help="Convert all linear layers including final scoring and bbox prediction heads to FP8")
    parser.add_argument('--log_file', type=str, default=None, help="Path to JSONL log file")
    parser.add_argument('--save_dir', type=str, default='runs/train_rtdetr_apa', help="Directory to save checkpoints")
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help="Filter dataset and model to specific class IDs (e.g. --classes 0 for person only)")
    parser.add_argument('--max_steps', type=int, default=0,
                        help="Stop training after reaching max_steps (0 = train for full epochs)")
    parser.add_argument('--no_save', action='store_true', help="Do not save model checkpoint .pt files")
    parser.add_argument('--smoke_test', action='store_true', help="Run 3 steps smoke test and verify 0 errors")
    return parser.parse_args()

def main():
    args = get_args()
    print("=" * 70)
    print("  DINO-DETR / RT-DETR Training Pipeline with Adaptive Precision (APA)")
    print("=" * 70)
    print(f"Model: {args.model} | Dataset: {args.data} | Device: {args.device}")
    print(f"Precision Mode: {args.precision} | Batch size: {args.batch_size} | Img size: {args.imgsz}")

    # 1. Setup APA Configuration
    if args.apa_preset == 'research':
        config = APAConfig.research_default()
    elif args.apa_preset == 'conservative':
        config = APAConfig.conservative_default()
    else:
        config = APAConfig.aggressive_default()

    config.device = args.device
    config.check_interval = args.check_interval
    config.fp8_output_dtype = 'float32'
    if args.log_file:
        config.log_file = args.log_file

    if args.precision == 'fp8':
        config.freeze_level = LEVEL_FP8
    elif args.precision == 'fp16':
        config.freeze_level = LEVEL_FP16
    elif args.precision == 'tf32':
        config.freeze_level = LEVEL_TF32

    # 2. Setup Ultralytics RT-DETR Model
    cfg = get_cfg()
    cfg.data = args.data
    cfg.model = args.model
    cfg.batch = args.batch_size
    cfg.imgsz = args.imgsz
    cfg.device = args.device

    trainer = RTDETRTrainer(overrides=cfg)
    if args.classes is not None:
        trainer.data['nc'] = len(args.classes)
        trainer.data['names'] = {i: trainer.data['names'][c] for i, c in enumerate(args.classes)}
    trainer.setup_model()
    model = trainer.model.to(args.device)
    model.args = cfg
    model.nc = trainer.data['nc']

    manager = None
    if args.precision != 'fp32_baseline':
        preserve_critical = not args.all_apa_layers
        convert_rtdetr_to_apa(model, config=config, preserve_critical_layers=preserve_critical)
        manager = APAManager(model, config=config)
        print(f"APA Enabled: Successfully managing {len(manager.apa_modules)} Transformer Linear layers!")
        params = manager.get_trainable_parameters()
    else:
        print("Running in FP32 Baseline mode (No APA).")
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 3. Setup Dataset and DataLoader
    data_dict = check_det_dataset(cfg.data)
    dataset = build_yolo_dataset(cfg, data_dict['train'], batch=args.batch_size, data=data_dict, mode='train', rect=False, stride=32)

    if args.classes is not None:
        import numpy as np
        target_classes = set(args.classes)
        class_map = {old_c: new_c for new_c, old_c in enumerate(args.classes)}
        removed_count = 0
        for item in dataset.labels:
            keep = np.array([int(c[0]) in target_classes for c in item['cls']], dtype=bool) if len(item['cls']) else np.zeros(0, dtype=bool)
            removed_count += (~keep).sum()
            if len(keep) > 0 and keep.any():
                item['cls'] = np.array([[class_map[int(c[0])]] for c in item['cls'][keep]], dtype=np.float32)
            else:
                item['cls'] = np.zeros((0, 1), dtype=np.float32)
            item['bboxes'] = item['bboxes'][keep]
            if 'segments' in item and len(item['segments']) == len(keep):
                item['segments'] = [s for s, k in zip(item['segments'], keep) if k]
            if 'keypoints' in item and item['keypoints'] is not None:
                item['keypoints'] = item['keypoints'][keep]
        print(f"Filtered dataset to class(es) {args.classes} ({trainer.data['names']}): removed {removed_count} annotations outside selected classes.")

    dataloader = build_dataloader(dataset, batch=args.batch_size, workers=args.workers, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} images, {len(dataloader)} batches per epoch.")

    # 4. Training Loop
    if args.log_file:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_file, 'w', encoding='utf-8') as f:
            header_data = {
                "model": args.model,
                "dataset": args.data,
                "mode": args.precision,
                "classes": args.classes,
                "args": vars(args),
                "config": config.__dict__ if config is not None else {"precision": args.precision}
            }
            f.write(json.dumps(header_data) + "\n")
        print(f"Metrics will be logged to JSONL: {args.log_file}")

    model.train()
    total_steps = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_giou_loss = 0.0
        epoch_l1_loss = 0.0
        batches_accepted = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            step_start = time.perf_counter()
            if manager is not None:
                manager.pre_step()
            optimizer.zero_grad()

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device, non_blocking=True)
            batch['img'] = batch['img'].float() / 255.0

            loss, loss_items = model(batch)
            total_loss = loss.sum() if isinstance(loss, torch.Tensor) else torch.tensor(float(loss), device=args.device)
            total_loss.backward()

            accepted = True
            if manager is not None:
                accepted = manager.post_backward_sync_and_eval()

            if accepted:
                optimizer.step()
                batches_accepted += 1
                epoch_loss += total_loss.item()
                if isinstance(loss_items, dict):
                    epoch_cls_loss += loss_items.get('cls_loss', torch.tensor(0.0)).item()
                    epoch_giou_loss += loss_items.get('giou_loss', torch.tensor(0.0)).item()
                    epoch_l1_loss += loss_items.get('l1_loss', torch.tensor(0.0)).item()
            else:
                pbar.write(f"  [Step {step}] Batch rejected due to overflow. Precision escalated.")

            total_steps += 1
            step_time_ms = (time.perf_counter() - step_start) * 1000.0

            if args.log_file and (step % 50 == 0 or step == len(dataloader) - 1 or (args.max_steps > 0 and step % 5 == 0)):
                with open(args.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "event": "step_metric",
                        "epoch": epoch + 1,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": round(total_loss.item(), 4),
                        "cls_loss": round(loss_items.get('cls_loss', torch.tensor(0.0)).item(), 4) if isinstance(loss_items, dict) else 0.0,
                        "giou_loss": round(loss_items.get('giou_loss', torch.tensor(0.0)).item(), 4) if isinstance(loss_items, dict) else 0.0,
                        "step_time_ms": round(step_time_ms, 2),
                    }) + "\n")

            postfix = {
                'loss': f"{total_loss.item():.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            if isinstance(loss_items, dict):
                if 'cls_loss' in loss_items:
                    postfix['cls'] = f"{loss_items['cls_loss'].item():.2f}"
                if 'giou_loss' in loss_items:
                    postfix['giou'] = f"{loss_items['giou_loss'].item():.2f}"
            if manager is not None:
                level_dist = {}
                for m in manager.apa_modules.values():
                    level_dist[m.level] = level_dist.get(m.level, 0) + 1
                postfix['levels'] = f"F8:{level_dist.get(0, 0)}/F16:{level_dist.get(1, 0)}/TF:{level_dist.get(2, 0)}"
            pbar.set_postfix(postfix)

            if args.smoke_test and total_steps >= 3:
                if args.log_file:
                    with open(args.log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "event": "smoke_test_complete",
                            "steps": total_steps,
                            "last_loss": round(total_loss.item(), 4),
                            "status": "passed"
                        }) + "\n")
                print("\n[SMOKE TEST PASSED] Verified 3 steps with 0 errors!")
                return

            if args.max_steps > 0 and total_steps >= args.max_steps:
                pbar.close()
                break

        avg_loss = epoch_loss / max(1, batches_accepted)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} Complete: Avg Loss={avg_loss:.4f} | LR={current_lr:.6f} | Accepted Batches: {batches_accepted}/{len(dataloader) if args.max_steps <= 0 else min(len(dataloader), args.max_steps)} | Time: {epoch_time:.2f}s")

        if args.log_file:
            record = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 4),
                "cls_loss": round(epoch_cls_loss / max(1, batches_accepted), 4),
                "giou_loss": round(epoch_giou_loss / max(1, batches_accepted), 4),
                "l1_loss": round(epoch_l1_loss / max(1, batches_accepted), 4),
                "lr": round(current_lr, 6),
                "epoch_time_sec": round(epoch_time, 2),
                "precision": args.precision,
                "mode": args.precision,
                "total_steps": total_steps,
                "accepted_batches": batches_accepted
            }
            if torch.cuda.is_available():
                record["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                record["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
            if manager is not None:
                level_dist = {}
                for m in manager.apa_modules.values():
                    level_dist[m.level] = level_dist.get(m.level, 0) + 1
                record["precision_distribution"] = {
                    "fp8": level_dist.get(0, 0),
                    "fp16": level_dist.get(1, 0),
                    "tf32": level_dist.get(2, 0),
                }
            with open(args.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")

        if args.max_steps > 0 and total_steps >= args.max_steps:
            print(f"\n[BENCHMARK COMPLETE] Reached {total_steps} max steps.")
            break

        if not args.smoke_test and not args.no_save and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            model_base = os.path.splitext(os.path.basename(args.model))[0]
            ckpt_name = f'{model_base}_{args.precision}_last.pt'
            ckpt_path = os.path.join(args.save_dir, ckpt_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'apa_levels': {name: m.level for name, m in manager.apa_modules.items()} if manager else {}
            }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({total_steps} steps).")
    if args.log_file:
        print(f"Final JSONL metrics saved to: {args.log_file}")

if __name__ == '__main__':
    main()
