import sys
import os
import argparse
import time
import json
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import convert_yolov8_to_apa

try:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
except ImportError as e:
    raise ImportError("ultralytics package is required. Install via: pip install ultralytics") from e

def get_args():
    parser = argparse.ArgumentParser(description="Train or Pre-train YOLOv8 with Adaptive Precision Architecture (APA).")
    parser.add_argument('--model', type=str, default='yolov8n.yaml', help="YOLOv8 model config or checkpoint (e.g. yolov8n.yaml, yolov8n.pt)")
    parser.add_argument('--data', type=str, default='coco8.yaml', help="Dataset config (e.g. coco8.yaml, coco128.yaml, coco.yaml)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Target device (cuda or cpu)")
    parser.add_argument('--workers', type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument('--precision', type=str, default='apa', choices=['apa', 'fp8', 'fp16', 'tf32', 'fp32_baseline'],
                        help="Precision mode: 'apa' (Adaptive), 'fp8' (Fixed FP8), 'fp16' (Fixed FP16), 'tf32' (Fixed TF32), 'fp32_baseline' (Pure PyTorch FP32)")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['research', 'conservative', 'aggressive'], help="APA config preset")
    parser.add_argument('--check_interval', type=int, default=1, help="Evaluation interval for soft checks")
    parser.add_argument('--no_dynamic_scaling', action='store_true', help="Disable dynamic delayed scaling (Trick B)")
    parser.add_argument('--no_dual_fp8', action='store_true', help="Disable dual FP8 format (Trick A)")
    parser.add_argument('--all_apa_layers', action='store_true', help="Convert all layers including Detect head to FP8")
    parser.add_argument('--forensic', action='store_true', help="Enable forensic logging on escalation events")
    parser.add_argument('--log_file', type=str, default=None, help="Path to JSONL log file")
    parser.add_argument('--smoke_test', action='store_true', help="Run 3 steps smoke test and verify 0 errors")
    return parser.parse_args()

def main():
    args = get_args()
    print("=" * 60)
    print("  YOLOv8 Training Pipeline with Adaptive Precision Architecture (APA)")
    print("=" * 60)
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
    if args.no_dynamic_scaling:
        config.enable_dynamic_scaling = False
    if args.no_dual_fp8:
        config.use_dual_fp8 = False
    if args.forensic:
        config.enable_forensic_logging = True
    if args.log_file:
        config.log_file = args.log_file

    if args.precision == 'fp8':
        config.freeze_level = LEVEL_FP8
    elif args.precision == 'fp16':
        config.freeze_level = LEVEL_FP16
    elif args.precision == 'tf32':
        config.freeze_level = LEVEL_TF32

    # 2. Setup Ultralytics YOLOv8 Model
    cfg = get_cfg()
    cfg.data = args.data
    cfg.model = args.model
    cfg.batch = args.batch_size
    cfg.imgsz = args.imgsz
    cfg.device = args.device

    trainer = DetectionTrainer(overrides=cfg)
    trainer.setup_model()
    model = trainer.model.to(args.device)
    model.args = cfg

    manager = None
    if args.precision != 'fp32_baseline':
        preserve_critical = not args.all_apa_layers
        convert_yolov8_to_apa(model, config=config, preserve_critical_layers=preserve_critical)
        manager = APAManager(model, config=config)
        print(f"APA Enabled: Successfully managing {len(manager.apa_modules)} layers!")
        params = manager.get_trainable_parameters()
    else:
        print("Running in FP32 Baseline mode (No APA).")
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)

    # 3. Setup Dataset and DataLoader
    data_dict = check_det_dataset(cfg.data)
    dataset = build_yolo_dataset(cfg, data_dict['train'], batch=args.batch_size, data=data_dict, mode='train', rect=False, stride=32)
    dataloader = build_dataloader(dataset, batch=args.batch_size, workers=args.workers, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} images, {len(dataloader)} batches per epoch.")

    # 4. Training Loop
    model.train()
    total_steps = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        epoch_loss = 0.0
        batches_accepted = 0

        for step, batch in enumerate(dataloader):
            if manager is not None:
                manager.pre_step()
            optimizer.zero_grad()

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device, non_blocking=True)
            batch['img'] = batch['img'].float() / 255.0

            loss, loss_items = model(batch)
            total_loss = loss.sum()
            total_loss.backward()

            accepted = True
            if manager is not None:
                accepted = manager.post_backward_sync_and_eval()

            if accepted:
                optimizer.step()
                batches_accepted += 1
                epoch_loss += total_loss.item()
            else:
                print(f"  [Step {step}] Batch rejected due to overflow. Precision escalated.")

            total_steps += 1

            if step % 5 == 0 or args.smoke_test:
                level_dist = {}
                if manager is not None:
                    for m in manager.apa_modules.values():
                        level_dist[m.level] = level_dist.get(m.level, 0) + 1
                    level_str = f"Levels: FP8={level_dist.get(0, 0)}, FP16={level_dist.get(1, 0)}, TF32={level_dist.get(2, 0)}"
                else:
                    level_str = "FP32"

                items_str = " | ".join([f"{k}: {v.item():.3f}" for k, v in loss_items.items()])
                print(f"Step {step}/{len(dataloader)}: Total Loss={total_loss.item():.4f} ({items_str}) | {level_str}")

            if args.smoke_test and total_steps >= 3:
                print("\n[SMOKE TEST PASSED] Verified 3 steps with 0 errors!")
                return

        avg_loss = epoch_loss / max(1, batches_accepted)
        print(f"Epoch {epoch + 1} Complete: Avg Loss={avg_loss:.4f} | Accepted Batches: {batches_accepted}/{len(dataloader)}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s ({total_steps} steps).")

if __name__ == '__main__':
    main()
