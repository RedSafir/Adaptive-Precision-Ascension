import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from apa import APAConfig, APAManager
from model import VGG16

def get_args():
    parser = argparse.ArgumentParser(description="Train VGG-16 on CIFAR-10 with APA or pure FP32 baseline.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'], help="APA config preset")
    parser.add_argument('--fp8_sim', action='store_true', help="Use FP8 simulation mode")
    parser.add_argument('--fp32_baseline', action='store_true', help="Run pure FP32 baseline training WITHOUT APA (uses standard nn.Linear)")
    parser.add_argument('--forensic', action='store_true', help="Enable detailed forensic logging on escalation")
    parser.add_argument('--forensic_argmax', action='store_true', help="Capture flat argmax index in forensic snapshots (opt-in)")
    parser.add_argument('--telemetry_interval', type=int, default=0, help="Periodic step interval to log per-layer amax and underflow time series (defaults to 50 if --forensic is enabled, else 0)")
    parser.add_argument('--log_file', type=str, default='result/apa_vgg16_log.jsonl', help="Path to output JSONL log file")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_apa = not args.fp32_baseline
    mode_str = "APA (Adaptive Precision)" if use_apa else "Pure FP32 Baseline (No APA)"
    
    print("=" * 65)
    print(f"VGG-16 CIFAR-10 Training")
    print(f"Mode      : {mode_str}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Log File  : {args.log_file}")
    if use_apa and args.forensic:
        print(f"Forensic  : ENABLED (Detailed telemetry on escalation)")
    print("=" * 65)
    
    # Ensure log output directory exists
    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Dataset Preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model & Config Setup
    if use_apa:
        preset_map = {
            'conservative': APAConfig.conservative,
            'aggressive': APAConfig.aggressive,
            'research': APAConfig.research_default,
        }
        telemetry_int = args.telemetry_interval if args.telemetry_interval > 0 else (50 if args.forensic else 0)
        config_kwargs = {
            'device': str(device),
            'log_file': args.log_file,
            'enable_forensic_logging': args.forensic,
            'forensic_capture_argmax_index': args.forensic_argmax,
            'telemetry_log_interval': telemetry_int,
        }
        config = preset_map[args.apa_preset](**config_kwargs)
        if args.fp8_sim:
            config.fp8_simulation_mode = True
            
        model = VGG16(num_classes=10, config=config, use_apa=True).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = VGG16(num_classes=10, config=None, use_apa=False).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize Log File
    with open(args.log_file, 'w', encoding='utf-8') as f:
        header_data = {
            "model": "VGG-16",
            "dataset": "CIFAR-10",
            "mode": "apa" if use_apa else "fp32_baseline",
            "args": vars(args),
            "config": config.__dict__ if config is not None else {"precision": "FP32"}
        }
        f.write(json.dumps(header_data) + "\n")
        
    total_training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{mode_str}]")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if apa_manager is not None:
                apa_manager.pre_step()
                
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            
            if apa_manager is not None:
                if apa_manager.post_backward_sync_and_eval():
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                
            total_loss += loss.item()
            _, preds = out.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluation Loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                loss = F.cross_entropy(out, y)
                test_loss += loss.item()
                _, preds = out.max(1)
                test_total += y.size(0)
                test_correct += preds.eq(y).sum().item()
                
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        epoch_time = time.time() - epoch_start
        
        # Precision Summary
        if use_apa and apa_manager is not None:
            dist = {0: 0, 1: 0, 2: 0}
            layer_status = {}
            for name, m in apa_manager.apa_modules.items():
                dist[m.level] += 1
                lvl_name = {0: 'FP8', 1: 'FP16', 2: 'TF32'}[m.level]
                layer_status[name] = lvl_name
                
            total_layers = len(apa_manager.apa_modules)
            prec_summary_str = f"FP8={dist[0]} ({dist[0]/total_layers*100:.1f}%), FP16={dist[1]}, TF32={dist[2]} (Total: {total_layers} APA layers)"
        else:
            dist = {0: 0, 1: 0, 2: 0}
            layer_status = {}
            prec_summary_str = "Pure FP32 Baseline (No APA)"
            
        print(f"Epoch {epoch+1:3d}/{args.epochs} Summary ({epoch_time:.1f}s): Train Loss={train_loss:.4f} Train Acc={train_acc*100:.2f}% | Test Loss={test_loss:.4f} Test Acc={test_acc*100:.2f}%")
        if use_apa:
            print(f"         Precision State: {prec_summary_str}")
            
        # Log Epoch Summary
        with open(args.log_file, 'a', encoding='utf-8') as f:
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "epoch_time_sec": round(epoch_time, 2),
                "mode": "apa" if use_apa else "fp32_baseline",
                "precision_distribution": {"fp8": dist[0], "fp16": dist[1], "tf32": dist[2]} if use_apa else {"fp32": "100%"},
                "layer_precisions": layer_status if use_apa else {}
            }
            f.write(json.dumps(epoch_record) + "\n")

    total_training_time = time.time() - total_training_start
    print("=" * 65)
    print(f"VGG-16 Training Complete! Total Time: {total_training_time:.2f}s")
    print(f"Logs saved to: {args.log_file}")
    if use_apa and args.forensic:
        base = args.log_file.rsplit('.', 1)
        forensic_file = base[0] + '_forensic.jsonl' if len(base) == 2 else args.log_file + '_forensic.jsonl'
        print(f"Forensic log : {forensic_file}")
    print("=" * 65)

if __name__ == '__main__':
    main()
