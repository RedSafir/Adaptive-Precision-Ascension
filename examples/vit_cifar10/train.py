import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from apa import APAConfig, APAManager
from model import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on CIFAR-10 with APA or pure FP32 baseline.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'], help="APA config preset")
    parser.add_argument('--fp8_sim', action='store_true', help="Use FP8 simulation mode")
    parser.add_argument('--fp32_baseline', action='store_true', help="Run pure FP32 baseline training WITHOUT APA (uses standard nn.Linear)")
    parser.add_argument('--log_file', type=str, default='apa_vit_log.jsonl', help="Path to output JSONL log file")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_apa = not args.fp32_baseline
    mode_str = "APA (Adaptive Precision)" if use_apa else "Pure FP32 Baseline (No APA)"
    
    print("=" * 60)
    print(f"Vision Transformer CIFAR-10 Training")
    print(f"Mode   : {mode_str}")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)
    
    # Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
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
        config = preset_map[args.apa_preset](device=str(device), log_file=args.log_file)
        if args.fp8_sim:
            config.fp8_simulation_mode = True
            
        model = VisionTransformer(config=config, use_apa=True).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = VisionTransformer(config=None, use_apa=False).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize Log File
    with open(args.log_file, 'w', encoding='utf-8') as f:
        header_data = {
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
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{mode_str}]")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
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
                
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{(total_correct / total_samples)*100:.2f}%"
                })
                
        scheduler.step()
        epoch_duration = time.time() - epoch_start
        
        # Evaluation Loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = F.cross_entropy(out, y)
                test_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=-1)
                test_correct += (preds == y).sum().item()
                test_samples += x.size(0)
                
        train_acc = total_correct / total_samples
        test_acc = test_correct / test_samples
        
        # Summary & Logging
        if use_apa:
            precision_names = {0: "FP8", 1: "FP16", 2: "TF32"}
            layer_precisions = {name: precision_names.get(mod.level, f"Level_{mod.level}") for name, mod in apa_manager.apa_modules.items()}
            counts = {"FP8": 0, "FP16": 0, "TF32": 0}
            for lvl in layer_precisions.values():
                if lvl in counts:
                    counts[lvl] += 1
            total_apa_layers = len(layer_precisions)
            fp8_pct = (counts['FP8'] / total_apa_layers * 100) if total_apa_layers > 0 else 0
            
            print(f"Epoch {epoch+1:3d}/{args.epochs} Summary ({epoch_duration:.1f}s): "
                  f"Train Loss={total_loss/total_samples:.4f} Train Acc={train_acc*100:.2f}% | "
                  f"Test Loss={test_loss/test_samples:.4f} Test Acc={test_acc*100:.2f}%")
            print(f"         Precision State: FP8={counts['FP8']} ({fp8_pct:.1f}%), FP16={counts['FP16']}, TF32={counts['TF32']} (Total: {total_apa_layers} layers)")
            
            log_data = {
                "event": "epoch_summary",
                "epoch": epoch + 1,
                "mode": "apa",
                "epoch_time_sec": round(epoch_duration, 2),
                "train_loss": total_loss / total_samples,
                "train_acc": train_acc,
                "test_loss": test_loss / test_samples,
                "test_acc": test_acc,
                "precision_distribution": {
                    "fp8": counts["FP8"],
                    "fp16": counts["FP16"],
                    "tf32": counts["TF32"],
                    "fp8_percentage": f"{fp8_pct:.1f}%"
                },
                "layer_precisions": layer_precisions
            }
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} Summary ({epoch_duration:.1f}s): "
                  f"Train Loss={total_loss/total_samples:.4f} Train Acc={train_acc*100:.2f}% | "
                  f"Test Loss={test_loss/test_samples:.4f} Test Acc={test_acc*100:.2f}% (FP32 Baseline)")
                  
            log_data = {
                "event": "epoch_summary",
                "epoch": epoch + 1,
                "mode": "fp32_baseline",
                "epoch_time_sec": round(epoch_duration, 2),
                "train_loss": total_loss / total_samples,
                "train_acc": train_acc,
                "test_loss": test_loss / test_samples,
                "test_acc": test_acc,
                "precision": "FP32_pure"
            }
            
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + "\n")
            
    total_duration = time.time() - total_training_start
    print("=" * 60)
    print(f"Training Complete! Total Time: {total_duration/60:.2f} minutes")
    print(f"Log saved to: {args.log_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
