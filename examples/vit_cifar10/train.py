import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APAConfig, APAManager
from model import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'])
    parser.add_argument('--fp8_sim', action='store_true')
    parser.add_argument('--log_file', type=str, default='apa_vit_log.jsonl')
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Model
    preset_map = {
        'conservative': APAConfig.conservative,
        'aggressive': APAConfig.aggressive,
        'research': APAConfig.research_default,
    }
    config = preset_map[args.apa_preset](device=str(device), log_file=args.log_file)
    if args.fp8_sim:
        config.fp8_simulation_mode = True
        
    model = VisionTransformer(config=config).to(device)
    
    # APA Manager
    apa_manager = APAManager(model, config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(apa_manager.get_trainable_parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    with open(args.log_file, 'w') as f:
        f.write(json.dumps({"config": config.__dict__, "args": vars(args)}) + "\n")
        
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            apa_manager.pre_step()
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            
            if apa_manager.post_backward_sync_and_eval():
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({'loss': loss.item(), 'acc': total_correct / total_samples})
                
        scheduler.step()
        
        # Eval
        model.eval()
        test_loss = 0
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
        print(f"Epoch {epoch+1} Summary: Train Loss={total_loss/total_samples:.4f} Train Acc={train_acc:.4f} | Test Loss={test_loss/test_samples:.4f} Test Acc={test_acc:.4f}")
        
        log_data = {
            "epoch": epoch + 1,
            "train_loss": total_loss/total_samples,
            "train_acc": train_acc,
            "test_loss": test_loss/test_samples,
            "test_acc": test_acc
        }
        with open(args.log_file, 'a') as f:
            f.write(json.dumps(log_data) + "\n")

if __name__ == '__main__':
    main()
