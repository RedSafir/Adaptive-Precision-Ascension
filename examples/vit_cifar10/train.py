import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F
try:
    import torch._dynamo
except ImportError:
    pass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from apa import APAConfig, APAManager, APACUDAGraphRunner
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on CIFAR-10 with APA or pure FP32 baseline.")
    parser.add_argument('--precision', type=lambda s: s.lower(), default=None, choices=['apa', 'fp8', 'fp16', 'fp16_apa', 'tf32', 'fp32'],
                        help="Precision mode: 'apa' (Adaptive), 'fp8' (Fixed FP8 via Triton), 'fp16' (Mixed Precision AMP), 'fp16_apa' (Controlled FP16 via APALinear), 'tf32' (Standard TF32 FP32), or 'fp32' (Strict IEEE 754 Single Precision)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'], help="APA config preset")
    parser.add_argument('--theta_underflow', type=float, default=None, help="Override underflow ratio threshold (e.g. 0.80 or 0.95)")
    parser.add_argument('--disable_underflow', action='store_true', help="Disable silent underflow escalation entirely")
    parser.add_argument('--check_interval', type=int, default=None, help="Override telemetry check interval in steps")
    parser.add_argument('--fp8_sim', action='store_true', help="Use FP8 simulation mode")
    parser.add_argument('--fp32_baseline', '--no_apa', action='store_true', dest='fp32_baseline', help="Run pure FP32 baseline training WITHOUT APA (uses standard nn.Linear)")
    parser.add_argument('--strict_fp32', action='store_true', help="Force strict IEEE 754 FP32 math (disables TF32 Tensor Cores, slower but exact 23-bit mantissa)")
    parser.add_argument('--no_dynamic_scaling', action='store_true', help="Disable Trick B (Dynamic Amax Delayed Scaling)")
    parser.add_argument('--no_dual_fp8', action='store_true', help="Disable Trick A (forces E4M3 for backward gradients instead of E5M2)")
    parser.add_argument('--all_apa_layers', action='store_true', help="Apply APA to patch_embed and head layers as well (disables boundary layer preservation)")
    parser.add_argument('--forensic', action='store_true', help="Enable detailed forensic logging on escalation")
    parser.add_argument('--forensic_argmax', action='store_true', help="Capture flat argmax index in forensic snapshots (opt-in)")
    parser.add_argument('--interval_telemetry', action='store_true', help="Sample telemetry only on check_interval steps instead of every step (default: False)")
    parser.add_argument('--telemetry_interval', type=int, default=0, help="Periodic step interval to log per-layer amax and underflow time series (defaults to 50 if --forensic is enabled, else 0)")
    parser.add_argument('--log_file', type=str, default='apa_vit_log.jsonl', help="Path to output JSONL log file")
    parser.add_argument('--compile', action='store_true', help="Compile model with torch.compile(backend='inductor') for maximum throughput")
    parser.add_argument('--cuda_graph', action='store_true', help="Use Custom APA CUDA Graph Engine for 0-freeze compiled execution")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'imagefolder'], help="Dataset: cifar10, cifar100, imagenet")
    parser.add_argument('--data_dir', type=str, default=None, help="Custom dataset path (e.g. for ImageNet ImageFolder)")
    parser.add_argument('--dim', type=int, default=256, help="ViT hidden dimension (e.g. 256, 768, 1024, 1536)")
    parser.add_argument('--depth', type=int, default=6, help="ViT depth (number of blocks)")
    parser.add_argument('--heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--image_size', type=int, default=32, help="Image resolution (e.g. 32, 224)")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size (e.g. 4 for 32x32, 16 for 224x224)")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of output classes (e.g. 10, 100, 1000)")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    freeze_level = None
    use_amp = False
    if args.precision == 'fp8':
        use_apa = True
        freeze_level = LEVEL_FP8
        mode_str = "Pure FP8 (Fixed Level 0 with Native Triton Kernel)"
        math_mode = "FP8 E4M3/E5M2 Tensor Cores"
    elif args.precision == 'fp16_apa':
        use_apa = True
        freeze_level = LEVEL_FP16
        mode_str = "Controlled FP16 (APALinear Level 1, Isolated Bit-Width Benchmark)"
        math_mode = "FP16 Half Precision via APALinear"
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
    elif args.precision == 'apa':
        use_apa = True
        mode_str = "APA (Adaptive Precision)"
        math_mode = "Adaptive (FP8 -> FP16 -> TF32)"
    else:
        use_apa = not args.fp32_baseline
        mode_str = "APA (Adaptive Precision)" if use_apa else "Pure FP32 Baseline (No APA)"
        math_mode = "Standard FP32 (TF32 Tensor Cores Enabled)" if not args.strict_fp32 else "Strict IEEE 754 Single Precision (TF32 Disabled)"
    
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
    
    print("=" * 60)
    print(f"Vision Transformer CIFAR-10 Training")
    print(f"Mode      : {mode_str}")
    print(f"Math Dtype: {math_mode}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)
    
    # Dataset
    data_dir = args.data_dir if args.data_dir is not None else os.path.join(os.path.dirname(__file__), 'data')
    if args.dataset in ('imagenet', 'imagefolder'):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(args.image_size * 1.14)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dir = os.path.join(data_dir, 'val') if os.path.exists(os.path.join(data_dir, 'val')) else data_dir
        train_dir = os.path.join(data_dir, 'train') if os.path.exists(os.path.join(data_dir, 'train')) else data_dir
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
        test_dataset = datasets.ImageFolder(root=val_dir, transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize(args.image_size) if args.image_size != 32 else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.image_size) if args.image_size != 32 else transforms.ToTensor(),
            transforms.ToTensor() if args.image_size == 32 else transforms.Lambda(lambda x: x),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        os.makedirs(data_dir, exist_ok=True)
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    else: # CIFAR-10 default
        transform_train = transforms.Compose([
            transforms.Resize(args.image_size) if args.image_size != 32 else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.image_size) if args.image_size != 32 else transforms.ToTensor(),
            transforms.ToTensor() if args.image_size == 32 else transforms.Lambda(lambda x: x),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        os.makedirs(data_dir, exist_ok=True)
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
            'enable_dynamic_scaling': not args.no_dynamic_scaling,
            'use_dual_fp8': not args.no_dual_fp8,
            'interval_telemetry': args.interval_telemetry,
            'freeze_level': freeze_level,
        }
        if args.theta_underflow is not None:
            config_kwargs['theta_underflow'] = args.theta_underflow
        if args.disable_underflow:
            config_kwargs['theta_underflow'] = 1.01
        if args.check_interval is not None:
            config_kwargs['check_interval'] = args.check_interval

        config = preset_map[args.apa_preset](**config_kwargs)
        if args.fp8_sim:
            config.fp8_simulation_mode = True
            
        preserve_crit = not args.all_apa_layers
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.dim * 4 if args.dim >= 768 else args.dim * 2,
            config=config,
            use_apa=True,
            preserve_critical_layers=preserve_crit
        ).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.dim * 4 if args.dim >= 768 else args.dim * 2,
            config=None,
            use_apa=False
        ).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            if hasattr(torch, '_dynamo'):
                torch._dynamo.config.force_parameter_static_shapes = False
                torch._dynamo.config.suppress_errors = True
            print("Compiling model via torch.compile(backend='inductor')...", end="", flush=True)
            model = torch.compile(model, dynamic=False)
            print(" Done.\n")
            if use_apa and freeze_level is None and device.type == 'cuda':
                print("  [Pre-warming Inductor cache for all precision levels (FP8, FP16, TF32)]...", end="", flush=True)
                sample_batch = next(iter(train_loader))
                bx, by = sample_batch[0].to(device, non_blocking=True), sample_batch[1].to(device, non_blocking=True)
                for p_lvl in [LEVEL_FP8, LEVEL_FP16, LEVEL_TF32]:
                    for m in apa_manager.apa_modules.values():
                        m.level = p_lvl
                    apa_manager.pre_step()
                    out_sample = model(bx)
                    loss_sample = F.cross_entropy(out_sample, by)
                    loss_sample.backward()
                    apa_manager.post_backward_sync_and_eval()
                for m in apa_manager.apa_modules.values():
                    m.level = LEVEL_FP8
                for p in trainable_params:
                    p.grad = None
                print(" Done. (Transitions will now happen with 0 ms freeze!)")
        else:
            print("[WARN: torch.compile not available in this PyTorch version]\n")

    # Optimizer & Scheduler & Scaler
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    opt_kwargs = {'lr': args.lr, 'weight_decay': 0.05}
    if getattr(args, 'cuda_graph', False):
        opt_kwargs['capturable'] = True
    optimizer = torch.optim.AdamW(trainable_params, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    cuda_graph_runner = None
    if getattr(args, 'cuda_graph', False) and device.type == 'cuda':
        print("Capturing Custom APA CUDA Graph Engine (0-freeze mode)...", end="", flush=True)
        sample_batch = next(iter(train_loader))
        sx = sample_batch[0].to(device, non_blocking=True)
        sy = sample_batch[1].to(device, non_blocking=True)
        autocast_dt = torch.float16 if (use_amp or use_apa) else None
        cuda_graph_runner = APACUDAGraphRunner(
            model=model,
            optimizer=optimizer,
            sample_x=sx,
            sample_y=sy,
            apa_manager=apa_manager,
            loss_fn=F.cross_entropy,
            autocast_dtype=autocast_dt,
            warmup_steps=3,
        )
        print(" Done. (0-freeze compiled execution ready!)\n")

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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if cuda_graph_runner is not None:
                loss_val = cuda_graph_runner.step(x, y)
                total_loss += loss_val * x.size(0)
                total_samples += x.size(0)
                if batch_idx % 50 == 0:
                    pbar.set_postfix({'loss': f"{loss_val:.4f}"})
                continue

            if apa_manager is not None:
                apa_manager.pre_step()
                
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{total_correct/total_samples*100:.1f}%"})
                
        epoch_duration = time.time() - epoch_start
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = model(x)
                else:
                    out = model(x)
                loss = F.cross_entropy(out, y)
                test_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=-1)
                test_correct += (preds == y).sum().item()
                test_samples += x.size(0)
                
        train_acc = total_correct / total_samples
        test_acc = test_correct / test_samples
        
        # Summary & Logging
        if use_apa and config.freeze_level is None:
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
                  f"Test Loss={test_loss/test_samples:.4f} Test Acc={test_acc*100:.2f}% ({mode_str})")
                  
            log_data = {
                "event": "epoch_summary",
                "epoch": epoch + 1,
                "mode": args.precision or ("apa" if use_apa else "fp32_baseline"),
                "epoch_time_sec": round(epoch_duration, 2),
                "train_loss": total_loss / total_samples,
                "train_acc": train_acc,
                "test_loss": test_loss / test_samples,
                "test_acc": test_acc,
                "precision": mode_str
            }
            
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + "\n")

        if use_apa and args.forensic and config.forensic_log_file and config.forensic_log_file != args.log_file:
            try:
                with open(config.forensic_log_file, 'a', encoding='utf-8') as ff:
                    ff.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            
    total_duration = time.time() - total_training_start
    print("=" * 60)
    print(f"Training Complete! Total Time: {total_duration/60:.2f} minutes")
    print(f"Log saved to: {args.log_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
