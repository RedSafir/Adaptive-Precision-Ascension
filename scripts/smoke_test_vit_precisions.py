#!/usr/bin/env python3
"""
ViT Precision Smoke Test & Benchmark: FP8 vs FP16 vs TF32 (and APA)

Fast, isolated benchmark comparing training throughput, latency, VRAM footprint,
and numerical sanity across different precision modes for Vision Transformer on CIFAR-10.

Usage:
    # Quick 15-second smoke test with synthetic data:
    python scripts/smoke_test_vit_precisions.py --synthetic --steps 50

    # Smoke test on real CIFAR-10:
    python scripts/smoke_test_vit_precisions.py --steps 50 --batch_size 128
"""

import sys
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'vit_cifar10'))

from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import VisionTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="ViT Precision Smoke Test & Benchmark: FP8 vs FP16 vs TF32")
    parser.add_argument('--steps', type=int, default=50, help="Number of benchmark steps per method (default: 50)")
    parser.add_argument('--warmup', type=int, default=10, help="Number of warmup steps (default: 10)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument('--dim', type=int, default=256, help="ViT hidden dimension (default: 256, try 384 or 512 for larger GEMMs)")
    parser.add_argument('--depth', type=int, default=6, help="ViT depth / number of transformer blocks (default: 6)")
    parser.add_argument('--image_size', type=int, default=32, help="Input image resolution (default: 32, try 224 for ImageNet scale)")
    parser.add_argument('--patch_size', type=int, default=4, help="Patch size (default: 4 for CIFAR, 16 for ImageNet)")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of output classes (default: 10, 100 for ImageNet-100)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument('--methods', nargs='+', default=['fp8', 'fp16', 'tf32', 'fp32', 'apa'],
                        choices=['fp8', 'fp16', 'tf32', 'fp32', 'apa'],
                        type=lambda s: 'fp8' if s.lower() == 'fp8_fast' else s.lower(),
                        help="Methods to benchmark (default: fp8 fp16 tf32 fp32 apa)")
    parser.add_argument('--synthetic', action='store_true',
                        help="Use synthetic random data (instant, no CIFAR-10 download needed)")
    parser.add_argument('--data_dir', type=str, default=os.path.join('examples', 'vit_cifar10', 'data'),
                        help="Path to CIFAR-10 dataset (used if not --synthetic)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to benchmark on")
    parser.add_argument('--save_json', type=str, default=None,
                        help="Optional path to save benchmark metrics as JSON")
    return parser.parse_args()

def get_data_loader(args):
    """Returns a data iterator yielding (x, y) batches."""
    if args.synthetic or not torch.cuda.is_available():
        class SyntheticDataset:
            def __init__(self, count, batch_size, image_size, num_classes):
                self.count = count
                self.batch_size = batch_size
                self.image_size = image_size
                self.num_classes = num_classes
            def __iter__(self):
                for _ in range(self.count):
                    x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
                    y = torch.randint(0, self.num_classes, (self.batch_size,))
                    yield x, y
            def __len__(self):
                return self.count
        total_batches = args.warmup + args.steps + 10
        return SyntheticDataset(total_batches, args.batch_size, args.image_size, args.num_classes)
    else:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        os.makedirs(args.data_dir, exist_ok=True)
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

def benchmark_single_method(method, args, data_batches):
    """Runs warmup + timed steps for a single precision method."""
    device = torch.device(args.device)
    
    # Configure precision environment
    fp8_output_dtype = 'float16'
    if method == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        use_apa = False
        freeze_level = None
        use_amp = False
        backend_desc = "TF32 Tensor Cores (IEEE 754 19-bit)"
    elif method == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')
        use_apa = False
        freeze_level = None
        use_amp = False
        backend_desc = "Strict FP32 (IEEE 754 23-bit, TF32 Off)"
    elif method == 'fp16':
        use_apa = False
        freeze_level = None
        use_amp = True
        backend_desc = "PyTorch AMP FP16 Tensor Cores"
    elif method == 'fp8':
        use_apa = True
        freeze_level = LEVEL_FP8
        use_amp = False
        fp8_output_dtype = 'float16'
        backend_desc = "Native FP8 (E4M3/E5M2 Tensor Cores)"
    elif method == 'apa':
        use_apa = True
        freeze_level = None
        use_amp = False
        fp8_output_dtype = 'float16'
        backend_desc = "APA Adaptive (FP8 -> FP16 -> TF32)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build model & optimizer
    if use_apa:
        config = APAConfig.research_default(
            device=str(device),
            interval_telemetry=True,
            freeze_level=freeze_level,
            fp8_output_dtype=fp8_output_dtype
        )
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            config=config, use_apa=True, preserve_critical_layers=True,
            dim=args.dim, depth=args.depth, mlp_dim=args.dim * 2
        ).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            config=None, use_apa=False,
            dim=args.dim, depth=args.depth, mlp_dim=args.dim * 2
        ).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    model.train()

    # Reset VRAM counters
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Warmup Phase (prime GPU caches and Triton JIT compilation)
    print(f"  [Warmup {args.warmup} steps]...", end="", flush=True)
    batch_idx = 0
    for _ in range(args.warmup):
        x, y = data_batches[batch_idx]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_idx += 1
        
        if apa_manager is not None:
            apa_manager.pre_step()
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
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
                apa_manager.post_backward_sync_and_eval()
            optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(" Done.")

    # 2. Timed Benchmark Phase
    print(f"  [Benchmark {args.steps} steps]...", end="", flush=True)
    initial_loss = None
    final_loss = None

    start_event = None
    end_event = None
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
    wall_start = time.perf_counter()

    for step in range(args.steps):
        x, y = data_batches[batch_idx]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_idx += 1
        
        if apa_manager is not None:
            apa_manager.pre_step()
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
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
                apa_manager.post_backward_sync_and_eval()
            optimizer.step()

        curr_loss = loss.detach().item()
        if step == 0:
            initial_loss = curr_loss
        if step == args.steps - 1:
            final_loss = curr_loss

    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
    else:
        elapsed_sec = time.perf_counter() - wall_start
        elapsed_ms = elapsed_sec * 1000.0

    print(" Done.")

    # Memory & Throughput Metrics
    total_samples = args.steps * args.batch_size
    img_per_sec = total_samples / elapsed_sec if elapsed_sec > 0 else 0.0
    ms_per_step = elapsed_ms / args.steps if args.steps > 0 else 0.0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == 'cuda' else 0.0

    # Cleanup
    del model, optimizer, apa_manager, trainable_params
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "method": method.upper(),
        "backend": backend_desc,
        "elapsed_sec": round(elapsed_sec, 3),
        "ms_per_step": round(ms_per_step, 2),
        "throughput_fps": round(img_per_sec, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "initial_loss": round(initial_loss, 4) if initial_loss is not None else 0.0,
        "final_loss": round(final_loss, 4) if final_loss is not None else 0.0,
    }

def main():
    args = parse_args()

    print("=" * 75)
    print("🔬 ViT Precision Smoke Test & Hardware Throughput Benchmark")
    print("=" * 75)
    print(f"Device      : {args.device}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(args.device)
        print(f"GPU Model   : {prop.name} (Compute {prop.major}.{prop.minor}, VRAM: {prop.total_memory / (1024**3):.1f} GB)")
    print(f"Dataset     : {'Synthetic Tensors (Fast Mode)' if args.synthetic else 'CIFAR-10'}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Steps       : {args.steps} (+ {args.warmup} warmup)")
    print(f"Methods     : {', '.join([m.upper() for m in args.methods])}")
    print("=" * 75)

    # Pre-generate or load batches to ensure identical data across methods
    print("\nPreparing batches...")
    loader = get_data_loader(args)
    data_batches = []
    needed_batches = args.warmup + args.steps + 5
    for batch in loader:
        data_batches.append(batch)
        if len(data_batches) >= needed_batches:
            break
    
    # If CIFAR-10 loader yielded fewer batches than needed, loop it
    while len(data_batches) < needed_batches:
        for batch in loader:
            data_batches.append(batch)
            if len(data_batches) >= needed_batches:
                break
    print(f"Ready: {len(data_batches)} batches cached for testing.\n")

    results = []
    for method in args.methods:
        print(f">>> Running Method: [{method.upper()}]")
        res = benchmark_single_method(method, args, data_batches)
        results.append(res)
        print(f"    -> Latency   : {res['ms_per_step']} ms/step")
        print(f"    -> Throughput: {res['throughput_fps']} img/sec")
        print(f"    -> Peak VRAM : {res['peak_vram_mb']} MB")
        print(f"    -> Loss      : {res['initial_loss']} -> {res['final_loss']}\n")

    # Determine baseline for speedup calculation (TF32 if present, else FP32, else last method)
    tf32_ms = next((r['ms_per_step'] for r in results if r['method'] == 'TF32'), None)
    if tf32_ms is None:
        tf32_ms = next((r['ms_per_step'] for r in results if r['method'] == 'FP32'), None)
    if tf32_ms is None and len(results) > 0:
        tf32_ms = results[-1]['ms_per_step']

    for r in results:
        if tf32_ms and tf32_ms > 0:
            speedup = tf32_ms / r['ms_per_step']
            r['speedup'] = f"{speedup:.2f}x"
        else:
            r['speedup'] = "1.00x"

    # Display Pretty Table
    print("=" * 95)
    print("📊 BENCHMARK COMPARISON SUMMARY (Vision Transformer CIFAR-10)")
    print("=" * 95)
    header = f"{'Method':<8} | {'Backend / Dtype':<34} | {'Latency':<11} | {'Throughput':<12} | {'Speedup':<9} | {'Peak VRAM':<10}"
    print(header)
    print("-" * 95)
    for r in results:
        row = (
            f"{r['method']:<8} | "
            f"{r['backend']:<34} | "
            f"{r['ms_per_step']:>7.2f} ms | "
            f"{r['throughput_fps']:>7.1f} img/s | "
            f"{r['speedup']:>7} | "
            f"{r['peak_vram_mb']:>7.1f} MB"
        )
        print(row)
    print("=" * 95)

    # Save to JSON if requested
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump({
                "args": vars(args),
                "device": str(args.device),
                "results": results
            }, f, indent=2)
        print(f"Results saved to: {args.save_json}")

if __name__ == '__main__':
    main()
