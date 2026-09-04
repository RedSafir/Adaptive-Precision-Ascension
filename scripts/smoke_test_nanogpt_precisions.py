#!/usr/bin/env python3
"""
nanoGPT Precision Smoke Test & Benchmark: FP8 vs FP16 vs TF32 (and APA)

Fast, isolated benchmark comparing training throughput, latency, VRAM footprint,
and numerical sanity across different precision modes for nanoGPT Character-level LM.

Usage:
    python scripts/smoke_test_nanogpt_precisions.py --steps 100 --batch_size 32
"""

import sys
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'nanogpt_char'))

from apa import APAConfig, APAManager
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import GPT

def parse_args():
    parser = argparse.ArgumentParser(description="nanoGPT Precision Smoke Test & Benchmark: FP8 vs FP16 vs TF32")
    parser.add_argument('--steps', type=int, default=100, help="Number of benchmark steps per method (default: 100)")
    parser.add_argument('--warmup', type=int, default=15, help="Number of warmup steps (default: 15)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--n_embd', type=int, default=384, help="Embedding dimension (default: 384)")
    parser.add_argument('--n_layer', type=int, default=6, help="Number of transformer layers (default: 6)")
    parser.add_argument('--n_head', type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument('--block_size', type=int, default=256, help="Context length (default: 256)")
    parser.add_argument('--vocab_size', type=int, default=65, help="Vocabulary size (default: 65)")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument('--methods', nargs='+', default=['fp8', 'fp16', 'tf32', 'apa'],
                        choices=['fp8', 'fp16', 'tf32', 'apa'],
                        help="Methods to benchmark (default: fp8 fp16 tf32 apa)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to benchmark on")
    parser.add_argument('--save_json', type=str, default=None,
                        help="Optional path to save benchmark metrics as JSON")
    return parser.parse_args()

def generate_batches(args, total_batches):
    """Generates synthetic character-level token batches for reproducible timing."""
    batches = []
    for _ in range(total_batches):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size))
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size))
        batches.append((x, y))
    return batches

def benchmark_single_method(method, args, data_batches):
    """Runs warmup + timed steps for a single precision method."""
    device = torch.device(args.device)
    
    # Configure precision environment
    if method == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_apa = False
        freeze_level = None
        use_amp = False
        backend_desc = "TF32 Tensor Cores (Standard FP32)"
    elif method == 'fp16':
        use_apa = False
        freeze_level = None
        use_amp = True
        backend_desc = "PyTorch AMP FP16 Tensor Cores"
    elif method == 'fp8':
        use_apa = True
        freeze_level = LEVEL_FP8
        use_amp = False
        backend_desc = "Native FP8 (FP16 Accum / Fast SDPA)"
    elif method == 'apa':
        use_apa = True
        freeze_level = None
        use_amp = False
        backend_desc = "APA Adaptive (FP8 -> FP16 -> TF32)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build model & optimizer
    if use_apa:
        config = APAConfig.research_default(
            device=str(device),
            interval_telemetry=True,
            freeze_level=freeze_level,
            fp8_output_dtype='float16'
        )
        model = GPT(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            config=config,
            use_apa=True
        ).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = GPT(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            config=None,
            use_apa=False
        ).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
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
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
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
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
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
    total_tokens = args.steps * args.batch_size * args.block_size
    tokens_per_sec = total_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
    ms_per_step = elapsed_ms / args.steps if args.steps > 0 else 0.0
    it_per_sec = args.steps / elapsed_sec if elapsed_sec > 0 else 0.0
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
        "it_per_sec": round(it_per_sec, 1),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "initial_loss": round(initial_loss, 4) if initial_loss is not None else 0.0,
        "final_loss": round(final_loss, 4) if final_loss is not None else 0.0,
    }

def main():
    args = parse_args()

    print("=" * 80)
    print("🔬 nanoGPT Precision Smoke Test & Hardware Throughput Benchmark")
    print("=" * 80)
    print(f"Device      : {args.device}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(args.device)
        print(f"GPU Model   : {prop.name} (Compute {prop.major}.{prop.minor}, VRAM: {prop.total_memory / (1024**3):.1f} GB)")
    print(f"Model Config: {args.n_layer} layers, {args.n_head} heads, {args.n_embd} dim, context={args.block_size}")
    print(f"Batch Size  : {args.batch_size} (Tokens per batch: {args.batch_size * args.block_size:,})")
    print(f"Steps       : {args.steps} (+ {args.warmup} warmup)")
    print(f"Methods     : {', '.join([m.upper() for m in args.methods])}")
    print("=" * 80)

    print("\nPreparing batches...")
    needed_batches = args.warmup + args.steps + 5
    data_batches = generate_batches(args, needed_batches)
    print(f"Ready: {len(data_batches)} batches cached for testing.\n")

    results = []
    for method in args.methods:
        print(f">>> Running Method: [{method.upper()}]")
        res = benchmark_single_method(method, args, data_batches)
        results.append(res)
        print(f"    -> Latency   : {res['ms_per_step']} ms/step ({res['it_per_sec']} it/s)")
        print(f"    -> Throughput: {res['tokens_per_sec']:,.0f} tokens/sec")
        print(f"    -> Peak VRAM : {res['peak_vram_mb']} MB")
        print(f"    -> Loss      : {res['initial_loss']} -> {res['final_loss']}\n")

    # Determine baseline for speedup calculation (TF32 if present, else first method)
    tf32_ms = next((r['ms_per_step'] for r in results if r['method'] == 'TF32'), None)
    if tf32_ms is None and len(results) > 0:
        tf32_ms = results[-1]['ms_per_step']

    for r in results:
        if tf32_ms and tf32_ms > 0:
            speedup = tf32_ms / r['ms_per_step']
            r['speedup'] = f"{speedup:.2f}x"
        else:
            r['speedup'] = "1.00x"

    # Display Pretty Table
    print("=" * 105)
    print("📊 BENCHMARK COMPARISON SUMMARY (nanoGPT Language Model)")
    print("=" * 105)
    header = f"{'Method':<8} | {'Backend / Dtype':<34} | {'Latency':<11} | {'Throughput':<15} | {'Speedup':<9} | {'Peak VRAM':<10}"
    print(header)
    print("-" * 105)
    for r in results:
        tok_str = f"{r['tokens_per_sec']:,.0f} tok/s"
        row = (
            f"{r['method']:<8} | "
            f"{r['backend']:<34} | "
            f"{r['ms_per_step']:>7.2f} ms | "
            f"{tok_str:>13} | "
            f"{r['speedup']:>7} | "
            f"{r['peak_vram_mb']:>7.1f} MB"
        )
        print(row)
    print("=" * 105)

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
