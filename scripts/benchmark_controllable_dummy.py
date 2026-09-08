#!/usr/bin/env python3
"""
Interactive & Controllable Dummy Input & Dimension Benchmark for APA.

Provides full transparency into how:
1. Matrix Dimensions: M (tokens), K (in_features), N (out_features)
2. Input Characteristics: Mean, Scale/Std, Distribution, Outlier Spikes
3. Precision Modes: Native FP8 vs PyTorch FP16 AMP vs TF32
4. Execution Engines: CUDA Graph (0-freeze) vs Eager Mode

reveal the exact performance boundaries and crossover points where FP8 surpasses FP16.
"""

import sys
import os
import time
import argparse
import csv
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'vit_cifar10')))

from apa import APAConfig, APALinear, APAManager, APACUDAGraphRunner
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32
from model import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controllable Dummy Input & Dimension Benchmark for APA (FP8 vs FP16 vs TF32)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. Mode Selection
    parser.add_argument('--mode', type=str, default='layer', choices=['layer', 'vit'],
                        help="Benchmark scope: 'layer' (pure isolated GEMM) or 'vit' (full Vision Transformer)")

    # 2. Controllable Dimensions
    parser.add_argument('--M', type=int, default=8192,
                        help="Token count / batch rows (M = Batch * Tokens) for layer mode")
    parser.add_argument('--K', type=int, default=768,
                        help="Input feature dimension (in_features / hidden dim)")
    parser.add_argument('--N', type=int, default=768,
                        help="Output feature dimension (out_features / projection dim)")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size (used for ViT mode or M = batch_size * seq_len)")
    parser.add_argument('--seq_len', type=int, default=65,
                        help="Sequence length / tokens per sample for ViT mode")
    parser.add_argument('--depth', type=int, default=6,
                        help="Number of transformer blocks for ViT mode")

    # 3. Controllable Dummy Input Values
    parser.add_argument('--input_scale', type=float, default=1.0,
                        help="Scale / standard deviation of dummy input tensors")
    parser.add_argument('--input_mean', type=float, default=0.0,
                        help="Mean / offset of dummy input tensors")
    parser.add_argument('--input_dist', type=str, default='normal', choices=['normal', 'uniform', 'constant'],
                        help="Statistical distribution of dummy inputs")
    parser.add_argument('--outlier_ratio', type=float, default=0.0,
                        help="Fraction of elements injected with extreme outliers [0.0 - 1.0]")
    parser.add_argument('--outlier_val', type=float, default=500.0,
                        help="Magnitude of outlier spikes (exceeds FP8 ceiling 448.0 to test escalation)")

    # 4. Benchmarking Controls
    parser.add_argument('--steps', type=int, default=50,
                        help="Number of timed benchmark steps")
    parser.add_argument('--warmup', type=int, default=10,
                        help="Number of warmup steps")
    parser.add_argument('--cuda_graph', action='store_true',
                        help="Use APA CUDA Graph Engine for 0-freeze compiled execution")
    parser.add_argument('--sweep_dim', action='store_true',
                        help="Automatically sweep hidden dimensions (256 -> 4096) to find FP8 crossover point")
    parser.add_argument('--sweep_m', action='store_true',
                        help="Automatically sweep token count M (1024 -> 32768) to show batch scaling curve")
    parser.add_argument('--methods', nargs='+', default=['fp8', 'fp16', 'tf32'],
                        help="Precision methods to benchmark (e.g. --methods fp8 fp16)")
    parser.add_argument('--save_csv', type=str, default=None,
                        help="Optional path to save sweep benchmark results as CSV")
    parser.add_argument('--plot', type=str, default=None,
                        help="Optional path to save comparison plot as PNG (e.g. result/plots/crossover.png)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to benchmark on")

    return parser.parse_args()


def generate_dummy_tensor(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    dist: str = 'normal',
    scale: float = 1.0,
    mean: float = 0.0,
    outlier_ratio: float = 0.0,
    outlier_val: float = 500.0,
) -> torch.Tensor:
    """Generates a controllable synthetic tensor with explicit scale, mean, and outlier spikes."""
    if dist == 'normal':
        x = torch.randn(shape, dtype=dtype, device=device) * scale + mean
    elif dist == 'uniform':
        x = (torch.rand(shape, dtype=dtype, device=device) * 2.0 - 1.0) * scale + mean
    elif dist == 'constant':
        x = torch.full(shape, scale, dtype=dtype, device=device)
    else:
        x = torch.randn(shape, dtype=dtype, device=device) * scale + mean

    if outlier_ratio > 0.0:
        mask = torch.rand(shape, device=device) < outlier_ratio
        x[mask] = outlier_val

    return x


def benchmark_layer_single_precision(
    method: str,
    M: int,
    K: int,
    N: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """Micro-benchmarks a single Linear / GEMM layer in forward + backward + sync."""
    # Ensure TF32 settings
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    # Create dummy input with controllable parameters
    x = generate_dummy_tensor(
        shape=(M, K),
        device=device,
        dtype=torch.float32,
        dist=args.input_dist,
        scale=args.input_scale,
        mean=args.input_mean,
        outlier_ratio=args.outlier_ratio,
        outlier_val=args.outlier_val,
    )

    # Initialize layer according to precision method
    if method == 'fp8':
        config = APAConfig.research_default(
            device=str(device),
            interval_telemetry=True,
            freeze_level=LEVEL_FP8,
            fp8_output_dtype='float16',
        )
        layer = APALinear(K, N, bias=True, config=config).to(device)
        apa_manager = APAManager(layer, config)
    elif method == 'apa':
        config = APAConfig.research_default(
            device=str(device),
            interval_telemetry=True,
            freeze_level=None,  # Dynamic escalation enabled
            fp8_output_dtype='float16',
        )
        layer = APALinear(K, N, bias=True, config=config).to(device)
        apa_manager = APAManager(layer, config)
    elif method == 'fp16':
        layer = nn.Linear(K, N, bias=True).to(device)
        apa_manager = None
    elif method == 'tf32':
        layer = nn.Linear(K, N, bias=True).to(device)
        apa_manager = None
    else:
        raise ValueError(f"Unknown method: {method}")

    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    opt_kwargs = {'lr': 1e-3, 'weight_decay': 0.01}
    if args.cuda_graph and device.type == 'cuda':
        opt_kwargs['capturable'] = True
    optimizer = torch.optim.AdamW(trainable_params, **opt_kwargs)

    # Reset VRAM
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Target loss function for single layer
    loss_fn = lambda out, target: out.sum()
    target_dummy = torch.zeros((), device=device)

    cuda_graph_runner = None
    if args.cuda_graph and device.type == 'cuda':
        autocast_dt = torch.float16 if method in ('fp8', 'apa', 'fp16') else None
        cuda_graph_runner = APACUDAGraphRunner(
            model=layer,
            optimizer=optimizer,
            sample_x=x,
            sample_y=target_dummy,
            apa_manager=apa_manager,
            loss_fn=loss_fn,
            autocast_dtype=autocast_dt,
            warmup_steps=args.warmup,
        )
    else:
        # Standard Warmup
        for _ in range(args.warmup):
            if apa_manager is not None:
                apa_manager.pre_step()
            optimizer.zero_grad(set_to_none=True)

            if method in ('fp8', 'apa'):
                out = layer(x)
                loss = out.sum()
                loss.backward()
                if apa_manager is not None:
                    apa_manager.post_backward_sync_and_eval()
            elif method == 'fp16':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = layer(x)
                    loss = out.sum()
                loss.backward()
            else:  # TF32
                out = layer(x)
                loss = out.sum()
                loss.backward()

            optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # Timed Benchmark
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

    if device.type == 'cuda':
        start_event.record()
    wall_start = time.perf_counter()

    for _ in range(args.steps):
        if cuda_graph_runner is not None:
            cuda_graph_runner.step(x, target_dummy)
        else:
            if apa_manager is not None:
                apa_manager.pre_step()
            optimizer.zero_grad(set_to_none=True)

            if method in ('fp8', 'apa'):
                out = layer(x)
                loss = out.sum()
                loss.backward()
                if apa_manager is not None:
                    apa_manager.post_backward_sync_and_eval()
            elif method == 'fp16':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = layer(x)
                    loss = out.sum()
                loss.backward()
            else:  # TF32
                out = layer(x)
                loss = out.sum()
                loss.backward()

            optimizer.step()

    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0

    latency_ms = elapsed_ms / args.steps
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == 'cuda' else 0.0

    # Calculate Total FLOPs for 1 Forward + 2 Backward GEMMs = 6 * M * K * N
    total_flops = 6.0 * M * K * N
    tflops = (total_flops / (latency_ms * 1e-3)) / 1e12

    # Final precision level report for APA
    final_level_str = "N/A"
    if method == 'apa' and apa_manager is not None:
        lvl_names = {0: "FP8", 1: "FP16", 2: "TF32"}
        final_level_str = lvl_names.get(layer.level, f"Level {layer.level}")

    del layer, optimizer, x
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'latency_ms': latency_ms,
        'tflops': tflops,
        'peak_vram_mb': peak_vram_mb,
        'final_level': final_level_str,
    }


def run_single_layer_benchmark(args):
    device = torch.device(args.device)
    prop = torch.cuda.get_device_properties(device) if device.type == 'cuda' else None
    gpu_name = prop.name if prop else "CPU"

    print("=" * 85)
    print(f"🔬 Controllable Layer Micro-Benchmark (GEMM Dimensions: M={args.M}, K={args.K}, N={args.N})")
    print("=" * 85)
    print(f"Device       : {args.device} ({gpu_name})")
    print(f"Dummy Input  : Shape=({args.M}, {args.K}) | Scale={args.input_scale} | Mean={args.input_mean} | Dist={args.input_dist}")
    print(f"Outliers     : {args.outlier_ratio * 100:.2f}% (Magnitude: {args.outlier_val})")
    print(f"Engine       : {'APA CUDA Graph Engine (0-freeze)' if args.cuda_graph else 'Eager PyTorch Execution'}")
    print(f"Steps        : {args.steps} (+ {args.warmup} warmup)")
    print("=" * 85)

    methods = [m.lower() for m in args.methods]
    if args.outlier_ratio > 0.0 and 'apa' not in methods:
        methods.append('apa')  # Also test dynamic escalation if outliers are injected

    results = {}
    for m in methods:
        print(f">>> Benchmarking Method: [{m.upper()}]...", end="", flush=True)
        res = benchmark_layer_single_precision(m, args.M, args.K, args.N, args, device)
        results[m] = res
        print(f" Done. ({res['latency_ms']:.3f} ms, {res['tflops']:.1f} TFLOPS)")

    # Display comparison table
    ref_lat = results.get('tf32', results.get('fp16', list(results.values())[0]))['latency_ms']
    fp16_lat = results.get('fp16', {}).get('latency_ms', None)

    print("\n" + "=" * 88)
    speedup_header = "Speedup vs TF32" if 'tf32' in results else "Speedup vs Baseline"
    print(f"{'Method':<8} | {'Latency':<10} | {'Throughput (TFLOPS)':<20} | {speedup_header:<16} | {'Speedup vs FP16':<16} | {'VRAM':<10}")
    print("-" * 88)
    for m in methods:
        res = results[m]
        sp_ref = ref_lat / res['latency_ms'] if res['latency_ms'] > 0 else 1.0
        sp_fp16 = (fp16_lat / res['latency_ms']) if (fp16_lat is not None and res['latency_ms'] > 0) else 1.0
        extra_note = f" (Escalated: {res['final_level']})" if m == 'apa' and res['final_level'] != 'N/A' else ""
        sp_fp16_str = f"{sp_fp16:>14.2f}x" if fp16_lat is not None else "           N/A"
        print(f"{m.upper() + extra_note:<8} | {res['latency_ms']:>7.3f} ms | {res['tflops']:>15.1f} TFLOPS | {sp_ref:>14.2f}x | {sp_fp16_str} | {res['peak_vram_mb']:>7.1f} MB")
    print("=" * 88)

    # Diagnostic Insight
    if 'fp8' in results and 'fp16' in results:
        fp8_lat = results['fp8']['latency_ms']
        if fp8_lat < fp16_lat:
            ratio = fp16_lat / fp8_lat
            print(f"💡 [INSIGHT]: Di dimensi ini (M={args.M}, K={args.K}, N={args.N}), FP8 LEBIH CEPAT {ratio:.2f}x dibanding FP16!")
            print("   Beban komputasi berada di zona COMPUTE-BOUND (Tensor Cores FP8 mendominasi).")
        else:
            diff_us = (fp8_lat - fp16_lat) * 1000.0
            print(f"💡 [INSIGHT]: FP16 masih lebih cepat tipis (+{diff_us:.0f} µs) dibanding FP8.")
            print("   Beban komputasi berada di zona MEMORY/LAUNCH-BOUND. Gunakan dimensi K/N lebih besar atau M lebih tinggi.")


def run_dimension_sweep(args):
    """Sweeps hidden dimensions to map the exact crossover curve where FP8 surpasses FP16."""
    device = torch.device(args.device)
    prop = torch.cuda.get_device_properties(device) if device.type == 'cuda' else None
    gpu_name = prop.name if prop else "CPU"

    dim_list = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    M = args.M

    print("=" * 95)
    print(f"📈 AUTOMATIC DIMENSION SWEEP (Fixed Tokens M={M} | Hidden Dim K=N: 256 -> 4096)")
    print("=" * 95)
    print(f"Device    : {args.device} ({gpu_name})")
    print(f"Engine    : {'APA CUDA Graph Engine (0-freeze)' if args.cuda_graph else 'Eager PyTorch'}")
    print(f"Searching : Titik temu (crossover point) di mana TFLOPS FP8 menyalip FP16...")
    print("=" * 95)

    print(f"\n{'Dim (K=N)':<10} | {'FP8 Latency':<12} | {'FP16 Latency':<13} | {'TF32 Latency':<13} | {'FP8 TFLOPS':<11} | {'FP16 TFLOPS':<12} | {'FP8 vs FP16':<12}")
    print("-" * 95)

    crossover_found = None
    sweep_records = []

    for dim in dim_list:
        res_fp8 = benchmark_layer_single_precision('fp8', M, dim, dim, args, device)
        res_fp16 = benchmark_layer_single_precision('fp16', M, dim, dim, args, device)
        res_tf32 = benchmark_layer_single_precision('tf32', M, dim, dim, args, device)

        ratio = res_fp16['latency_ms'] / res_fp8['latency_ms']
        symbol = "🏆 FP8 Menang" if ratio > 1.0 else "FP16 Menang"

        if ratio > 1.0 and crossover_found is None:
            crossover_found = dim

        sweep_records.append({
            'dim': dim,
            'fp8_latency_ms': res_fp8['latency_ms'],
            'fp16_latency_ms': res_fp16['latency_ms'],
            'tf32_latency_ms': res_tf32['latency_ms'],
            'fp8_tflops': res_fp8['tflops'],
            'fp16_tflops': res_fp16['tflops'],
            'tf32_tflops': res_tf32['tflops'],
            'speedup_fp8_vs_fp16': ratio,
            'speedup_fp8_vs_tf32': res_tf32['latency_ms'] / res_fp8['latency_ms'],
        })

        print(f"{dim:<10} | {res_fp8['latency_ms']:>8.3f} ms | {res_fp16['latency_ms']:>9.3f} ms | {res_tf32['latency_ms']:>9.3f} ms | {res_fp8['tflops']:>8.1f} TF | {res_fp16['tflops']:>9.1f} TF | {ratio:>5.2f}x ({symbol})")

    print("=" * 95)
    if crossover_found:
        print(f"🎯 [KESIMPULAN CROSSOVER]: Titik di mana FP8 mulai mengalahkan FP16 adalah pada Dimensi K=N >= {crossover_found}!")
    else:
        print("ℹ️ [INFO]: FP16 masih unggul pada rentang dimensi ini. Coba naikkan M dengan flag `--M 16384` untuk meningkatkan saturasi Tensor Cores.")

    # Save to CSV
    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        keys = sweep_records[0].keys()
        with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(sweep_records)
        print(f"💾 Data sweep berhasil disimpan ke: {args.save_csv}")

    # Plot figure
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
            dims = [r['dim'] for r in sweep_records]
            fp8_tf = [r['fp8_tflops'] for r in sweep_records]
            fp16_tf = [r['fp16_tflops'] for r in sweep_records]
            tf32_tf = [r['tf32_tflops'] for r in sweep_records]
            speedups = [r['speedup_fp8_vs_fp16'] for r in sweep_records]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            
            # Panel 1: Throughput (TFLOPS)
            ax1.plot(dims, fp8_tf, label='FP8 (Native Tensor Cores)', color='#10B981', linewidth=2.5, marker='o')
            ax1.plot(dims, fp16_tf, label='FP16 (AMP Tensor Cores)', color='#3B82F6', linewidth=2.2, marker='s')
            ax1.plot(dims, tf32_tf, label='TF32 (Baseline)', color='#9CA3AF', linewidth=1.8, linestyle='--', marker='^')
            if crossover_found:
                ax1.axvline(x=crossover_found, color='#EF4444', linestyle=':', label=f'Crossover K={crossover_found}')
            ax1.set_title(f'Compute Throughput (TFLOPS) vs Hidden Dim (M={M})', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Hidden Dimension (K=N)', fontweight='bold')
            ax1.set_ylabel('Effective TFLOPS', fontweight='bold')
            ax1.legend(frameon=True)
            ax1.grid(True, alpha=0.3)

            # Panel 2: Speedup Ratio (FP8 / FP16)
            ax2.plot(dims, speedups, label='Speedup (FP8 / FP16)', color='#8B5CF6', linewidth=2.5, marker='D')
            ax2.axhline(y=1.0, color='#EF4444', linestyle='--', label='1.0x Parity Line')
            if crossover_found:
                ax2.axvline(x=crossover_found, color='#EF4444', linestyle=':', label=f'Crossover ({crossover_found})')
            ax2.set_title('FP8 Relative Speedup over FP16', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Hidden Dimension (K=N)', fontweight='bold')
            ax2.set_ylabel('Speedup Factor', fontweight='bold')
            ax2.legend(frameon=True)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.plot, dpi=300)
            plt.close()
            print(f"📊 Grafik crossover berhasil disimpan ke: {args.plot}")
        except Exception as e:
            print(f"[WARN] Gagal membuat plot grafik: {e}")


def main():
    args = parse_args()

    if args.sweep_dim:
        run_dimension_sweep(args)
    else:
        run_single_layer_benchmark(args)


if __name__ == '__main__':
    main()
