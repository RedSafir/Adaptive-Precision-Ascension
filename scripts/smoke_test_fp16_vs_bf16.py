#!/usr/bin/env python3
"""
Smoke Test & Benchmark: FP16 vs BF16 Latency & Throughput Comparison
Tested on NVIDIA Tensor Cores (Blackwell / Ada Lovelace)

Compares:
1. Pure GEMM (Linear Layer) across Small, Medium, Large matrix dimensions (Forward & Backward)
2. Conv2d Layer across typical vision feature map dimensions
3. Full Transformer Layer (Multi-Head Attention + MLP FFN + LayerNorm)
4. PyTorch AMP Mode (FP16 with GradScaler vs BF16 without GradScaler)
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def benchmark_cuda_fn(fn, warmup=25, iters=100):
    """Accurately measure GPU execution time using CUDA Events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms

def run_gemm_benchmark():
    print("\n" + "=" * 80)
    print("  [TEST 1] TENSOR CORE GEMM BENCHMARK (LINEAR / MATMUL)")
    print("=" * 80)

    configs = [
        {"name": "Small (YOLOS / ViT-Small scale)", "M": 1600, "K": 384, "N": 384},
        {"name": "Medium (ViT-Base / BERT scale)", "M": 2048, "K": 768, "N": 768},
        {"name": "Large (LLM FFN / Mega-GEMM)", "M": 4096, "K": 1024, "N": 2048},
        {"name": "Extra Large (4K x 4K x 4K)", "M": 4096, "K": 4096, "N": 4096},
    ]

    print(f"{'Matrix Scale (M x K x N)':<36} | {'FP16 Fwd':<10} | {'BF16 Fwd':<10} | {'Fwd Ratio':<10} | {'FP16 Bwd':<10} | {'BF16 Bwd':<10} | {'Bwd Ratio':<10}")
    print("-" * 105)

    results = []

    for cfg in configs:
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        flops = 2.0 * M * N * K

        # Setup FP16
        x_16 = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
        w_16 = torch.randn(N, K, device='cuda', dtype=torch.float16, requires_grad=True)
        g_16 = torch.randn(M, N, device='cuda', dtype=torch.float16)

        # Setup BF16
        x_bf = torch.randn(M, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        w_bf = torch.randn(N, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        g_bf = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

        # 1. Forward benchmark
        fwd_16 = lambda: F.linear(x_16, w_16)
        fwd_bf = lambda: F.linear(x_bf, w_bf)

        t_fwd_16 = benchmark_cuda_fn(fwd_16)
        t_fwd_bf = benchmark_cuda_fn(fwd_bf)
        ratio_fwd = t_fwd_bf / max(t_fwd_16, 1e-6)

        # 2. Backward benchmark
        out_16 = F.linear(x_16, w_16)
        bwd_16 = lambda: torch.autograd.backward(out_16, g_16, retain_graph=True)

        out_bf = F.linear(x_bf, w_bf)
        bwd_bf = lambda: torch.autograd.backward(out_bf, g_bf, retain_graph=True)

        t_bwd_16 = benchmark_cuda_fn(bwd_16)
        t_bwd_bf = benchmark_cuda_fn(bwd_bf)
        ratio_bwd = t_bwd_bf / max(t_bwd_16, 1e-6)

        shape_str = f"{M}x{K}x{N} ({cfg['name'].split()[0]})"
        print(f"{shape_str:<36} | {t_fwd_16*1000:6.1f} µs | {t_fwd_bf*1000:6.1f} µs | {ratio_fwd:6.2f}x    | {t_bwd_16*1000:6.1f} µs | {t_bwd_bf*1000:6.1f} µs | {ratio_bwd:6.2f}x")

        results.append({
            "name": cfg["name"],
            "shape": f"{M}x{K}x{N}",
            "t_fwd_16": t_fwd_16,
            "t_fwd_bf": t_fwd_bf,
            "t_bwd_16": t_bwd_16,
            "t_bwd_bf": t_bwd_bf
        })

    return results

def run_conv2d_benchmark():
    print("\n" + "=" * 80)
    print("  [TEST 2] TENSOR CORE CONV2D BENCHMARK")
    print("=" * 80)

    configs = [
        {"name": "Standard ResNet Conv (3x3)", "B": 16, "C_in": 64, "C_out": 128, "H": 64, "W": 64, "k": 3},
        {"name": "Bottleneck 1x1 Conv", "B": 16, "C_in": 256, "C_out": 64, "H": 64, "W": 64, "k": 1},
        {"name": "Deep Feature Conv (3x3)", "B": 16, "C_in": 256, "C_out": 512, "H": 32, "W": 32, "k": 3},
    ]

    print(f"{'Conv Configuration':<36} | {'FP16 Fwd':<10} | {'BF16 Fwd':<10} | {'Fwd Ratio':<10} | {'FP16 Bwd':<10} | {'BF16 Bwd':<10} | {'Bwd Ratio':<10}")
    print("-" * 105)

    results = []

    for cfg in configs:
        B, C_in, C_out, H, W, k = cfg["B"], cfg["C_in"], cfg["C_out"], cfg["H"], cfg["W"], cfg["k"]
        pad = k // 2

        # Setup FP16
        conv_16 = nn.Conv2d(C_in, C_out, kernel_size=k, padding=pad).cuda().half()
        x_16 = torch.randn(B, C_in, H, W, device='cuda', dtype=torch.float16, requires_grad=True)
        g_16 = torch.randn(B, C_out, H, W, device='cuda', dtype=torch.float16)

        # Setup BF16
        conv_bf = nn.Conv2d(C_in, C_out, kernel_size=k, padding=pad).cuda().bfloat16()
        x_bf = torch.randn(B, C_in, H, W, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        g_bf = torch.randn(B, C_out, H, W, device='cuda', dtype=torch.bfloat16)

        fwd_16 = lambda: conv_16(x_16)
        fwd_bf = lambda: conv_bf(x_bf)

        t_fwd_16 = benchmark_cuda_fn(fwd_16)
        t_fwd_bf = benchmark_cuda_fn(fwd_bf)
        ratio_fwd = t_fwd_bf / max(t_fwd_16, 1e-6)

        out_16 = conv_16(x_16)
        bwd_16 = lambda: torch.autograd.backward(out_16, g_16, retain_graph=True)

        out_bf = conv_bf(x_bf)
        bwd_bf = lambda: torch.autograd.backward(out_bf, g_bf, retain_graph=True)

        t_bwd_16 = benchmark_cuda_fn(bwd_16)
        t_bwd_bf = benchmark_cuda_fn(bwd_bf)
        ratio_bwd = t_bwd_bf / max(t_bwd_16, 1e-6)

        desc = f"B={B}, C={C_in}->{C_out}, {k}x{k}"
        print(f"{desc:<36} | {t_fwd_16*1000:6.1f} µs | {t_fwd_bf*1000:6.1f} µs | {ratio_fwd:6.2f}x    | {t_bwd_16*1000:6.1f} µs | {t_bwd_bf*1000:6.1f} µs | {ratio_bwd:6.2f}x")

        results.append({
            "name": cfg["name"],
            "desc": desc,
            "t_fwd_16": t_fwd_16,
            "t_fwd_bf": t_fwd_bf,
            "t_bwd_16": t_bwd_16,
            "t_bwd_bf": t_bwd_bf
        })

    return results

class TransformerBlock(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

def run_transformer_block_benchmark():
    print("\n" + "=" * 80)
    print("  [TEST 3] FULL TRANSFORMER BLOCK BENCHMARK (MHA + MLP + LAYERNORM)")
    print("=" * 80)

    B = 16
    Seq = 100  # Token count (as in YOLOS)
    Dim = 384  # YOLOS-Small dimension
    Heads = 6

    # Model in FP32
    block = TransformerBlock(dim=Dim, heads=Heads).cuda()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)

    x_fp32 = torch.randn(B, Seq, Dim, device='cuda')
    target = torch.randn(B, Seq, Dim, device='cuda')

    # A. Pure FP16 Model (model.half())
    block_16 = TransformerBlock(dim=Dim, heads=Heads).cuda().half()
    x_16 = x_fp32.clone().half()
    target_16 = target.clone().half()
    opt_16 = torch.optim.AdamW(block_16.parameters(), lr=1e-4)

    def step_pure_fp16():
        opt_16.zero_grad(set_to_none=True)
        out = block_16(x_16)
        loss = F.mse_loss(out, target_16)
        loss.backward()
        opt_16.step()

    # B. Pure BF16 Model (model.bfloat16())
    block_bf = TransformerBlock(dim=Dim, heads=Heads).cuda().bfloat16()
    x_bf = x_fp32.clone().bfloat16()
    target_bf = target.clone().bfloat16()
    opt_bf = torch.optim.AdamW(block_bf.parameters(), lr=1e-4)

    def step_pure_bf16():
        opt_bf.zero_grad(set_to_none=True)
        out = block_bf(x_bf)
        loss = F.mse_loss(out, target_bf)
        loss.backward()
        opt_bf.step()

    # C. AMP FP16 with GradScaler
    scaler_16 = torch.amp.GradScaler('cuda')
    block_amp_16 = TransformerBlock(dim=Dim, heads=Heads).cuda()
    opt_amp_16 = torch.optim.AdamW(block_amp_16.parameters(), lr=1e-4)

    def step_amp_fp16():
        opt_amp_16.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = block_amp_16(x_fp32)
            loss = F.mse_loss(out, target)
        scaler_16.scale(loss).backward()
        scaler_16.step(opt_amp_16)
        scaler_16.update()

    # D. AMP BF16 (No GradScaler needed in PyTorch because exponent matches FP32!)
    block_amp_bf = TransformerBlock(dim=Dim, heads=Heads).cuda()
    opt_amp_bf = torch.optim.AdamW(block_amp_bf.parameters(), lr=1e-4)

    def step_amp_bf16():
        opt_amp_bf.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = block_amp_bf(x_fp32)
            loss = F.mse_loss(out, target)
        loss.backward()
        opt_amp_bf.step()

    # Run benchmarks
    t_pure_16 = benchmark_cuda_fn(step_pure_fp16)
    t_pure_bf = benchmark_cuda_fn(step_pure_bf16)

    t_amp_16 = benchmark_cuda_fn(step_amp_fp16)
    t_amp_bf = benchmark_cuda_fn(step_amp_bf16)

    print(f"1. Pure Half Model (model.half() vs model.bfloat16()):")
    print(f"   • Pure FP16 Step Latency : {t_pure_16:.3f} ms / step  (Baseline 1.00x)")
    print(f"   • Pure BF16 Step Latency : {t_pure_bf:.3f} ms / step  ({t_pure_16 / max(t_pure_bf, 1e-6):.2f}x speedup relative)")
    print()
    print(f"2. PyTorch AMP Training (Full Forward + Backward + Optimizer Step):")
    print(f"   • AMP FP16 (+ GradScaler): {t_amp_16:.3f} ms / step  (Throughput: {(B*1000)/t_amp_16:.1f} fps)")
    print(f"   • AMP BF16 (No Scaler)   : {t_amp_bf:.3f} ms / step  (Throughput: {(B*1000)/t_amp_bf:.1f} fps)")
    print(f"   • Perbandingan Latensi   : BF16 {'LEBIH CEPAT' if t_amp_bf < t_amp_16 else 'SEBANDING / SEDIKIT LEBIH LAMBAT'} ({abs((t_amp_16 - t_amp_bf)/t_amp_16)*100:.1f}%)")

def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print("\n" + "#" * 80)
    print(f"  SMOKE TEST & BENCHMARK LATENSI: FP16 VS BF16")
    print(f"  Hardware GPU : {device_name} (Compute Capability: {capability[0]}.{capability[1]})")
    print(f"  PyTorch CUDA : {torch.version.cuda} | BF16 Supported: {torch.cuda.is_bf16_supported()}")
    print("#" * 80)

    # 1. Pure GEMM
    run_gemm_benchmark()

    # 2. Conv2d
    run_conv2d_benchmark()

    # 3. Transformer Block (Pure & AMP)
    run_transformer_block_benchmark()

    # 4. Summary & Architectural differences
    print("\n" + "=" * 80)
    print("  [RINGKASAN TEKNIS & PERBANDINGAN ARSITEKTURAL FP16 VS BF16]")
    print("=" * 80)
    print("  • Format Bit:")
    print("    - FP16  : 1 bit sign, 5 bit exponent, 10 bit mantissa (Presisi tinggi, rentang sempit)")
    print("    - BF16  : 1 bit sign, 8 bit exponent,  7 bit mantissa (Rentang luas setara FP32, presisi mantissa lebih rendah)")
    print("  • Batas Nilai Maksimum:")
    print("    - FP16  : max 65,504, min normal 6.1e-5 (Wajib GradScaler untuk mencegah overflow/underflow)")
    print("    - BF16  : max ~3.4e38, min normal ~1.17e-38 (Bebas GradScaler, imun terhadap overflow)")
    print("  • Throughput Hardware Tensor Core:")
    print("    - Pada arsitektur Ada Lovelace & Blackwell (RTX 40/50 series), Tensor Cores FP16 dan BF16")
    print("      memiliki throughput TFLOPs teoritis yang SAMA PERSIS.")
    print("    - Perbedaan latensi di dunia nyata umumnya dipengaruhi oleh overhead GradScaler (FP16 butuh scaler, BF16 tidak)")
    print("      serta fusi kernel cuBLAS/cuDNN.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
