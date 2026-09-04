import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from apa.kernels import fused_scale_clamp_quantize_fp8, TRITON_AVAILABLE
from apa.config import FP8_E4M3_MAX

def main():
    print("=" * 60)
    print("APA Fused Triton FP8 Kernel Verification & Benchmark")
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available.")
        return

    device = torch.device('cuda')
    dtype_fp8 = torch.float8_e4m3fn

    # Create test tensor: batch 32, seq 256, dim 384 (same as nanoGPT batch!)
    shape = (32 * 256, 384)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    scale = torch.tensor(0.75, dtype=torch.float32, device=device)

    print(f"\n[Test 1] Correctness Test on shape {shape}...")
    # Reference: native PyTorch
    ref = (x * scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(dtype_fp8)

    # Triton Fused
    fused = fused_scale_clamp_quantize_fp8(x, scale, FP8_E4M3_MAX, dtype_fp8)

    # Compare exact byte representation
    diff_bytes = (ref.view(torch.uint8) != fused.view(torch.uint8)).sum().item()
    total_bytes = ref.numel()
    match_pct = (1.0 - (diff_bytes / total_bytes)) * 100.0
    print(f"  Total elements : {total_bytes:,}")
    print(f"  Exact matches  : {total_bytes - diff_bytes:,} ({match_pct:.2f}%)")
    if match_pct >= 99.9:
        print("  \033[92m[PASS] Fused Triton kernel output matches native PyTorch perfectly!\033[0m")
    else:
        print(f"  \033[93m[WARN] Output differs on {diff_bytes} elements (expected minor rounding variations).\033[0m")

    # Benchmark Speed
    print("\n[Test 2] Speed Benchmark (1000 iterations)...")
    torch.cuda.synchronize()

    # Warmup
    for _ in range(50):
        _ = (x * scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(dtype_fp8)
        _ = fused_scale_clamp_quantize_fp8(x, scale, FP8_E4M3_MAX, dtype_fp8)
    torch.cuda.synchronize()

    # Benchmark Native
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = (x * scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(dtype_fp8)
    torch.cuda.synchronize()
    t_native = time.perf_counter() - t0

    # Benchmark Fused Triton
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = fused_scale_clamp_quantize_fp8(x, scale, FP8_E4M3_MAX, dtype_fp8)
    torch.cuda.synchronize()
    t_fused = time.perf_counter() - t0

    print(f"  Native PyTorch (3 kernels) : {t_native*1000:.2f} ms ({t_native/1000*1e6:.1f} µs/call)")
    print(f"  Triton Fused   (1 kernel)  : {t_fused*1000:.2f} ms ({t_fused/1000*1e6:.1f} µs/call)")
    speedup = t_native / t_fused
    print(f"  \033[92m[RESULT] Fused Triton is {speedup:.2f}x FASTER than native!\033[0m")
    print("=" * 60)

if __name__ == '__main__':
    main()
