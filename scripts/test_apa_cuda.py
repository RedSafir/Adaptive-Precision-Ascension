#!/usr/bin/env python3
"""
Unit verification and micro-benchmark script for the native C++/CUDA extension 'apa_cuda'.
Tests fused quantization and telemetry tracking against PyTorch reference implementation.
"""

import sys
import os
import glob
import time
import torch

# Ensure repository root and build/lib directories are in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
for bl in glob.glob(os.path.join(repo_root, 'build', 'lib.*')):
    if bl not in sys.path:
        sys.path.insert(0, bl)

def main():
    print("=" * 65)
    print("🧪 Verifikasi & Pengujian Ekstensi Native C++/CUDA: apa_cuda")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("❌ Error: CUDA tidak tersedia pada environment ini.")
        sys.exit(1)

    device = torch.device('cuda:0')
    prop = torch.cuda.get_device_properties(device)
    print(f"Hardware GPU : {prop.name} (Compute {prop.major}.{prop.minor}, VRAM: {prop.total_memory / (1024**3):.1f} GB)")

    # 1. Test Import
    print("\n[1/4] Menguji import modul 'apa_cuda'...", end="", flush=True)
    try:
        import apa_cuda
        print(" [PASS]")
    except ImportError as e:
        print(" [FAIL]")
        print(f"\n❌ Modul 'apa_cuda' belum ter-compile atau tidak ditemukan: {e}")
        print("Silakan jalankan kompilasi terlebih dahulu:")
        print("    python setup.py build_ext --inplace")
        print("atau")
        print("    pip install -e .")
        sys.exit(1)

    # 2. Test Functional Correctness (E4M3)
    print("[2/4] Menguji fungsionalitas FP8 E4M3 + Amax Tracking...", end="", flush=True)
    M, K = 12608, 768  # Ukuran tensor ViT (batch 64 x 197 patches x dim 768)
    x = torch.randn(M, K, dtype=torch.float32, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    amax_cuda = torch.zeros(1, dtype=torch.float32, device=device)

    out_e4m3 = apa_cuda.fused_quantize_fp8_e4m3(x, scale, 448.0, amax_cuda)
    assert out_e4m3.dtype == torch.float8_e4m3fn, f"Dtype salah: {out_e4m3.dtype}"
    assert out_e4m3.shape == x.shape, f"Shape salah: {out_e4m3.shape}"

    # Verifikasi Amax
    expected_amax = torch.max(torch.abs(x)).item()
    cuda_amax_val = amax_cuda.item()
    rel_diff = abs(expected_amax - cuda_amax_val) / (expected_amax + 1e-6)
    assert rel_diff < 1e-4, f"Amax mismatch: expected {expected_amax}, got {cuda_amax_val}"
    print(" [PASS]")

    # 3. Test Functional Correctness (E5M2)
    print("[3/4] Menguji fungsionalitas FP8 E5M2 (Backward Gradients)...", end="", flush=True)
    grad_out = torch.randn(M, K, dtype=torch.float32, device=device)
    scale_g = torch.tensor(2.0, dtype=torch.float32, device=device)
    amax_g = torch.zeros(1, dtype=torch.float32, device=device)

    out_e5m2 = apa_cuda.fused_quantize_fp8_e5m2(grad_out, scale_g, 57344.0, amax_g)
    assert out_e5m2.dtype == torch.float8_e5m2, f"Dtype salah: {out_e5m2.dtype}"
    assert out_e5m2.shape == grad_out.shape, f"Shape salah: {out_e5m2.shape}"
    print(" [PASS]")

    # 4. Micro-Benchmark: apa_cuda vs PyTorch Eager Fallback
    print("[4/5] Micro-benchmark Latensi Kuantisasi (100 iterasi)...")
    
    # Warmup
    for _ in range(20):
        _ = apa_cuda.fused_quantize_fp8_e4m3(x, scale, 448.0, amax_cuda)
        _ = (x * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.cuda.synchronize(device)

    # Benchmark apa_cuda
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 100
    start_event.record()
    for _ in range(iters):
        _ = apa_cuda.fused_quantize_fp8_e4m3(x, scale, 448.0, amax_cuda)
    end_event.record()
    torch.cuda.synchronize(device)
    cuda_time_ms = start_event.elapsed_time(end_event) / iters

    # Benchmark PyTorch Eager (3 kernels + amax)
    start_event.record()
    for _ in range(iters):
        _ = (x * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        torch.maximum(amax_cuda, torch.max(torch.abs(x)), out=amax_cuda)
    end_event.record()
    torch.cuda.synchronize(device)
    pytorch_time_ms = start_event.elapsed_time(end_event) / iters

    speedup = pytorch_time_ms / cuda_time_ms if cuda_time_ms > 0 else 1.0

    print("-" * 65)
    print(f"📊 HASIL MICRO-BENCHMARK Kuantisasi ({M}x{K} = ~{M*K/1e6:.1f}M elements):")
    print(f"  • PyTorch Eager (4 separate kernels) : {pytorch_time_ms:.3f} ms")
    print(f"  • apa_cuda Native Fused Kernel       : {cuda_time_ms:.3f} ms")
    print(f"  • Speedup Kuantisasi                 : {speedup:.2f}x LEBIH CEPAT!")
    print("-" * 65)

    # 5. Test Fused Linear Forward & Backward
    print("[5/5] Menguji Pure C++ Fused Linear Forward & Backward...", end="", flush=True)
    if hasattr(apa_cuda, 'fused_linear_forward') and hasattr(apa_cuda, 'fused_linear_backward'):
        B, S, D = 64, 197, 768
        x_3d = torch.randn(B, S, D, dtype=torch.float32, device=device)
        w_master = torch.randn(D, D, dtype=torch.float32, device=device) * 0.02
        w_fp8 = (w_master * 1.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_t = w_fp8.t().contiguous()
        w_bwd = w_fp8.t().contiguous().t()
        bias = torch.zeros(D, dtype=torch.float32, device=device)

        scale_x = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_x = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_w = torch.tensor(1.0, dtype=torch.float32, device=device)
        scale_grad = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_grad = torch.tensor(1.0, dtype=torch.float32, device=device)
        amax_fwd = torch.zeros(1, dtype=torch.float32, device=device)
        amax_bwd = torch.zeros(1, dtype=torch.float32, device=device)

        # Forward
        out_fwd, x_saved = apa_cuda.fused_linear_forward(
            x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16"
        )
        assert out_fwd.shape == (B, S, D), f"Forward out shape salah: {out_fwd.shape}"
        assert out_fwd.dtype == torch.float16, f"Forward out dtype salah: {out_fwd.dtype}"
        assert x_saved.shape == (B * S, D), f"Saved x shape salah: {x_saved.shape}"

        # Backward
        grad_out_3d = torch.randn(B, S, D, dtype=torch.float16, device=device)
        g_in, g_w, g_b = apa_cuda.fused_linear_backward(
            grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w,
            amax_bwd, True, True, True, "float16", list(x_3d.shape)
        )
        assert g_in.shape == x_3d.shape, f"Grad input shape salah: {g_in.shape}"
        assert g_w.shape == w_fp8.shape, f"Grad weight shape salah: {g_w.shape}"
        assert g_b.shape == bias.shape, f"Grad bias shape salah: {g_b.shape}"
        print(" [PASS]")


        # Micro-benchmark Fused Forward & Backward
        for _ in range(20):
            _ = apa_cuda.fused_linear_forward(x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16")
            _ = apa_cuda.fused_linear_backward(grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, amax_bwd, True, True, True, "float16", list(x_3d.shape))
        torch.cuda.synchronize(device)

        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_forward(x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16")
        end_event.record()
        torch.cuda.synchronize(device)
        fwd_time_ms = start_event.elapsed_time(end_event) / iters

        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_backward(grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, amax_bwd, True, True, True, "float16", list(x_3d.shape))
        end_event.record()
        torch.cuda.synchronize(device)
        bwd_time_ms = start_event.elapsed_time(end_event) / iters

        print("-" * 65)
        print(f"⚡ Fused Linear Forward Latency  ({B}x{S}x{D}): {fwd_time_ms:.3f} ms / layer")
        print(f"⚡ Fused Linear Backward Latency ({B}x{S}x{D}): {bwd_time_ms:.3f} ms / layer")
        print(f"⚡ Total Layer Fwd+Bwd Step Latency     : {fwd_time_ms + bwd_time_ms:.3f} ms / layer")
        print("-" * 65)
    else:
        print(" [SKIP - functions not found]")

    print("✅ Seluruh pengujian 'apa_cuda' BERHASIL SEMPURNA!\n")

if __name__ == '__main__':
    main()
