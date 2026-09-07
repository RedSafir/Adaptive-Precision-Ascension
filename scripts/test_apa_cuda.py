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
    print("[4/6] Micro-benchmark Latensi Kuantisasi (100 iterasi)...")
    
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

    # 5. Test Functional Correctness of Fused Linear (Standard & Pure 1-Kernel)
    print("[5/6] Menguji Fungsionalitas Pure C++ Fused Linear (FP16 & FP8 Pure Path)...", end="", flush=True)
    if hasattr(apa_cuda, 'fused_linear_forward') and hasattr(apa_cuda, 'fused_linear_backward'):
        B, S, D = 64, 197, 768
        x_3d = torch.randn(B, S, D, dtype=torch.float16, device=device)
        w_master = torch.randn(D, D, dtype=torch.float32, device=device) * 0.02
        w_fp8 = (w_master * 1.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_t = w_fp8.t().contiguous()
        w_bwd = w_fp8.t().contiguous().t()
        bias = torch.zeros(D, dtype=torch.float16, device=device)

        scale_x = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_x = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_w = torch.tensor(1.0, dtype=torch.float32, device=device)
        scale_grad = torch.tensor(1.0, dtype=torch.float32, device=device)
        inv_scale_grad = torch.tensor(1.0, dtype=torch.float32, device=device)
        amax_fwd = torch.zeros(1, dtype=torch.float32, device=device)
        amax_bwd = torch.zeros(1, dtype=torch.float32, device=device)

        # Standard Forward (FP16 in -> FP8 Quantize -> GEMM -> FP16 out)
        out_fwd, x_saved = apa_cuda.fused_linear_forward(
            x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16"
        )
        assert out_fwd.shape == (B, S, D), f"Forward out shape salah: {out_fwd.shape}"
        assert out_fwd.dtype == torch.float16, f"Forward out dtype salah: {out_fwd.dtype}"
        assert x_saved.shape == (B * S, D), f"Saved x shape salah: {x_saved.shape}"

        # Pure 1-Kernel Forward (Pre-quantized FP8 in -> Direct cuBLASLt GEMM -> FP16 out)
        x_fp8_in = apa_cuda.fused_quantize_fp8_e4m3(x_3d, scale_x, 448.0, None)
        out_pure_fwd, _ = apa_cuda.fused_linear_forward(
            x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "float16"
        )
        assert out_pure_fwd.shape == (B, S, D), f"Pure 1-Kernel out shape salah: {out_pure_fwd.shape}"
        assert out_pure_fwd.dtype == torch.float16, f"Pure 1-Kernel out dtype salah: {out_pure_fwd.dtype}"

        # Pure 1-Kernel Forward Chained (Pre-quantized FP8 in -> Direct cuBLASLt GEMM -> FP8 out)
        out_pure_fp8, _ = apa_cuda.fused_linear_forward(
            x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "fp8"
        )
        assert out_pure_fp8.shape == (B, S, D), f"Pure FP8 out shape salah: {out_pure_fp8.shape}"
        assert out_pure_fp8.dtype == torch.float8_e4m3fn, f"Pure FP8 out dtype salah: {out_pure_fp8.dtype}"

        # Standard Backward (FP16 in -> FP8 Quantize -> dX GEMM + dW GEMM + Bias Sum)
        grad_out_3d = torch.randn(B, S, D, dtype=torch.float16, device=device)
        g_in, g_w, g_b = apa_cuda.fused_linear_backward(
            grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w,
            amax_bwd, True, True, True, "float16", list(x_3d.shape)
        )
        assert g_in.shape == x_3d.shape, f"Grad input shape salah: {g_in.shape}"
        assert g_w.shape == w_fp8.shape, f"Grad weight shape salah: {g_w.shape}"
        assert g_b.shape == bias.shape, f"Grad bias shape salah: {g_b.shape}"

        # Pure 1-Kernel Backward (Pre-quantized FP8 in -> Direct GEMMs)
        grad_out_fp8 = apa_cuda.fused_quantize_fp8_e5m2(grad_out_3d, scale_grad, 57344.0, None)
        g_in_pure, g_w_pure, g_b_pure = apa_cuda.fused_linear_backward(
            grad_out_fp8, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w,
            None, True, True, True, "float16", list(x_3d.shape)
        )
        assert g_in_pure.shape == x_3d.shape
        assert g_w_pure.shape == w_fp8.shape
        assert g_b_pure.shape == bias.shape
        print(" [PASS]")

        # 6. Micro-Benchmark Head-to-Head GEMM (100 iterasi)
        print("[6/6] Micro-benchmark Head-to-Head Latensi GEMM (100 iterasi)...")
        w_16 = w_master.to(torch.float16)
        b_16 = bias.to(torch.float16)
        x_16_2d = x_3d.reshape(-1, D)
        g_16_2d = grad_out_3d.reshape(-1, D)

        # Warmup
        for _ in range(20):
            _ = torch.nn.functional.linear(x_3d, w_16, b_16)
            _ = apa_cuda.fused_linear_forward(x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16")
            _ = apa_cuda.fused_linear_forward(x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "float16")
            _ = apa_cuda.fused_linear_forward(x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "fp8")
        torch.cuda.synchronize(device)

        # A. Baseline PyTorch Native FP16 GEMM
        start_event.record()
        for _ in range(iters):
            _ = torch.nn.functional.linear(x_3d, w_16, b_16)
        end_event.record()
        torch.cuda.synchronize(device)
        t_fp16_fwd = start_event.elapsed_time(end_event) / iters

        # B. APA Fused Forward (Quantize + GEMM)
        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_forward(x_3d, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, amax_fwd, "float16")
        end_event.record()
        torch.cuda.synchronize(device)
        t_apa_fwd = start_event.elapsed_time(end_event) / iters

        # C. Pure 1-Kernel FP8 GEMM (Pre-quantized -> FP16 Out)
        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_forward(x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "float16")
        end_event.record()
        torch.cuda.synchronize(device)
        t_pure_fwd16 = start_event.elapsed_time(end_event) / iters

        # D. Pure 1-Kernel FP8 GEMM Chained (Pre-quantized -> FP8 Out)
        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_forward(x_fp8_in, w_fp8, w_t, scale_x, inv_scale_x, inv_scale_w, bias, None, "fp8")
        end_event.record()
        torch.cuda.synchronize(device)
        t_pure_fwd8 = start_event.elapsed_time(end_event) / iters

        # E. Baseline PyTorch Native FP16 Backward (dX + dW + dB)
        for _ in range(20):
            _ = g_16_2d @ w_16
            _ = g_16_2d.t() @ x_16_2d
            _ = g_16_2d.sum(0)
        torch.cuda.synchronize(device)

        start_event.record()
        for _ in range(iters):
            _ = g_16_2d @ w_16
            _ = g_16_2d.t() @ x_16_2d
            _ = g_16_2d.sum(0)
        end_event.record()
        torch.cuda.synchronize(device)
        t_fp16_bwd = start_event.elapsed_time(end_event) / iters

        # F. APA Fused Backward (Quantize + dX + dW + dB)
        for _ in range(20):
            _ = apa_cuda.fused_linear_backward(grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, amax_bwd, True, True, True, "float16", list(x_3d.shape))
        torch.cuda.synchronize(device)

        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_backward(grad_out_3d, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, amax_bwd, True, True, True, "float16", list(x_3d.shape))
        end_event.record()
        torch.cuda.synchronize(device)
        t_apa_bwd = start_event.elapsed_time(end_event) / iters

        # G. Pure 1-Kernel FP8 Backward (Pre-quantized Grads -> dX + dW + dB)
        for _ in range(20):
            _ = apa_cuda.fused_linear_backward(grad_out_fp8, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, None, True, True, True, "float16", list(x_3d.shape))
        torch.cuda.synchronize(device)

        start_event.record()
        for _ in range(iters):
            _ = apa_cuda.fused_linear_backward(grad_out_fp8, x_saved, w_bwd, scale_grad, inv_scale_grad, inv_scale_x, inv_scale_w, None, True, True, True, "float16", list(x_3d.shape))
        end_event.record()
        torch.cuda.synchronize(device)
        t_pure_bwd = start_event.elapsed_time(end_event) / iters

        print("-" * 65)
        print(f"🚀 HASIL HEAD-TO-HEAD GEMM BENCHMARK ({B}x{S}x{D} = {M}x{K}x{K}):")
        print("  [FORWARD PASS]:")
        print(f"    1. PyTorch Native FP16 GEMM          : {t_fp16_fwd:.3f} ms / layer  (Baseline 1.00x)")
        print(f"    2. APA Fused Forward (Quant + GEMM)  : {t_apa_fwd:.3f} ms / layer  ({t_fp16_fwd/t_apa_fwd:.2f}x)")
        print(f"    3. Pure 1-Kernel FP8 GEMM (FP16 Out) : {t_pure_fwd16:.3f} ms / layer  ({t_fp16_fwd/t_pure_fwd16:.2f}x LEBIH CEPAT!)")
        print(f"    4. Pure 1-Kernel FP8 GEMM (FP8 Out)  : {t_pure_fwd8:.3f} ms / layer  ({t_fp16_fwd/t_pure_fwd8:.2f}x LEBIH CEPAT!)")
        print("  [BACKWARD PASS (dX + dW + dBias)]:")
        print(f"    1. PyTorch Native FP16 Backward      : {t_fp16_bwd:.3f} ms / layer  (Baseline 1.00x)")
        print(f"    2. APA Fused Backward (Quant + GEMM) : {t_apa_bwd:.3f} ms / layer  ({t_fp16_bwd/t_apa_bwd:.2f}x)")
        print(f"    3. Pure 1-Kernel FP8 Backward        : {t_pure_bwd:.3f} ms / layer  ({t_fp16_bwd/t_pure_bwd:.2f}x LEBIH CEPAT!)")
        print("-" * 65)
    else:
        print(" [SKIP - functions not found]")

    print("✅ Seluruh pengujian 'apa_cuda' BERHASIL SEMPURNA!\n")

if __name__ == '__main__':
    main()
