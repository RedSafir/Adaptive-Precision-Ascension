#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>

/**
 * Fused Scale, Clamp, Quantize to FP8 E4M3 with simultaneous Amax Tracking (Single Output).
 */
at::Tensor fused_scale_clamp_quantize_cuda_e4m3(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);

/**
 * Fused Scale, Clamp, Quantize to FP8 E5M2 with simultaneous Amax Tracking (Single Output).
 */
at::Tensor fused_scale_clamp_quantize_cuda_e5m2(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);

/**
 * Dual-Layout Fused Quantization for FP8 E4M3:
 * In a SINGLE GPU pass, produces BOTH:
 * 1. out_row: Standard row-major [M, K]
 * 2. out_col: Raw transposed matrix [K, M] whose .t() is column-major [M, K]
 * Eliminates all runtime .t().contiguous() transposition overhead.
 */
std::tuple<at::Tensor, at::Tensor> fused_quantize_dual_layout_cuda_e4m3(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);

/**
 * Dual-Layout Fused Quantization for FP8 E5M2 (Backward Pass):
 * In a SINGLE GPU pass, produces BOTH:
 * 1. out_row: Row-major grad_output [M, N] for grad_input GEMM
 * 2. out_t: Row-major contiguous transposed grad_output [N, M] for grad_weight GEMM
 * Eliminates all runtime .t().contiguous() transposition overhead in backward.
 */
std::tuple<at::Tensor, at::Tensor> fused_quantize_dual_layout_cuda_e5m2(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);
