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
 * Fused Scale, Clamp, Quantize to FP8 E5M2 with simultaneous Amax Tracking.
 */
at::Tensor fused_scale_clamp_quantize_cuda_e5m2(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);

