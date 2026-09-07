#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>

/**
 * Fused Scale, Clamp, Quantize to FP8 E4M3 with simultaneous Amax Tracking.
 * 
 * Performs in a single GPU memory pass:
 * 1. abs_max = max(|x|)
 * 2. scaled = x * scale
 * 3. clamped = clamp(scaled, -max_val, max_val)
 * 4. out = static_cast<__nv_fp8_e4m3>(clamped)
 * 5. *gpu_amax = max(*gpu_amax, abs_max) [atomic]
 * 
 * @param x Input tensor (float32, float16, or bfloat16) on CUDA
 * @param scale Scalar tensor (float32) on CUDA
 * @param max_val Maximum saturation bound (e.g. 448.0 for FP8 E4M3)
 * @param amax_out Optional 1-element float32 tensor on CUDA to accumulate running amax
 * @return Tensor with dtype torch.float8_e4m3fn, same shape as x
 */
at::Tensor fused_scale_clamp_quantize_cuda_e4m3(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);

/**
 * Fused Scale, Clamp, Quantize to FP8 E5M2 with simultaneous Amax Tracking.
 * Used primarily for backward gradients.
 * 
 * @param x Input tensor (float32, float16, or bfloat16) on CUDA
 * @param scale Scalar tensor (float32) on CUDA
 * @param max_val Maximum saturation bound (e.g. 57344.0 for FP8 E5M2)
 * @param amax_out Optional 1-element float32 tensor on CUDA to accumulate running amax
 * @return Tensor with dtype torch.float8_e5m2, same shape as x
 */
at::Tensor fused_scale_clamp_quantize_cuda_e5m2(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
);
