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


/**
 * Helper to safely extract the output tensor from at::_scaled_mm across PyTorch versions,
 * handling both `at::Tensor` and `std::tuple<at::Tensor, at::Tensor>` return signatures.
 */
template <typename T>
inline at::Tensor extract_scaled_mm_out(T&& res) {
    if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
        return res;
    } else {
        return std::get<0>(res);
    }
}

/**
 * Fused Linear Forward in Pure C++ / CUDA.
 * Performs in a single C++ dispatch:
 * 1. Flattens x to 2D [M, K].
 * 2. Quantizes x to FP8 E4M3 with simultaneous amax tracking.
 * 3. Dispatches cuBLASLt GEMM (at::_scaled_mm) with fused bias epilogue.
 * 4. Reshapes out_2d to original batch shape.
 * Returns: (out, x_fp8_saved)
 */
std::tuple<at::Tensor, at::Tensor> fused_linear_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_fp8,
    const at::Tensor& weight_t,
    const at::Tensor& scale_x,
    const at::Tensor& inv_scale_x,
    const at::Tensor& inv_scale_w,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& amax_x,
    const std::string& out_dtype_str
);

/**
 * Fused Linear Backward in Pure C++ / CUDA.
 * Performs in a single C++ dispatch:
 * 1. Flattens grad_output to 2D [M, N].
 * 2. Quantizes grad_output to FP8 E5M2 with simultaneous amax tracking.
 * 3. Dispatches grad_input cuBLASLt GEMM (dY @ W) directly in C++.
 * 4. Transposes layouts and dispatches grad_weight cuBLASLt GEMM (dY.t @ X) in C++.
 * 5. Computes grad_bias via hardware sum reduction in C++.
 * Returns: (grad_input, grad_weight, grad_bias)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_linear_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& x_fp8,
    const at::Tensor& weight_bwd,
    const at::Tensor& scale_grad,
    const at::Tensor& inv_scale_grad,
    const at::Tensor& inv_scale_x,
    const at::Tensor& inv_scale_w,
    const c10::optional<at::Tensor>& amax_grad,
    bool needs_grad_input,
    bool needs_grad_weight,
    bool needs_grad_bias,
    const std::string& target_act_dtype_str,
    c10::IntArrayRef orig_x_shape
);


