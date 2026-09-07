#include "apa_cuda.h"
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// Helper: atomicMax for positive float values using IEEE 754 monotonic integer representation
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    if (val <= 0.0f || address == nullptr) return;
    int* address_as_int = reinterpret_cast<int*>(address);
    int val_as_int = __float_as_int(val);
    atomicMax(address_as_int, val_as_int);
}

// Helper: Convert float to FP8 (E4M3 or E5M2)
template <typename fp8_t>
__device__ __forceinline__ fp8_t float_to_fp8(float f);

template <>
__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8<__nv_fp8_e4m3>(float f) {
    return __nv_fp8_e4m3(f);
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 float_to_fp8<__nv_fp8_e5m2>(float f) {
    return __nv_fp8_e5m2(f);
}

// Unified CUDA Kernel: Fused Scale + Clamp + Quantize + Amax Tracking
template <typename scalar_t, typename fp8_t>
__global__ void fused_scale_clamp_quantize_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale_ptr,
    fp8_t* __restrict__ out,
    float* __restrict__ amax_out,
    int64_t numel,
    float max_val
) {
    const float scale = *scale_ptr;
    const int64_t tid = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    float thread_amax = 0.0f;

    for (int64_t i = tid; i < numel; i += stride) {
        float val = static_cast<float>(x[i]);
        float abs_val = fabsf(val);
        if (abs_val > thread_amax) {
            thread_amax = abs_val;
        }

        // Fused scale and clamp directly in registers
        float scaled = val * scale;
        float clamped = fminf(fmaxf(scaled, -max_val), max_val);
        out[i] = float_to_fp8<fp8_t>(clamped);
    }

    // Warp-level and Block-level Amax Reduction if amax_out is provided
    if (amax_out != nullptr) {
        // 1. Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_amax = fmaxf(thread_amax, __shfl_down_sync(0xffffffff, thread_amax, offset));
        }

        // 2. Shared memory inter-warp reduction
        __shared__ float s_amax[32];
        const int lane = threadIdx.x % 32;
        const int wid = threadIdx.x / 32;

        if (lane == 0) {
            s_amax[wid] = thread_amax;
        }
        __syncthreads();

        // 3. First warp reduces the block maximum
        if (wid == 0) {
            const int num_warps = (blockDim.x + 31) / 32;
            float block_amax = (lane < num_warps) ? s_amax[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                block_amax = fmaxf(block_amax, __shfl_down_sync(0xffffffff, block_amax, offset));
            }

            if (lane == 0 && block_amax > 0.0f) {
                atomicMaxFloat(amax_out, block_amax);
            }
        }
    }
}

template <typename fp8_t>
at::Tensor dispatch_fused_quantize(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out,
    c10::ScalarType target_dtype
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be on CUDA");
    TORCH_CHECK(scale.is_cuda(), "Scale tensor must be on CUDA");

    at::Tensor x_contig = x.contiguous();
    at::Tensor scale_contig = scale.contiguous().to(at::kFloat);
    const int64_t numel = x_contig.numel();

    at::Tensor out = at::empty(x_contig.sizes(), x_contig.options().dtype(target_dtype));
    if (numel == 0) {
        return out;
    }

    float* amax_ptr = nullptr;
    if (amax_out.has_value() && amax_out.value().defined()) {
        TORCH_CHECK(amax_out.value().is_cuda(), "amax_out must be on CUDA");
        TORCH_CHECK(amax_out.value().scalar_type() == at::kFloat, "amax_out must have float32 dtype");
        amax_ptr = amax_out.value().data_ptr<float>();
    }

    const int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((numel + threads - 1) / threads, 65535));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x_contig.scalar_type(),
        "fused_scale_clamp_quantize_kernel",
        ([&] {
            fused_scale_clamp_quantize_kernel<scalar_t, fp8_t><<<blocks, threads, 0, stream>>>(
                x_contig.data_ptr<scalar_t>(),
                scale_contig.data_ptr<float>(),
                reinterpret_cast<fp8_t*>(out.data_ptr()),
                amax_ptr,
                numel,
                max_val
            );
        })
    );

    return out;
}

} // anonymous namespace

at::Tensor fused_scale_clamp_quantize_cuda_e4m3(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
) {
    return dispatch_fused_quantize<__nv_fp8_e4m3>(x, scale, max_val, amax_out, at::kFloat8_e4m3fn);
}

at::Tensor fused_scale_clamp_quantize_cuda_e5m2(
    const at::Tensor& x,
    const at::Tensor& scale,
    float max_val,
    c10::optional<at::Tensor> amax_out
) {
    return dispatch_fused_quantize<__nv_fp8_e5m2>(x, scale, max_val, amax_out, at::kFloat8_e5m2);
}
