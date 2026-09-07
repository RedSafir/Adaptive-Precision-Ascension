#include "apa_cuda.h"

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
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(weight_fp8.is_cuda(), "weight_fp8 must be on CUDA");

    const int64_t K = x.size(-1);
    const int64_t numel = x.numel();
    const int64_t M = numel / K;
    const int64_t N = weight_fp8.size(0);

    // 1. Flatten input to 2D [M, K]
    at::Tensor x_2d = (x.dim() == 2 && x.is_contiguous()) ? x : x.reshape({M, K}).contiguous();

    // 2. Dual Fused scale, clamp, quantize with simultaneous amax tracking directly in CUDA
    // Produces: x_fp8 [M, K] (row-major) and x_t [K, M] (row-major transposed)
    auto dual_x = fused_scale_clamp_quantize_dual_cuda_e4m3(x_2d, scale_x, 448.0f, amax_x);
    at::Tensor x_fp8 = std::get<0>(dual_x);
    at::Tensor x_t = std::get<1>(dual_x); // [K, M] row-major -> x_t.t() is [M, K] with stride(0) == 1 (column-major)
    at::Tensor x_col = x_t.t();           // Zero-copy column-major view for backward dW!

    // 3. Select output precision
    at::ScalarType target_dtype = (out_dtype_str == "float16") ? at::kHalf : at::kFloat;

    // 4. Ensure weight_t is column-major for scaled_mm mat2 (stride(0) == 1)
    at::Tensor w_mat2 = weight_t;
    if (!w_mat2.defined() || !w_mat2.t().is_contiguous()) {
        w_mat2 = weight_fp8.t().contiguous();
    }

    // Cast bias to target_dtype if present
    c10::optional<at::Tensor> bias_fwd = c10::nullopt;
    if (bias.has_value() && bias.value().defined()) {
        bias_fwd = bias.value().to(target_dtype);
    }

    // 5. NVIDIA FP8 Tensor Core alignment: M, N, K must be multiples of 16
    const int64_t pad_m = (16 - (M % 16)) % 16;
    const int64_t pad_k = (16 - (K % 16)) % 16;
    const int64_t pad_n = (16 - (N % 16)) % 16;

    at::Tensor mm_x = x_fp8;
    at::Tensor mm_w = w_mat2;

    if (pad_m > 0 || pad_k > 0) {
        mm_x = at::zeros({M + pad_m, K + pad_k}, x_fp8.options());
        mm_x.slice(0, 0, M).slice(1, 0, K).copy_(x_fp8);
    }

    if (pad_k > 0 || pad_n > 0) {
        mm_w = at::zeros({K + pad_k, N + pad_n}, w_mat2.options());
        mm_w.slice(0, 0, K).slice(1, 0, N).copy_(w_mat2);
        mm_w = mm_w.t().contiguous().t(); // Ensure column-major
    }

    // 6. Native Scaled Matrix Multiplication (cuBLASLt)
    at::Tensor out_2d;
    bool bias_added = false;
    if (bias_fwd.has_value() && pad_n == 0) {
        try {
            auto res = at::_scaled_mm(mm_x, mm_w, inv_scale_x, inv_scale_w, bias_fwd, c10::nullopt, target_dtype, /*use_fast_accum=*/false);
            out_2d = extract_scaled_mm_out(res);
            bias_added = true;
        } catch (...) {
            auto res = at::_scaled_mm(mm_x, mm_w, inv_scale_x, inv_scale_w, c10::nullopt, c10::nullopt, target_dtype, /*use_fast_accum=*/false);
            out_2d = extract_scaled_mm_out(res);
        }
    } else {
        auto res = at::_scaled_mm(mm_x, mm_w, inv_scale_x, inv_scale_w, c10::nullopt, c10::nullopt, target_dtype, /*use_fast_accum=*/false);
        out_2d = extract_scaled_mm_out(res);
    }

    // 7. Remove padding if applied
    if (pad_m > 0 || pad_n > 0) {
        out_2d = out_2d.slice(0, 0, M).slice(1, 0, N);
    }

    if (bias_fwd.has_value() && !bias_added) {
        out_2d.add_(bias_fwd.value());
    }

    // 8. Reshape to original batch dimensions
    at::Tensor out;
    if (x.dim() > 2) {
        std::vector<int64_t> out_shape = x.sizes().vec();
        out_shape.back() = N;
        out = out_2d.view(out_shape);
    } else {
        out = out_2d;
    }

    return std::make_tuple(out, x_col);
}

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
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
    TORCH_CHECK(x_fp8.is_cuda(), "x_fp8 must be on CUDA");

    const int64_t N = grad_output.size(-1);
    const int64_t numel = grad_output.numel();
    const int64_t M = numel / N;

    // 1. Flatten grad_output to 2D [M, N]
    at::Tensor g_out_2d = (grad_output.dim() == 2 && grad_output.is_contiguous())
        ? grad_output : grad_output.reshape({M, N}).contiguous();

    // 2. Dual Fused scale, clamp, quantize with simultaneous amax tracking directly in CUDA (E5M2)
    // Produces g_fp8 [M, N] (row-major) AND g_fp8_t [N, M] (row-major!) in ONE pass with ZERO extra allocations!
    auto dual_g = fused_scale_clamp_quantize_dual_cuda_e5m2(g_out_2d, scale_grad, 57344.0f, amax_grad);
    at::Tensor g_fp8 = std::get<0>(dual_g);
    at::Tensor g_fp8_t = std::get<1>(dual_g); // [N, M] row-major for dW = g_t @ X

    at::ScalarType act_dtype = (target_act_dtype_str == "float16") ? at::kHalf : at::kFloat;

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    // 3. Compute grad_input: dX = dY @ W
    if (needs_grad_input) {
        const int64_t K = (x_fp8.dim() == 2) ? x_fp8.size(1) : x_fp8.size(-1);

        // Ensure weight_bwd is column-major: [N, K] with stride(0) == 1
        at::Tensor w_bwd_col = weight_bwd;
        if (!w_bwd_col.defined() || !w_bwd_col.t().is_contiguous()) {
            w_bwd_col = weight_bwd.t().contiguous().t();
        }

        const int64_t pad_m = (16 - (M % 16)) % 16;
        const int64_t pad_n = (16 - (N % 16)) % 16;
        const int64_t pad_k = (16 - (K % 16)) % 16;

        at::Tensor mm_g = g_fp8;
        at::Tensor mm_w = w_bwd_col;

        if (pad_m > 0 || pad_n > 0) {
            mm_g = at::zeros({M + pad_m, N + pad_n}, g_fp8.options());
            mm_g.slice(0, 0, M).slice(1, 0, N).copy_(g_fp8);
        }
        if (pad_n > 0 || pad_k > 0) {
            mm_w = at::zeros({N + pad_n, K + pad_k}, w_bwd_col.options());
            mm_w.slice(0, 0, N).slice(1, 0, K).copy_(w_bwd_col);
            mm_w = mm_w.t().contiguous().t(); // column-major
        }

        auto res_in = at::_scaled_mm(mm_g, mm_w, inv_scale_grad, inv_scale_w, c10::nullopt, c10::nullopt, act_dtype, /*use_fast_accum=*/false);
        at::Tensor grad_input_2d = extract_scaled_mm_out(res_in);
        if (pad_m > 0 || pad_k > 0) {
            grad_input_2d = grad_input_2d.slice(0, 0, M).slice(1, 0, K);
        }

        if (orig_x_shape.size() > 0) {
            grad_input = grad_input_2d.view(orig_x_shape);
        } else {
            grad_input = grad_input_2d;
        }
    }

    // 4. Compute grad_weight: dW = dY.t() @ X
    if (needs_grad_weight) {
        const int64_t K = (x_fp8.dim() == 2) ? x_fp8.size(1) : x_fp8.size(-1);

        // g_fp8_t is ALREADY [N, M] row-major!
        // x_fp8 is ALREADY [M, K] column-major (stride(0) == 1) saved from forward!
        // ZERO .contiguous() allocations!
        at::Tensor x_col = x_fp8;
        if (!x_col.t().is_contiguous()) {
            x_col = x_fp8.t().contiguous().t();
        }

        const int64_t pad_n = (16 - (N % 16)) % 16;
        const int64_t pad_m = (16 - (M % 16)) % 16;
        const int64_t pad_k = (16 - (K % 16)) % 16;

        at::Tensor mm_gt = g_fp8_t;
        at::Tensor mm_x = x_col;

        if (pad_n > 0 || pad_m > 0) {
            mm_gt = at::zeros({N + pad_n, M + pad_m}, g_fp8_t.options());
            mm_gt.slice(0, 0, N).slice(1, 0, M).copy_(g_fp8_t);
        }
        if (pad_m > 0 || pad_k > 0) {
            mm_x = at::zeros({M + pad_m, K + pad_k}, x_col.options());
            mm_x.slice(0, 0, M).slice(1, 0, K).copy_(x_col);
            mm_x = mm_x.t().contiguous().t(); // column-major
        }

        auto res_w = at::_scaled_mm(mm_gt, mm_x, inv_scale_grad, inv_scale_x, c10::nullopt, c10::nullopt, at::kFloat, /*use_fast_accum=*/false);
        at::Tensor grad_w_out = extract_scaled_mm_out(res_w);
        if (pad_n > 0 || pad_k > 0) {
            grad_w_out = grad_w_out.slice(0, 0, N).slice(1, 0, K);
        }
        grad_weight = grad_w_out;
    }

    // 5. Compute grad_bias: sum dY along batch/sequence dimension
    if (needs_grad_bias) {
        grad_bias = g_out_2d.sum(0, /*keepdim=*/false, at::kFloat);
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "APA CUDA Extension: Fused FP8 Quantization and Telemetry Acceleration for Blackwell / Hopper / Ada";

    m.def(
        "fused_quantize_fp8_e4m3",
        &fused_scale_clamp_quantize_cuda_e4m3,
        "Fused scale, clamp, and quantize to FP8 E4M3 with running amax tracking",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 448.0f,
        py::arg("amax_out") = py::none()
    );

    m.def(
        "fused_quantize_fp8_e5m2",
        &fused_scale_clamp_quantize_cuda_e5m2,
        "Fused scale, clamp, and quantize to FP8 E5M2 with running amax tracking",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 57344.0f,
        py::arg("amax_out") = py::none()
    );

    m.def(
        "fused_quantize_fp8_dual_e4m3",
        &fused_scale_clamp_quantize_dual_cuda_e4m3,
        "Dual fused scale, clamp, and quantize to FP8 E4M3 with simultaneous transpose and amax",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 448.0f,
        py::arg("amax_out") = py::none()
    );

    m.def(
        "fused_quantize_fp8_dual_e5m2",
        &fused_scale_clamp_quantize_dual_cuda_e5m2,
        "Dual fused scale, clamp, and quantize to FP8 E5M2 with simultaneous transpose and amax",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 57344.0f,
        py::arg("amax_out") = py::none()
    );

    m.def(
        "fused_linear_forward",
        &fused_linear_forward_cuda,
        "Pure Native C++ Fused Linear Forward (Quantize + cuBLASLt GEMM + Epilogue Bias)",
        py::arg("x"),
        py::arg("weight_fp8"),
        py::arg("weight_t"),
        py::arg("scale_x"),
        py::arg("inv_scale_x"),
        py::arg("inv_scale_w"),
        py::arg("bias") = py::none(),
        py::arg("amax_x") = py::none(),
        py::arg("out_dtype_str") = "float16"
    );

    m.def(
        "fused_linear_backward",
        &fused_linear_backward_cuda,
        "Pure Native C++ Fused Linear Backward (Quantize + dX GEMM + dW GEMM + Bias Sum)",
        py::arg("grad_output"),
        py::arg("x_fp8"),
        py::arg("weight_bwd"),
        py::arg("scale_grad"),
        py::arg("inv_scale_grad"),
        py::arg("inv_scale_x"),
        py::arg("inv_scale_w"),
        py::arg("amax_grad") = py::none(),
        py::arg("needs_grad_input") = true,
        py::arg("needs_grad_weight") = true,
        py::arg("needs_grad_bias") = false,
        py::arg("target_act_dtype_str") = "float16",
        py::arg("orig_x_shape") = std::vector<int64_t>()
    );
}
