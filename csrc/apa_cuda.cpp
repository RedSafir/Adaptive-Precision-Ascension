#include "apa_cuda.h"

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
        "fused_quantize_dual_layout_e4m3",
        &fused_quantize_dual_layout_cuda_e4m3,
        "Dual-layout fused quantization for FP8 E4M3 (returns row-major and column-major raw [K, M])",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 448.0f,
        py::arg("amax_out") = py::none()
    );

    m.def(
        "fused_quantize_dual_layout_e5m2",
        &fused_quantize_dual_layout_cuda_e5m2,
        "Dual-layout fused quantization for FP8 E5M2 (returns row-major [M, N] and transposed row-major [N, M])",
        py::arg("x"),
        py::arg("scale"),
        py::arg("max_val") = 57344.0f,
        py::arg("amax_out") = py::none()
    );
}
