import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import (
    APAConfig, LEVEL_FP8, LEVEL_FP16, LEVEL_TF32,
    DTYPE_MAP, DTYPE_BACKWARD_MAP,
    THRESHOLDS_MIN, THRESHOLDS_MAX,
    THRESHOLDS_MIN_BWD, THRESHOLDS_MAX_BWD,
    FP8_E4M3_MAX, FP8_E5M2_MAX
)
from .telemetry import track_telemetry_on_tensor, compute_underflow_ratio
from .kernels import fused_scale_clamp_quantize_fp8, APA_CUDA_AVAILABLE, apa_cuda


class APABoundaryCast(nn.Module):
    def __init__(self, parent_linear):
        super().__init__()
        # Use object.__setattr__ to prevent PyTorch from registering parent_linear
        # as a child submodule in self._modules, which causes an infinite recursion loop in model.to(device)
        object.__setattr__(self, 'parent_linear', parent_linear)

    def forward(self, x):
        # If dynamic scaling is active at FP8, keep x in full precision (float32)
        # so that APALinearFunction can accurately compute amax and scale factors before quantizing.
        if self.parent_linear.level == LEVEL_FP8 and self.parent_linear.config.enable_dynamic_scaling:
            return x.to(torch.float32) if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) else x
        working_dtype = self.parent_linear.working_dtype
        if x.dtype != working_dtype:
            return x.to(working_dtype)
        return x

def _safe_amax(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            t = tensor.to(torch.float32)
        else:
            t = tensor
        return torch.max(torch.abs(t)).float()

_SCALE_CACHE = {}

def _get_scale_one(device):
    dev_key = str(device)
    if dev_key not in _SCALE_CACHE:
        _SCALE_CACHE[dev_key] = torch.tensor(1.0, device=device, dtype=torch.float32)
    return _SCALE_CACHE[dev_key]

_warned_scaled_mm = False

def _call_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, bias: Optional[torch.Tensor] = None, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    global _warned_scaled_mm
    m, k = a.shape
    k2, n = b.shape

    # Zero-Copy Fast Path: M, K, N are 16-aligned, 'a' is row-major contiguous, and 'b' is column-major
    if (m % 16 == 0 and k % 16 == 0 and n % 16 == 0 and
            a.is_contiguous() and b.t().is_contiguous()):
        try:
            if bias is not None:
                try:
                    res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype)
                    return res[0] if isinstance(res, tuple) else res
                except Exception:
                    res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                    res = res[0] if isinstance(res, tuple) else res
                    return res + bias
            else:
                res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                return res[0] if isinstance(res, tuple) else res
        except Exception:
            pass  # Fall through to standard / padded path on error

    # mat1 (a) must be row-major (contiguous)
    if not a.is_contiguous():
        a = a.contiguous()
    # mat2 (b) must be column-major for torch._scaled_mm
    if not b.t().is_contiguous():
        b = b.t().contiguous().t()

    # NVIDIA FP8 Tensor Core alignment constraint: M, N, K must be multiples of 16
    pad_m = (16 - (m % 16)) % 16
    pad_k = (16 - (k % 16)) % 16
    pad_n = (16 - (n % 16)) % 16

    if pad_m > 0 or pad_k > 0 or pad_n > 0:
        if pad_m > 0 or pad_k > 0:
            a_padded = torch.zeros((m + pad_m, k + pad_k), dtype=a.dtype, device=a.device)
            a_padded[:m, :k] = a
        else:
            a_padded = a

        if pad_k > 0 or pad_n > 0:
            b_padded = torch.zeros((k2 + pad_k, n + pad_n), dtype=b.dtype, device=b.device)
            b_padded[:k2, :n] = b
            # mat2 must be strictly column-major (stride(0) == 1) for torch._scaled_mm
            b_padded = b_padded.t().contiguous().t()
        else:
            b_padded = b

        if not a_padded.is_contiguous():
            a_padded = a_padded.contiguous()
        if not b_padded.t().is_contiguous():
            b_padded = b_padded.t().contiguous().t()


        try:
            if bias is not None:
                try:
                    res = torch._scaled_mm(a_padded, b_padded, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype)
                    res = res[0] if isinstance(res, tuple) else res
                    return res[:m, :n]
                except Exception:
                    res = torch._scaled_mm(a_padded, b_padded, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                    res = res[0] if isinstance(res, tuple) else res
                    return res[:m, :n] + bias
            else:
                res = torch._scaled_mm(a_padded, b_padded, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                res = res[0] if isinstance(res, tuple) else res
                return res[:m, :n]
        except Exception as e:
            if not _warned_scaled_mm:
                print(f"[APA WARN] torch._scaled_mm failed on padded tensor: {e}. Falling back to float32 matmul.")
                _warned_scaled_mm = True
            out = (a.to(torch.float32) @ b.to(torch.float32)) * (scale_a * scale_b)
            if bias is not None:
                out = out + bias
            return out
    else:
        try:
            if bias is not None:
                try:
                    res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype)
                    return res[0] if isinstance(res, tuple) else res
                except Exception:
                    res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                    res = res[0] if isinstance(res, tuple) else res
                    return res + bias
            else:
                res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
                return res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if not _warned_scaled_mm:
                print(f"[APA WARN] torch._scaled_mm failed: {e}. Falling back to float32 matmul.")
                _warned_scaled_mm = True
            out = (a.to(torch.float32) @ b.to(torch.float32)) * (scale_a * scale_b)
            if bias is not None:
                out = out + bias
            return out


def _update_forensic_role(
    tensor: torch.Tensor,
    role: str,
    forensic_amax: Optional[dict],
    forensic_shape: Optional[dict],
    forensic_stats: Optional[dict],
    capture_argmax: bool,
    forensic_argmax: Optional[dict],
) -> None:
    """Update per-role forensic buffers (CPU dicts).

    Called only when forensic mode is active. Performs a CPU-GPU sync
    (``.item()``) per tensor — this is intentionally simple at the cost of
    training speed.  Do not use forensic mode in production training runs.

    Args:
        tensor: The tensor to inspect (any dtype; cast to float32 internally).
        role: Semantic role string, e.g. ``"input_activation"``, ``"weight"``.
        forensic_amax: Dict[role -> float] to update with max(abs(tensor)).
        forensic_shape: Dict[role -> list[int]] to update with tensor shape.
        forensic_stats: Dict[role -> dict] to update with mean/std, or None
            when ``forensic_capture_tensor_stats`` is False.
        capture_argmax: If True, also compute and store flat argmax index.
        forensic_argmax: Dict[role -> int] to update, or None.
    """
    if forensic_amax is None:
        return
    with torch.no_grad():
        t = tensor.detach()
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            t = t.to(torch.float32)
        abs_t = torch.abs(t)
        # Keep as GPU scalar tensor asynchronously during training loop.
        # Synchronous .item() calls on every tensor in every batch stall the GPU
        # and trigger cudaErrorLaunchTimeout from the Linux X11/display watchdog.
        # .item() is deferred to escalation time in _capture_forensic_snapshot.
        forensic_amax[role] = abs_t.max()
        forensic_shape[role] = list(tensor.shape)

        if forensic_stats is not None:
            t_f32 = t.float() if t.dtype != torch.float32 else t
            forensic_stats[role] = {
                'mean': t_f32.mean(),
                'std': t_f32.std() if t_f32.numel() > 1 else torch.tensor(0.0, device=t.device),
            }

        if capture_argmax and forensic_argmax is not None:
            forensic_argmax[role] = abs_t.flatten().argmax()


class APALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, weight, bias,
        config, level,
        gpu_amax, gpu_has_nonfinite,
        update_underflow_metric,
        working_bwd_dtype,
        scale_x, inv_scale_x, inv_scale_w, scale_grad, inv_scale_grad,
        # Forensic buffers — all None when forensic mode is off (zero overhead).
        forensic_amax, forensic_shape, forensic_stats, forensic_argmax,
        is_telemetry_step: bool = True,
        weight_t: Optional[torch.Tensor] = None,
        gpu_amax_x: Optional[torch.Tensor] = None,
        gpu_amax_grad: Optional[torch.Tensor] = None,
        weight_bwd: Optional[torch.Tensor] = None,
    ):
        """Forward pass for APALinear with NVIDIA-style Delayed Scaling."""
        ctx.config = config
        ctx.level = level
        ctx.update_underflow_metric = update_underflow_metric
        ctx.working_bwd_dtype = working_bwd_dtype
        ctx.forensic_amax = forensic_amax
        ctx.forensic_shape = forensic_shape
        ctx.forensic_stats = forensic_stats
        ctx.forensic_argmax = forensic_argmax
        ctx.capture_argmax_index = config.forensic_capture_argmax_index
        ctx.is_telemetry_step = is_telemetry_step
        ctx.x_shape = x.shape

        if is_telemetry_step:
            with torch.no_grad():
                # Avoid redundant telemetry when fused quantization already updates amax
                if level != LEVEL_FP8 or not config.enable_dynamic_scaling:
                    track_telemetry_on_tensor(x, gpu_amax, gpu_has_nonfinite)
                    if gpu_amax_x is not None:
                        track_telemetry_on_tensor(x, gpu_amax_x, gpu_has_nonfinite)

                if forensic_amax is not None:
                    _update_forensic_role(
                        x, 'input_activation',
                        forensic_amax, forensic_shape, forensic_stats,
                        config.forensic_capture_argmax_index, forensic_argmax,
                    )
                    _update_forensic_role(
                        weight, 'weight',
                        forensic_amax, forensic_shape, forensic_stats,
                        config.forensic_capture_argmax_index, forensic_argmax,
                    )

        if level == LEVEL_FP8:
            if not config.fp8_simulation_mode and APA_CUDA_AVAILABLE and hasattr(apa_cuda, 'fused_linear_forward') and x.is_cuda and not config.enable_forensic_logging:
                # Fast-Path: Pure Native C++ Fused Forward (Quantize + cuBLASLt GEMM + Epilogue Bias)
                out_dtype_str = getattr(config, 'fp8_output_dtype', 'float16')
                w_for_mm = weight_t if weight_t is not None else weight.t().contiguous()
                amax_tensor = gpu_amax if is_telemetry_step else None

                result, x_fp8 = apa_cuda.fused_linear_forward(
                    x, weight, w_for_mm, scale_x, inv_scale_x, inv_scale_w, bias, amax_tensor, out_dtype_str
                )

                w_bwd_saved = weight_bwd if weight_bwd is not None else weight.t().contiguous().t()
                ctx.save_for_backward(x_fp8, w_bwd_saved, bias, inv_scale_x, inv_scale_w, scale_grad, inv_scale_grad, gpu_amax, gpu_has_nonfinite, gpu_amax_grad)
            else:
                if config.enable_dynamic_scaling:
                    fwd_dtype = DTYPE_MAP[LEVEL_FP8] if DTYPE_MAP[LEVEL_FP8] is not None else torch.float32
                    x_fp8 = fused_scale_clamp_quantize_fp8(x, scale_x, FP8_E4M3_MAX, fwd_dtype, gpu_amax=(gpu_amax if is_telemetry_step else None))
                    w_fp8 = weight  # pre-scaled and pre-quantized in refresh_working_copy

                    s_a = inv_scale_x
                    s_b = inv_scale_w
                else:
                    x_fp8 = x.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else x
                    w_fp8 = weight.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else weight
                    s_a = _get_scale_one(x.device)
                    s_b = _get_scale_one(weight.device)

                if config.fp8_simulation_mode or DTYPE_MAP[LEVEL_FP8] is None:
                    ctx.save_for_backward(x_fp8, w_fp8, bias, s_a, s_b, scale_grad, inv_scale_grad, gpu_amax, gpu_has_nonfinite, gpu_amax_grad)
                    if config.enable_dynamic_scaling:
                        result = F.linear(x_fp8.float() * s_a, w_fp8.float() * s_b, bias.float() if bias is not None else None)
                    else:
                        result = F.linear(x_fp8.float(), w_fp8.float(), bias.float() if bias is not None else None)
                else:
                    original_shape = x.shape
                    x_2d = x_fp8.view(-1, x_fp8.shape[-1])
                    w_for_mm = weight_t if weight_t is not None else w_fp8.t().contiguous()

                    out_dtype = torch.float16 if getattr(config, 'fp8_output_dtype', 'float32') == 'float16' else torch.float32
                    bias_fwd = bias.to(out_dtype) if bias is not None else None
                    out_2d = _call_scaled_mm(
                        x_2d,
                        w_for_mm,
                        scale_a=s_a,
                        scale_b=s_b,
                        bias=bias_fwd,
                        out_dtype=out_dtype
                    )

                    result = out_2d.view(*original_shape[:-1], weight.shape[0])

                    # Pre-format tensors for zero-redundancy, zero-copy backward pass:
                    # 1. w_bwd_saved: column-major weight for grad_input (dX = dY @ W)
                    w_bwd_saved = weight_bwd if weight_bwd is not None else w_fp8.t().contiguous().t()
                    # 2. x_saved_bwd: activation saved for grad_weight (dX = dY.t() @ X -> X column-major)
                    x_saved_bwd = x_2d.t().contiguous().t()
                    ctx.save_for_backward(x_saved_bwd, w_bwd_saved, bias, s_a, s_b, scale_grad, inv_scale_grad, gpu_amax, gpu_has_nonfinite, gpu_amax_grad)
        elif level == LEVEL_FP16:
            dummy = _get_scale_one(x.device)
            ctx.save_for_backward(x, weight, bias, dummy, dummy, dummy, dummy, gpu_amax, gpu_has_nonfinite, gpu_amax_grad)
            result = F.linear(x, weight, bias)
        else:  # LEVEL_TF32
            dummy = _get_scale_one(x.device)
            ctx.save_for_backward(x, weight, bias, dummy, dummy, dummy, dummy, gpu_amax, gpu_has_nonfinite, gpu_amax_grad)
            result = F.linear(x, weight, bias)

        if is_telemetry_step:
            with torch.no_grad():
                track_telemetry_on_tensor(result, gpu_amax, gpu_has_nonfinite)
                if forensic_amax is not None:
                    _update_forensic_role(
                        result, 'output',
                        forensic_amax, forensic_shape, forensic_stats,
                        config.forensic_capture_argmax_index, forensic_argmax,
                    )

        return result

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x_saved, w_saved, bias, s_a, s_b, scale_grad, inv_scale_grad, gpu_amax, gpu_has_nonfinite, gpu_amax_grad = saved
        config = ctx.config
        level = ctx.level
        update_underflow_metric = ctx.update_underflow_metric
        working_bwd_dtype = ctx.working_bwd_dtype
        forensic_amax = ctx.forensic_amax
        forensic_shape = ctx.forensic_shape
        forensic_stats = ctx.forensic_stats
        forensic_argmax = ctx.forensic_argmax

        if ctx.is_telemetry_step:
            with torch.no_grad():
                # Avoid redundant telemetry when fused quantization already updates amax
                if level != LEVEL_FP8 or not config.enable_dynamic_scaling:
                    track_telemetry_on_tensor(grad_output, gpu_amax, gpu_has_nonfinite)
                    if gpu_amax_grad is not None:
                        track_telemetry_on_tensor(grad_output, gpu_amax_grad, gpu_has_nonfinite)
                if forensic_amax is not None:
                    _update_forensic_role(
                        grad_output, 'grad_output',
                        forensic_amax, forensic_shape, forensic_stats,
                        config.forensic_capture_argmax_index, forensic_argmax,
                    )

        grad_input = grad_weight = grad_bias = None

        if level == LEVEL_FP8:
            if not config.fp8_simulation_mode and APA_CUDA_AVAILABLE and hasattr(apa_cuda, 'fused_linear_backward') and grad_output.is_cuda and not config.enable_forensic_logging:
                # Fast-Path: Pure Native C++ Fused Backward (Quantize + dX GEMM + dW GEMM + Bias Sum)
                target_act_dtype_str = getattr(config, 'fp8_output_dtype', 'float16')
                amax_tensor = gpu_amax if ctx.is_telemetry_step else None
                grad_input, grad_weight, grad_bias = apa_cuda.fused_linear_backward(
                    grad_output,
                    x_saved,
                    w_saved,
                    scale_grad,
                    inv_scale_grad,
                    s_a,
                    s_b,
                    amax_tensor,
                    bool(ctx.needs_input_grad[0]),
                    bool(ctx.needs_input_grad[1]),
                    bool(bias is not None and ctx.needs_input_grad[2]),
                    target_act_dtype_str,
                    list(ctx.x_shape) if hasattr(ctx, 'x_shape') else []
                )
            else:
                bwd_dtype = working_bwd_dtype
                v_max_bwd = FP8_E5M2_MAX if (config.use_dual_fp8 and bwd_dtype == DTYPE_BACKWARD_MAP[0]) else FP8_E4M3_MAX

                if config.enable_dynamic_scaling:
                    target_bwd = bwd_dtype if bwd_dtype is not None else torch.float32
                    g_fp8 = fused_scale_clamp_quantize_fp8(grad_output, scale_grad, v_max_bwd, target_bwd, gpu_amax=(gpu_amax if ctx.is_telemetry_step else None))
                    s_g = inv_scale_grad
                    s_x = s_a
                    s_w = s_b
                else:
                    g_fp8 = grad_output.to(bwd_dtype if bwd_dtype is not None else torch.float32)
                    s_g = _get_scale_one(grad_output.device)
                    s_x = _get_scale_one(x_saved.device)
                    s_w = _get_scale_one(w_saved.device)

                if config.fp8_simulation_mode or DTYPE_MAP[LEVEL_FP8] is None:
                    g_out_f32 = g_fp8.float() * s_g if config.enable_dynamic_scaling else g_fp8.float()
                    x_f32 = x_saved.float() * s_x if config.enable_dynamic_scaling else x_saved.float()
                    w_f32 = w_saved.float() * s_w if config.enable_dynamic_scaling else w_saved.float()

                    if ctx.needs_input_grad[0]:
                        grad_input = (g_out_f32 @ w_f32).view(*ctx.x_shape) if hasattr(ctx, 'x_shape') else (g_out_f32 @ w_f32).view_as(x_saved)
                    if ctx.needs_input_grad[1]:
                        g_out_2d = g_out_f32.reshape(-1, g_out_f32.shape[-1])
                        x_2d = x_f32.reshape(-1, x_f32.shape[-1])
                        grad_weight = g_out_2d.t() @ x_2d
                    if bias is not None and ctx.needs_input_grad[2]:
                        g_out_2d = g_out_f32.reshape(-1, g_out_f32.shape[-1])
                        grad_bias = g_out_2d.sum(dim=0)
                else:
                    g_out_2d = g_fp8.reshape(-1, g_fp8.shape[-1])

                    out_dtype = torch.float16 if getattr(config, 'fp8_output_dtype', 'float32') == 'float16' else torch.float32
                    if ctx.needs_input_grad[0]:
                        # Zero-copy fast-path: g_out_2d is row-major, w_saved is pre-transposed column-major
                        grad_input_2d = _call_scaled_mm(
                            g_out_2d,
                            w_saved,
                            scale_a=s_g,
                            scale_b=s_w,
                            out_dtype=out_dtype
                        )
                        grad_input = grad_input_2d.view(*ctx.x_shape) if hasattr(ctx, 'x_shape') else grad_input_2d.view_as(x_saved)
                    if ctx.needs_input_grad[1]:
                        # Transpose to [N, M] row-major for scaled_mm
                        g_out_2d_t = g_out_2d.t().contiguous()
                        # grad_weight = g_out_2d_t [N, M] @ x_saved [M, K]
                        grad_weight = _call_scaled_mm(
                            g_out_2d_t,
                            x_saved,
                            scale_a=s_g,
                            scale_b=s_x,
                            out_dtype=torch.float32
                        )
                    if bias is not None and ctx.needs_input_grad[2]:
                        # Avoid 1.85 GB allocations per step by summing directly with dtype=float32
                        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0, dtype=torch.float32)
        else:
            g_out = grad_output.to(w_saved.dtype) if grad_output.dtype != w_saved.dtype else grad_output
            if ctx.needs_input_grad[0]:
                grad_input = g_out @ w_saved
            if ctx.needs_input_grad[1]:
                g_out_2d = g_out.reshape(-1, g_out.shape[-1])
                x_saved_cast = x_saved.to(w_saved.dtype) if x_saved.dtype != w_saved.dtype else x_saved
                x_2d = x_saved_cast.reshape(-1, x_saved_cast.shape[-1])
                grad_weight = g_out_2d.t() @ x_2d
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0, dtype=torch.float32)

        if grad_input is not None:
            target_act_dtype = torch.float16 if getattr(config, 'fp8_output_dtype', 'float32') == 'float16' else torch.float32
            grad_input = grad_input.to(target_act_dtype)
        if grad_weight is not None:
            grad_weight = grad_weight.to(torch.float32)
            if ctx.is_telemetry_step and update_underflow_metric is not None:
                update_underflow_metric(grad_weight)

            if forensic_amax is not None:
                _update_forensic_role(
                    grad_weight, 'grad_weight',
                    forensic_amax, forensic_shape, forensic_stats,
                    config.forensic_capture_argmax_index, forensic_argmax,
                )
        if grad_bias is not None:
            grad_bias = grad_bias.to(torch.float32)

        if grad_input is not None and forensic_amax is not None:
            _update_forensic_role(
                grad_input, 'grad_input',
                forensic_amax, forensic_shape, forensic_stats,
                config.forensic_capture_argmax_index, forensic_argmax,
            )

        # 23 outputs matching forward parameter count
        return (grad_input, grad_weight, grad_bias) + (None,) * 20


class APALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, config: APAConfig = APAConfig()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        self.weight_master = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32, device=config.device))
        if bias:
            self.bias_master = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=config.device))
        else:
            self.register_parameter('bias_master', None)

        object.__setattr__(self, 'weight_work', None)
        object.__setattr__(self, 'bias_work', None)

        self.level = LEVEL_FP8

        self.register_buffer('gpu_amax', torch.zeros(1, dtype=torch.float32, device=config.device))
        self.register_buffer('gpu_underflow_ratio', torch.zeros(1, dtype=torch.float32, device=config.device))
        self.register_buffer('gpu_has_nonfinite', torch.zeros(1, dtype=torch.int32, device=config.device))

        # Delayed Scaling state (NVIDIA TransformerEngine style)
        self.register_buffer('gpu_amax_x', torch.zeros(1, dtype=torch.float32, device=config.device), persistent=False)
        self.register_buffer('gpu_amax_grad', torch.zeros(1, dtype=torch.float32, device=config.device), persistent=False)
        self.register_buffer('scale_x', torch.tensor([1.0], dtype=torch.float32, device=config.device))
        self.register_buffer('scale_w', torch.tensor([1.0], dtype=torch.float32, device=config.device))
        self.register_buffer('scale_grad', torch.tensor([1.0], dtype=torch.float32, device=config.device))
        self.register_buffer('inv_scale_x', torch.tensor([1.0], dtype=torch.float32, device=config.device))
        self.register_buffer('inv_scale_w', torch.tensor([1.0], dtype=torch.float32, device=config.device))
        self.register_buffer('inv_scale_grad', torch.tensor([1.0], dtype=torch.float32, device=config.device))

        self.ema_underflow_ratio = 0.0
        self.boundary_cast = APABoundaryCast(self)
        self._weight_scale_initialized = False
        self.is_telemetry_step: bool = True
        self.weight_work_t = None
        self.weight_work_bwd = None

        # ---------------------------------------------------------------------------
        # Forensic per-role buffers (CPU dicts, only populated when forensic ON)
        # ---------------------------------------------------------------------------
        self._forensic_role_amax: dict = {}    # role -> float (max abs value)
        self._forensic_last_shape: dict = {}   # role -> list[int]
        self._forensic_role_stats: dict = {}   # role -> {mean, std} or empty
        self._forensic_role_argmax: dict = {}  # role -> int (flat index) or empty

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias_master is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_master)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_master, -bound, bound)

    @property
    def working_dtype(self):
        return DTYPE_MAP[self.level] if DTYPE_MAP[self.level] is not None else torch.float32

    @property
    def working_bwd_dtype(self):
        if self.level == LEVEL_FP8 and self.config.use_dual_fp8:
            return DTYPE_BACKWARD_MAP[0] if DTYPE_BACKWARD_MAP[0] is not None else torch.float32
        return DTYPE_BACKWARD_MAP.get(self.level, torch.float32)

    @property
    def current_threshold_min(self):
        if self.level == LEVEL_FP8 and self.config.use_dual_fp8:
            return THRESHOLDS_MIN_BWD[0]
        return THRESHOLDS_MIN[self.level]

    @property
    def current_threshold_max(self):
        return THRESHOLDS_MAX[self.level]

    def update_delayed_scales(self, update_weight: bool = False):
        """Update delayed scales asynchronously using tracked running amax (NVIDIA style, 100% on GPU)."""
        if not self.config.enable_dynamic_scaling or self.level != LEVEL_FP8:
            return
        with torch.no_grad():
            eps = 1e-4
            # 1. Update scale for activations x (100% GPU asynchronous)
            amax_x_src = torch.where(self.gpu_amax_x > 0, self.gpu_amax_x, self.gpu_amax)
            amax_val_x = torch.clamp(amax_x_src, min=eps)
            target_scale_x = torch.clamp(
                (self.config.scale_margin * FP8_E4M3_MAX) / amax_val_x,
                self.config.scale_min, self.config.scale_max
            )
            self.scale_x.copy_(torch.where(amax_x_src > 0, target_scale_x, self.scale_x))
            self.inv_scale_x.copy_(1.0 / self.scale_x)

            # 2. Update scale for backward gradient (100% GPU asynchronous)
            v_max_bwd = FP8_E5M2_MAX if self.config.use_dual_fp8 else FP8_E4M3_MAX
            amax_grad_src = torch.where(self.gpu_amax_grad > 0, self.gpu_amax_grad, self.gpu_amax)
            amax_val_grad = torch.clamp(amax_grad_src, min=eps)
            target_scale_grad = torch.clamp(
                (self.config.scale_margin * v_max_bwd) / amax_val_grad,
                self.config.scale_min, self.config.scale_max
            )
            self.scale_grad.copy_(torch.where(amax_grad_src > 0, target_scale_grad, self.scale_grad))
            self.inv_scale_grad.copy_(1.0 / self.scale_grad)

            # 3. Weight scales change very slowly: update periodically or initially
            if update_weight or not self._weight_scale_initialized:
                w_amax = _safe_amax(self.weight_master).clamp(min=eps)
                self.scale_w.copy_(
                    torch.clamp(
                        (self.config.scale_margin * FP8_E4M3_MAX) / w_amax,
                        self.config.scale_min, self.config.scale_max
                    )
                )
                self.inv_scale_w.copy_(1.0 / self.scale_w)
                self._weight_scale_initialized = True

    def refresh_working_copy(self):
        with torch.no_grad():
            w_detached = self.weight_master.detach()
            b_detached = self.bias_master.detach() if self.bias_master is not None else None

            if self.level == LEVEL_FP8:
                if self.config.enable_dynamic_scaling:
                    fwd_dtype = DTYPE_MAP[LEVEL_FP8] if DTYPE_MAP[LEVEL_FP8] is not None else torch.float32
                    w_scaled = fused_scale_clamp_quantize_fp8(w_detached, self.scale_w, FP8_E4M3_MAX, fwd_dtype)
                    object.__setattr__(self, 'weight_work', w_scaled.requires_grad_(self.weight_master.requires_grad))

                    # Pre-transpose to column-major layout for torch._scaled_mm (stride (1, in_features))
                    object.__setattr__(self, 'weight_work_t', w_scaled.t())
                    # Pre-transpose to column-major layout for backward pass grad_input (dX = dY @ W)
                    object.__setattr__(self, 'weight_work_bwd', w_scaled.t().contiguous().t())
                    if b_detached is not None:
                        object.__setattr__(self, 'bias_work', b_detached.to(torch.float32).requires_grad_(self.bias_master.requires_grad))
                else:
                    w_fp8 = w_detached.to(self.working_dtype)
                    object.__setattr__(self, 'weight_work', w_fp8.requires_grad_(self.weight_master.requires_grad))
                    object.__setattr__(self, 'weight_work_t', w_fp8.t())
                    object.__setattr__(self, 'weight_work_bwd', w_fp8.t().contiguous().t())
                    if b_detached is not None:
                        object.__setattr__(self, 'bias_work', b_detached.to(self.working_dtype).requires_grad_(self.bias_master.requires_grad))
            else:
                object.__setattr__(self, 'weight_work', w_detached.to(self.working_dtype).requires_grad_(self.weight_master.requires_grad))
                object.__setattr__(self, 'weight_work_t', None)
                object.__setattr__(self, 'weight_work_bwd', None)
                if b_detached is not None:
                    object.__setattr__(self, 'bias_work', b_detached.to(self.working_dtype).requires_grad_(self.bias_master.requires_grad))

    def track_telemetry(self, tensor: torch.Tensor, role: str = 'unspecified'):
        """Update the running-max amax and nonfinite flag for this module.

        Args:
            tensor: The tensor to track (any dtype).
            role: Semantic role of the tensor.  Used only when forensic logging
                is enabled (``config.enable_forensic_logging=True``).  When
                forensic mode is off the ``role`` argument is ignored entirely
                and this method behaves identically to the original
                implementation — no overhead is added.

                Valid roles: ``"input_activation"``, ``"weight"``,
                ``"output"``, ``"grad_output"``, ``"grad_weight"``,
                ``"grad_input"``, ``"unspecified"``.
        """
        track_telemetry_on_tensor(tensor, self.gpu_amax, self.gpu_has_nonfinite)

        # Per-role forensic tracking: only active when forensic mode is on.
        # Performs a CPU-GPU sync (.item()) — intentionally trades speed for
        # data completeness. See APAConfig.enable_forensic_logging warning.
        if self.config.enable_forensic_logging:
            _update_forensic_role(
                tensor, role,
                self._forensic_role_amax,
                self._forensic_last_shape,
                self._forensic_role_stats if self.config.forensic_capture_tensor_stats else None,
                self.config.forensic_capture_argmax_index,
                self._forensic_role_argmax,
            )

    def update_underflow_metric(self, grad: torch.Tensor):
        with torch.no_grad():
            ratio = compute_underflow_ratio(grad, self.current_threshold_min)
            torch.maximum(self.gpu_underflow_ratio, ratio, out=self.gpu_underflow_ratio)

    def forward(self, x):
        # Terminal Ceiling Fast-Path:
        # If this layer has reached LEVEL_TF32 (maximum ceiling), it can never escalate further.
        # Bypass telemetry reductions, custom autograd Function, working copies, and grad sync.
        if self.level == LEVEL_TF32 and not self.config.enable_forensic_logging:
            return F.linear(x, self.weight_master, self.bias_master)

        if self.weight_work is None:
            self.refresh_working_copy()

        # Boundary cast: cast input according to current precision level
        if self.level == LEVEL_FP8 and self.config.enable_dynamic_scaling:
            x_cast = x.to(torch.float32) if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) else x
        else:
            w_dtype = self.working_dtype
            x_cast = x.to(w_dtype) if x.dtype != w_dtype else x


        # Pass forensic dicts only when forensic mode is on; None otherwise
        # so APALinearFunction fast-paths around all forensic operations.
        if self.config.enable_forensic_logging:
            f_amax = self._forensic_role_amax
            f_shape = self._forensic_last_shape
            f_stats = self._forensic_role_stats if self.config.forensic_capture_tensor_stats else None
            f_argmax = self._forensic_role_argmax if self.config.forensic_capture_argmax_index else None
        else:
            f_amax = f_shape = f_stats = f_argmax = None

        out = APALinearFunction.apply(
            x_cast,
            self.weight_work,
            self.bias_work,
            self.config,
            self.level,
            self.gpu_amax,
            self.gpu_has_nonfinite,
            self.update_underflow_metric,
            self.working_bwd_dtype,
            self.scale_x,
            self.inv_scale_x,
            self.inv_scale_w,
            self.scale_grad,
            self.inv_scale_grad,
            # Forensic args (None = forensic off, no overhead)
            f_amax, f_shape, f_stats, f_argmax,
            self.is_telemetry_step,
            self.weight_work_t,
            self.gpu_amax_x,
            self.gpu_amax_grad,
            self.weight_work_bwd,
        )

        return out
