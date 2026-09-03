import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import APAConfig, LEVEL_FP8, LEVEL_FP16, LEVEL_TF32, DTYPE_MAP, THRESHOLDS_MIN
from .telemetry import track_telemetry_on_tensor, compute_underflow_ratio

class APABoundaryCast(nn.Module):
    def __init__(self, parent_linear):
        super().__init__()
        # Use object.__setattr__ to prevent PyTorch from registering parent_linear
        # as a child submodule in self._modules, which causes an infinite recursion loop in model.to(device)
        object.__setattr__(self, 'parent_linear', parent_linear)

    def forward(self, x):
        working_dtype = self.parent_linear.working_dtype
        if x.dtype != working_dtype:
            return x.to(working_dtype)
        return x

def _call_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape

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
        else:
            b_padded = b

        try:
            res = torch._scaled_mm(a_padded, b_padded, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
            if isinstance(res, tuple):
                res = res[0]
            return res[:m, :n]
        except Exception:
            # Fallback to simulated FP8 matmul if hardware scaled_mm fails on padded tensor
            return a.to(torch.float32) @ b.to(torch.float32)
    else:
        try:
            res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
            if isinstance(res, tuple):
                return res[0]
            return res
        except Exception:
            return a.to(torch.float32) @ b.to(torch.float32)


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
        # Forensic buffers — all None when forensic mode is off (zero overhead).
        forensic_amax, forensic_shape, forensic_stats, forensic_argmax,
    ):
        """Forward pass for APALinear with optional per-role forensic tracking.

        When ``forensic_amax`` is None (default, forensic mode off) this
        function is identical to the original implementation.  No additional
        operations are performed and there is no overhead.

        When forensic mode is on, ``_update_forensic_role`` is called for
        each tracked tensor performing a CPU-GPU sync per tensor.  This
        intentionally slows training — forensic mode is for analysis only.
        """
        ctx.config = config
        ctx.level = level
        ctx.update_underflow_metric = update_underflow_metric
        ctx.forensic_amax = forensic_amax
        ctx.forensic_shape = forensic_shape
        ctx.forensic_stats = forensic_stats
        ctx.forensic_argmax = forensic_argmax
        ctx.capture_argmax_index = config.forensic_capture_argmax_index
        ctx.save_for_backward(x, weight, bias, gpu_amax, gpu_has_nonfinite)

        with torch.no_grad():
            # --- Core telemetry (always runs) ---
            track_telemetry_on_tensor(x, gpu_amax, gpu_has_nonfinite)
            track_telemetry_on_tensor(weight, gpu_amax, gpu_has_nonfinite)

            # --- Per-role forensic tracking (only when forensic mode is on) ---
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
            if config.fp8_simulation_mode or DTYPE_MAP[LEVEL_FP8] is None:
                x_fp8 = x.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else x
                w_fp8 = weight.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else weight

                result = F.linear(x_fp8.to(torch.float32), w_fp8.to(torch.float32), bias.to(torch.float32) if bias is not None else None)
            else:
                original_shape = x.shape
                x_2d = x.view(-1, x.shape[-1])

                out_2d = _call_scaled_mm(
                    x_2d,
                    weight.t(),
                    scale_a=torch.tensor(1.0, device=x.device),
                    scale_b=torch.tensor(1.0, device=weight.device),
                    out_dtype=torch.float32
                )

                result = out_2d.view(*original_shape[:-1], weight.shape[0])
                if bias is not None:
                    result += bias.to(torch.float32)
        elif level == LEVEL_FP16:
            result = F.linear(x, weight, bias)
        else:  # LEVEL_TF32
            result = F.linear(x, weight, bias)

        with torch.no_grad():
            # --- Core telemetry ---
            track_telemetry_on_tensor(result, gpu_amax, gpu_has_nonfinite)

            # --- Per-role forensic ---
            if forensic_amax is not None:
                _update_forensic_role(
                    result, 'output',
                    forensic_amax, forensic_shape, forensic_stats,
                    config.forensic_capture_argmax_index, forensic_argmax,
                )

        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, gpu_amax, gpu_has_nonfinite = ctx.saved_tensors
        config = ctx.config
        level = ctx.level
        update_underflow_metric = ctx.update_underflow_metric
        forensic_amax = ctx.forensic_amax
        forensic_shape = ctx.forensic_shape
        forensic_stats = ctx.forensic_stats
        forensic_argmax = ctx.forensic_argmax

        with torch.no_grad():
            # --- Core telemetry ---
            track_telemetry_on_tensor(grad_output, gpu_amax, gpu_has_nonfinite)

            # --- Per-role forensic ---
            if forensic_amax is not None:
                _update_forensic_role(
                    grad_output, 'grad_output',
                    forensic_amax, forensic_shape, forensic_stats,
                    config.forensic_capture_argmax_index, forensic_argmax,
                )

        grad_input = grad_weight = grad_bias = None

        if level == LEVEL_FP8:
            if config.fp8_simulation_mode or DTYPE_MAP[LEVEL_FP8] is None:
                g_out = grad_output.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else grad_output
                x_fp8 = x.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else x
                w_fp8 = weight.to(DTYPE_MAP[LEVEL_FP8]) if DTYPE_MAP[LEVEL_FP8] is not None else weight

                g_out_f32 = g_out.to(torch.float32)
                x_f32 = x_fp8.to(torch.float32)
                w_f32 = w_fp8.to(torch.float32)

                if ctx.needs_input_grad[0]:
                    grad_input = g_out_f32 @ w_f32
                if ctx.needs_input_grad[1]:
                    g_out_2d = g_out_f32.reshape(-1, g_out_f32.shape[-1])
                    x_2d = x_f32.reshape(-1, x_f32.shape[-1])
                    grad_weight = g_out_2d.t() @ x_2d
                if bias is not None and ctx.needs_input_grad[2]:
                    g_out_2d = g_out_f32.reshape(-1, g_out_f32.shape[-1])
                    grad_bias = g_out_2d.sum(dim=0)
            else:
                g_out_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(DTYPE_MAP[LEVEL_FP8])
                x_2d = x.reshape(-1, x.shape[-1]).to(DTYPE_MAP[LEVEL_FP8])
                w_fp8 = weight.to(DTYPE_MAP[LEVEL_FP8])

                if ctx.needs_input_grad[0]:
                    grad_input_2d = _call_scaled_mm(
                        g_out_2d,
                        w_fp8,
                        scale_a=torch.tensor(1.0, device=x.device),
                        scale_b=torch.tensor(1.0, device=weight.device),
                        out_dtype=torch.float32
                    )
                    grad_input = grad_input_2d.view_as(x)
                if ctx.needs_input_grad[1]:
                    grad_weight_2d = _call_scaled_mm(
                        g_out_2d.t(),
                        x_2d,
                        scale_a=torch.tensor(1.0, device=x.device),
                        scale_b=torch.tensor(1.0, device=weight.device),
                        out_dtype=torch.float32
                    )
                    grad_weight = grad_weight_2d.view_as(weight)
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32).sum(dim=0)
        else:
            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight
            if ctx.needs_input_grad[1]:
                g_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
                x_2d = x.reshape(-1, x.shape[-1])
                grad_weight = g_out_2d.t() @ x_2d
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)

        if grad_input is not None:
            grad_input = grad_input.to(torch.float32)
        if grad_weight is not None:
            grad_weight = grad_weight.to(torch.float32)
            if update_underflow_metric is not None:
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

        # Return None for all non-tensor args (config, level, gpu_amax, etc.)
        # and for the 4 forensic dict args — dicts are passed by reference so
        # they have already been mutated in-place above.
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None


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

        self.weight_work = None
        self.bias_work = None

        self.level = LEVEL_FP8

        self.register_buffer('gpu_amax', torch.zeros(1, dtype=torch.float32, device=config.device))
        self.register_buffer('gpu_underflow_ratio', torch.zeros(1, dtype=torch.float32, device=config.device))
        self.register_buffer('gpu_has_nonfinite', torch.zeros(1, dtype=torch.int32, device=config.device))

        self.ema_underflow_ratio = 0.0
        self.boundary_cast = APABoundaryCast(self)

        # ---------------------------------------------------------------------------
        # Forensic per-role buffers (CPU dicts, only populated when forensic ON)
        # ---------------------------------------------------------------------------
        # These dicts accumulate the max absolute value seen for each tensor role
        # (input_activation, weight, output, grad_output, grad_weight, grad_input)
        # during each step.  They are reset at the start of each step by
        # APAManager.pre_step() when forensic mode is active.
        #
        # NOTE: When forensic mode is OFF these dicts remain empty and are never
        # touched — zero overhead on the normal training path.
        #
        # NOTE: When forensic mode is ON, _update_forensic_role() calls .item()
        # on the GPU tensor synchronously for every tracked tensor.  This is a
        # deliberate design choice (simplicity > speed) — training will be
        # significantly slower.  See APAConfig.enable_forensic_logging docs.
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
    def current_threshold_min(self):
        return THRESHOLDS_MIN[self.level]

    def refresh_working_copy(self):
        with torch.no_grad():
            self.weight_work = self.weight_master.to(self.working_dtype).requires_grad_(self.weight_master.requires_grad)
            if self.bias_master is not None:
                self.bias_work = self.bias_master.to(self.working_dtype).requires_grad_(self.bias_master.requires_grad)

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
        if self.weight_work is None:
            self.refresh_working_copy()

        x_cast = self.boundary_cast(x)

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
            # Forensic args (None = forensic off, no overhead)
            f_amax, f_shape, f_stats, f_argmax,
        )

        return out
