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
    res = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
    if isinstance(res, tuple):
        return res[0]
    return res

class APALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, config, level, gpu_amax, gpu_has_nonfinite, update_underflow_metric):
        ctx.config = config
        ctx.level = level
        ctx.update_underflow_metric = update_underflow_metric
        ctx.save_for_backward(x, weight, bias, gpu_amax, gpu_has_nonfinite)
        
        with torch.no_grad():
            track_telemetry_on_tensor(x, gpu_amax, gpu_has_nonfinite)
            track_telemetry_on_tensor(weight, gpu_amax, gpu_has_nonfinite)
        
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
        else: # LEVEL_TF32
            result = F.linear(x, weight, bias)
            
        with torch.no_grad():
            track_telemetry_on_tensor(result, gpu_amax, gpu_has_nonfinite)
            
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, gpu_amax, gpu_has_nonfinite = ctx.saved_tensors
        config = ctx.config
        level = ctx.level
        update_underflow_metric = ctx.update_underflow_metric
        
        with torch.no_grad():
            track_telemetry_on_tensor(grad_output, gpu_amax, gpu_has_nonfinite)
            
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
        if grad_bias is not None:
            grad_bias = grad_bias.to(torch.float32)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

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

    def track_telemetry(self, tensor: torch.Tensor):
        track_telemetry_on_tensor(tensor, self.gpu_amax, self.gpu_has_nonfinite)

    def update_underflow_metric(self, grad: torch.Tensor):
        with torch.no_grad():
            ratio = compute_underflow_ratio(grad, self.current_threshold_min)
            torch.maximum(self.gpu_underflow_ratio, ratio, out=self.gpu_underflow_ratio)

    def forward(self, x):
        if self.weight_work is None:
            self.refresh_working_copy()
            
        x_cast = self.boundary_cast(x)
        
        out = APALinearFunction.apply(
            x_cast, 
            self.weight_work, 
            self.bias_work, 
            self.config, 
            self.level, 
            self.gpu_amax, 
            self.gpu_has_nonfinite,
            self.update_underflow_metric
        )
        
        return out
