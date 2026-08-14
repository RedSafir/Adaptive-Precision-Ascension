import torch
import torch.nn as nn
from typing import List, Dict
import math

from .config import APAConfig, LEVEL_FP8, LEVEL_FP16, LEVEL_TF32, THRESHOLDS_MAX
from .layers import APALinear
from .telemetry import APAEventLogger, track_telemetry_on_tensor

class APAManager:
    def __init__(self, model: nn.Module, config: APAConfig = APAConfig()):
        self.model = model
        self.config = config
        self.step_count = 0
        
        self.apa_modules: Dict[str, APALinear] = {}
        self.other_modules: Dict[str, nn.Module] = {}
        
        self._global_nonfinite = torch.zeros(1, dtype=torch.int32, device=config.device)
        self.logger = APAEventLogger(config.log_file)
        
        self._register_modules_and_hooks()

    def _register_modules_and_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, APALinear):
                self.apa_modules[name] = module
            elif len(list(module.children())) == 0:
                self.other_modules[name] = module
                module.register_buffer('gpu_amax', torch.zeros(1, dtype=torch.float32, device=self.config.device), persistent=False)
                module.register_buffer('gpu_has_nonfinite', torch.zeros(1, dtype=torch.int32, device=self.config.device), persistent=False)
                
                def hook_fn(mod, grad_input, grad_output):
                    for g in grad_output:
                        if g is not None:
                            track_telemetry_on_tensor(g, mod.gpu_amax, mod.gpu_has_nonfinite)
                
                module.register_full_backward_hook(hook_fn)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        param_ids = set()
        
        # 1. Collect master parameters from all APALinear modules
        for name, module in self.apa_modules.items():
            if module.weight_master.requires_grad:
                params.append(module.weight_master)
                param_ids.add(id(module.weight_master))
            if module.bias_master is not None and module.bias_master.requires_grad:
                params.append(module.bias_master)
                param_ids.add(id(module.bias_master))
                
        # 2. Collect trainable parameters from non-APA modules (LayerNorm, Embedding, etc.)
        for p in self.model.parameters():
            if p.requires_grad and id(p) not in param_ids:
                params.append(p)
                param_ids.add(id(p))
                
        # 3. Validation: Verify that all trainable parameters in the model are captured
        expected_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        assert len(params) == expected_count, \
            f"Parameter count mismatch: found {len(params)}, expected {expected_count}"
                    
        return params

    def pre_step(self):
        with torch.no_grad():
            self._global_nonfinite.zero_()
            for module in self.apa_modules.values():
                module.refresh_working_copy()
                module.gpu_has_nonfinite.zero_()
                
            for module in self.other_modules.values():
                module.gpu_has_nonfinite.zero_()

    def _check_hard_overflow(self) -> bool:
        with torch.no_grad():
            self._global_nonfinite.zero_()
            for module in self.apa_modules.values():
                self._global_nonfinite.bitwise_or_(module.gpu_has_nonfinite)
            for module in self.other_modules.values():
                self._global_nonfinite.bitwise_or_(module.gpu_has_nonfinite)
                
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(self._global_nonfinite, op=torch.distributed.ReduceOp.MAX)
                
            return self._global_nonfinite.item() > 0

    def _sync_grads_to_master(self):
        with torch.no_grad():
            for module in self.apa_modules.values():
                if module.weight_work is not None and module.weight_work.grad is not None:
                    if module.weight_master.grad is None:
                        module.weight_master.grad = module.weight_work.grad.to(torch.float32).clone()
                    else:
                        module.weight_master.grad.copy_(module.weight_work.grad.to(torch.float32))
                        
                if module.bias_work is not None and module.bias_work.grad is not None:
                    if module.bias_master.grad is None:
                        module.bias_master.grad = module.bias_work.grad.to(torch.float32).clone()
                    else:
                        module.bias_master.grad.copy_(module.bias_work.grad.to(torch.float32))

    def _do_full_evaluation(self) -> bool:
        batch_has_overflow = False
        trigger_modules = []
        
        with torch.no_grad():
            for name, module in self.apa_modules.items():
                amax_val = module.gpu_amax.item()
                underflow_val = module.gpu_underflow_ratio.item()
                has_nonfinite = module.gpu_has_nonfinite.item() > 0
                
                if has_nonfinite or math.isinf(amax_val) or math.isnan(amax_val) or amax_val > THRESHOLDS_MAX[module.level]:
                    batch_has_overflow = True
                    trigger_modules.append(name)
                    trigger_val = float('nan') if has_nonfinite and not math.isinf(amax_val) and not math.isnan(amax_val) and amax_val <= THRESHOLDS_MAX[module.level] else amax_val
                    self._escalate_module(name, module, "OVERFLOW", trigger_val)
                else:
                    module.ema_underflow_ratio = (self.config.ema_alpha * module.ema_underflow_ratio + 
                                                  (1 - self.config.ema_alpha) * underflow_val)
                    if module.ema_underflow_ratio > self.config.theta_underflow:
                        self._escalate_module(name, module, "SILENT_UNDERFLOW", module.ema_underflow_ratio)
                        
                module.gpu_amax.zero_()
                module.gpu_underflow_ratio.zero_()
                module.gpu_has_nonfinite.zero_()
                
            min_apa_level = min([m.level for m in self.apa_modules.values()]) if self.apa_modules else LEVEL_FP16
            
            for name, module in self.other_modules.items():
                amax_val = module.gpu_amax.item()
                has_nonfinite = module.gpu_has_nonfinite.item() > 0
                if has_nonfinite or math.isinf(amax_val) or math.isnan(amax_val) or amax_val > THRESHOLDS_MAX.get(min_apa_level, float('inf')):
                    batch_has_overflow = True
                    trigger_modules.append(name)
                    trigger_val = float('nan') if has_nonfinite and not math.isinf(amax_val) and not math.isnan(amax_val) else amax_val
                    self.logger.log_escalation(
                        self.step_count, name, "NON_PARAMETRIC_OVERFLOW", 
                        min_apa_level, min_apa_level, trigger_val
                    )
                
                module.gpu_amax.zero_()
                module.gpu_has_nonfinite.zero_()
                
        if batch_has_overflow:
            self.logger.log_skip_batch(self.step_count, "OVERFLOW_DETECTED", trigger_modules)
            
        return not batch_has_overflow

    def _escalate_module(self, name: str, module: APALinear, reason: str, trigger_value: float):
        old_level = module.level
        if old_level < LEVEL_TF32:
            module.level += 1
            module.ema_underflow_ratio = 0.0
            self.logger.log_escalation(
                self.step_count, name, reason, 
                old_level, module.level, trigger_value
            )

    def post_backward_sync_and_eval(self) -> bool:
        self.step_count += 1
        self._sync_grads_to_master()
        
        has_hard_overflow = self._check_hard_overflow()
        
        if has_hard_overflow or (self.step_count % self.config.check_interval == 0):
            return self._do_full_evaluation()
            
        return True
