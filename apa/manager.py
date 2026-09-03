import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math

from .config import APAConfig, LEVEL_FP8, LEVEL_FP16, LEVEL_TF32, THRESHOLDS_MAX
from .layers import APALinear
from .telemetry import APAEventLogger, APAForensicLogger, track_telemetry_on_tensor

_LEVEL_NAME = {LEVEL_FP8: 'FP8', LEVEL_FP16: 'FP16', LEVEL_TF32: 'TF32'}
_KNOWN_ROLES = ('input_activation', 'weight', 'output', 'grad_output', 'grad_weight', 'grad_input')


class APAManager:
    """Manages adaptive precision escalation for a model with APALinear layers.

    Responsibilities:
    - Module discovery & backward-hook registration for non-APA leaves
    - Two-speed overflow/underflow detection (hard check every step,
      soft check every ``config.check_interval`` steps)
    - Master/working weight gradient synchronisation
    - Optional forensic logging (see ``APAConfig.enable_forensic_logging``)

    Forensic Logging:
        When ``config.enable_forensic_logging`` is True, the manager writes a
        detailed JSON snapshot to ``config.forensic_log_file`` for every
        escalation event.  The snapshot identifies which tensor role
        (input_activation / weight / output / grad_output / grad_weight /
        grad_input) had the highest amax at the time of escalation, the
        preceding module in forward-execution order, and optional argmax
        element index.

        **Performance warning**: forensic mode performs a CPU-GPU sync
        (``.item()``) on every tensor tracked by ``track_telemetry()`` while
        active.  Training throughput will be significantly lower.  This is an
        intentional design decision — forensic mode is optimised for analysis
        completeness, not speed.  Never enable it in production benchmarks.
    """

    def __init__(self, model: nn.Module, config: APAConfig = APAConfig()):
        self.model = model
        self.config = config
        self.step_count = 0

        self.apa_modules: Dict[str, APALinear] = {}
        self.other_modules: Dict[str, nn.Module] = {}

        self._global_nonfinite = torch.zeros(1, dtype=torch.int32, device=config.device)
        self.logger = APAEventLogger(config.log_file)

        # Forensic logging (opt-in) ----------------------------------------
        self.forensic_logger: Optional[APAForensicLogger] = None
        # Records the forward-execution order of APALinear modules in the
        # current step.  Reset at the start of each step in pre_step() so
        # that the list always reflects a single step's execution sequence.
        # Used to determine the "preceding module" for forensic snapshots.
        self._forward_execution_order: List[str] = []
        self._forward_hook_handles = []  # for potential cleanup

        self._register_modules_and_hooks()

        if config.enable_forensic_logging:
            if not config.forensic_log_file:
                if config.log_file is not None:
                    base = config.log_file.rsplit('.', 1)
                    config.forensic_log_file = (
                        base[0] + '_forensic.jsonl' if len(base) == 2
                        else config.log_file + '_forensic.jsonl'
                    )
                else:
                    config.forensic_log_file = 'apa_forensic.jsonl'
            self.forensic_logger = APAForensicLogger(config.forensic_log_file)
            self._register_forensic_forward_hooks()

    # ------------------------------------------------------------------
    # Module & hook registration
    # ------------------------------------------------------------------

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

    def _register_forensic_forward_hooks(self):
        """Register lightweight forward hooks on APALinear modules to record
        the order in which they execute.

        These hooks are only registered when forensic mode is active.  Each
        hook appends the module name to ``_forward_execution_order`` so that,
        at escalation time, we can identify the module that executed
        immediately before the one that overflowed.

        Limitation: this is a linear execution-order record, not a true
        dataflow graph.  In models with parallel branches (e.g. residual
        connections) the recorded predecessor may not be the true upstream
        data source.  See README section "Forensic Logging — Known Limitations"
        for details.
        """
        for name, module in self.apa_modules.items():
            # Capture `name` in closure via default argument
            def make_hook(module_name):
                def _fwd_hook(mod, inp, out):
                    self._forward_execution_order.append(module_name)
                return _fwd_hook

            handle = module.register_forward_hook(make_hook(name))
            self._forward_hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Parameter registry
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------

    def pre_step(self):
        """Reset per-step tracking state before each forward pass.

        Must be called at the start of every training step, before the forward
        pass.

        In addition to refreshing working copies and resetting nonfinite flags
        (existing behaviour), when forensic mode is active this method also
        clears ``_forward_execution_order`` so that the list always contains
        only the execution sequence of the *current* step — not an accumulation
        across multiple steps.  This is critical for correctness: if the reset
        were deferred to ``_do_full_evaluation()`` (which runs only every
        ``check_interval`` steps), the recorded order would mix sequences from
        multiple steps and ``preceding_module_in_forward_order`` in the
        forensic log would point to the wrong module.
        """
        with torch.no_grad():
            self._global_nonfinite.zero_()
            for module in self.apa_modules.values():
                module.refresh_working_copy()
                module.gpu_has_nonfinite.zero_()

            for module in self.other_modules.values():
                module.gpu_has_nonfinite.zero_()

        # Forensic: reset execution order and per-step role telemetry to avoid memory retention
        if self.config.enable_forensic_logging:
            self._forward_execution_order.clear()
            for module in self.apa_modules.values():
                module._forensic_role_amax.clear()
                module._forensic_last_shape.clear()
                module._forensic_role_stats.clear()
                module._forensic_role_argmax.clear()

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

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
        """Evaluate overflow/underflow metrics for all modules.

        Note on forensic mode: ``_forward_execution_order`` is NOT reset here.
        It is reset in ``pre_step()`` (called once per step, before the forward
        pass) so that the recorded sequence always belongs to a single step.
        Resetting here would be incorrect because ``_do_full_evaluation()`` is
        called only every ``check_interval`` steps (soft check) — not every
        step — so the order accumulated since the last reset would cover
        multiple steps.
        """
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
                    self._escalate_module(name, module, 'OVERFLOW', trigger_val)
                else:
                    module.ema_underflow_ratio = (self.config.ema_alpha * module.ema_underflow_ratio +
                                                  (1 - self.config.ema_alpha) * underflow_val)
                    if module.ema_underflow_ratio > self.config.theta_underflow:
                        self._escalate_module(name, module, 'SILENT_UNDERFLOW', module.ema_underflow_ratio)

            if self.config.telemetry_log_interval > 0 and (self.step_count % self.config.telemetry_log_interval == 0):
                periodic_data = {}
                for name, module in self.apa_modules.items():
                    periodic_data[name] = {
                        "amax": float(module.gpu_amax.item()),
                        "underflow_ratio": float(module.gpu_underflow_ratio.item()),
                        "ema_underflow": float(module.ema_underflow_ratio),
                        "level": module.level,
                        "threshold_max": THRESHOLDS_MAX.get(module.level, float('inf'))
                    }
                self.logger.log_periodic_telemetry(self.step_count, periodic_data)

            for name, module in self.apa_modules.items():
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
                        self.step_count, name, 'NON_PARAMETRIC_OVERFLOW',
                        min_apa_level, min_apa_level, trigger_val
                    )

                module.gpu_amax.zero_()
                module.gpu_has_nonfinite.zero_()

        if batch_has_overflow:
            self.logger.log_skip_batch(self.step_count, 'OVERFLOW_DETECTED', trigger_modules)

        return not batch_has_overflow

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def _escalate_module(self, name: str, module: APALinear, reason: str, trigger_value: float):
        """Escalate the precision level of a module.

        After logging the standard escalation event, if forensic logging is
        active, captures a detailed forensic snapshot that includes per-role
        tensor amax values, the culprit role, shape, and the preceding module
        in forward-execution order.

        Args:
            name: Module name (as returned by ``model.named_modules()``).
            module: The ``APALinear`` instance to escalate.
            reason: ``"OVERFLOW"`` or ``"SILENT_UNDERFLOW"``.
            trigger_value: The amax or EMA underflow ratio that triggered
                escalation.
        """
        old_level = module.level
        if old_level < LEVEL_TF32:
            module.level += 1
            module.ema_underflow_ratio = 0.0
            self.logger.log_escalation(
                self.step_count, name, reason,
                old_level, module.level, trigger_value
            )

            # Forensic snapshot — only when mode is active
            if self.config.enable_forensic_logging and self.forensic_logger is not None:
                self._capture_forensic_snapshot(name, module, reason, old_level, trigger_value)

    def _capture_forensic_snapshot(
        self,
        name: str,
        module: APALinear,
        reason: str,
        old_level: int,
        trigger_value: float,
    ) -> None:
        """Capture and write a forensic snapshot for an escalation event.

        Collects the per-role amax dict from ``module._forensic_role_amax``,
        determines which role was the culprit (highest amax), looks up the
        preceding module in the forward-execution order recorded this step,
        and writes one JSON line to ``forensic_log_file``.

        This method is called only when ``config.enable_forensic_logging``
        is True and escalation actually occurs — it is *not* called every step.

        Args:
            name: Name of the escalated module.
            module: The ``APALinear`` instance that was escalated.
            reason: ``"OVERFLOW"`` or ``"SILENT_UNDERFLOW"``.
            old_level: Precision level before escalation.
            trigger_value: The amax or EMA underflow ratio that triggered it.
        """
        # Unpack GPU scalar tensors to CPU Python floats at escalation time (deferred sync)
        role_amax = {}
        for role, val in module._forensic_role_amax.items():
            if isinstance(val, torch.Tensor):
                role_amax[role] = float(val.item())
            elif val is not None:
                role_amax[role] = float(val)
            else:
                role_amax[role] = None

        # Determine culprit: role with highest observed amax
        culprit_role = None
        culprit_amax = -1.0
        for role, val in role_amax.items():
            if val is not None and val > culprit_amax:
                culprit_amax = val
                culprit_role = role

        # Build per-role amax dict with all known roles (null for unseen ones)
        per_role_amax = {r: role_amax.get(r) for r in _KNOWN_ROLES}

        # Shape of culprit tensor
        tensor_shape = module._forensic_last_shape.get(culprit_role) if culprit_role else None

        # Per-role stats if enabled (convert scalar GPU tensors to float)
        per_role_stats = None
        if self.config.forensic_capture_tensor_stats and module._forensic_role_stats:
            per_role_stats = {}
            for r in _KNOWN_ROLES:
                st = module._forensic_role_stats.get(r)
                if isinstance(st, dict):
                    per_role_stats[r] = {
                        k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                        for k, v in st.items() if v is not None
                    }
                else:
                    per_role_stats[r] = None

        # Argmax index if enabled
        argmax_index = None
        if self.config.forensic_capture_argmax_index and culprit_role:
            raw_idx = module._forensic_role_argmax.get(culprit_role)
            if isinstance(raw_idx, torch.Tensor):
                argmax_index = int(raw_idx.item())
            elif raw_idx is not None:
                argmax_index = int(raw_idx)

        # Preceding module in forward-execution order
        preceding_module = None
        try:
            idx = self._forward_execution_order.index(name)
            if idx > 0:
                preceding_module = self._forward_execution_order[idx - 1]
        except ValueError:
            # Module was not recorded in execution order this step
            # (e.g. escalation triggered on a step where this module was not called)
            preceding_module = None

        # Dtype string at the time of escalation (old level before promotion)
        dtype_map_str = {LEVEL_FP8: 'float8_e4m3fn', LEVEL_FP16: 'float16', LEVEL_TF32: 'float32'}
        dtype_at_time = dtype_map_str.get(old_level, 'unknown')

        underflow_ratio = module.ema_underflow_ratio if reason == 'SILENT_UNDERFLOW' else None

        record = {
            'step': self.step_count,
            'module_name': name,
            'reason': reason,
            'level_before': _LEVEL_NAME.get(old_level, str(old_level)),
            'level_after': _LEVEL_NAME.get(module.level, str(module.level)),
            'culprit_tensor_role': culprit_role,
            'per_role_amax': per_role_amax,
            'amax_value': float(trigger_value),
            'threshold_at_time': THRESHOLDS_MAX.get(old_level, float('inf')),
            'tensor_shape': tensor_shape,
            'dtype_at_time': dtype_at_time,
            'preceding_module_in_forward_order': preceding_module,
            'underflow_ratio': underflow_ratio,
            'argmax_flat_index': argmax_index,
            'per_role_stats': per_role_stats,
        }

        self.forensic_logger.log_forensic_event(record)

    # ------------------------------------------------------------------
    # Main training-loop entry point
    # ------------------------------------------------------------------

    def post_backward_sync_and_eval(self) -> bool:
        self.step_count += 1
        self._sync_grads_to_master()

        has_hard_overflow = self._check_hard_overflow()
        periodic_due = (self.config.telemetry_log_interval > 0 and (self.step_count % self.config.telemetry_log_interval == 0))
        if has_hard_overflow or (self.step_count % self.config.check_interval == 0) or periodic_due:
            return self._do_full_evaluation()

        return True
