import torch
from dataclasses import dataclass
from typing import Optional

LEVEL_FP8 = 0
LEVEL_FP16 = 1
LEVEL_TF32 = 2

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FP16_MAX = 65504.0

THRESHOLDS_MAX = {
    0: 0.90 * FP8_E4M3_MAX,
    1: 0.90 * FP16_MAX,
    2: float('inf')
}

# Backward thresholds (Trick A: E5M2 has wider dynamic range)
THRESHOLDS_MAX_BWD = {
    0: 0.90 * FP8_E5M2_MAX,
    1: 0.90 * FP16_MAX,
    2: float('inf')
}

THRESHOLDS_MIN = {
    0: 0.015625,      # 2^-6 (E4M3 normal limit)
    1: 6.1035e-5,     # 2^-14 (FP16 normal limit)
    2: 0.0
}

# Backward floor with E5M2 (Trick A: underflow floor is 256x deeper)
THRESHOLDS_MIN_BWD = {
    0: 6.1035e-5,     # 2^-14 (E5M2 normal limit, matching FP16)
    1: 6.1035e-5,     # 2^-14
    2: 0.0
}

try:
    _float8_e4m3fn = torch.float8_e4m3fn
except AttributeError:
    _float8_e4m3fn = None

try:
    _float8_e5m2 = torch.float8_e5m2
except AttributeError:
    _float8_e5m2 = None

# Forward dtype map (E4M3 -> FP16 -> TF32)
DTYPE_MAP = {
    0: _float8_e4m3fn,
    1: torch.float16,
    2: torch.float32
}

# Backward dtype map (Trick A: E5M2 -> FP16 -> TF32)
DTYPE_BACKWARD_MAP = {
    0: _float8_e5m2 if _float8_e5m2 is not None else _float8_e4m3fn,
    1: torch.float16,
    2: torch.float32
}

@dataclass
class APAConfig:
    gamma: float = 0.90
    theta_underflow: float = 0.40
    ema_alpha: float = 0.95
    check_interval: int = 1
    fp8_simulation_mode: bool = False
    log_file: Optional[str] = None
    device: str = 'cuda'

    # ---------------------------------------------------------------------------
    # Advanced FP8 Enhancements: Trick A (Dual FP8) & Trick B (Dynamic Scaling)
    # ---------------------------------------------------------------------------
    enable_dynamic_scaling: bool = True   # Trick B: Amax-based delayed dynamic scaling
    use_dual_fp8: bool = True             # Trick A: E4M3 for forward, E5M2 for backward
    scale_min: float = 1e-4
    scale_max: float = 65536.0
    scale_margin: float = 0.75
    interval_telemetry: bool = False  # When True, samples telemetry only on check_interval steps
    freeze_level: Optional[int] = None  # If set (0=FP8, 1=FP16, 2=TF32), fixes precision and disables escalation
    fp8_output_dtype: str = 'float16'   # 'float16' (default: fast FP16 output accumulator), 'bfloat16' / 'bf16' (BF16 output accumulator), or 'float32'

    def get_threshold_max(self, level: int) -> float:
        """Return dynamic overflow threshold for the given precision level using self.gamma."""
        if level == LEVEL_FP8:
            return float(self.gamma) * FP8_E4M3_MAX
        elif level == LEVEL_FP16:
            return float(self.gamma) * FP16_MAX
        return float('inf')

    # ---------------------------------------------------------------------------
    # Forensic Logging (opt-in, disabled by default)
    # ---------------------------------------------------------------------------
    # When enabled, APA records a detailed JSON snapshot for every escalation
    # event: which tensor role (input/weight/output/gradient) had the highest
    # amax, its shape, the preceding module in forward-execution order, etc.
    #
    # WARNING: Forensic mode performs a CPU-GPU sync (.item()) on *every* tensor
    # tracked by track_telemetry() while the mode is active. This significantly
    # slows training — by design. This mode is intended for post-training
    # analysis, NOT production training runs. Never enable in benchmarks.
    enable_forensic_logging: bool = False
    forensic_log_file: Optional[str] = None  # auto-derived if None when enabled
    forensic_capture_argmax_index: bool = False  # opt-in: flat index of extreme
    # element via torch.argmax — extra overhead per tensor, keep False unless
    # you need precise element-level debug info.
    forensic_capture_tensor_stats: bool = True   # capture mean/std alongside amax
    telemetry_log_interval: int = 0              # Periodic step interval to log per-layer
    # amax and underflow ratio time series (0 = disabled, e.g. 50 = every 50 steps)

    # ---------------------------------------------------------------------------
    # Frobenius Norm Telemetry & Instability Diagnostics (opt-in, disabled by default)
    # ---------------------------------------------------------------------------
    enable_frobenius_telemetry: bool = False   # Opt-in trigger for Frobenius norm diagnostic telemetry
    frobenius_check_interval: int = 4         # Evaluation step interval for Frobenius telemetry
    frobenius_log_file: Optional[str] = None  # Auto-derived if None when enabled

    def __post_init__(self):
        if DTYPE_MAP[0] is None:
            self.fp8_simulation_mode = True
        elif torch.cuda.is_available():
            try:
                dev = self.device if (isinstance(self.device, str) and 'cuda' in self.device) else 0
                if torch.cuda.get_device_capability(dev) < (8, 9):
                    self.fp8_simulation_mode = True
            except Exception:
                pass

        # Auto-derive forensic_log_file so callers never get a silent no-op.
        if self.enable_forensic_logging and self.forensic_log_file is None:
            if self.log_file is not None:
                base = self.log_file.rsplit('.', 1)
                self.forensic_log_file = (
                    base[0] + '_forensic.jsonl' if len(base) == 2
                    else self.log_file + '_forensic.jsonl'
                )
            else:
                self.forensic_log_file = 'apa_forensic.jsonl'

        # Auto-derive frobenius_log_file so callers never get a silent no-op.
        if self.enable_frobenius_telemetry and self.frobenius_log_file is None:
            if self.log_file is not None:
                base = self.log_file.rsplit('.', 1)
                self.frobenius_log_file = (
                    base[0] + '_frobenius.jsonl' if len(base) == 2
                    else self.log_file + '_frobenius.jsonl'
                )
            else:
                self.frobenius_log_file = 'apa_frobenius.jsonl'

    @classmethod
    def conservative(cls, **kwargs):
        config_kwargs = dict(gamma=0.80, theta_underflow=0.30, check_interval=1)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)

    @classmethod
    def conservative_default(cls, **kwargs):
        return cls.conservative(**kwargs)

    @classmethod
    def aggressive(cls, **kwargs):
        config_kwargs = dict(gamma=0.95, theta_underflow=0.50, check_interval=8)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)

    @classmethod
    def aggressive_default(cls, **kwargs):
        return cls.aggressive(**kwargs)

    @classmethod
    def research_default(cls, **kwargs):
        config_kwargs = dict(gamma=0.90, theta_underflow=0.40, check_interval=4)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)

    @classmethod
    def apa_bf(cls, **kwargs):
        """Variant APA_bf: FP8_bf mode (BF16 output accumulator) without delayed scaling, adaptive ascension."""
        config_kwargs = dict(
            fp8_output_dtype='bfloat16',
            enable_dynamic_scaling=False,
            gamma=0.90,
            theta_underflow=0.40,
            check_interval=4
        )
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)

    @classmethod
    def apa_fp8_bf(cls, **kwargs):
        """Variant APA_fp8_bf: Fixed FP8_bf mode (BF16 output accumulator) without delayed scaling."""
        config_kwargs = dict(
            fp8_output_dtype='bfloat16',
            enable_dynamic_scaling=False,
            freeze_level=LEVEL_FP8,
            check_interval=1
        )
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)

