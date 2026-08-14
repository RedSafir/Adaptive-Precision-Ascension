import torch
from dataclasses import dataclass
from typing import Optional

LEVEL_FP8 = 0
LEVEL_FP16 = 1
LEVEL_TF32 = 2

THRESHOLDS_MAX = {
    0: 0.90 * 448.0,
    1: 0.90 * 65504.0,
    2: float('inf')
}

THRESHOLDS_MIN = {
    0: 0.015625,
    1: 6.1035e-5,
    2: 0.0
}

try:
    _float8_e4m3fn = torch.float8_e4m3fn
except AttributeError:
    _float8_e4m3fn = None

DTYPE_MAP = {
    0: _float8_e4m3fn,
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
    
    def __post_init__(self):
        if DTYPE_MAP[0] is None:
            self.fp8_simulation_mode = True
        
    @classmethod
    def conservative(cls, **kwargs):
        config_kwargs = dict(gamma=0.80, theta_underflow=0.30, check_interval=1)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)
        
    @classmethod
    def aggressive(cls, **kwargs):
        config_kwargs = dict(gamma=0.95, theta_underflow=0.50, check_interval=8)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)
        
    @classmethod
    def research_default(cls, **kwargs):
        config_kwargs = dict(gamma=0.90, theta_underflow=0.40, check_interval=4)
        config_kwargs.update(kwargs)
        return cls(**config_kwargs)
