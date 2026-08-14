import torch
import json
import os
from datetime import datetime
from typing import Optional

def track_telemetry_on_tensor(tensor: torch.Tensor, gpu_amax: torch.Tensor, gpu_has_nonfinite: torch.Tensor) -> None:
    """Updates running-max amax and OR-accumulates nonfinite flag on GPU."""
    with torch.no_grad():
        val = torch.max(torch.abs(tensor)).to(torch.float32)
        torch.maximum(gpu_amax, val, out=gpu_amax)
        nonfinite = (~torch.isfinite(val)).to(torch.int32).view(1)
        torch.maximum(gpu_has_nonfinite, nonfinite, out=gpu_has_nonfinite)

def compute_underflow_ratio(grad_tensor: torch.Tensor, v_min_threshold: float) -> torch.Tensor:
    """Returns scalar ratio of non-zero elements below v_min."""
    with torch.no_grad():
        abs_grad = torch.abs(grad_tensor)
        non_zero_mask = abs_grad > 0
        non_zero_count = non_zero_mask.sum().to(torch.float32)
        
        if non_zero_count.item() == 0:
            return torch.tensor(0.0, device=grad_tensor.device, dtype=torch.float32)
            
        underflow_mask = (abs_grad < v_min_threshold) & non_zero_mask
        underflow_count = underflow_mask.sum().to(torch.float32)
        return underflow_count / non_zero_count

class APAEventLogger:
    def __init__(self, log_file: Optional[str]):
        self.log_file = log_file

    def _log_event(self, event_data: dict):
        event_data['timestamp'] = datetime.utcnow().isoformat() + "Z"
        json_str = json.dumps(event_data)
        print(f"APA Event: {json_str}")
        
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
                with open(self.log_file, 'a') as f:
                    f.write(json_str + '\n')
            except Exception as e:
                print(f"Failed to write to APA log file {self.log_file}: {e}")

    def log_escalation(self, step: int, module_name: str, reason: str, old_level: int, new_level: int, trigger_value: float):
        self._log_event({
            "event": "escalation",
            "step": step,
            "module": module_name,
            "reason": reason,
            "old_level": old_level,
            "new_level": new_level,
            "trigger_value": float(trigger_value)
        })

    def log_skip_batch(self, step: int, reason: str, trigger_modules: list):
        self._log_event({
            "event": "skip_batch",
            "step": step,
            "reason": reason,
            "trigger_modules": trigger_modules
        })
