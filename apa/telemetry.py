import torch
import json
import os
from datetime import datetime
from typing import Optional

def track_telemetry_on_tensor(tensor: torch.Tensor, gpu_amax: torch.Tensor, gpu_has_nonfinite: torch.Tensor) -> None:
    """Updates running-max amax and OR-accumulates nonfinite flag on GPU.
    
    Tensors are cast to float32 before reductions to support dtypes (like FP8)
    where max/abs reduction kernels are not implemented in PyTorch.
    """
    with torch.no_grad():
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            t = tensor.to(torch.float32)
        else:
            t = tensor
        val = torch.max(torch.abs(t)).to(torch.float32)
        torch.maximum(gpu_amax, val, out=gpu_amax)
        nonfinite = (~torch.isfinite(val)).to(torch.int32).view(1)
        torch.maximum(gpu_has_nonfinite, nonfinite, out=gpu_has_nonfinite)

def compute_underflow_ratio(grad_tensor: torch.Tensor, v_min_threshold: float) -> torch.Tensor:
    """Returns scalar ratio of non-zero elements below v_min."""
    with torch.no_grad():
        if grad_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            t = grad_tensor.to(torch.float32)
        else:
            t = grad_tensor
        abs_grad = torch.abs(t)
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
        # Events are logged silently to log_file when provided to keep stdout clean
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

    def log_periodic_telemetry(self, step: int, telemetry: dict):
        self._log_event({
            "event": "periodic_telemetry",
            "step": step,
            "telemetry": telemetry
        })


class APAForensicLogger:
    """Writes detailed forensic snapshots to a dedicated JSONL file.

    Each line in the forensic log corresponds to a single escalation event and
    contains per-role tensor amax values, the culprit tensor role, shape info,
    and the preceding module in forward-execution order.

    This logger is instantiated only when ``APAConfig.enable_forensic_logging``
    is True.  All I/O is append-only to ``forensic_log_file``.

    Note:
        Forensic log entries are written *only* at escalation time (once per
        escalation event), NOT every step.  The volume of this file is bounded
        by the total number of escalation events throughout training.
    """

    def __init__(self, forensic_log_file: Optional[str] = None) -> None:
        if not forensic_log_file:
            forensic_log_file = "apa_forensic.jsonl"
        self.forensic_log_file = forensic_log_file
        # Ensure parent directory exists up-front so log_forensic_event() never
        # fails silently on the first write.
        try:
            parent = os.path.dirname(os.path.abspath(self.forensic_log_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception as e:
            print(f"[APA Forensic] Warning: could not create log directory: {e}")

    def log_forensic_event(self, record: dict) -> None:
        """Append one forensic snapshot record to the forensic log file.

        ``record`` must be a JSON-serialisable dict.  A ``timestamp_utc`` field
        is added automatically before writing.

        Args:
            record: Dict containing forensic snapshot fields (see
                ``_capture_forensic_snapshot`` in ``APAManager``).
        """
        record = dict(record)  # shallow copy so caller's dict is unchanged
        record['timestamp_utc'] = datetime.utcnow().isoformat() + 'Z'
        json_str = json.dumps(record, default=str)
        print(f"[APA Forensic] {json_str}")
        try:
            with open(self.forensic_log_file, 'a', encoding='utf-8') as f:
                f.write(json_str + '\n')
        except Exception as e:
            print(f"[APA Forensic] Warning: failed to write forensic log: {e}")
