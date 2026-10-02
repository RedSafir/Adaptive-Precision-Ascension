"""
Frobenius Norm Diagnostic Telemetry for Adaptive Precision Architecture (APA).

Provides real-time and periodic telemetry to detect three primary modes of numerical instability:
1. Sebaran Varians Bobot Inisialisasi (Weight Variance Disparity)
2. Akumulasi Perambatan Mundur (Backward Pass Exponential Decay/Explosion / Gradient Slope)
3. Keberadaan Lapisan Anomali (Layer Outlier / Bottleneck)
"""

import os
import math
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

from ..config import APAConfig
from ..layers import APAModule, APALinear, APAConv2d


class FrobeniusTelemetry:
    """Diagnostic analyzer utilizing Frobenius matrix norms for deep learning stability.

    Computes:
    - Weight & Gradient Frobenius Norm: ||W||_F and ||∇W||_F
    - Weight & Gradient Scale Disparity: max(||∇W||_F) / (min(||∇W||_F) + 1e-12)
    - Backward Gradient Slope: Linear regression slope of log10(||∇W_l||_F) across layers
      (slope > +0.2 -> gradual vanishing towards input; slope < -0.2 -> gradual exploding towards input)
    - Layer Outlier Detection: Local ratio ||∇W_l||_F / (||∇W_{l-1}||_F + 1e-12)
      (ratio > 1e4 -> anomaly amplifier/bottleneck; ratio < 1e-4 -> anomaly cliff)
    """

    def __init__(
        self,
        config: Optional[APAConfig] = None,
        model: Optional[nn.Module] = None,
        log_file: Optional[str] = None,
    ):
        self.config = config or APAConfig()
        self.log_file = log_file or getattr(self.config, 'frobenius_log_file', None)
        self.model = model

        # Thresholds
        self.disparity_threshold: float = 1e5
        self.slope_vanishing_threshold: float = 0.2
        self.slope_exploding_threshold: float = -0.2
        self.ratio_outlier_high: float = 1e4
        self.ratio_outlier_low: float = 1e-4
        self.consecutive_outlier_threshold: int = 2

        # State tracking
        self.consecutive_outliers: Dict[str, int] = {}
        self.last_outliers: Dict[str, Dict[str, Any]] = {}
        self.last_evaluation: Optional[Dict[str, Any]] = None

    def collect_layer_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Collect atomic layers with trainable weights ordered by forward execution / naming."""
        layers = []
        for name, module in model.named_modules():
            # Target APALinear, APAConv2d, APAModule or native Linear/Conv2d leaf modules
            if isinstance(module, APAModule):
                layers.append((name, module))
            elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                # If module has no children (leaf module) and has weight
                if len(list(module.children())) == 0 and hasattr(module, 'weight') and module.weight is not None:
                    layers.append((name, module))
        return layers

    @torch.no_grad()
    def collect_step_metrics(
        self,
        model: nn.Module,
        step: int,
        lr: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Collect Frobenius norms of weights, gradients, and update ratios across layers.

        Args:
            model: PyTorch model.
            step: Current training step.
            lr: Optional learning rate. If None, computes raw ||∇W|| / ||W||.

        Returns:
            List of per-layer metric dictionaries.
        """
        layer_modules = self.collect_layer_modules(model)
        metrics = []

        for name, module in layer_modules:
            # 1. Resolve weight tensor
            if isinstance(module, APAModule):
                weight = module.weight_master if getattr(module, 'weight_master', None) is not None else getattr(module, 'weight_work', module.weight)
            else:
                weight = module.weight

            if weight is None:
                continue

            # 2. Resolve gradient tensor
            grad = None
            if weight.grad is not None:
                grad = weight.grad
            elif isinstance(module, APAModule) and getattr(module, 'weight_work', None) is not None and module.weight_work.grad is not None:
                grad = module.weight_work.grad

            # 3. Calculate Frobenius norms
            weight_norm = float(torch.norm(weight.float(), p='fro').item())
            grad_norm = float(torch.norm(grad.float(), p='fro').item()) if grad is not None else 0.0

            # 4. Compute update ratio
            effective_lr = float(lr) if lr is not None else 1.0
            update_ratio = (effective_lr * grad_norm) / (weight_norm + 1e-12)

            metrics.append({
                "name": name,
                "weight_norm": weight_norm,
                "grad_norm": grad_norm,
                "update_ratio": float(update_ratio),
                "level": getattr(module, 'level', None)
            })

        return metrics

    def evaluate_instability(
        self,
        step: int,
        layer_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate the 3 numerical instability indicators from collected Frobenius metrics.

        Indicators:
        1. Weight & Gradient Scale Disparity
        2. Backward Multiplicative Chain (Gradient Slope)
        3. Local Layer Outlier / Bottleneck
        """
        if not layer_metrics:
            return {
                "step": step,
                "num_layers": 0,
                "weight_disparity": 1.0,
                "grad_disparity": 1.0,
                "disparity_warning": False,
                "weight_disparity_warning": False,
                "grad_slope": 0.0,
                "slope_status": "NO_LAYERS",
                "outliers": [],
                "layer_metrics": []
            }

        L = len(layer_metrics)
        grad_norms = [m["grad_norm"] for m in layer_metrics]
        weight_norms = [m["weight_norm"] for m in layer_metrics]

        # -------------------------------------------------------------------
        # 1. Weight & Gradient Variance Disparity
        # -------------------------------------------------------------------
        max_grad = max(grad_norms) if grad_norms else 0.0
        min_grad = min(grad_norms) if grad_norms else 0.0
        grad_disparity = max_grad / (min_grad + 1e-12)
        disparity_warning = bool(grad_disparity > self.disparity_threshold)

        max_weight = max(weight_norms) if weight_norms else 0.0
        min_weight = min(weight_norms) if weight_norms else 0.0
        weight_disparity = max_weight / (min_weight + 1e-12)
        weight_disparity_warning = bool(weight_disparity > self.disparity_threshold)

        # -------------------------------------------------------------------
        # 2. Backward Multiplicative Chain (Gradient Slope)
        # Sequence ordered from input (l=1) to output (l=L)
        # -------------------------------------------------------------------
        if L >= 2 and max_grad > 0.0:
            indices = list(range(1, L + 1))
            # log10 with clamp floor at 1e-12
            log_norms = [math.log10(max(g, 1e-12)) for g in grad_norms]

            mean_l = sum(indices) / L
            mean_y = sum(log_norms) / L

            cov_ly = sum((indices[i] - mean_l) * (log_norms[i] - mean_y) for i in range(L)) / L
            var_l = sum((indices[i] - mean_l) ** 2 for i in range(L)) / L

            slope = cov_ly / (var_l + 1e-12)

            if slope > self.slope_vanishing_threshold:
                slope_status = "GRADUAL_VANISHING_GRADIENT"
            elif slope < self.slope_exploding_threshold:
                slope_status = "GRADUAL_EXPLODING_GRADIENT"
            else:
                slope_status = "STABLE"
        else:
            slope = 0.0
            slope_status = "STABLE" if L < 2 else "ZERO_GRADIENTS"

        # -------------------------------------------------------------------
        # 3. Layer Outlier Detection (Rasio_Lokal)
        # Rasio_Lokal = ||∇W_l||_F / (||∇W_{l-1}||_F + 1e-12)
        # -------------------------------------------------------------------
        outliers = []
        current_outlier_names = set()

        for i in range(1, L):
            prev_norm = layer_metrics[i - 1]["grad_norm"]
            curr_norm = layer_metrics[i]["grad_norm"]
            curr_name = layer_metrics[i]["name"]

            local_ratio = curr_norm / (prev_norm + 1e-12)
            outlier_type = None

            if local_ratio > self.ratio_outlier_high:
                outlier_type = "ANOMALY_AMPLIFIER"
            elif local_ratio < self.ratio_outlier_low:
                outlier_type = "ANOMALY_CLIFF"

            if outlier_type is not None:
                current_outlier_names.add(curr_name)
                self.consecutive_outliers[curr_name] = self.consecutive_outliers.get(curr_name, 0) + 1
                consecutive_count = self.consecutive_outliers[curr_name]

                outlier_record = {
                    "layer": curr_name,
                    "prev_layer": layer_metrics[i - 1]["name"],
                    "ratio": float(local_ratio),
                    "type": outlier_type,
                    "consecutive": consecutive_count,
                    "is_persistent": bool(consecutive_count >= self.consecutive_outlier_threshold)
                }
                outliers.append(outlier_record)
                self.last_outliers[curr_name] = outlier_record

        # Reset consecutive counts for layers that have returned to normal
        for name in list(self.consecutive_outliers.keys()):
            if name not in current_outlier_names:
                self.consecutive_outliers[name] = 0
                self.last_outliers.pop(name, None)

        evaluation_result = {
            "step": step,
            "num_layers": L,
            "weight_disparity": float(weight_disparity),
            "grad_disparity": float(grad_disparity),
            "disparity_warning": disparity_warning,
            "weight_disparity_warning": weight_disparity_warning,
            "grad_slope": float(slope),
            "slope_status": slope_status,
            "outliers": outliers,
            "layer_metrics": layer_metrics
        }

        self.last_evaluation = evaluation_result
        return evaluation_result

    def evaluate_at_initialization(self, model: nn.Module) -> Dict[str, Any]:
        """Record initial weight distribution and scale disparity prior to training (step 0)."""
        layer_modules = self.collect_layer_modules(model)
        weight_norms = []
        metrics = []

        with torch.no_grad():
            for name, module in layer_modules:
                if isinstance(module, APAModule):
                    weight = module.weight_master if getattr(module, 'weight_master', None) is not None else getattr(module, 'weight_work', module.weight)
                else:
                    weight = module.weight

                if weight is not None:
                    w_norm = float(torch.norm(weight.float(), p='fro').item())
                    weight_norms.append(w_norm)
                    metrics.append({
                        "name": name,
                        "weight_norm": w_norm,
                        "grad_norm": 0.0,
                        "update_ratio": 0.0,
                        "level": getattr(module, 'level', None)
                    })

        max_w = max(weight_norms) if weight_norms else 0.0
        min_w = min(weight_norms) if weight_norms else 0.0
        weight_disparity = max_w / (min_w + 1e-12)
        weight_disparity_warning = bool(weight_disparity > self.disparity_threshold)

        init_result = {
            "step": 0,
            "event": "frobenius_initialization",
            "num_layers": len(metrics),
            "weight_disparity": float(weight_disparity),
            "weight_disparity_warning": weight_disparity_warning,
            "grad_disparity": 0.0,
            "disparity_warning": False,
            "grad_slope": 0.0,
            "slope_status": "INITIALIZATION",
            "outliers": [],
            "layer_metrics": metrics
        }

        self.log_step(0, init_result, event_name="frobenius_initialization")
        return init_result

    def get_module_outlier_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve recent outlier information for a specific module, if active."""
        if self.consecutive_outliers.get(module_name, 0) >= self.consecutive_outlier_threshold:
            return self.last_outliers.get(module_name)
        return None

    def log_step(
        self,
        step: int,
        eval_result: Dict[str, Any],
        event_name: str = "frobenius_telemetry"
    ) -> None:
        """Append structured Frobenius telemetry evaluation to JSONL log file."""
        if not self.log_file:
            return

        payload = {
            "event": event_name,
            "step": step,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "weight_disparity": eval_result.get("weight_disparity", 1.0),
            "grad_disparity": eval_result.get("grad_disparity", 1.0),
            "disparity_warning": eval_result.get("disparity_warning", False),
            "grad_slope": eval_result.get("grad_slope", 0.0),
            "slope_status": eval_result.get("slope_status", "STABLE"),
            "outliers_count": len(eval_result.get("outliers", [])),
            "outliers": eval_result.get("outliers", []),
            "layers": [
                {
                    "name": lm["name"],
                    "weight_norm": round(lm["weight_norm"], 6),
                    "grad_norm": round(lm["grad_norm"], 6),
                    "update_ratio": round(lm["update_ratio"], 8),
                    "level": lm.get("level")
                }
                for lm in eval_result.get("layer_metrics", [])
            ]
        }

        try:
            log_dir = os.path.dirname(os.path.abspath(self.log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"[FrobeniusTelemetry] Failed to write log to {self.log_file}: {e}")
