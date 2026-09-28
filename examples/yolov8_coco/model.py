import sys
import os
import torch
import torch.nn as nn
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APAConv2d, APALinear, APAConfig
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError:
    YOLO = None
    DetectionModel = None

def convert_yolov8_to_apa(
    model: nn.Module,
    config: APAConfig = APAConfig(),
    preserve_critical_layers: bool = True
) -> nn.Module:
    """Convert standard YOLOv8 convolutional layers to APAConv2d layers.

    Args:
        model: A YOLOv8 DetectionModel or nn.Module containing YOLOv8 layers.
        config: APAConfig configuration for precision escalation and telemetry.
        preserve_critical_layers: If True, keep the final Detect head in LEVEL_TF32
            to protect non-linear box coordinates (DFL / CIoU) and class classification
            from numerical underflow/divergence while running the Backbone & Neck in FP8.
            If False, all convolutions including Detect are converted to FP8.

    Returns:
        The modified model with APAConv2d layers.
    """
    detect_idx = len(model.model) - 1 if hasattr(model, 'model') else -1

    def _replace_convs(module: nn.Module, is_critical: bool):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                init_lvl = LEVEL_TF32 if is_critical else LEVEL_FP8
                apa_conv = APAConv2d.from_conv2d(child, config=config, initial_level=init_lvl)
                setattr(module, child_name, apa_conv)
            elif isinstance(child, nn.Linear):
                init_lvl = LEVEL_TF32 if is_critical else LEVEL_FP8
                apa_lin = APALinear.from_linear(child, config=config, initial_level=init_lvl)
                setattr(module, child_name, apa_lin)
            else:
                _replace_convs(child, is_critical=is_critical)

    if hasattr(model, 'model') and isinstance(model.model, (nn.Sequential, nn.ModuleList)):
        for i, layer in enumerate(model.model):
            is_detect = (i == detect_idx)
            is_critical = preserve_critical_layers and is_detect
            _replace_convs(layer, is_critical=is_critical)
    else:
        _replace_convs(model, is_critical=False)

    return model

def create_yolov8_apa(
    model_name_or_cfg: str = 'yolov8n.yaml',
    config: Optional[APAConfig] = None,
    preserve_critical_layers: bool = True,
    device: str = 'cuda'
):
    """Load or build a YOLOv8 model and convert it to APA.

    Args:
        model_name_or_cfg: Path to yaml config (e.g. 'yolov8n.yaml') or checkpoint (e.g. 'yolov8n.pt').
        config: APAConfig configuration. If None, uses APAConfig.research_default().
        preserve_critical_layers: Whether to preserve Detect head in LEVEL_TF32.
        device: Device to place the model on ('cuda' or 'cpu').

    Returns:
        (yolo_wrapper, inner_model)
    """
    if YOLO is None:
        raise ImportError("ultralytics is required to use create_yolov8_apa. Run: pip install ultralytics")

    if config is None:
        config = APAConfig.research_default()
    config.device = device

    yolo = YOLO(model_name_or_cfg)
    model = yolo.model.to(device)
    convert_yolov8_to_apa(model, config=config, preserve_critical_layers=preserve_critical_layers)
    return yolo, model
