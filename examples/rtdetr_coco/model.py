import sys
import os
import torch
import torch.nn as nn
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig
from apa.config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

try:
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr import RTDETRDetectionModel
except ImportError:
    RTDETR = None
    RTDETRDetectionModel = None

def convert_rtdetr_to_apa(
    model: nn.Module,
    config: APAConfig = APAConfig(),
    preserve_critical_layers: bool = True
) -> nn.Module:
    """Convert Linear layers in RT-DETR (DINO-DETR) Transformer to APALinear layers.

    Args:
        model: An RTDETRDetectionModel or nn.Module containing RT-DETR layers.
        config: APAConfig configuration for precision escalation and telemetry.
        preserve_critical_layers: If True, keep the final scoring and bbox prediction
            heads in LEVEL_TF32 to protect Hungarian matcher and loss calculation
            from divergence, while running all Multi-Head Attention and MLP FFN layers
            in native FP8 Tensor Cores.

    Returns:
        The modified model with APALinear layers.
    """
    critical_subnames = ('score_head', 'bbox_head', 'denoising_class')

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            continue
        is_critical = preserve_critical_layers and any(crit in name for crit in critical_subnames)
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                default_lvl = config.freeze_level if config.freeze_level is not None else LEVEL_FP8
                init_lvl = LEVEL_TF32 if is_critical else default_lvl
                apa_lin = APALinear.from_linear(child, config=config, initial_level=init_lvl)
                setattr(module, child_name, apa_lin)

    return model

def create_rtdetr_apa(
    model_name_or_cfg: str = 'rtdetr-l.yaml',
    config: Optional[APAConfig] = None,
    preserve_critical_layers: bool = True,
    device: str = 'cuda'
):
    """Load or build an RT-DETR (DINO-DETR) model and convert it to APA."""
    if RTDETR is None:
        raise ImportError("ultralytics is required to use RT-DETR. Run: pip install ultralytics")

    if config is None:
        config = APAConfig.research_default()
    config.device = device
    config.fp8_output_dtype = 'float32'

    rtdetr = RTDETR(model_name_or_cfg)
    model = rtdetr.model.to(device)
    convert_rtdetr_to_apa(model, config=config, preserve_critical_layers=preserve_critical_layers)
    return rtdetr, model
